"""
Phantom Pharmacology - Streaming Data Processing Pipeline
==========================================================
Handles arbitrarily large FAERS datasets by streaming one quarter at a time.
Only count matrices are held in memory — never all reports simultaneously.

Architecture
------------
  Streaming path (default):
    Pass 1 — vocabulary  : stream all quarters, count drug/reaction/pair frequencies
    Pass 2 — accumulate  : stream again, fill count matrices + graph edge counts
    Compute              : ROR/IC entirely from matrices — no DataFrame needed

  Legacy path (if reports.parquet exists from a previous run):
    Loads parquet and runs the old path — backward compatible.

Usage:
    python pipeline.py                  # normal run
    python pipeline.py --clear-cache    # rerun signals/graph/model (keep counts)
    python pipeline.py --clear-all      # full reset

Why two passes?
    Pass 1 determines the top-N drugs/reactions/pairs so we know what size
    to allocate the count matrices in Pass 2. Trying to do it in one pass
    would require storing all reports in memory first.
"""

import argparse
import json
import math
import itertools
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FAERS_DATA_DIR  = Path("Phantom Data")
CACHE_DIR       = Path("faers_cache")
CACHE_DIR.mkdir(exist_ok=True)

REPORTS_FILE    = CACHE_DIR / "reports.parquet"     # legacy
COUNTS_FILE     = CACHE_DIR / "streaming_counts.pkl" # new streaming cache
ROR_FILE        = CACHE_DIR / "ror_signals.parquet"
GRAPH_FILE      = CACHE_DIR / "graph_data.pkl"
MODEL_FILE      = CACHE_DIR / "gnn_model.pkl"
DRUGS_FILE      = CACHE_DIR / "all_drugs.json"

MAX_QUARTERS      = 0    # 0 = all quarters; set to e.g. 4 to limit
TOP_DRUGS         = 500  # for graph node set
ROR_TOP_PAIRS     = 200
ROR_TOP_REACTIONS = 50
ROR_MIN_CASES     = 3
ROR_LOWER_CI      = 1.0

DANGEROUS_REACTIONS = {
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}


# ─────────────────────────────────────────────────────────────────────────────
# FAERS QUARTER LOADING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def find_faers_folders():
    quarters = []
    if not FAERS_DATA_DIR.exists():
        return quarters
    for folder in sorted(FAERS_DATA_DIR.iterdir()):
        if folder.is_dir() and folder.name.startswith("faers_ascii_"):
            ascii_dir = folder / "ASCII"
            if ascii_dir.exists():
                quarters.append(ascii_dir)
    return quarters


def load_faers_quarter(ascii_dir: Path):
    """Load DRUG and REAC files from one FAERS quarter. Returns DataFrame or None."""
    drug_file = reac_file = None
    for f in ascii_dir.iterdir():
        fname = f.name.upper()
        if fname.startswith("DRUG") and fname.endswith(".TXT"):
            drug_file = f
        elif fname.startswith("REAC") and fname.endswith(".TXT"):
            reac_file = f

    if not drug_file or not reac_file:
        return None

    drug_df = pd.read_csv(drug_file, sep="$", dtype=str, low_memory=False,
                          usecols=["primaryid", "drugname", "role_cod"])
    drug_df = drug_df[drug_df["role_cod"].isin(["PS", "SS"])]
    drug_df["drugname"] = drug_df["drugname"].str.lower().str.strip()
    drug_df = drug_df[["primaryid", "drugname"]].dropna()

    reac_df = pd.read_csv(reac_file, sep="$", dtype=str, low_memory=False,
                          usecols=["primaryid", "pt"])
    reac_df["pt"] = reac_df["pt"].str.lower().str.strip()
    reac_df = reac_df.dropna()

    drugs_g = drug_df.groupby("primaryid")["drugname"].apply(list).reset_index()
    drugs_g.columns = ["primaryid", "drugs"]
    del drug_df

    reacs_g = reac_df.groupby("primaryid")["pt"].apply(list).reset_index()
    reacs_g.columns = ["primaryid", "reactions"]
    del reac_df

    result = drugs_g.merge(reacs_g, on="primaryid", how="inner")
    del drugs_g, reacs_g

    result = result[result["drugs"].apply(len) >= 2]
    result = result[result["reactions"].apply(len) >= 1]
    return result


def _get_quarters():
    quarters = find_faers_folders()
    if not quarters:
        raise FileNotFoundError("No FAERS data found in 'Phantom Data' folder")
    if MAX_QUARTERS > 0 and len(quarters) > MAX_QUARTERS:
        quarters = quarters[-MAX_QUARTERS:]
        print(f"    Limiting to {MAX_QUARTERS} most recent quarters")
    return quarters


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING PASS 1 — VOCABULARY
# Streams all quarters once, counting frequencies only.
# Memory: O(unique drugs + reactions) — a few MB regardless of dataset size.
# ─────────────────────────────────────────────────────────────────────────────

def _stream_pass1_vocab(quarters: list) -> tuple:
    """
    Returns (n_reports, drug_count, rxn_count, pair_count).

    Note: seen_ids is rebuilt independently in each pass so that both passes
    process exactly the same deduplicated set of reports without needing to
    hold the full ID set from pass 1 in memory during pass 2.
    """
    drug_count = defaultdict(int)
    rxn_count  = defaultdict(int)
    pair_count = defaultdict(int)
    seen_ids   = set()   # int64 primaryids — ~8 bytes each, 4x smaller than strings
    n_reports  = 0

    for ascii_dir in tqdm(quarters, desc="  Pass 1/2 — vocabulary"):
        df = load_faers_quarter(ascii_dir)
        if df is None or df.empty:
            continue

        # Convert primaryid to int for compact deduplication set
        df["pid"] = pd.to_numeric(df["primaryid"], errors="coerce")
        df = df.dropna(subset=["pid"])
        df["pid"] = df["pid"].astype(np.int64)
        df = df[~df["pid"].isin(seen_ids)]
        seen_ids.update(df["pid"].tolist())

        for drugs, rxns in zip(df["drugs"], df["reactions"]):
            drugs = list(set(drugs))   # deduplicate within report
            rxns  = list(set(rxns))
            n_reports += 1
            for d in drugs:
                drug_count[d] += 1
            for r in rxns:
                rxn_count[r] += 1
            for pair in itertools.combinations(sorted(drugs), 2):
                pair_count[frozenset(pair)] += 1

        del df

    return n_reports, drug_count, rxn_count, pair_count


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING PASS 2 — COUNT ACCUMULATION
# Streams all quarters again, filling fixed-size count matrices.
# Memory: O(n_top_drugs × n_top_reactions + n_top_pairs × n_top_reactions)
#         ≈ 500×50 + 200×50 = 35,000 integers — completely negligible.
# ─────────────────────────────────────────────────────────────────────────────

def _stream_pass2_accumulate(quarters: list,
                              drug_to_idx: dict, rxn_to_idx: dict,
                              pair_to_idx: dict, all_drug_to_idx: dict) -> tuple:
    """
    Returns (N, drug_totals, rxn_totals, drug_rxn_co,
             pair_totals, pair_rxn_co, all_drug_totals, edge_counter).

    Count matrix semantics
    ----------------------
    drug_totals[d]      : reports containing drug d (top-pair drugs only)
    rxn_totals[r]       : reports containing reaction r (top reactions only)
    drug_rxn_co[d, r]   : reports containing drug d AND reaction r
    pair_totals[p]      : reports containing both drugs in pair p
    pair_rxn_co[p, r]   : reports containing both drugs in pair p AND reaction r
    all_drug_totals[d]  : reports containing drug d (ALL drugs, for GNN features)
    edge_counter[(i,j)] : co-occurrence count for graph edges (all drugs)
    """
    nd  = len(drug_to_idx)
    nr  = len(rxn_to_idx)
    np_ = len(pair_to_idx)
    ng  = len(all_drug_to_idx)

    drug_totals     = np.zeros(nd,         dtype=np.int32)
    rxn_totals      = np.zeros(nr,         dtype=np.int32)
    drug_rxn_co     = np.zeros((nd, nr),   dtype=np.int32)
    pair_totals     = np.zeros(np_,        dtype=np.int32)
    pair_rxn_co     = np.zeros((np_, nr),  dtype=np.int32)
    all_drug_totals = np.zeros(ng,         dtype=np.int32)
    edge_counter    = defaultdict(int)
    seen_ids        = set()  # rebuilt independently — see pass 1 docstring
    N               = 0

    for ascii_dir in tqdm(quarters, desc="  Pass 2/2 — accumulate"):
        df = load_faers_quarter(ascii_dir)
        if df is None or df.empty:
            continue

        df["pid"] = pd.to_numeric(df["primaryid"], errors="coerce")
        df = df.dropna(subset=["pid"])
        df["pid"] = df["pid"].astype(np.int64)
        df = df[~df["pid"].isin(seen_ids)]
        seen_ids.update(df["pid"].tolist())

        for drugs, rxns in zip(df["drugs"], df["reactions"]):
            drug_set = set(drugs)
            rxn_set  = set(rxns)

            # ── Indices into vocabularies ─────────────────────────────────────
            d_top  = [(d, drug_to_idx[d])     for d in drug_set if d in drug_to_idx]
            r_idxs = [rxn_to_idx[r]           for r in rxn_set  if r in rxn_to_idx]
            d_all  = [all_drug_to_idx[d]      for d in drug_set if d in all_drug_to_idx]

            N += 1

            # ── All-drug totals (GNN node features) ───────────────────────────
            for gi in d_all:
                all_drug_totals[gi] += 1

            # ── Drug marginals + drug-rxn co-occurrence ───────────────────────
            for d_name, di in d_top:
                drug_totals[di] += 1
                for ri in r_idxs:
                    drug_rxn_co[di, ri] += 1

            # ── Reaction marginals ────────────────────────────────────────────
            for ri in r_idxs:
                rxn_totals[ri] += 1

            # ── Pair counts + pair-rxn co-occurrence ──────────────────────────
            d_top_sorted = sorted(d_top, key=lambda x: x[0])
            for (da, ia), (db, ib) in itertools.combinations(d_top_sorted, 2):
                pkey = frozenset([da, db])
                if pkey in pair_to_idx:
                    pi = pair_to_idx[pkey]
                    pair_totals[pi] += 1
                    for ri in r_idxs:
                        pair_rxn_co[pi, ri] += 1

            # ── Graph edges (all drugs, not just top) ─────────────────────────
            d_all_sorted = sorted(d_all)
            for ga, gb in itertools.combinations(d_all_sorted, 2):
                edge_counter[(ga, gb)] += 1

        del df

    return (N, drug_totals, rxn_totals, drug_rxn_co,
            pair_totals, pair_rxn_co, all_drug_totals, edge_counter)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING COUNTS CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

class StreamingCounts:
    """All count matrices and vocabulary indices from the two streaming passes."""
    __slots__ = [
        "N", "drug_totals", "rxn_totals", "drug_rxn_co",
        "pair_totals", "pair_rxn_co", "all_drug_totals", "edge_counter",
        "all_drugs", "top_pair_drugs", "top_reactions", "top_pairs",
        "drug_to_idx", "rxn_to_idx", "pair_to_idx", "all_drug_to_idx",
    ]
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD OR BUILD STREAMING COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_counts() -> StreamingCounts:
    """
    Build StreamingCounts via two streaming passes, or load from cache.
    This replaces load_all_faers_data() — no large DataFrame is ever created.

    If reports.parquet exists from a previous run (legacy), that path is
    NOT used here — delete it with --clear-all if you want a fresh start.
    """
    if COUNTS_FILE.exists():
        print(f"[1/5] Loading cached streaming counts from {COUNTS_FILE}")
        with open(COUNTS_FILE, "rb") as f:
            return pickle.load(f)

    quarters = _get_quarters()
    print(f"[1/5] Streaming {len(quarters)} quarters (two passes, one at a time)...")

    # ── Pass 1: vocabulary ────────────────────────────────────────────────────
    n_reports, drug_count, rxn_count, pair_count = _stream_pass1_vocab(quarters)
    print(f"    Pass 1 complete: {n_reports:,} reports | "
          f"{len(drug_count):,} drugs | {len(rxn_count):,} reactions")

    # Determine top-N subsets
    top_pairs     = sorted(pair_count, key=pair_count.get, reverse=True)[:ROR_TOP_PAIRS]
    top_reactions = [r for r, _ in
                     sorted(rxn_count.items(), key=lambda x: -x[1])[:ROR_TOP_REACTIONS]]
    # ROR solo table only needs drugs that appear in top pairs
    top_pair_drugs = sorted({d for pair in top_pairs for d in pair})
    # Full drug list for graph and UI
    all_drugs      = sorted(drug_count.keys())

    # Build index mappings
    drug_to_idx     = {d: i for i, d in enumerate(top_pair_drugs)}
    rxn_to_idx      = {r: i for i, r in enumerate(top_reactions)}
    pair_to_idx     = {p: i for i, p in enumerate(top_pairs)}
    all_drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

    # ── Pass 2: accumulate ────────────────────────────────────────────────────
    (N, drug_totals, rxn_totals, drug_rxn_co,
     pair_totals, pair_rxn_co, all_drug_totals,
     edge_counter) = _stream_pass2_accumulate(
        quarters, drug_to_idx, rxn_to_idx, pair_to_idx, all_drug_to_idx
    )
    print(f"    Pass 2 complete: N={N:,} deduplicated reports")

    # Pack
    sc = StreamingCounts()
    sc.N               = N
    sc.drug_totals     = drug_totals
    sc.rxn_totals      = rxn_totals
    sc.drug_rxn_co     = drug_rxn_co
    sc.pair_totals     = pair_totals
    sc.pair_rxn_co     = pair_rxn_co
    sc.all_drug_totals = all_drug_totals
    sc.edge_counter    = edge_counter
    sc.all_drugs       = all_drugs
    sc.top_pair_drugs  = top_pair_drugs
    sc.top_reactions   = top_reactions
    sc.top_pairs       = top_pairs
    sc.drug_to_idx     = drug_to_idx
    sc.rxn_to_idx      = rxn_to_idx
    sc.pair_to_idx     = pair_to_idx
    sc.all_drug_to_idx = all_drug_to_idx

    with open(COUNTS_FILE, "wb") as f:
        pickle.dump(sc, f)
    print(f"[1/5] Saved streaming counts to {COUNTS_FILE}")

    return sc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_graph(sc: StreamingCounts):
    if GRAPH_FILE.exists():
        print(f"[2/5] Loading cached graph from {GRAPH_FILE}")
        with open(GRAPH_FILE, "rb") as f:
            return pickle.load(f)

    print(f"[2/5] Building drug graph ({len(sc.all_drugs):,} drugs)...")

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("[2/5] torch-geometric not installed — skipping graph")
        return None

    all_drugs    = sc.all_drugs
    drug_to_idx  = sc.all_drug_to_idx
    edge_counter = sc.edge_counter

    edges        = list(edge_counter.keys())
    edge_weights = [edge_counter[e] for e in edges]

    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])

    # Node features: log-frequency from all_drug_totals
    x = torch.tensor(
        [[math.log1p(int(sc.all_drug_totals[i]))] for i in range(len(all_drugs))],
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index)
    data.drug_to_idx = drug_to_idx
    data.idx_to_drug = {v: k for k, v in drug_to_idx.items()}
    data.edge_weights = edge_weights

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[2/5] Saved graph to {GRAPH_FILE}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN GNN MODEL  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_model(graph_data):
    if MODEL_FILE.exists():
        print(f"[3/5] Loading cached model from {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    if graph_data is None:
        print("[3/5] No graph data — skipping model training")
        return None

    print("[3/5] Training GNN model...")

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
    except ImportError:
        print("[3/5] torch-geometric not installed — skipping")
        return None

    in_ch = graph_data.x.shape[1]

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_ch, 64)
            self.conv2 = GCNConv(64, 32)
            self.head  = torch.nn.Linear(64, 1)

        def encode(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

        def decode(self, z, src, dst):
            return torch.sigmoid(self.head(torch.cat([z[src], z[dst]], dim=-1)))

    model = GCN()
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)
    crit  = torch.nn.BCELoss()

    src = graph_data.edge_index[0]
    dst = graph_data.edge_index[1]
    neg_src = torch.randint(0, graph_data.x.shape[0], (src.shape[0],))
    neg_dst = torch.randint(0, graph_data.x.shape[0], (src.shape[0],))

    all_src    = torch.cat([src, neg_src])
    all_dst    = torch.cat([dst, neg_dst])
    all_labels = torch.cat([torch.ones(src.shape[0]), torch.zeros(neg_src.shape[0])])

    for epoch in tqdm(range(50), desc="Training GNN"):
        model.train()
        opt.zero_grad()
        z    = model.encode(graph_data.x, graph_data.edge_index)
        pred = model.decode(z, all_src, all_dst).squeeze()
        loss = crit(pred, all_labels)
        loss.backward()
        opt.step()

    model_data = {
        "state_dict":  model.state_dict(),
        "drug_to_idx": graph_data.drug_to_idx,
        "idx_to_drug": graph_data.idx_to_drug,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[3/5] Saved model to {MODEL_FILE}")
    return model_data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE BIAS-ADJUSTED ROR FROM COUNT MATRICES
#
# Key insight: every contingency cell (a, b, c, d) for any (pair, reaction)
# can be derived algebraically from the accumulated count matrices.
# No per-report iteration is needed at this stage.
#
# Derivations
# -----------
# For solo ROR of drug d against reaction r:
#   a = drug_rxn_co[d, r]
#   b = drug_totals[d] - a
#   c = rxn_totals[r]  - a
#   d = N - drug_totals[d] - rxn_totals[r] + a
#
# For pair (da, db) against reaction r:
#   a = pair_rxn_co[p, r]
#   b = pair_totals[p] - a
#   c = drug_rxn_co[da,r] + drug_rxn_co[db,r] - 2*pair_rxn_co[p,r]
#   d = (drug_totals[da] + drug_totals[db] - 2*pair_totals[p]) - c
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic(observed: int, expected: float,
                alpha: float = 0.5, beta: float = 0.5) -> tuple:
    """
    Information Component with empirical Bayes shrinkage (Noren 2013).
    IC025 > 0 is the WHO Uppsala signal criterion.
    High-frequency drugs get high 'expected', shrinking IC toward 0 unless
    the pair-reaction signal genuinely exceeds the base rate.
    """
    obs_s = observed + alpha
    exp_s = expected + beta
    ic    = math.log2(obs_s / exp_s)
    var   = (1 / (obs_s * math.log(2) ** 2)) * (1 - obs_s / (exp_s + obs_s))
    se    = math.sqrt(max(var, 0.0))
    return ic, ic - 1.96 * se, ic + 1.96 * se


def compute_ror_from_counts(sc: StreamingCounts) -> pd.DataFrame:
    """
    Compute bias-adjusted ROR/IC/combo_lift from count matrices.
    Never reads the raw reports — entirely matrix arithmetic.
    """
    if ROR_FILE.exists():
        print(f"[4/5] Loading cached ROR signals from {ROR_FILE}")
        return pd.read_parquet(ROR_FILE)

    print("[4/5] Computing bias-adjusted ROR from count matrices...")

    N             = sc.N
    drug_totals   = sc.drug_totals.astype(np.float64)
    rxn_totals    = sc.rxn_totals.astype(np.float64)
    drug_rxn_co   = sc.drug_rxn_co.astype(np.float64)
    pair_totals   = sc.pair_totals
    pair_rxn_co   = sc.pair_rxn_co.astype(np.float64)
    drug_to_idx   = sc.drug_to_idx
    top_pairs     = sc.top_pairs
    top_reactions = sc.top_reactions

    # ── Solo ROR matrix — fully vectorized, computed once ────────────────────
    # Shape: (n_top_pair_drugs, n_top_reactions)
    print("    Building solo ROR matrix (vectorized)...")
    A_s = drug_rxn_co                                              # (nd, nr)
    B_s = drug_totals[:, None] - A_s                              # (nd, nr)
    C_s = rxn_totals[None, :]  - A_s                              # (nd, nr)
    D_s = N - drug_totals[:, None] - rxn_totals[None, :] + A_s   # (nd, nr)

    valid_solo = (A_s >= ROR_MIN_CASES) & (B_s > 0) & (C_s > 0) & (D_s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        solo_ror_mat = np.where(valid_solo, (A_s * D_s) / (B_s * C_s), np.nan)

    # ── Pair screening ────────────────────────────────────────────────────────
    print(f"    Screening {len(top_pairs)} pairs x {len(top_reactions)} reactions...")
    results = []

    for pi, pair in enumerate(tqdm(top_pairs, desc="Pair ROR")):
        drugs = list(pair)
        if len(drugs) < 2:
            continue
        da, db = drugs[0], drugs[1]
        if da not in drug_to_idx or db not in drug_to_idx:
            continue

        ia     = drug_to_idx[da]
        ib     = drug_to_idx[db]
        n_pair = int(pair_totals[pi])
        if n_pair == 0:
            continue

        # Contingency cells for all reactions at once
        A = pair_rxn_co[pi]                                          # (nr,)
        B = n_pair - A                                               # (nr,)
        C = drug_rxn_co[ia] + drug_rxn_co[ib] - 2.0 * pair_rxn_co[pi]  # (nr,)
        D = (drug_totals[ia] + drug_totals[ib] - 2.0 * n_pair) - C  # (nr,)

        for rj, rxn in enumerate(top_reactions):
            aj, bj, cj, dj = float(A[rj]), float(B[rj]), float(C[rj]), float(D[rj])

            # ── Raw ROR ───────────────────────────────────────────────────────
            if aj < ROR_MIN_CASES or bj <= 0 or cj <= 0 or dj <= 0:
                continue

            ror   = (aj * dj) / (bj * cj)
            se    = math.sqrt(1/aj + 1/bj + 1/cj + 1/dj)
            lower = math.exp(math.log(ror) - 1.96 * se)
            upper = math.exp(math.log(ror) + 1.96 * se)

            if ror <= 2.0 or lower <= ROR_LOWER_CI:
                continue

            # ── IC with empirical Bayes shrinkage ─────────────────────────────
            n_rxn = float(rxn_totals[rj])
            exp   = (n_pair * n_rxn / N) if N > 0 else 0.0
            if exp == 0.0:
                continue

            ic, ic025, ic975 = _compute_ic(int(aj), exp)

            # ── Combo lift ────────────────────────────────────────────────────
            sa_val = solo_ror_mat[ia, rj]
            sb_val = solo_ror_mat[ib, rj]
            solo_a = None if np.isnan(sa_val) else float(sa_val)
            solo_b = None if np.isnan(sb_val) else float(sb_val)

            if solo_a is not None and solo_b is not None:
                max_solo   = max(solo_a, solo_b)
                combo_lift = ror / max_solo if max_solo > 0 else None
            elif solo_a is not None:
                combo_lift = ror / solo_a   if solo_a > 0 else None
            elif solo_b is not None:
                combo_lift = ror / solo_b   if solo_b > 0 else None
            else:
                combo_lift = None  # no solo signal — most novel case

            results.append({
                "drug_a":       da,
                "drug_b":       db,
                "reaction":     rxn,
                "ror":          round(ror,   2),
                "ci_lower":     round(lower, 2),
                "ci_upper":     round(upper, 2),
                "n_cases":      int(aj),
                "ic":           round(ic,    3),
                "ic025":        round(ic025, 3),
                "ic975":        round(ic975, 3),
                "combo_lift":   round(combo_lift, 3) if combo_lift is not None else None,
                "solo_ror_a":   round(solo_a,  2)   if solo_a    is not None else None,
                "solo_ror_b":   round(solo_b,  2)   if solo_b    is not None else None,
                "is_dangerous": rxn in DANGEROUS_REACTIONS,
            })

    ror_df = pd.DataFrame(results)

    if not ror_df.empty:
        n_before = len(ror_df)
        # IC025 > 0: WHO Uppsala shrinkage criterion
        ror_df = ror_df[ror_df["ic025"] > 0]
        # combo_lift >= 0.8 OR null: filter signals explained by one drug alone
        ror_df = ror_df[ror_df["combo_lift"].isna() | (ror_df["combo_lift"] >= 0.8)]
        print(f"    Bias filters: {n_before} -> {len(ror_df)} signals "
              f"({n_before - len(ror_df)} removed)")
        ror_df = ror_df.sort_values("ic025", ascending=False).reset_index(drop=True)

    ror_df.to_parquet(ROR_FILE)
    print(f"[4/5] Saved {len(ror_df)} signals to {ROR_FILE}")
    return ror_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE DRUG LIST
# ─────────────────────────────────────────────────────────────────────────────

def save_drug_list(sc: StreamingCounts):
    with open(DRUGS_FILE, "w") as f:
        json.dump(sc.all_drugs, f)
    print(f"[5/5] Saved {len(sc.all_drugs):,} drugs to {DRUGS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clear_cache(keep_counts: bool = True):
    """
    keep_counts=True  : only clear derived outputs (fast — keeps the two-pass work)
    keep_counts=False : full reset including streaming counts (re-streams everything)
    """
    targets = [ROR_FILE, GRAPH_FILE, MODEL_FILE, DRUGS_FILE]
    if not keep_counts:
        targets += [COUNTS_FILE, REPORTS_FILE]
    for f in targets:
        if f.exists():
            f.unlink()
            print(f"Deleted {f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHANTOM PHARMACOLOGY - Streaming Pipeline")
    print("=" * 60)
    print()

    # Step 1: Build/load count matrices (two streaming passes)
    sc = load_or_build_counts()
    print(f"    {sc.N:,} reports | {len(sc.all_drugs):,} unique drugs\n")

    # Step 2: Build graph
    graph_data = build_and_save_graph(sc)
    print()

    # Step 3: Train GNN
    train_and_save_model(graph_data)
    print()

    # Step 4: Compute bias-adjusted ROR from count matrices
    ror_df = compute_ror_from_counts(sc)
    print()

    # Step 5: Save drug list for UI
    save_drug_list(sc)
    print()

    print("=" * 60)
    print("  Pipeline complete! Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phantom Pharmacology Pipeline")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete ROR/graph/model cache and rerun "
                             "(keeps streaming counts — fast)")
    parser.add_argument("--clear-all", action="store_true",
                        help="Delete ALL cache including streaming counts "
                             "(re-streams all quarters from scratch)")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache(keep_counts=True)
    if args.clear_all:
        clear_cache(keep_counts=False)

    main()
