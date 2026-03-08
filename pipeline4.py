"""
Phantom Pharmacology - Streaming Data Processing Pipeline v3
=============================================================
Handles arbitrarily large FAERS datasets by streaming one quarter at a time.
Only count matrices are held in memory — never all reports simultaneously.

NEW IN v3: Added PRR (Proportional Reporting Ratio) and BCPNN (Bayesian Confidence
Propagation Neural Network) metrics alongside ROR and IC.

FIX 1 — RICHER NODE FEATURES
------------------------------
The original pipeline used a single log-frequency feature per drug node.
This means the GNN's embeddings carried no pharmacological signal — a drug
that causes many dangerous reactions looked identical to one that causes the
same number of harmless ones.

We now build a 5-dimensional feature vector per drug node:

  [0] log1p(report_count)
        — How often this drug appears in any FAERS report.
          High-frequency drugs aren't necessarily risky; this is just a
          popularity prior.

  [1] n_distinct_reactions
        — How many distinct adverse reaction types have been co-reported
          with this drug. A broad profile (many reaction types) can indicate
          a non-specific or systemic drug; a narrow profile suggests a more
          targeted one.

  [2] n_dangerous_reaction_reports
        — Total number of reports where this drug co-occurred with any
          reaction in DANGEROUS_REACTIONS. Raw count, not fraction.

  [3] fraction_of_reports_dangerous
        — n_dangerous_reaction_reports / report_count.
          Normalises for drug frequency. A rare drug with a high fraction
          is a stronger signal than a common one with the same raw count.

  [4] reaction_entropy
        — Shannon entropy of this drug's reaction distribution.
          High entropy = reactions are spread evenly across many types
            (non-specific side-effect profile, or confounding polypharmacy).
          Low entropy = one or two reactions dominate
            (mechanistically specific toxicity — more informative signal).

These features are computed entirely from the count matrices already built
in the two streaming passes — no extra data loading is required.

Architecture
------------
  Streaming path (default):
    Pass 1 — vocabulary  : stream all quarters, count drug/reaction/pair frequencies
    Pass 2 — accumulate  : stream again, fill count matrices + graph edge counts
    Compute              : ROR/IC/PRR/BCPNN entirely from matrices — no DataFrame needed

  Legacy path (if reports.parquet exists from a previous run):
    Loads parquet and runs the old path — backward compatible.

Usage:
    python pipeline3.py                  # normal run
    python pipeline3.py --clear-cache    # rerun signals/graph/model (keep counts)
    python pipeline3.py --clear-all      # full reset

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
CACHE_DIR       = Path("faers_cache_3")
CACHE_DIR.mkdir(exist_ok=True)

REPORTS_FILE    = CACHE_DIR / "reports.parquet"      # legacy
COUNTS_FILE     = CACHE_DIR / "streaming_counts.pkl" # streaming cache
ROR_FILE        = CACHE_DIR / "ror_signals.parquet"
GRAPH_FILE      = CACHE_DIR / "graph_data.pkl"
MODEL_FILE      = CACHE_DIR / "gnn_model.pkl"
DRUGS_FILE      = CACHE_DIR / "all_drugs.json"

MAX_QUARTERS      = 0    # 0 = all quarters; set to e.g. 4 to limit
TOP_DRUGS         = 500  # for graph node set
ROR_TOP_PAIRS     = 200
ROR_TOP_REACTIONS = 100  # Increased from 50 to capture more dangerous reactions
ROR_MIN_CASES     = 3
ROR_LOWER_CI      = 1.0

# Number of features per drug node in the GNN (Fix 1)
NODE_FEATURE_DIM  = 5

DANGEROUS_REACTIONS = {
    # Death / Severe outcomes (high frequency in FAERS)
    "death",
    "completed suicide", "suicide attempt",
    "coma",
    "multiple organ dysfunction syndrome",

    # Cardiac (present in FAERS)
    "cardiac arrest", "cardiac failure", "cardiac failure congestive",
    "cardiac disorder", "cardiogenic shock",
    "myocardial infarction", "cerebral infarction",
    "qt prolongation", "torsade de pointes", "ventricular fibrillation",

    # Respiratory (common in FAERS)
    "respiratory failure", "acute respiratory failure",
    "respiratory distress", "respiratory disorder",
    "pneumonia", "pneumonia aspiration",
    "pulmonary embolism",

    # Bleeding / Thrombosis (common in FAERS)
    "haemorrhage", "hemorrhage",
    "cerebral haemorrhage", "gastrointestinal haemorrhage",
    "rectal haemorrhage", "vaginal haemorrhage",
    "injection site haemorrhage", "skin haemorrhage",
    "thrombosis", "deep vein thrombosis",

    # Liver (common in FAERS)
    "hepatic failure", "liver failure",
    "drug-induced liver injury", "hepatotoxicity",
    "hepatic necrosis", "liver injury", "liver disorder",
    "hepatic cytolysis", "hepatic cirrhosis",

    # Kidney (common in FAERS)
    "renal failure", "acute kidney injury", "renal impairment",
    "renal failure acute", "renal disorder",

    # Immune / Shock (common in FAERS)
    "anaphylactic reaction", "anaphylactic shock", "anaphylaxis",
    "sepsis", "septic shock", "shock",
    "cytokine release syndrome",
    "serotonin syndrome", "stevens-johnson syndrome",

    # Neurological (common in FAERS)
    "seizure", "generalised tonic-clonic seizure", "convulsion",
    "neurotoxicity",
    "immune effector cell-associated neurotoxicity syndrome",

    # Metabolic
    "hypoglycaemia", "hypoglycemia",

    # Overdose
    "overdose", "accidental overdose", "intentional overdose",

    # Falls (important for polypharmacy in elderly)
    "fall",

    # Muscle
    "rhabdomyolysis",
}


# ─────────────────────────────────────────────────────────────────────────────
# FAERS QUARTER LOADING
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
# ─────────────────────────────────────────────────────────────────────────────

def _stream_pass1_vocab(quarters: list) -> tuple:
    drug_count = defaultdict(int)
    rxn_count  = defaultdict(int)
    pair_count = defaultdict(int)
    seen_ids   = set()
    n_reports  = 0

    for ascii_dir in tqdm(quarters, desc="  Pass 1/2 — vocabulary"):
        df = load_faers_quarter(ascii_dir)
        if df is None or df.empty:
            continue

        df["pid"] = pd.to_numeric(df["primaryid"], errors="coerce")
        df = df.dropna(subset=["pid"])
        df["pid"] = df["pid"].astype(np.int64)
        df = df[~df["pid"].isin(seen_ids)]
        seen_ids.update(df["pid"].tolist())

        for drugs, rxns in zip(df["drugs"], df["reactions"]):
            drugs = list(set(drugs))
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
# ─────────────────────────────────────────────────────────────────────────────

def _stream_pass2_accumulate(quarters: list,
                              drug_to_idx: dict, rxn_to_idx: dict,
                              pair_to_idx: dict, all_drug_to_idx: dict) -> tuple:
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
    seen_ids        = set()
    N               = 0

    # ── Fix 1: extra per-drug accumulators for richer node features ───────────
    # These are indexed by all_drug_to_idx (same as all_drug_totals).
    # We need them to build the 5-feature vector in build_and_save_graph().

    # Set of distinct reactions seen per drug (we track counts, not sets,
    # to avoid storing one set per drug in memory for very large vocabs).
    # Instead: all_drug_rxn_co[drug_idx, rxn_idx] counts co-occurrences,
    # analogous to drug_rxn_co but over ALL drugs × ALL tracked reactions.
    # We reuse rxn_to_idx (top reactions) as the reaction axis — this covers
    # the dangerous reactions and most common ones, which is all we need for
    # the feature computation.
    all_drug_rxn_co = np.zeros((ng, nr), dtype=np.int32)

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

            d_top  = [(d, drug_to_idx[d])     for d in drug_set if d in drug_to_idx]
            r_idxs = [rxn_to_idx[r]           for r in rxn_set  if r in rxn_to_idx]
            d_all  = [all_drug_to_idx[d]      for d in drug_set if d in all_drug_to_idx]

            N += 1

            # All-drug totals (existing)
            for gi in d_all:
                all_drug_totals[gi] += 1

            # ── Fix 1: accumulate all-drug × top-reaction co-occurrence ───────
            for gi in d_all:
                for ri in r_idxs:
                    all_drug_rxn_co[gi, ri] += 1

            # Drug marginals + drug-rxn co-occurrence (existing)
            for d_name, di in d_top:
                drug_totals[di] += 1
                for ri in r_idxs:
                    drug_rxn_co[di, ri] += 1

            # Reaction marginals (existing)
            for ri in r_idxs:
                rxn_totals[ri] += 1

            # Pair counts + pair-rxn co-occurrence (existing)
            d_top_sorted = sorted(d_top, key=lambda x: x[0])
            for (da, ia), (db, ib) in itertools.combinations(d_top_sorted, 2):
                pkey = frozenset([da, db])
                if pkey in pair_to_idx:
                    pi = pair_to_idx[pkey]
                    pair_totals[pi] += 1
                    for ri in r_idxs:
                        pair_rxn_co[pi, ri] += 1

            # Graph edges (existing)
            d_all_sorted = sorted(d_all)
            for ga, gb in itertools.combinations(d_all_sorted, 2):
                edge_counter[(ga, gb)] += 1

        del df

    return (N, drug_totals, rxn_totals, drug_rxn_co,
            pair_totals, pair_rxn_co, all_drug_totals,
            edge_counter, all_drug_rxn_co)


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
        # Fix 1: new accumulator
        "all_drug_rxn_co",
    ]
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD OR BUILD STREAMING COUNTS
# ─────────────────────────────────────────────────────────────────────────────

def load_or_build_counts() -> StreamingCounts:
    if COUNTS_FILE.exists():
        print(f"[1/6] Loading cached streaming counts from {COUNTS_FILE}")
        sc = pickle.load(open(COUNTS_FILE, "rb"))
        # Backwards compatibility: old cache may not have all_drug_rxn_co.
        # If missing, invalidate the cache so the user gets the richer features.
        if not hasattr(sc, "all_drug_rxn_co") or sc.all_drug_rxn_co is None:
            print("    Cache is from an older version (missing all_drug_rxn_co).")
            print("    Delete faers_cache_3/streaming_counts.pkl to rebuild with "
                  "richer node features, or run with --clear-all.")
            print("    Continuing with legacy single-feature nodes for now.")
        return sc

    quarters = _get_quarters()
    print(f"[1/6] Streaming {len(quarters)} quarters (two passes, one at a time)...")

    # Pass 1: vocabulary
    n_reports, drug_count, rxn_count, pair_count = _stream_pass1_vocab(quarters)
    print(f"    Pass 1 complete: {n_reports:,} reports | "
          f"{len(drug_count):,} drugs | {len(rxn_count):,} reactions")

    top_pairs      = sorted(pair_count, key=pair_count.get, reverse=True)[:ROR_TOP_PAIRS]
    top_reactions  = [r for r, _ in
                      sorted(rxn_count.items(), key=lambda x: -x[1])[:ROR_TOP_REACTIONS]]
    top_pair_drugs = sorted({d for pair in top_pairs for d in pair})
    all_drugs      = sorted(drug_count.keys())

    drug_to_idx     = {d: i for i, d in enumerate(top_pair_drugs)}
    rxn_to_idx      = {r: i for i, r in enumerate(top_reactions)}
    pair_to_idx     = {p: i for i, p in enumerate(top_pairs)}
    all_drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

    # Pass 2: accumulate
    (N, drug_totals, rxn_totals, drug_rxn_co,
     pair_totals, pair_rxn_co, all_drug_totals,
     edge_counter, all_drug_rxn_co) = _stream_pass2_accumulate(
        quarters, drug_to_idx, rxn_to_idx, pair_to_idx, all_drug_to_idx
    )
    print(f"    Pass 2 complete: N={N:,} deduplicated reports")

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
    sc.all_drug_rxn_co = all_drug_rxn_co

    with open(COUNTS_FILE, "wb") as f:
        pickle.dump(sc, f)
    print(f"[1/6] Saved streaming counts to {COUNTS_FILE}")
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: BUILD RICHER NODE FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_features(sc: StreamingCounts) -> np.ndarray:
    """
    Build a (n_all_drugs, NODE_FEATURE_DIM) float32 feature matrix.

    Features (all derived from count matrices — no external data needed):

      [0] log1p(report_count)
            Raw popularity prior. A drug that appears in 100,000 reports is
            very different from one in 10, independent of its risk profile.

      [1] n_distinct_reactions   (normalised by log1p(max_across_drugs))
            How many distinct reaction types have been co-reported.
            Broad reaction profile → non-specific or confounding drug.
            Narrow → possibly more mechanistically specific toxicity.
            We normalise to keep this feature on a similar scale to others.

      [2] n_dangerous_reaction_reports   (log1p-scaled)
            Raw count of reports where this drug co-occurred with any
            reaction in DANGEROUS_REACTIONS. Captures absolute severity
            exposure, before normalising for frequency.

      [3] fraction_of_reports_dangerous
            Feature [2] / report_count. Normalises for drug frequency.
            A rare drug with 0.8 fraction is a much stronger signal than a
            common one with the same raw count.

      [4] reaction_entropy   (normalised by log2(n_tracked_reactions))
            Shannon entropy of the drug's reaction distribution over the
            top-N tracked reactions.
            High  → reactions spread evenly (non-specific / polypharmacy noise)
            Low   → one or two reactions dominate (specific mechanistic toxicity)
            We normalise to [0, 1] by dividing by log2(n_reactions).

    All features are L2-normalised row-wise after construction so that the
    GNN's dot-product similarity in message passing is meaningful regardless
    of scale differences between features.
    """
    ng = len(sc.all_drugs)
    nr = sc.all_drug_rxn_co.shape[1] if sc.all_drug_rxn_co is not None else 0

    X = np.zeros((ng, NODE_FEATURE_DIM), dtype=np.float32)

    # Build set of reaction indices that are "dangerous" within our vocab
    dangerous_rxn_idxs = set()
    if sc.rxn_to_idx is not None:
        for rxn, idx in sc.rxn_to_idx.items():
            if rxn in DANGEROUS_REACTIONS:
                dangerous_rxn_idxs.add(idx)

    # ── Feature 0: log1p(report_count) ───────────────────────────────────────
    X[:, 0] = np.log1p(sc.all_drug_totals.astype(np.float32))

    if sc.all_drug_rxn_co is not None and nr > 0:
        co = sc.all_drug_rxn_co.astype(np.float32)  # (ng, nr)

        # ── Feature 1: n_distinct_reactions (normalised) ──────────────────────
        # "Distinct" = at least one co-report.
        n_distinct = (co > 0).sum(axis=1).astype(np.float32)          # (ng,)
        max_distinct = max(n_distinct.max(), 1.0)
        X[:, 1] = np.log1p(n_distinct) / np.log1p(max_distinct)

        # ── Feature 2: n_dangerous_reaction_reports (log1p-scaled) ───────────
        if dangerous_rxn_idxs:
            dang_cols = np.array(sorted(dangerous_rxn_idxs), dtype=np.int64)
            n_dang = co[:, dang_cols].sum(axis=1)                      # (ng,)
        else:
            n_dang = np.zeros(ng, dtype=np.float32)
        X[:, 2] = np.log1p(n_dang)

        # ── Feature 3: fraction_of_reports_dangerous ─────────────────────────
        report_counts = sc.all_drug_totals.astype(np.float32)
        safe_counts   = np.where(report_counts > 0, report_counts, 1.0)
        X[:, 3] = (n_dang / safe_counts).astype(np.float32)

        # ── Feature 4: reaction entropy (normalised Shannon) ─────────────────
        # p[i, r] = co[i, r] / sum_r(co[i, r])  (reaction probability dist)
        row_sums = co.sum(axis=1, keepdims=True)                       # (ng, 1)
        # Avoid /0 for drugs with no tracked reactions
        row_sums_safe = np.where(row_sums > 0, row_sums, 1.0)
        p = co / row_sums_safe                                         # (ng, nr)
        # Shannon entropy: -sum(p * log2(p)), ignoring zeros
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p > 0, np.log2(p), 0.0)
        entropy = -(p * log_p).sum(axis=1)                             # (ng,)
        # Normalise by maximum possible entropy = log2(nr)
        max_entropy = math.log2(nr) if nr > 1 else 1.0
        X[:, 4] = (entropy / max_entropy).astype(np.float32)

    else:
        # Old cache without all_drug_rxn_co — fall back to single feature,
        # leave features 1-4 as zeros. Graph will still train but without
        # the pharmacological signal. User is warned in load_or_build_counts.
        pass

    # ── L2 row-normalisation ─────────────────────────────────────────────────
    # Prevents features with large scale (log counts) from dominating
    # dot-product similarity in GNN message passing.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    X = X / norms

    return X


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_graph(sc: StreamingCounts):
    if GRAPH_FILE.exists():
        print(f"[2/6] Loading cached graph from {GRAPH_FILE}")
        with open(GRAPH_FILE, "rb") as f:
            return pickle.load(f)

    print(f"[2/6] Building drug graph ({len(sc.all_drugs):,} drugs)...")
    print(f"      Node feature dimensions: {NODE_FEATURE_DIM} "
          f"(Fix 1: richer pharmacological features)")

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("[2/6] torch-geometric not installed — skipping graph")
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

    # ── Fix 1: use richer 5-dimensional feature matrix ───────────────────────
    # Previously: x = [[log1p(count)] for each drug]  — shape (n, 1)
    # Now:        x = _build_node_features(sc)        — shape (n, 5)
    #
    # The five features give the GNN meaningful pharmacological signal to
    # distinguish drugs by their adverse event profile, not just frequency.
    print("      Computing 5-dimensional node feature matrix...")
    X_np = _build_node_features(sc)
    x    = torch.tensor(X_np, dtype=torch.float)
    print(f"      Feature matrix shape: {tuple(x.shape)}")
    print(f"      Feature column means: {X_np.mean(axis=0).round(4).tolist()}")
    print(f"      Feature column stds:  {X_np.std(axis=0).round(4).tolist()}")

    data = Data(x=x, edge_index=edge_index)
    data.drug_to_idx  = drug_to_idx
    data.idx_to_drug  = {v: k for k, v in drug_to_idx.items()}
    data.edge_weights = edge_weights

    # Store feature names alongside the data for interpretability
    data.feature_names = [
        "log1p_report_count",
        "n_distinct_reactions_norm",
        "n_dangerous_reports_log",
        "fraction_dangerous",
        "reaction_entropy_norm",
    ]

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"[2/6] Saved graph to {GRAPH_FILE}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN GNN MODEL (Risk-Based Training)
# ─────────────────────────────────────────────────────────────────────────────
#
# FIX 2 — RISK-BASED TRAINING LABELS
# -----------------------------------
# The original GNN was trained as a link-prediction model: it learned to
# predict whether two drugs co-occur in FAERS reports. This meant that
# commonly co-prescribed drugs (like ibuprofen + metformin) got high scores
# even if they're perfectly safe together.
#
# We now train the GNN to predict INTERACTION RISK using ROR signals:
#
#   Positive samples:
#       Drug pairs with elevated ROR signals (especially dangerous reactions).
#       Labels are continuous risk scores derived from ROR magnitude + danger.
#
#   Negative samples:
#       Drug pairs that co-occur frequently in FAERS but have NO elevated ROR.
#       These are "confirmed safe" combinations — frequently co-prescribed
#       without generating adverse event signals.
#
#   Weighted loss:
#       Each sample is weighted by log1p(n_cases) so high-confidence signals
#       (many reports) have more influence than low-confidence ones.
#
# This lets the GNN learn patterns like:
#   "Drugs with high dangerous-reaction fractions that are combined with
#    other high-entropy drugs tend to produce interaction signals."
#
# For drug pairs with LIMITED data, the GNN generalizes by comparing their
# pharmacological profiles (node features) to drugs with KNOWN interactions.
# ─────────────────────────────────────────────────────────────────────────────

def _compute_risk_score(ror: float, is_dangerous: bool, ic025: float) -> float:
    """
    Convert ROR signal metrics into a 0-1 risk score for GNN training.

    Components:
      - Base risk: log-scaled ROR (capped at ROR=100 → 1.0)
      - Danger boost: +0.15 for dangerous reactions
      - Confidence boost: +0.10 for IC025 > 2 (strong statistical signal)

    The result is clipped to [0.1, 1.0] — we use 0.1 as floor (not 0) because
    these are all pairs that DID trigger an ROR signal, so they have some risk.
    """
    # Log-scale the ROR: ROR=2 → ~0.15, ROR=10 → ~0.5, ROR=100 → 1.0
    base = min(math.log(max(ror, 1.0)) / math.log(100), 1.0)
    danger_boost = 0.15 if is_dangerous else 0.0
    confidence_boost = 0.10 if ic025 > 2.0 else 0.0
    return max(0.1, min(base + danger_boost + confidence_boost, 1.0))


def train_and_save_model(graph_data, ror_df: pd.DataFrame, sc: StreamingCounts):
    """
    Train GAT (Graph Attention Network) to predict interaction RISK.

    Why GAT over GCN:
      - Learns attention weights for each neighbor, so it can focus on
        pharmacologically relevant drug connections rather than treating
        all co-occurring drugs equally.
      - Multi-head attention captures different types of relationships
        (e.g., CYP450 interactions vs QT-prolonging combinations).
      - Attention weights provide interpretability: we can see which
        neighboring drugs influenced a prediction.

    Args:
        graph_data: PyG Data object with node features and edge index
        ror_df: DataFrame of ROR signals (computed in step 3)
        sc: StreamingCounts with edge_counter for negative sampling
    """
    if MODEL_FILE.exists():
        print(f"[5/6] Loading cached model from {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    if graph_data is None:
        print("[5/6] No graph data — skipping model training")
        return None

    if ror_df is None or ror_df.empty:
        print("[5/6] No ROR signals — skipping model training")
        return None

    in_ch = graph_data.x.shape[1]
    print(f"[5/6] Training GAT model (risk-based labels, input dim={in_ch})...")

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
    except ImportError:
        print("[5/6] torch-geometric not installed — skipping")
        return None

    # ── Build GAT architecture with Bilinear Decoder ──────────────────────────
    #
    # FIX 3: Bilinear decoder for better interaction modeling
    # --------------------------------------------------------
    # The original concat + linear decoder often collapses to predicting
    # a constant value because concatenation loses interaction information.
    #
    # A bilinear decoder computes: score = sigmoid(z_a^T W z_b + MLP([z_a, z_b, z_a*z_b]))
    # This explicitly models how drug A's features INTERACT with drug B's features,
    # rather than just stacking them.
    #
    # We also add element-wise product (Hadamard) features which capture
    # "both drugs have high dangerous fraction" type patterns.
    #
    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Layer 1: 4 attention heads, each producing 16-dim output → 64-dim
            self.conv1 = GATConv(in_ch, 16, heads=4, dropout=0.3)
            # Layer 2: 4 attention heads averaged → 32-dim output
            self.conv2 = GATConv(16 * 4, 32, heads=4, concat=False, dropout=0.3)

            # Bilinear interaction layer: z_a^T W z_b
            self.bilinear = torch.nn.Bilinear(32, 32, 16)

            # MLP decoder on combined features:
            # [z_a (32) | z_b (32) | z_a*z_b (32) | bilinear(16)] = 112-dim
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(112, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 1),
            )

        def encode(self, x, edge_index):
            # GAT layer 1 with ELU activation
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            # GAT layer 2
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z, src, dst):
            z_a = z[src]  # (batch, 32)
            z_b = z[dst]  # (batch, 32)

            # Bilinear interaction: captures z_a^T W z_b
            bilin = self.bilinear(z_a, z_b)  # (batch, 16)

            # Element-wise product: captures "both high" patterns
            hadamard = z_a * z_b  # (batch, 32)

            # Concatenate all features
            combined = torch.cat([z_a, z_b, hadamard, bilin], dim=-1)  # (batch, 112)

            # MLP prediction
            return torch.sigmoid(self.decoder(combined))

    # ── Build training data from ROR signals ─────────────────────────────────
    drug_to_idx = graph_data.drug_to_idx

    # Positive samples: pairs with ROR signals
    pos_src, pos_dst, pos_labels, pos_weights = [], [], [], []
    ror_pairs = set()  # Track which pairs have ROR signals

    for _, row in ror_df.iterrows():
        da, db = row["drug_a"], row["drug_b"]
        if da not in drug_to_idx or db not in drug_to_idx:
            continue

        ia, ib = drug_to_idx[da], drug_to_idx[db]
        ror = row.get("ror", 2.0)
        is_dangerous = row.get("is_dangerous", False)
        ic025 = row.get("ic025", 0.0)
        n_cases = row.get("n_cases", 3)

        risk_score = _compute_risk_score(ror, is_dangerous, ic025)
        weight = math.log1p(n_cases)  # Weight by confidence

        pos_src.append(ia)
        pos_dst.append(ib)
        pos_labels.append(risk_score)
        pos_weights.append(weight)
        ror_pairs.add(frozenset([da, db]))

    n_pos = len(pos_src)
    print(f"      Positive samples (ROR signals): {n_pos}")

    if n_pos == 0:
        print("[5/6] No valid positive samples — skipping model training")
        return None

    # ── Negative samples: co-occurring pairs WITHOUT ROR signals ─────────────
    # These are "confirmed safe" — frequently co-prescribed without problems
    neg_src, neg_dst, neg_weights = [], [], []
    idx_to_drug = {v: k for k, v in sc.all_drug_to_idx.items()}

    # Sort edges by co-occurrence count (prefer high-count negatives)
    sorted_edges = sorted(sc.edge_counter.items(), key=lambda x: -x[1])

    # Take up to 3x as many negatives as positives (for class balance)
    max_neg = n_pos * 3

    for (ia, ib), count in sorted_edges:
        if len(neg_src) >= max_neg:
            break

        da = idx_to_drug.get(ia)
        db = idx_to_drug.get(ib)
        if da is None or db is None:
            continue
        if da not in drug_to_idx or db not in drug_to_idx:
            continue

        # Only use as negative if NOT in ROR signals
        if frozenset([da, db]) in ror_pairs:
            continue

        # Must have reasonable co-occurrence count to be "confirmed safe"
        if count < 5:
            continue

        neg_src.append(drug_to_idx[da])
        neg_dst.append(drug_to_idx[db])
        neg_weights.append(math.log1p(count))

    n_neg = len(neg_src)
    print(f"      Negative samples (safe pairs):  {n_neg}")

    if n_neg < n_pos // 2:
        print("      Warning: few negative samples — adding random negatives")
        # Fallback: add some random negatives
        n_nodes = graph_data.x.shape[0]
        while len(neg_src) < n_pos:
            ia = np.random.randint(0, n_nodes)
            ib = np.random.randint(0, n_nodes)
            if ia != ib:
                neg_src.append(ia)
                neg_dst.append(ib)
                neg_weights.append(1.0)

    # ── Combine into training tensors ────────────────────────────────────────
    all_src = torch.tensor(pos_src + neg_src, dtype=torch.long)
    all_dst = torch.tensor(pos_dst + neg_dst, dtype=torch.long)
    all_labels = torch.tensor(
        pos_labels + [0.0] * len(neg_src),  # Negatives have risk = 0
        dtype=torch.float
    )
    all_weights = torch.tensor(
        pos_weights + neg_weights,
        dtype=torch.float
    )
    # Normalize weights
    all_weights = all_weights / all_weights.mean()

    # Store positive/negative indices for margin loss
    pos_indices = torch.arange(n_pos)
    neg_indices = torch.arange(n_pos, n_pos + len(neg_src))

    # ── Train model ──────────────────────────────────────────────────────────
    model = GAT()

    # Initialize decoder bias to spread predictions
    # This prevents the model from starting at a collapsed state
    with torch.no_grad():
        # Set final layer bias to encourage ~0.5 initial output
        model.decoder[-1].bias.fill_(0.0)

    # Use AdamW with lower learning rate for stability
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=5e-4)

    # Learning rate scheduler: reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=20
    )

    # ── Loss function: BCE + Margin Ranking + Contrastive ────────────────────
    # FIX 4: Better loss function to prevent collapse
    #
    # Key changes:
    # 1. BCE instead of MSE - more appropriate for [0,1] scores
    # 2. Stronger margin loss (weight 1.0 instead of 0.5)
    # 3. Much stronger variance regularization (1.0 instead of 0.1)
    # 4. Add contrastive loss that explicitly pushes pos/neg apart
    #
    margin_loss_fn = torch.nn.MarginRankingLoss(margin=0.4)
    bce_loss_fn = torch.nn.BCELoss(reduction='none')

    def combined_loss(pred, target, weight, pos_idx, neg_idx):
        # 1. Weighted BCE for regression (better for [0,1] range)
        # Clamp predictions to avoid log(0)
        pred_clamped = torch.clamp(pred, 1e-6, 1 - 1e-6)
        bce = (weight * bce_loss_fn(pred_clamped, target)).mean()

        # 2. Margin ranking loss: pos should be > neg by margin
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            # Sample pairs for ranking loss (avoid O(n^2))
            n_pairs = min(len(pos_idx), len(neg_idx), 1000)
            pos_sample = pos_idx[torch.randperm(len(pos_idx))[:n_pairs]]
            neg_sample = neg_idx[torch.randperm(len(neg_idx))[:n_pairs]]

            pos_scores = pred[pos_sample]
            neg_scores = pred[neg_sample]
            # Target: +1 means first input should be higher
            rank_target = torch.ones(n_pairs)
            margin = margin_loss_fn(pos_scores, neg_scores, rank_target)

            # 3. Contrastive: push means apart
            pos_mean = pred[pos_idx].mean()
            neg_mean = pred[neg_idx].mean()
            # We want pos_mean > neg_mean, so penalize when neg_mean >= pos_mean
            contrastive = F.relu(neg_mean - pos_mean + 0.2)
        else:
            margin = torch.tensor(0.0)
            contrastive = torch.tensor(0.0)

        # 4. Variance regularization: encourage spread in predictions
        # Much stronger weight to fight collapse
        var_loss = -torch.var(pred) * 2.0

        return bce + 1.0 * margin + 0.5 * contrastive + var_loss

    best_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0

    print("      Training with MSE + Margin Ranking + Variance regularization...")

    for epoch in tqdm(range(200), desc="Training GAT (risk-based)"):
        model.train()
        opt.zero_grad()

        z = model.encode(graph_data.x, graph_data.edge_index)
        pred = model.decode(z, all_src, all_dst).squeeze()
        loss = combined_loss(pred, all_labels, all_weights, pos_indices, neg_indices)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        opt.step()
        scheduler.step(loss)

        # Track best model
        if loss.item() < best_loss - 0.0005:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
                break

        # Periodic diagnostics
        if epoch % 50 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(graph_data.x, graph_data.edge_index)
                pred = model.decode(z, all_src, all_dst).squeeze()
                pos_mean = pred[:n_pos].mean().item()
                neg_mean = pred[n_pos:].mean().item()
                print(f"      Epoch {epoch}: loss={loss.item():.4f}, "
                      f"pos_mean={pos_mean:.3f}, neg_mean={neg_mean:.3f}, "
                      f"sep={pos_mean - neg_mean:.3f}")
            model.train()

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Evaluate on training data ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        z = model.encode(graph_data.x, graph_data.edge_index)
        pred = model.decode(z, all_src, all_dst).squeeze()

        pos_preds = pred[:n_pos].numpy()
        neg_preds = pred[n_pos:].numpy()

        print(f"\n      === Final Model Evaluation ===")
        print(f"      Positive predictions: mean={pos_preds.mean():.3f}, "
              f"std={pos_preds.std():.3f}, min={pos_preds.min():.3f}, max={pos_preds.max():.3f}")
        print(f"      Negative predictions: mean={neg_preds.mean():.3f}, "
              f"std={neg_preds.std():.3f}, min={neg_preds.min():.3f}, max={neg_preds.max():.3f}")

        # Check separation
        separation = pos_preds.mean() - neg_preds.mean()
        print(f"      Separation (pos - neg): {separation:.3f}")

        # Check if model collapsed
        total_std = pred.numpy().std()
        if total_std < 0.05:
            print(f"      ⚠️  WARNING: Low prediction variance ({total_std:.4f}) - model may have collapsed!")
        elif separation < 0.1:
            print(f"      ⚠️  WARNING: Low separation ({separation:.3f}) - model may not discriminate well")
        else:
            print(f"      ✓  Model shows good separation between risky and safe pairs")

    model_data = {
        "state_dict":    model.state_dict(),
        "drug_to_idx":   graph_data.drug_to_idx,
        "idx_to_drug":   graph_data.idx_to_drug,
        "feature_names": getattr(graph_data, "feature_names", []),
        "in_ch":         in_ch,
        "training_info": {
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pos_mean": float(pos_preds.mean()),
            "neg_mean": float(neg_preds.mean()),
            "separation": float(separation),
        },
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[5/6] Saved model to {MODEL_FILE}")
    return model_data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE BIAS-ADJUSTED ROR, PRR, IC (BCPNN) FROM COUNT MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ic(observed: int, expected: float,
                alpha: float = 0.5, beta: float = 0.5) -> tuple:
    """
    Information Component (IC) with empirical Bayes shrinkage (Noren 2013).
    IC025 > 0 is the WHO Uppsala signal criterion.
    Returns: (IC, IC025, IC975)
    """
    obs_s = observed + alpha
    exp_s = expected + beta
    ic    = math.log2(obs_s / exp_s)
    var   = (1 / (obs_s * math.log(2) ** 2)) * (1 - obs_s / (exp_s + obs_s))
    se    = math.sqrt(max(var, 0.0))
    return ic, ic - 1.96 * se, ic + 1.96 * se


def _compute_prr(a: float, b: float, c: float, d: float) -> tuple:
    """
    Proportional Reporting Ratio (PRR) with 95% CI and chi-square.
    Returns: (PRR, PRR_lower, PRR_upper, chi_square)
    """
    if (a + b) == 0 or (c + d) == 0 or c == 0 or a <= 0:
        return None, None, None, None

    prr = (a / (a + b)) / (c / (c + d))

    try:
        se_ln     = math.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        prr_lower = math.exp(math.log(prr) - 1.96 * se_ln)
        prr_upper = math.exp(math.log(prr) + 1.96 * se_ln)
    except (ValueError, ZeroDivisionError):
        prr_lower = prr_upper = None

    n_total = a + b + c + d
    if n_total == 0:
        return prr, prr_lower, prr_upper, None

    e_a = (a + b) * (a + c) / n_total
    e_b = (a + b) * (b + d) / n_total
    e_c = (c + d) * (a + c) / n_total
    e_d = (c + d) * (b + d) / n_total

    chi_sq = sum(
        ((abs(obs - exp) - 0.5) ** 2) / exp
        for obs, exp in [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]
        if exp > 0
    )
    return prr, prr_lower, prr_upper, chi_sq


def compute_ror_from_counts(sc: StreamingCounts) -> pd.DataFrame:
    """
    Compute bias-adjusted ROR/PRR/IC (BCPNN)/combo_lift from count matrices.
    """
    if ROR_FILE.exists():
        print(f"[3/6] Loading cached ROR signals from {ROR_FILE}")
        return pd.read_parquet(ROR_FILE)

    print("[3/6] Computing bias-adjusted ROR/PRR/IC from count matrices...")

    N             = sc.N
    drug_totals   = sc.drug_totals.astype(np.float64)
    rxn_totals    = sc.rxn_totals.astype(np.float64)
    drug_rxn_co   = sc.drug_rxn_co.astype(np.float64)
    pair_totals   = sc.pair_totals
    pair_rxn_co   = sc.pair_rxn_co.astype(np.float64)
    drug_to_idx   = sc.drug_to_idx
    top_pairs     = sc.top_pairs
    top_reactions = sc.top_reactions

    print("    Building solo ROR matrix (vectorized)...")
    A_s = drug_rxn_co
    B_s = drug_totals[:, None] - A_s
    C_s = rxn_totals[None, :]  - A_s
    D_s = N - drug_totals[:, None] - rxn_totals[None, :] + A_s

    valid_solo = (A_s >= ROR_MIN_CASES) & (B_s > 0) & (C_s > 0) & (D_s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        solo_ror_mat = np.where(valid_solo, (A_s * D_s) / (B_s * C_s), np.nan)

    print(f"    Screening {len(top_pairs)} pairs x {len(top_reactions)} reactions...")
    results = []

    for pi, pair in enumerate(tqdm(top_pairs, desc="Pair ROR/PRR/IC")):
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

        A = pair_rxn_co[pi]
        B = n_pair - A
        C = rxn_totals - A
        D = N - n_pair - rxn_totals + A

        for rj, rxn in enumerate(top_reactions):
            aj, bj, cj, dj = float(A[rj]), float(B[rj]), float(C[rj]), float(D[rj])

            if aj < ROR_MIN_CASES or bj <= 0 or cj <= 0 or dj <= 0:
                continue

            ror   = (aj * dj) / (bj * cj)
            se    = math.sqrt(1/aj + 1/bj + 1/cj + 1/dj)
            lower = math.exp(math.log(ror) - 1.96 * se)
            upper = math.exp(math.log(ror) + 1.96 * se)

            if ror <= 2.0 or lower <= ROR_LOWER_CI:
                continue

            prr, prr_lower, prr_upper, chi_sq = _compute_prr(aj, bj, cj, dj)

            n_rxn = float(rxn_totals[rj])
            exp   = (n_pair * n_rxn / N) if N > 0 else 0.0
            if exp == 0.0:
                continue

            ic, ic025, ic975 = _compute_ic(int(aj), exp)

            sa_val = solo_ror_mat[ia, rj]
            sb_val = solo_ror_mat[ib, rj]
            solo_a = None if np.isnan(sa_val) else float(sa_val)
            solo_b = None if np.isnan(sb_val) else float(sb_val)

            if solo_a is not None and solo_b is not None:
                combo_lift = ror / max(solo_a, solo_b) if max(solo_a, solo_b) > 0 else None
            elif solo_a is not None:
                combo_lift = ror / solo_a if solo_a > 0 else None
            elif solo_b is not None:
                combo_lift = ror / solo_b if solo_b > 0 else None
            else:
                combo_lift = None

            results.append({
                "drug_a":       da,
                "drug_b":       db,
                "reaction":     rxn,
                "ror":          round(ror,   2),
                "ror_ci_lower": round(lower, 2),
                "ror_ci_upper": round(upper, 2),
                "prr":          round(prr, 2)       if prr       is not None else None,
                "prr_ci_lower": round(prr_lower, 2) if prr_lower is not None else None,
                "prr_ci_upper": round(prr_upper, 2) if prr_upper is not None else None,
                "prr_chi_sq":   round(chi_sq, 2)    if chi_sq    is not None else None,
                "ic":           round(ic,    3),
                "ic025":        round(ic025, 3),
                "ic975":        round(ic975, 3),
                "n_cases":      int(aj),
                "expected":     round(exp, 2),
                "combo_lift":   round(combo_lift, 3) if combo_lift is not None else None,
                "solo_ror_a":   round(solo_a,  2)   if solo_a    is not None else None,
                "solo_ror_b":   round(solo_b,  2)   if solo_b    is not None else None,
                "is_dangerous": rxn in DANGEROUS_REACTIONS,
            })

    ror_df = pd.DataFrame(results)

    if not ror_df.empty:
        n_before = len(ror_df)
        ror_df = ror_df[ror_df["ic025"] > 0]
        ror_df = ror_df[ror_df["combo_lift"].isna() | (ror_df["combo_lift"] >= 0.8)]
        print(f"    Bias filters: {n_before} -> {len(ror_df)} signals "
              f"({n_before - len(ror_df)} removed)")
        ror_df = ror_df.sort_values("ic025", ascending=False).reset_index(drop=True)

    ror_df.to_parquet(ROR_FILE)
    print(f"[3/6] Saved {len(ror_df)} signals to {ROR_FILE}")
    return ror_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE DRUG LIST
# ─────────────────────────────────────────────────────────────────────────────

def save_drug_list(sc: StreamingCounts):
    with open(DRUGS_FILE, "w") as f:
        json.dump(sc.all_drugs, f)
    print(f"[4/6] Saved {len(sc.all_drugs):,} drugs to {DRUGS_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def clear_cache(keep_counts: bool = True):
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
    print("  PHANTOM PHARMACOLOGY - Streaming Pipeline v4")
    print("  (ROR + PRR + BCPNN/IC)")
    print("  Fix 1: Richer GNN node features (5-dim pharmacological profile)")
    print("  Fix 2: Risk-based GNN training (not co-occurrence)")
    print("=" * 60)
    print()

    # Step 1: Load or build streaming counts
    sc = load_or_build_counts()
    print(f"    {sc.N:,} reports | {len(sc.all_drugs):,} unique drugs\n")

    # Step 2: Build drug co-occurrence graph
    graph_data = build_and_save_graph(sc)
    print()

    # Step 3: Compute ROR signals (BEFORE training — we need these as labels)
    ror_df = compute_ror_from_counts(sc)
    print()

    # Step 4: Save drug list
    save_drug_list(sc)
    print()

    # Step 5: Train GNN model using ROR signals as risk labels
    train_and_save_model(graph_data, ror_df, sc)
    print()

    print("=" * 60)
    print("  Pipeline complete! Now run: streamlit run appAF.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phantom Pharmacology Pipeline v3")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete ROR/graph/model cache (keeps streaming counts)")
    parser.add_argument("--clear-all", action="store_true",
                        help="Delete ALL cache including streaming counts")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache(keep_counts=True)
    if args.clear_all:
        clear_cache(keep_counts=False)

    main()
