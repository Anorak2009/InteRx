"""
Phantom Pharmacology - Data Processing Pipeline
================================================
Run this ONCE to process FAERS data and save results.
Then use app.py for the fast Streamlit UI.

Usage:
    python pipeline.py
"""

import json
import math
import itertools
import pickle
from pathlib import Path
import numpy as np
import math
import itertools
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FAERS_DATA_DIR = Path("Phantom Data")
CACHE_DIR = Path("faers_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Output files
REPORTS_FILE = CACHE_DIR / "reports.parquet"
ROR_FILE = CACHE_DIR / "ror_signals.parquet"
GRAPH_FILE = CACHE_DIR / "graph_data.pkl"
MODEL_FILE = CACHE_DIR / "gnn_model.pkl"
DRUGS_FILE = CACHE_DIR / "all_drugs.json"

# Parameters
MAX_QUARTERS = 12          # Limit quarters to save memory (0 = all)
TOP_DRUGS = 500           # For association rules
ROR_TOP_PAIRS = 200       # Number of drug pairs to screen
ROR_TOP_REACTIONS = 50    # Number of reactions to check
ROR_MIN_CASES = 3
ROR_LOWER_CI = 1.0

DANGEROUS_REACTIONS = {
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}

# ── Globals used by worker processes ─────────────────────────────────────────
_drug_matrix = None
_rxn_matrix  = None
_top_reactions = None

def _init_worker(drug_mat, rxn_mat, reactions):
    global _drug_matrix, _rxn_matrix, _top_reactions
    _drug_matrix   = drug_mat
    _rxn_matrix    = rxn_mat
    _top_reactions = reactions


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD FAERS DATA
# ─────────────────────────────────────────────────────────────────────────────

def find_faers_folders():
    """Find all FAERS ASCII data folders."""
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
    """Load DRUG and REAC files from a single FAERS quarter."""
    drug_file = reac_file = None
    for f in ascii_dir.iterdir():
        fname = f.name.upper()
        if fname.startswith("DRUG") and fname.endswith(".TXT"):
            drug_file = f
        elif fname.startswith("REAC") and fname.endswith(".TXT"):
            reac_file = f

    if not all([drug_file, reac_file]):
        return None

    # Load DRUG
    drug_df = pd.read_csv(
        drug_file, sep="$", dtype=str, low_memory=False,
        usecols=["primaryid", "drugname", "role_cod"]
    )
    drug_df = drug_df[drug_df["role_cod"].isin(["PS", "SS"])]
    drug_df["drugname"] = drug_df["drugname"].str.lower().str.strip()
    drug_df = drug_df[["primaryid", "drugname"]].dropna()

    # Load REAC
    reac_df = pd.read_csv(
        reac_file, sep="$", dtype=str, low_memory=False,
        usecols=["primaryid", "pt"]
    )
    reac_df["pt"] = reac_df["pt"].str.lower().str.strip()
    reac_df = reac_df.dropna()

    # Group by primaryid
    drugs_grouped = drug_df.groupby("primaryid")["drugname"].apply(list).reset_index()
    drugs_grouped.columns = ["primaryid", "drugs"]
    del drug_df

    reacs_grouped = reac_df.groupby("primaryid")["pt"].apply(list).reset_index()
    reacs_grouped.columns = ["primaryid", "reactions"]
    del reac_df

    # Merge
    result = drugs_grouped.merge(reacs_grouped, on="primaryid", how="inner")
    del drugs_grouped, reacs_grouped

    # Filter
    result = result[result["drugs"].apply(len) >= 2]
    result = result[result["reactions"].apply(len) >= 1]

    return result


def load_all_faers_data():
    """Load all FAERS data from Phantom Data folder."""
    if REPORTS_FILE.exists():
        print(f"[1/5] Loading cached reports from {REPORTS_FILE}")
        return pd.read_parquet(REPORTS_FILE)

    quarters = find_faers_folders()
    if not quarters:
        raise FileNotFoundError("No FAERS data found in 'Phantom Data' folder")

    # Limit quarters
    if MAX_QUARTERS > 0 and len(quarters) > MAX_QUARTERS:
        quarters = quarters[-MAX_QUARTERS:]
        print(f"[1/5] Limiting to {MAX_QUARTERS} most recent quarters")

    all_reports = []
    seen_ids = set()

    print(f"[1/5] Loading FAERS data from {len(quarters)} quarters...")
    for ascii_dir in tqdm(quarters, desc="Loading"):
        df = load_faers_quarter(ascii_dir)
        if df is not None and len(df) > 0:
            df = df[~df["primaryid"].isin(seen_ids)]
            seen_ids.update(df["primaryid"].tolist())
            all_reports.append(df)

    combined = pd.concat(all_reports, ignore_index=True)
    del all_reports, seen_ids

    combined.to_parquet(REPORTS_FILE)
    print(f"[1/5] Saved {len(combined)} reports to {REPORTS_FILE}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD DRUG GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_graph(df: pd.DataFrame, all_drugs: list):
    """Build drug co-occurrence graph and save."""
    if GRAPH_FILE.exists():
        print(f"[2/5] Loading cached graph from {GRAPH_FILE}")
        with open(GRAPH_FILE, "rb") as f:
            return pickle.load(f)

    print(f"[2/5] Building drug graph with {len(all_drugs)} drugs...")

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("[2/5] torch-geometric not installed, skipping graph")
        return None

    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    edge_counter = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph"):
        drugs = [drug_to_idx[d] for d in row["drugs"] if d in drug_to_idx]
        for a, b in itertools.combinations(sorted(drugs), 2):
            edge_counter[(a, b)] += 1

    edges = list(edge_counter.keys())
    edge_weights = [edge_counter[e] for e in edges]

    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])

    # Node features: log-frequency
    drug_freq = defaultdict(int)
    for drugs in df["drugs"]:
        for d in drugs:
            if d in drug_to_idx:
                drug_freq[drug_to_idx[d]] += 1

    x = torch.tensor(
        [[math.log1p(drug_freq[i])] for i in range(len(all_drugs))],
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
# STEP 3: TRAIN GNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_model(graph_data):
    """Train GNN model and save."""
    if MODEL_FILE.exists():
        print(f"[3/5] Loading cached model from {MODEL_FILE}")
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)

    if graph_data is None:
        print("[3/5] No graph data, skipping model training")
        return None

    print("[3/5] Training GNN model...")

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
    except ImportError:
        print("[3/5] torch-geometric not installed, skipping")
        return None

    in_ch = graph_data.x.shape[1]

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_ch, 64)
            self.conv2 = GCNConv(64, 32)
            self.head = torch.nn.Linear(64, 1)

        def encode(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z, src, dst):
            return torch.sigmoid(self.head(torch.cat([z[src], z[dst]], dim=-1)))

    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    src = graph_data.edge_index[0]
    dst = graph_data.edge_index[1]
    pos_labels = torch.ones(src.shape[0])

    neg_src = torch.randint(0, graph_data.x.shape[0], (src.shape[0],))
    neg_dst = torch.randint(0, graph_data.x.shape[0], (src.shape[0],))
    neg_labels = torch.zeros(neg_src.shape[0])

    all_src = torch.cat([src, neg_src])
    all_dst = torch.cat([dst, neg_dst])
    all_labels = torch.cat([pos_labels, neg_labels])

    for epoch in tqdm(range(50), desc="Training GNN"):
        model.train()
        opt.zero_grad()
        z = model.encode(graph_data.x, graph_data.edge_index)
        pred = model.decode(z, all_src, all_dst).squeeze()
        loss = criterion(pred, all_labels)
        loss.backward()
        opt.step()

    # Save model state and graph data reference
    model_data = {
        "state_dict": model.state_dict(),
        "drug_to_idx": graph_data.drug_to_idx,
        "idx_to_drug": graph_data.idx_to_drug,
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[3/5] Saved model to {MODEL_FILE}")

    return model_data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE ROR SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def _process_pair(args):
    """Worker: compute ROR for one drug pair across all reactions (vectorized)."""
    drug_a, drug_b, ia, ib = args

    pair_present = _drug_matrix[:, ia] & _drug_matrix[:, ib]
    either_present = _drug_matrix[:, ia] | _drug_matrix[:, ib]
    only_either = either_present & ~pair_present

    pp = pair_present[:, None]
    oe = only_either[:, None]
    rm = _rxn_matrix

    a = (pp & rm).sum(axis=0)
    b = (pp & ~rm).sum(axis=0)
    c = (oe & rm).sum(axis=0)
    d = (oe & ~rm).sum(axis=0)

    results = []
    for j, rxn in enumerate(_top_reactions):
        aj, bj, cj, dj = int(a[j]), int(b[j]), int(c[j]), int(d[j])
        if aj < ROR_MIN_CASES or bj == 0 or cj == 0 or dj == 0:
            continue
        ror = (aj * dj) / (bj * cj)
        se = math.sqrt(1 / aj + 1 / bj + 1 / cj + 1 / dj)
        lower = math.exp(math.log(ror) - 1.96 * se)
        upper = math.exp(math.log(ror) + 1.96 * se)
        if ror > 2.0 and lower > ROR_LOWER_CI:
            results.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "reaction": rxn,
                "ror": round(ror, 2),
                "ci_lower": round(lower, 2),
                "ci_upper": round(upper, 2),
                "n_cases": aj,
                "is_dangerous": rxn in DANGEROUS_REACTIONS,
            })
    return results


def _ror_math(a, b, c, d, drug_a, drug_b, top_reactions):
    """Shared ROR calculation used by both GPU paths after pulling data to CPU."""
    results = []
    for j, rxn in enumerate(top_reactions):
        aj, bj, cj, dj = int(a[j]), int(b[j]), int(c[j]), int(d[j])
        if aj < ROR_MIN_CASES or bj == 0 or cj == 0 or dj == 0:
            continue
        ror = (aj * dj) / (bj * cj)
        se = math.sqrt(1 / aj + 1 / bj + 1 / cj + 1 / dj)
        lower = math.exp(math.log(ror) - 1.96 * se)
        upper = math.exp(math.log(ror) + 1.96 * se)
        if ror > 2.0 and lower > ROR_LOWER_CI:
            results.append({
                "drug_a": drug_a,
                "drug_b": drug_b,
                "reaction": rxn,
                "ror": round(ror, 2),
                "ci_lower": round(lower, 2),
                "ci_upper": round(upper, 2),
                "n_cases": aj,
                "is_dangerous": rxn in DANGEROUS_REACTIONS,
            })
    return results


def _gpu_ror_cupy(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx):
    """NVIDIA GPU path via cupy."""
    import cupy as cp

    dm = cp.asarray(drug_matrix)
    rm = cp.asarray(rxn_matrix)

    results = []
    for pair in tqdm(top_pairs, desc="Computing ROR (NVIDIA/cupy)"):
        drugs = list(pair)
        if len(drugs) < 2 or drugs[0] not in drug_to_idx or drugs[1] not in drug_to_idx:
            continue
        drug_a, drug_b = drugs[0], drugs[1]
        ia, ib = drug_to_idx[drug_a], drug_to_idx[drug_b]

        pp = dm[:, ia] & dm[:, ib]
        oe = (dm[:, ia] | dm[:, ib]) & ~pp

        a = cp.asnumpy((pp[:, None] & rm).sum(axis=0))
        b = cp.asnumpy((pp[:, None] & ~rm).sum(axis=0))
        c = cp.asnumpy((oe[:, None] & rm).sum(axis=0))
        d = cp.asnumpy((oe[:, None] & ~rm).sum(axis=0))

        results.extend(_ror_math(a, b, c, d, drug_a, drug_b, top_reactions))

    return results


def _gpu_ror_mps(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx):
    """Apple Silicon GPU path via PyTorch MPS.

    Uses uint8 instead of bool — MPS does not support bitwise ops on bool tensors.
    """
    import torch

    device = torch.device("mps")
    # uint8 required — MPS doesn't support bitwise ops on bool tensors
    dm = torch.tensor(drug_matrix, dtype=torch.uint8).to(device)
    rm = torch.tensor(rxn_matrix, dtype=torch.uint8).to(device)

    results = []
    for pair in tqdm(top_pairs, desc="Computing ROR (Apple MPS)"):
        drugs = list(pair)
        if len(drugs) < 2 or drugs[0] not in drug_to_idx or drugs[1] not in drug_to_idx:
            continue
        drug_a, drug_b = drugs[0], drugs[1]
        ia, ib = drug_to_idx[drug_a], drug_to_idx[drug_b]

        pp = dm[:, ia] & dm[:, ib]
        oe = (dm[:, ia] | dm[:, ib]) & ~pp

        a = (pp[:, None] & rm).sum(dim=0).cpu().numpy()
        b = (pp[:, None] & ~rm).sum(dim=0).cpu().numpy()
        c = (oe[:, None] & rm).sum(dim=0).cpu().numpy()
        d = (oe[:, None] & ~rm).sum(dim=0).cpu().numpy()

        results.extend(_ror_math(a, b, c, d, drug_a, drug_b, top_reactions))

    return results


def _cpu_ror_multiprocessing(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx):
    """CPU multiprocessing fallback."""
    n_workers = max(1, os.cpu_count() - 1)
    print(f"    No GPU found — using {n_workers} CPU workers")

    tasks = []
    for pair in top_pairs:
        drugs = list(pair)
        if len(drugs) < 2 or drugs[0] not in drug_to_idx or drugs[1] not in drug_to_idx:
            continue
        tasks.append((drugs[0], drugs[1], drug_to_idx[drugs[0]], drug_to_idx[drugs[1]]))

    results = []
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(drug_matrix, rxn_matrix, top_reactions),
    ) as pool:
        futures = {pool.submit(_process_pair, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Computing ROR (CPU)"):
            results.extend(fut.result())

    return results


def compute_ror(df: pd.DataFrame, drug_a: str, drug_b: str, reaction: str):
    """Single-pair/reaction ROR — kept for external callers."""
    pair = {drug_a, drug_b}

    pair_present = df["drug_set"].apply(lambda d: pair.issubset(d))
    either_present = df["drug_set"].apply(lambda d: bool(d & pair))
    only_either = either_present & ~pair_present
    rxn_match = df["rxn_set"].apply(lambda r: reaction in r)

    a = int((pair_present & rxn_match).sum())
    b = int((pair_present & ~rxn_match).sum())
    c = int((only_either & rxn_match).sum())
    d = int((only_either & ~rxn_match).sum())

    if a < ROR_MIN_CASES or b == 0 or c == 0 or d == 0:
        return None

    ror = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lower = math.exp(math.log(ror) - 1.96 * se)
    upper = math.exp(math.log(ror) + 1.96 * se)
    return ror, lower, upper, a


def compute_all_ror_signals(df: pd.DataFrame):
    """Compute ROR using boolean matrices + best available backend:
       1. NVIDIA GPU  (cupy)
       2. Apple MPS   (torch)
       3. CPU multiprocessing (fallback)
    """
    if ROR_FILE.exists():
        print(f"[4/5] Loading cached ROR signals from {ROR_FILE}")
        return pd.read_parquet(ROR_FILE)

    print("[4/5] Computing ROR signals...")

    df = df.copy()
    df["drug_set"] = df["drugs"].apply(set)
    df["rxn_set"] = df["reactions"].apply(set)

    # ── Frequency counts ──────────────────────────────────────────────────────
    pair_count = defaultdict(int)
    for drug_set in df["drug_set"]:
        for pair in itertools.combinations(sorted(drug_set), 2):
            pair_count[frozenset(pair)] += 1
    top_pairs = sorted(pair_count, key=pair_count.get, reverse=True)[:ROR_TOP_PAIRS]

    rxn_count = defaultdict(int)
    for rxn_set in df["rxn_set"]:
        for r in rxn_set:
            rxn_count[r] += 1
    top_reactions = [r for r, _ in sorted(rxn_count.items(), key=lambda x: -x[1])[:ROR_TOP_REACTIONS]]

    # ── Build boolean matrices ────────────────────────────────────────────────
    print("    Building boolean matrices...")
    all_drugs = sorted({d for ds in df["drug_set"] for d in ds})
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    rxn_to_idx = {r: i for i, r in enumerate(top_reactions)}
    n = len(df)

    drug_matrix = np.zeros((n, len(all_drugs)), dtype=bool)
    rxn_matrix = np.zeros((n, len(top_reactions)), dtype=bool)

    for i, (ds, rs) in enumerate(zip(df["drug_set"], df["rxn_set"])):
        for d in ds:
            if d in drug_to_idx:
                drug_matrix[i, drug_to_idx[d]] = True
        for r in rs:
            if r in rxn_to_idx:
                rxn_matrix[i, rxn_to_idx[r]] = True

    print(f"    Screening {len(top_pairs)} pairs × {len(top_reactions)} reactions...")

    # ── Backend selection ─────────────────────────────────────────────────────
    try:
        import cupy as cp
        cp.array([1])  # trigger real init — import alone doesn't confirm GPU is usable
        print("    ✓ NVIDIA GPU detected — using cupy")
        results = _gpu_ror_cupy(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx)

    except (ImportError, Exception):
        try:
            import torch
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS not available")
            print("    ✓ Apple Silicon detected — using MPS")
            results = _gpu_ror_mps(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx)

        except (ImportError, RuntimeError):
            results = _cpu_ror_multiprocessing(drug_matrix, rxn_matrix, top_pairs, top_reactions, drug_to_idx)

    # ── Save ──────────────────────────────────────────────────────────────────
    ror_df = pd.DataFrame(results)
    if not ror_df.empty:
        ror_df = ror_df.sort_values("ror", ascending=False)

    ror_df.to_parquet(ROR_FILE)
    print(f"[4/5] Saved {len(ror_df)} signals to {ROR_FILE}")
    return ror_df
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SAVE DRUG LIST
# ─────────────────────────────────────────────────────────────────────────────

def save_drug_list(df: pd.DataFrame):
    """Save list of all drugs for the UI."""
    all_drugs = sorted({d for row in df["drugs"] for d in row})

    with open(DRUGS_FILE, "w") as f:
        json.dump(all_drugs, f)

    print(f"[5/5] Saved {len(all_drugs)} drugs to {DRUGS_FILE}")
    return all_drugs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHANTOM PHARMACOLOGY - Data Processing Pipeline")
    print("=" * 60)
    print()

    # Step 1: Load data
    df = load_all_faers_data()
    print(f"    Loaded {len(df)} reports\n")

    # Step 2: Build graph
    all_drugs = sorted({d for row in df["drugs"] for d in row})
    graph_data = build_and_save_graph(df, all_drugs)
    print()

    # Step 3: Train model
    train_and_save_model(graph_data)
    print()

    # Step 4: Compute ROR
    ror_df = compute_all_ror_signals(df)
    print()

    # Step 5: Save drug list
    save_drug_list(df)
    print()

    print("=" * 60)
    print("  Pipeline complete! Now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
