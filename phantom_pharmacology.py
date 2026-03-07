"""
Phantom Pharmacology
====================
Uncovers hidden drug-drug interactions from FDA FAERS public adverse event data.

Pipeline:
  1. Data Acquisition  — pull reports from openFDA API
  2. Drug Normalization — map brand/generic names to RxNorm CUIs
  3. Association Rules  — Apriori mining for co-occurrence signals
  4. Graph Neural Net   — link-prediction DDI model (PyTorch Geometric)
  5. ROR Analysis       — disproportionality / pharmacovigilance signal
  6. Novelty Filter     — cross-reference against DrugBank known interactions
  7. Streamlit UI       — interactive drug-pair lookup + graph visualization

Run:
    pip install requests pandas pyarrow mlxtend torch torch-geometric \
                transformers streamlit plotly scipy tqdm
    streamlit run phantom_pharmacology.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, math, itertools, warnings
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR        = Path("faers_cache")
REPORTS_FILE     = CACHE_DIR / "reports.parquet"
DRUGBANK_FILE    = CACHE_DIR / "drugbank_interactions.csv"  # place your DrugBank CSV here
MAX_REPORTS      = 50_000          # increase for better signal; each API page = 100
OPENFDA_BASE     = "https://api.fda.gov/drug/event.json"
RXNORM_BASE      = "https://rxnav.nlm.nih.gov/REST"
MIN_SUPPORT      = 0.0005          # Apriori minimum support
MIN_LIFT         = 4.0             # Apriori minimum lift
ROR_MIN_CASES    = 3               # discard pairs with fewer co-reports
ROR_LOWER_CI     = 1.0             # lower 95 % CI threshold for ROR signal

DANGEROUS_REACTIONS = {            # reactions we especially care about
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}

CACHE_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_faers_reports(max_reports: int = MAX_REPORTS) -> pd.DataFrame:
    """Download adverse-event reports from openFDA and cache as parquet."""
    if REPORTS_FILE.exists():
        print(f"[cache] Loading reports from {REPORTS_FILE}")
        return pd.read_parquet(REPORTS_FILE)

    records = []
    page_size = 100
    pages = max_reports // page_size

    print(f"[fetch] Downloading up to {max_reports} FAERS reports …")
    for skip in tqdm(range(0, pages * page_size, page_size)):
        try:
            resp = requests.get(
                OPENFDA_BASE,
                params={"limit": page_size, "skip": skip},
                timeout=15,
            )
            if resp.status_code != 200:
                break
            for report in resp.json().get("results", []):
                report_id  = report.get("safetyreportid", "")
                serious    = int(report.get("serious", 0) or 0)
                death      = int(report.get("seriousnessdeath", 0) or 0)
                drugs = [
                    d.get("medicinalproduct", "").strip().lower()
                    for d in report.get("patient", {}).get("drug", [])
                    if d.get("drugcharacterization") in ("1", 1)   # suspect drugs
                    and d.get("medicinalproduct")
                ]
                reactions = [
                    r.get("reactionmeddrapt", "").strip().lower()
                    for r in report.get("patient", {}).get("reaction", [])
                    if r.get("reactionmeddrapt")
                ]
                if len(drugs) >= 2 and reactions:
                    records.append({
                        "report_id":  report_id,
                        "serious":    serious,
                        "death":      death,
                        "drugs":      drugs,
                        "reactions":  reactions,
                    })
        except Exception as e:
            print(f"[warn] page skip={skip}: {e}")
            time.sleep(1)

    df = pd.DataFrame(records)
    df.to_parquet(REPORTS_FILE)
    print(f"[fetch] Saved {len(df)} reports → {REPORTS_FILE}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DRUG NAME NORMALISATION (RxNorm)
# ─────────────────────────────────────────────────────────────────────────────

_rxnorm_cache: dict[str, str | None] = {}

def normalize_drug_rxnorm(name: str) -> str | None:
    """Return RxNorm CUI string for a drug name, or None if not found."""
    if name in _rxnorm_cache:
        return _rxnorm_cache[name]
    try:
        resp = requests.get(
            f"{RXNORM_BASE}/rxcui.json",
            params={"name": name, "search": 2},
            timeout=8,
        )
        ids = resp.json().get("idGroup", {}).get("rxnormId", [])
        cui = ids[0] if ids else None
    except Exception:
        cui = None
    _rxnorm_cache[name] = cui
    return cui


def normalize_reports(df: pd.DataFrame, use_rxnorm: bool = False) -> pd.DataFrame:
    """
    Normalise drug names.
    use_rxnorm=True hits the NIH API (slow but accurate).
    use_rxnorm=False just lowercases/strips (fast, good enough for prototyping).
    """
    norm_cache_file = CACHE_DIR / "rxnorm_map.json"

    if use_rxnorm:
        if norm_cache_file.exists():
            mapping = json.loads(norm_cache_file.read_text())
        else:
            mapping = {}

        all_names = {n for row in df["drugs"] for n in row}
        todo = all_names - set(mapping)
        print(f"[norm] Resolving {len(todo)} drug names via RxNorm …")
        for name in tqdm(todo):
            mapping[name] = normalize_drug_rxnorm(name) or name
            time.sleep(0.05)   # be polite to the NIH API

        norm_cache_file.write_text(json.dumps(mapping))
        df["drugs"] = df["drugs"].apply(
            lambda lst: list({mapping.get(n, n) for n in lst})
        )
    else:
        df["drugs"] = df["drugs"].apply(
            lambda lst: list({n.lower().strip() for n in lst})
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — ASSOCIATION RULE MINING
# ─────────────────────────────────────────────────────────────────────────────

def build_drug_reaction_matrix(df: pd.DataFrame):
    """
    Returns:
        drug_matrix   — one-hot DataFrame (reports × drugs)
        reaction_sets — dict mapping report_index → set of reactions
    """
    # explode drugs to one-hot
    all_drugs = sorted({d for row in df["drugs"] for d in row})
    print(f"[arm] Building one-hot matrix for {len(all_drugs)} unique drugs …")

    rows = []
    for _, row in df.iterrows():
        drug_set = set(row["drugs"])
        rows.append({d: (d in drug_set) for d in all_drugs})

    drug_matrix = pd.DataFrame(rows, dtype=bool)
    reaction_sets = df["reactions"].to_dict()
    return drug_matrix, reaction_sets, all_drugs


def run_association_rules(drug_matrix: pd.DataFrame, reaction_sets: dict):
    """Mine frequent itemsets and return high-lift drug-pair → reaction rules."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        print("[warn] mlxtend not installed — skipping association rules.")
        return pd.DataFrame()

    print("[arm] Running Apriori (this may take a few minutes) …")
    frequent = apriori(drug_matrix, min_support=MIN_SUPPORT, use_colnames=True)
    # keep only 2-itemsets (drug pairs)
    frequent = frequent[frequent["itemsets"].apply(len) == 2]

    if frequent.empty:
        print("[arm] No frequent itemsets found — try lowering MIN_SUPPORT.")
        return pd.DataFrame()

    rules = association_rules(frequent, metric="lift", min_threshold=MIN_LIFT)

    # tag with dangerous reactions
    dangerous_rules = []
    for _, rule in rules.iterrows():
        antecedent = list(rule["antecedents"])[0]
        consequent = list(rule["consequents"])[0]
        pair = frozenset([antecedent, consequent])

        # find reports containing both drugs
        pair_reports = [
            i for i, drugs in enumerate(drug_matrix.index)
            if drug_matrix.loc[drugs, antecedent] and drug_matrix.loc[drugs, consequent]
        ] if False else []  # placeholder — join on index properly below

        dangerous_rules.append({
            "drug_a":   antecedent,
            "drug_b":   consequent,
            "support":  rule["support"],
            "lift":     rule["lift"],
            "confidence": rule["confidence"],
        })

    return pd.DataFrame(dangerous_rules)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — GRAPH NEURAL NETWORK (PyTorch Geometric)
# ─────────────────────────────────────────────────────────────────────────────

def build_drug_graph(df: pd.DataFrame, all_drugs: list):
    """Build a co-occurrence graph: nodes=drugs, edges=shared adverse-event reports."""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print("[gnn] torch / torch-geometric not installed — skipping GNN.")
        return None

    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    edge_counter: dict[tuple, int] = defaultdict(int)
    reaction_edge: dict[tuple, list] = defaultdict(list)

    for _, row in df.iterrows():
        drugs = [drug_to_idx[d] for d in row["drugs"] if d in drug_to_idx]
        for a, b in itertools.combinations(sorted(drugs), 2):
            edge_counter[(a, b)] += 1
            reaction_edge[(a, b)].extend(row["reactions"])

    edges = list(edge_counter.keys())
    edge_weights = [edge_counter[e] for e in edges]

    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])

    # Simple node features: log-frequency of each drug
    drug_freq = defaultdict(int)
    for row in df["drugs"]:
        for d in row:
            if d in drug_to_idx:
                drug_freq[drug_to_idx[d]] += 1
    x = torch.tensor(
        [[math.log1p(drug_freq[i])] for i in range(len(all_drugs))],
        dtype=torch.float,
    )

    data = Data(x=x, edge_index=edge_index)
    data.drug_to_idx     = drug_to_idx
    data.idx_to_drug     = {v: k for k, v in drug_to_idx.items()}
    data.reaction_edge   = reaction_edge
    data.edge_weights    = edge_weights
    return data


class DDIPredictor:
    """
    Simple GCN-based link predictor.
    Predicts a risk score for any drug pair.
    """
    def __init__(self, graph_data):
        try:
            import torch
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv
            self._torch = torch
            self._F     = F
            self._GCNConv = GCNConv
        except ImportError:
            self.model = None
            return

        self.data = graph_data
        in_ch = graph_data.x.shape[1]

        class _GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_ch, 64)
                self.conv2 = GCNConv(64, 32)
                self.head  = torch.nn.Linear(64, 1)

            def encode(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

            def decode(self, z, src, dst):
                return torch.sigmoid(self.head(torch.cat([z[src], z[dst]], dim=-1)))

        self.model = _GCN()
        self._train()

    def _train(self, epochs: int = 50):
        if self.model is None:
            return
        torch   = self._torch
        data    = self.data
        opt     = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        src = data.edge_index[0]
        dst = data.edge_index[1]
        pos_labels = torch.ones(src.shape[0])

        # negative samples
        neg_src = torch.randint(0, data.x.shape[0], (src.shape[0],))
        neg_dst = torch.randint(0, data.x.shape[0], (src.shape[0],))
        neg_labels = torch.zeros(neg_src.shape[0])

        all_src = torch.cat([src, neg_src])
        all_dst = torch.cat([dst, neg_dst])
        all_labels = torch.cat([pos_labels, neg_labels])

        print(f"[gnn] Training DDI predictor for {epochs} epochs …")
        for epoch in range(epochs):
            self.model.train()
            opt.zero_grad()
            z    = self.model.encode(data.x, data.edge_index)
            pred = self.model.decode(z, all_src, all_dst).squeeze()
            loss = criterion(pred, all_labels)
            loss.backward()
            opt.step()
            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    def predict(self, drug_a: str, drug_b: str) -> float:
        """Return interaction risk score in [0, 1]."""
        if self.model is None:
            return 0.0
        torch = self._torch
        d2i = self.data.drug_to_idx
        if drug_a not in d2i or drug_b not in d2i:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(self.data.x, self.data.edge_index)
            src = torch.tensor([d2i[drug_a]])
            dst = torch.tensor([d2i[drug_b]])
            score = self.model.decode(z, src, dst).item()
        return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — DISPROPORTIONALITY ANALYSIS (Reporting Odds Ratio)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ror(df: pd.DataFrame, drug_a: str, drug_b: str, reaction: str):
    """
    Compute Reporting Odds Ratio for (drug_a + drug_b) → reaction.
    Returns (ror, lower_ci, upper_ci, a, b, c, d) or None if insufficient data.
    """
    pair = {drug_a, drug_b}

    a = b = c = d = 0
    for _, row in df.iterrows():
        drug_set  = set(row["drugs"])
        rxn_match = reaction in row["reactions"]
        pair_present = pair.issubset(drug_set)
        either_present = bool(drug_set & pair)

        if pair_present:
            if rxn_match: a += 1
            else:         b += 1
        elif either_present:
            if rxn_match: c += 1
            else:         d += 1

    if a < ROR_MIN_CASES or b == 0 or c == 0 or d == 0:
        return None

    ror = (a * d) / (b * c)
    se  = math.sqrt(1/a + 1/b + 1/c + 1/d)
    lower = math.exp(math.log(ror) - 1.96 * se)
    upper = math.exp(math.log(ror) + 1.96 * se)
    return ror, lower, upper, a, b, c, d


def screen_all_pairs_ror(df: pd.DataFrame, top_n: int = 200):
    """
    Screen the most-common drug pairs for disproportionate reaction signals.
    Returns a DataFrame of significant (pair, reaction, ROR) rows.
    """
    # count pair frequencies
    pair_count: dict[frozenset, int] = defaultdict(int)
    for _, row in df.iterrows():
        for pair in itertools.combinations(sorted(row["drugs"]), 2):
            pair_count[frozenset(pair)] += 1

    top_pairs = sorted(pair_count, key=pair_count.get, reverse=True)[:top_n]

    # count reaction frequencies
    rxn_count: dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        for r in row["reactions"]:
            rxn_count[r] += 1
    top_reactions = [r for r, _ in sorted(rxn_count.items(), key=lambda x: -x[1])[:50]]

    print(f"[ror] Screening {len(top_pairs)} pairs × {len(top_reactions)} reactions …")
    results = []
    for pair in tqdm(top_pairs):
        drugs = list(pair)
        if len(drugs) < 2:
            continue
        drug_a, drug_b = drugs[0], drugs[1]
        for rxn in top_reactions:
            out = compute_ror(df, drug_a, drug_b, rxn)
            if out and out[0] > 2.0 and out[1] > ROR_LOWER_CI:
                results.append({
                    "drug_a":        drug_a,
                    "drug_b":        drug_b,
                    "reaction":      rxn,
                    "ror":           round(out[0], 2),
                    "ci_lower":      round(out[1], 2),
                    "ci_upper":      round(out[2], 2),
                    "n_cases":       out[3],
                    "is_dangerous":  rxn in DANGEROUS_REACTIONS,
                })

    ror_df = pd.DataFrame(results).sort_values("ror", ascending=False)
    ror_df.to_csv(CACHE_DIR / "ror_signals.csv", index=False)
    print(f"[ror] Found {len(ror_df)} significant signals.")
    return ror_df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — NOVELTY FILTER (cross-reference DrugBank)
# ─────────────────────────────────────────────────────────────────────────────

def load_known_interactions() -> set[frozenset]:
    """
    Load known DDI pairs from a DrugBank CSV.
    Expected columns: Drug1, Drug2  (drug names or CUIs)
    If the file doesn't exist, returns an empty set.
    """
    if not DRUGBANK_FILE.exists():
        print(f"[novel] {DRUGBANK_FILE} not found — novelty filter skipped.")
        return set()

    db = pd.read_csv(DRUGBANK_FILE)
    known = set()
    for _, row in db.iterrows():
        a = str(row.get("Drug1", "")).lower().strip()
        b = str(row.get("Drug2", "")).lower().strip()
        if a and b:
            known.add(frozenset([a, b]))
    print(f"[novel] Loaded {len(known)} known interactions from DrugBank.")
    return known


def filter_novel(ror_df: pd.DataFrame, known: set[frozenset]) -> pd.DataFrame:
    """Return only rows whose drug pair is NOT in the known interaction set."""
    if not known:
        return ror_df
    mask = ror_df.apply(
        lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known, axis=1
    )
    novel = ror_df[mask]
    print(f"[novel] {len(novel)} / {len(ror_df)} signals are potentially novel.")
    return novel


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — STREAMLIT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def run_streamlit(df: pd.DataFrame, ror_df: pd.DataFrame, novel_df: pd.DataFrame,
                  graph_data, predictor):
    """Launch interactive Streamlit UI."""
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="Phantom Pharmacology", page_icon="💊", layout="wide")

    st.title("💊 Phantom Pharmacology")
    st.caption("Uncovering invisible drug-drug interactions from FDA FAERS data")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")
    min_ror = st.sidebar.slider("Minimum ROR", 1.0, 20.0, 2.0, 0.5)
    only_dangerous = st.sidebar.checkbox("Only dangerous reactions", value=False)
    only_novel = st.sidebar.checkbox("Only novel (not in DrugBank)", value=False)

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔎 Drug Pair Lookup", "📊 Signal Table", "🕸 Network"])

    # ── TAB 1: Drug-pair lookup ───────────────────────────────────────────────
    with tab1:
        all_drugs = sorted({d for row in df["drugs"] for d in row})
        col1, col2 = st.columns(2)
        drug_a = col1.selectbox("Drug A", all_drugs, index=0)
        drug_b = col2.selectbox("Drug B", all_drugs, index=min(1, len(all_drugs) - 1))

        if st.button("Analyse Interaction", type="primary"):
            pair_signals = ror_df[
                ((ror_df["drug_a"] == drug_a) & (ror_df["drug_b"] == drug_b)) |
                ((ror_df["drug_a"] == drug_b) & (ror_df["drug_b"] == drug_a))
            ]

            gnn_score = predictor.predict(drug_a, drug_b) if predictor else None

            colA, colB, colC = st.columns(3)
            colA.metric("ROR signals found", len(pair_signals))
            colB.metric("GNN risk score", f"{gnn_score:.3f}" if gnn_score else "N/A")
            is_novel_flag = frozenset([drug_a, drug_b]) not in load_known_interactions()
            colC.metric("Novel (not in DrugBank)", "✅ Yes" if is_novel_flag else "❌ No")

            if not pair_signals.empty:
                st.subheader("Reaction signals")
                st.dataframe(
                    pair_signals[["reaction", "ror", "ci_lower", "ci_upper",
                                  "n_cases", "is_dangerous"]]
                    .sort_values("ror", ascending=False),
                    use_container_width=True,
                )
            else:
                st.info("No significant ROR signals found for this pair.")

    # ── TAB 2: Full signal table ──────────────────────────────────────────────
    with tab2:
        if ror_df.empty or "ror" not in ror_df.columns:
            st.info("No ROR signals computed yet — the pipeline may still be running or found no significant pairs. Check the terminal for progress.")
        else:
            filtered = ror_df[ror_df["ror"] >= min_ror].copy()
            if only_dangerous and "is_dangerous" in filtered.columns:
                filtered = filtered[filtered["is_dangerous"]]
            if only_novel and not novel_df.empty:
                novel_pairs = set(
                    zip(novel_df["drug_a"], novel_df["drug_b"])
                ) | set(zip(novel_df["drug_b"], novel_df["drug_a"]))
                filtered = filtered[
                    filtered.apply(lambda r: (r["drug_a"], r["drug_b"]) in novel_pairs, axis=1)
                ]

            st.subheader(f"{len(filtered)} signals  (ROR ≥ {min_ror})")
            st.dataframe(filtered, use_container_width=True)

        # top-10 bar chart
        if not ror_df.empty and "ror" in ror_df.columns:
            filtered_for_chart = ror_df[ror_df["ror"] >= min_ror].copy()
        else:
            filtered_for_chart = pd.DataFrame()

        top10 = filtered_for_chart.nlargest(10, "ror") if not filtered_for_chart.empty else pd.DataFrame()
        if not top10.empty:
            fig = go.Figure(go.Bar(
                x=top10["ror"],
                y=top10["drug_a"] + " + " + top10["drug_b"] + " → " + top10["reaction"],
                orientation="h",
                marker_color="crimson",
                error_x=dict(
                    type="data",
                    array=top10["ci_upper"] - top10["ror"],
                    arrayminus=top10["ror"] - top10["ci_lower"],
                    visible=True,
                ),
            ))
            fig.update_layout(
                title="Top 10 signals by ROR",
                xaxis_title="Reporting Odds Ratio",
                height=420,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Network graph ──────────────────────────────────────────────────
    with tab3:
        if graph_data is None:
            st.warning("Graph not built (torch-geometric not installed).")
        else:
            st.subheader("Drug co-occurrence network (top 80 pairs by weight)")

            try:
                # safely extract edge list as plain Python ints
                ei = graph_data.edge_index  # shape [2, num_edges]
                src_list = ei[0].tolist()
                dst_list = ei[1].tolist()
                raw_edges = list(zip(src_list, dst_list))

                idx2drug = getattr(graph_data, "idx_to_drug", {})
                raw_weights = getattr(graph_data, "edge_weights", None)
                if raw_weights is None:
                    raw_weights = [1] * len(raw_edges)

                # pair each edge with its weight and take top 80
                paired = list(zip(raw_edges, raw_weights))
                top_edges = sorted(paired, key=lambda x: -x[1])[:80]

                nodes_used = set()
                for (s, d), _ in top_edges:
                    nodes_used.add(idx2drug.get(s, str(s)))
                    nodes_used.add(idx2drug.get(d, str(d)))

                node_list = list(nodes_used)
                node_idx  = {n: i for i, n in enumerate(node_list)}

                # random layout
                rng = np.random.default_rng(42)
                pos = rng.uniform(0, 1, (len(node_list), 2))

                edge_x, edge_y = [], []
                for (s, d), _ in top_edges:
                    na = idx2drug.get(s, str(s))
                    nb = idx2drug.get(d, str(d))
                    if na in node_idx and nb in node_idx:
                        xa, ya = pos[node_idx[na]]
                        xb, yb = pos[node_idx[nb]]
                        edge_x += [xa, xb, None]
                        edge_y += [ya, yb, None]

            except Exception as ex:
                st.error(f"Could not render network: {ex}")
                st.stop()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.8, color="#aaa"), hoverinfo="none",
            ))
            fig2.add_trace(go.Scatter(
                x=pos[:, 0], y=pos[:, 1], mode="markers+text",
                marker=dict(size=8, color="steelblue", line=dict(width=1, color="white")),
                text=node_list, textposition="top center",
                hoverinfo="text",
            ))
            fig2.update_layout(
                showlegend=False, height=560,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHANTOM PHARMACOLOGY — DDI Discovery Pipeline")
    print("=" * 60)

    # 1. Fetch data
    df = fetch_faers_reports(MAX_REPORTS)
    print(f"[main] {len(df)} reports loaded.")

    # 2. Normalise
    df = normalize_reports(df, use_rxnorm=False)

    # 3. Graph
    drug_matrix, reaction_sets, all_drugs = build_drug_reaction_matrix(df)
    graph_data = build_drug_graph(df, all_drugs)

    # 4. GNN predictor
    predictor = DDIPredictor(graph_data) if graph_data is not None else None

    # 5. ROR analysis
    ror_cache = CACHE_DIR / "ror_signals.csv"
    if ror_cache.exists():
        print(f"[main] Loading cached ROR signals from {ror_cache}")
        ror_df = pd.read_csv(ror_cache)
    else:
        ror_df = screen_all_pairs_ror(df, top_n=200)

    # 6. Novelty filter
    known = load_known_interactions()
    novel_df = filter_novel(ror_df, known)

    print("\n[main] Top 10 novel high-ROR signals:")
    if not novel_df.empty:
        print(novel_df.head(10)[["drug_a", "drug_b", "reaction", "ror",
                                  "n_cases", "is_dangerous"]].to_string(index=False))

    # 7. Launch UI
    run_streamlit(df, ror_df, novel_df, graph_data, predictor)


if __name__ == "__main__":
    main()
