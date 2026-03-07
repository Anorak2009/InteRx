"""
InteRx - Streamlit UI
====================================
Fast UI that loads precomputed results from pipeline.py

Usage:
    1. First run: python pipeline.py
    2. Then run:  streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("faers_cache")
REPORTS_FILE = CACHE_DIR / "reports.parquet"
ROR_FILE = CACHE_DIR / "ror_signals.parquet"
GRAPH_FILE = CACHE_DIR / "graph_data.pkl"
MODEL_FILE = CACHE_DIR / "gnn_model.pkl"
DRUGS_FILE = CACHE_DIR / "all_drugs.json"
DRUGBANK_FILE = CACHE_DIR / "drugbank_interactions.csv"

DANGEROUS_REACTIONS = {
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (all cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_drugs():
    """Load list of all drugs."""
    if not DRUGS_FILE.exists():
        return []
    with open(DRUGS_FILE) as f:
        return json.load(f)


@st.cache_data
def load_ror_signals():
    """Load precomputed ROR signals."""
    if not ROR_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(ROR_FILE)


@st.cache_data
def load_known_interactions():
    """Load DrugBank known interactions."""
    if not DRUGBANK_FILE.exists():
        return set()
    db = pd.read_csv(DRUGBANK_FILE)
    known = set()
    for _, row in db.iterrows():
        a = str(row.get("Drug1", "")).lower().strip()
        b = str(row.get("Drug2", "")).lower().strip()
        if a and b:
            known.add(frozenset([a, b]))
    return known


@st.cache_resource
def load_gnn_predictor():
    """Load GNN model for predictions."""
    if not MODEL_FILE.exists() or not GRAPH_FILE.exists():
        return None

    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        # Load graph
        with open(GRAPH_FILE, "rb") as f:
            graph_data = pickle.load(f)

        # Load model
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)

        # Rebuild model
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
        model.load_state_dict(model_data["state_dict"])
        model.eval()

        return {
            "model": model,
            "graph": graph_data,
            "drug_to_idx": model_data["drug_to_idx"],
        }

    except Exception as e:
        st.warning(f"Could not load GNN model: {e}")
        return None


def predict_gnn_score(predictor, drug_a: str, drug_b: str) -> float | None:
    """Get GNN risk score for a drug pair."""
    if predictor is None:
        return None

    import torch

    d2i = predictor["drug_to_idx"]
    if drug_a not in d2i or drug_b not in d2i:
        return None

    model = predictor["model"]
    graph = predictor["graph"]

    with torch.no_grad():
        z = model.encode(graph.x, graph.edge_index)
        src = torch.tensor([d2i[drug_a]])
        dst = torch.tensor([d2i[drug_b]])
        score = model.decode(z, src, dst).item()

    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="InteRx",
        page_icon="💊",
        layout="wide"
    )

    st.title("InteRx")
    st.caption("Uncovering hidden drug-drug interactions from FDA FAERS data")

    # Check if pipeline has been run
    if not DRUGS_FILE.exists():
        st.error("No data found. Please run the pipeline first:")
        st.code("python pipeline.py", language="bash")
        st.stop()

    # Load data
    all_drugs = load_drugs()
    ror_df = load_ror_signals()
    known_interactions = load_known_interactions()
    predictor = load_gnn_predictor()

    # Sidebar
    st.sidebar.header("Settings")
    min_ror = st.sidebar.slider(
        "Minimum ROR",
        1.0, 20.0, 2.0, 0.5,
        help=(
            "Reporting Odds Ratio (ROR) measures how much more often a reaction "
            "is reported for a specific drug combination compared to all other drugs. "
            "An ROR of 2.0 means the reaction is reported twice as often for this pair. "
            "Values above 2 with a lower confidence interval above 1 are considered "
            "a meaningful pharmacovigilance signal, which is the same threshold used by FDA analysts."
        ),
    )
    only_dangerous = st.sidebar.checkbox("Only dangerous reactions", value=False)
    only_novel = st.sidebar.checkbox("Only novel (not in DrugBank)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(all_drugs)} drugs loaded")
    st.sidebar.caption(f"{len(ror_df)} ROR signals")
    st.sidebar.caption(f"GNN: {'✅' if predictor else '❌'}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Drug Pair Lookup", "Signal Table", "Network"])

    # TAB 1: Drug Pair Lookup
    with tab1:
        col1, col2 = st.columns(2)
        drug_a = col1.selectbox("Drug A", all_drugs, index=0)
        drug_b = col2.selectbox("Drug B", all_drugs, index=min(1, len(all_drugs) - 1))

        if st.button("Analyse Interaction", type="primary"):
            # Find ROR signals for this pair
            pair_signals = ror_df[
                ((ror_df["drug_a"] == drug_a) & (ror_df["drug_b"] == drug_b)) |
                ((ror_df["drug_a"] == drug_b) & (ror_df["drug_b"] == drug_a))
            ]

            # GNN score
            gnn_score = predict_gnn_score(predictor, drug_a, drug_b)

            # Display metrics
            colA, colB, colC = st.columns(3)
            colA.metric(
                "ROR signals found",
                len(pair_signals),
                help=(
                    "Number of adverse reactions where this drug pair shows a "
                    "statistically elevated Reporting Odds Ratio (ROR > 2, lower CI > 1). "
                    "A higher count means more reaction types are flagged for this combination."
                ),
            )
            colB.metric(
                "GNN risk score",
                f"{gnn_score:.3f}" if gnn_score is not None else "N/A",
                help="Graph Neural Network predicted interaction risk (0 = low, 1 = high).",
            )
            is_novel = frozenset([drug_a, drug_b]) not in known_interactions
            colC.metric(
                "Novel (not in DrugBank)",
                "✅ Yes" if is_novel else "❌ No",
                help="Whether this drug pair is absent from the DrugBank known-interactions database.",
            )

            if not pair_signals.empty:
                st.subheader("Reaction signals")
                st.dataframe(
                    pair_signals[["reaction", "ror", "ci_lower", "ci_upper", "n_cases", "is_dangerous"]]
                    .sort_values("ror", ascending=False),
                    use_container_width=True,
                )
            else:
                st.info("No significant ROR signals found for this pair.")

    # TAB 2: Signal Table
    with tab2:
        if ror_df.empty:
            st.info("No ROR signals found. Run pipeline.py to compute them.")
        else:
            filtered = ror_df[ror_df["ror"] >= min_ror].copy()

            if only_dangerous and "is_dangerous" in filtered.columns:
                filtered = filtered[filtered["is_dangerous"]]

            if only_novel and known_interactions:
                filtered = filtered[
                    filtered.apply(
                        lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions,
                        axis=1
                    )
                ]

            st.subheader(f"{len(filtered)} signals (ROR ≥ {min_ror})")
            st.dataframe(filtered, use_container_width=True)

            # Top 10 bar chart
            top10 = filtered.nlargest(10, "ror") if not filtered.empty else pd.DataFrame()
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

    # TAB 3: Network
    with tab3:
        if not GRAPH_FILE.exists():
            st.warning("Graph not built. Run pipeline.py first.")
        else:
            st.subheader("Drug co-occurrence network (top 80 pairs)")

            try:
                with open(GRAPH_FILE, "rb") as f:
                    graph_data = pickle.load(f)

                ei = graph_data.edge_index
                src_list = ei[0].tolist()
                dst_list = ei[1].tolist()
                raw_edges = list(zip(src_list, dst_list))

                idx2drug = graph_data.idx_to_drug
                raw_weights = graph_data.edge_weights or [1] * len(raw_edges)

                # Top 80 edges
                paired = list(zip(raw_edges, raw_weights))
                top_edges = sorted(paired, key=lambda x: -x[1])[:80]

                nodes_used = set()
                for (s, d), _ in top_edges:
                    nodes_used.add(idx2drug.get(s, str(s)))
                    nodes_used.add(idx2drug.get(d, str(d)))

                node_list = list(nodes_used)
                node_idx = {n: i for i, n in enumerate(node_list)}

                # Layout
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

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.8, color="#aaa"), hoverinfo="none",
                ))
                fig.add_trace(go.Scatter(
                    x=pos[:, 0], y=pos[:, 1], mode="markers+text",
                    marker=dict(size=8, color="steelblue", line=dict(width=1, color="white")),
                    text=node_list, textposition="top center",
                    hoverinfo="text",
                ))
                fig.update_layout(
                    showlegend=False, height=560,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Could not render network: {e}")


if __name__ == "__main__":
    main()
