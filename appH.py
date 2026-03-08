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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_lookup(drug_a, drug_b, ror_df, known_interactions, predictor):
    """Compute and return all lookup results for a drug pair."""
    pair_signals = ror_df[
        ((ror_df["drug_a"] == drug_a) & (ror_df["drug_b"] == drug_b)) |
        ((ror_df["drug_a"] == drug_b) & (ror_df["drug_b"] == drug_a))
    ].copy()
    gnn_score = predict_gnn_score(predictor, drug_a, drug_b)
    is_novel = frozenset([drug_a, drug_b]) not in known_interactions
    return pair_signals, gnn_score, is_novel


def render_lookup_results(pair_signals, gnn_score, is_novel, drug_a, drug_b):
    """Render metric cards, narrative summary, and signals table."""
    n_signals = len(pair_signals)
    n_dangerous = int(pair_signals["is_dangerous"].sum()) if not pair_signals.empty else 0

    # ── Metrics row ──────────────────────────────────────────────────────────
    colA, colB, colC = st.columns(3)
    colA.metric(
        "ROR signals found", n_signals,
        help=(
            "Number of adverse reactions where this drug pair shows a statistically "
            "elevated Reporting Odds Ratio (ROR > 2, lower CI > 1). "
            "A higher count means more reaction types are flagged for this combination."
        ),
    )
    colB.metric(
        "GNN risk score",
        f"{gnn_score:.3f}" if gnn_score is not None else "N/A",
        help="Graph Neural Network predicted interaction risk (0 = low, 1 = high).",
    )
    colC.metric(
        "Novel (not in DrugBank)",
        "✅ Yes" if is_novel else "❌ No",
        help="Whether this drug pair is absent from the DrugBank known-interactions database.",
    )

    # ── Narrative summary ────────────────────────────────────────────────────
    if n_signals > 0:
        top = pair_signals.sort_values("ror", ascending=False).iloc[0]
        danger_note = (
            f" **{n_dangerous} of these are classified as dangerous reactions.**"
            if n_dangerous > 0 else ""
        )
        novel_note = (
            " This pair is **not in DrugBank** and may represent a novel signal."
            if is_novel else
            " This pair is already documented in DrugBank."
        )
        st.info(
            f"**{drug_a.title()} + {drug_b.title()}** has **{n_signals}** flagged reaction(s)."
            f" The strongest signal is **{top['reaction']}** "
            f"(ROR {top['ror']:.1f}, 95% CI {top['ci_lower']:.1f}–{top['ci_upper']:.1f}, "
            f"n={int(top['n_cases'])})."
            f"{danger_note}{novel_note}"
        )
    else:
        st.info(
            f"No significant ROR signals found for **{drug_a.title()} + {drug_b.title()}**. "
            "This may mean the combination is safe, or simply that it is under-reported in FAERS."
        )

    # ── Signals table ────────────────────────────────────────────────────────
    if not pair_signals.empty:
        st.subheader("Reaction signals")
        display_df = pair_signals[
            ["reaction", "ror", "ci_lower", "ci_upper", "n_cases", "is_dangerous"]
        ].sort_values("ror", ascending=False).copy()
        display_df["is_dangerous"] = display_df["is_dangerous"].apply(
            lambda x: "⚠️ Yes" if x else ""
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "ror":          st.column_config.NumberColumn("ROR",       format="%.2f"),
                "ci_lower":     st.column_config.NumberColumn("CI Lower",  format="%.2f"),
                "ci_upper":     st.column_config.NumberColumn("CI Upper",  format="%.2f"),
                "n_cases":      st.column_config.NumberColumn("Reports"),
                "is_dangerous": st.column_config.TextColumn(
                    "Dangerous",
                    help="Flagged as a serious reaction category (e.g. rhabdomyolysis, cardiac arrest).",
                ),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="InteRx", page_icon="💊", layout="wide")

    # ── Session state initialisation ─────────────────────────────────────────
    if "lookup_result" not in st.session_state:
        st.session_state.lookup_result = None   # last analysed pair result
    if "history" not in st.session_state:
        st.session_state.history = []           # list of (drug_a, drug_b) tuples
    if "network_lookup" not in st.session_state:
        st.session_state.network_lookup = None  # drug clicked in network tab

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("InteRx")
    st.caption("Uncovering hidden drug-drug interactions from FDA FAERS data")

    # Check pipeline ran
    if not DRUGS_FILE.exists():
        st.error("No data found. Please run the pipeline first:")
        st.code("python pipeline.py", language="bash")
        st.stop()

    # Load data
    all_drugs        = load_drugs()
    ror_df           = load_ror_signals()
    known_interactions = load_known_interactions()
    predictor        = load_gnn_predictor()

    # ── Dashboard header metrics ─────────────────────────────────────────────
    n_novel = 0
    if not ror_df.empty and known_interactions:
        n_novel = int(ror_df.apply(
            lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1
        ).sum())
    n_dangerous_signals = int(ror_df["is_dangerous"].sum()) if not ror_df.empty and "is_dangerous" in ror_df.columns else 0

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Drugs indexed",      f"{len(all_drugs):,}")
    h2.metric("ROR signals",        f"{len(ror_df):,}")
    h3.metric("Novel signals",      f"{n_novel:,}",
              help="Signals not found in DrugBank known interactions.")
    h4.metric("Dangerous signals",  f"{n_dangerous_signals:,}",
              help="Signals involving a serious reaction category.")
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")
    min_ror = st.sidebar.slider(
        "Minimum ROR", 1.0, 20.0, 2.0, 0.5,
        help=(
            "Reporting Odds Ratio (ROR) measures how much more often a reaction "
            "is reported for a specific drug combination compared to all other drugs. "
            "An ROR of 2.0 means the reaction is reported twice as often for this pair. "
            "Values above 2 with a lower confidence interval above 1 are considered "
            "a meaningful pharmacovigilance signal, which is the same threshold used by FDA analysts."
        ),
    )
    only_dangerous = st.sidebar.checkbox("Only dangerous reactions", value=False)
    only_novel     = st.sidebar.checkbox("Only novel (not in DrugBank)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"GNN model: {'✅ loaded' if predictor else '❌ not found'}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Drug Pair Lookup", "Signal Table", "Network", "Summary"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Drug Pair Lookup
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        drug_a = col1.selectbox("Drug A", all_drugs, index=0,
                                placeholder="Type to search…")
        drug_b = col2.selectbox("Drug B", all_drugs,
                                index=min(1, len(all_drugs) - 1),
                                placeholder="Type to search…")

        # Pre-populate from network click
        if st.session_state.network_lookup:
            st.info(
                f"Jumped from Network tab: showing top pair for "
                f"**{st.session_state.network_lookup}**"
            )

        if st.button("Analyse Interaction", type="primary"):
            result = run_lookup(drug_a, drug_b, ror_df, known_interactions, predictor)
            st.session_state.lookup_result = (drug_a, drug_b, *result)

            # Add to history (keep last 5, no duplicates)
            entry = (drug_a, drug_b)
            st.session_state.history = [
                h for h in st.session_state.history if h != entry
            ][:4]
            st.session_state.history.insert(0, entry)
            st.session_state.network_lookup = None

        # Render persisted result
        if st.session_state.lookup_result:
            da, db, signals, gnn, novel = st.session_state.lookup_result
            st.markdown(f"#### Results for **{da.title()}** + **{db.title()}**")
            render_lookup_results(signals, gnn, novel, da, db)

        # Recent history
        if st.session_state.history:
            st.markdown("---")
            st.markdown("**Recently analysed pairs**")
            hist_cols = st.columns(min(len(st.session_state.history), 5))
            for i, (ha, hb) in enumerate(st.session_state.history):
                if hist_cols[i].button(f"{ha.title()} + {hb.title()}", key=f"hist_{i}"):
                    result = run_lookup(ha, hb, ror_df, known_interactions, predictor)
                    st.session_state.lookup_result = (ha, hb, *result)
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Signal Table
    # ─────────────────────────────────────────────────────────────────────────
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
                        axis=1,
                    )
                ]

            st.subheader(f"{len(filtered)} signals (ROR ≥ {min_ror})")

            # Export button
            st.download_button(
                label="⬇️ Export as CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="interx_signals.csv",
                mime="text/csv",
            )

            # Styled table
            display = filtered.copy()
            if "is_dangerous" in display.columns:
                display["is_dangerous"] = display["is_dangerous"].apply(
                    lambda x: "⚠️ Yes" if x else ""
                )
            st.dataframe(
                display, use_container_width=True,
                column_config={
                    "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                    "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                    "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                    "is_dangerous": st.column_config.TextColumn("Dangerous"),
                },
            )

            # Top 10 bar chart — accent color
            top10 = filtered.nlargest(10, "ror") if not filtered.empty else pd.DataFrame()
            if not top10.empty:
                fig = go.Figure(go.Bar(
                    x=top10["ror"],
                    y=top10["drug_a"] + " + " + top10["drug_b"] + " → " + top10["reaction"],
                    orientation="h",
                    marker_color="#00e9ab",
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
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#00e9ab"),
                    yaxis=dict(autorange="reversed"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Network
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        if not GRAPH_FILE.exists():
            st.warning("Graph not built. Run pipeline.py first.")
        else:
            st.subheader("Drug co-occurrence network (top 80 pairs)")
            st.caption("Click a node to look up its top interaction pair in the Lookup tab.")

            try:
                with open(GRAPH_FILE, "rb") as f:
                    graph_data = pickle.load(f)

                ei          = graph_data.edge_index
                src_list    = ei[0].tolist()
                dst_list    = ei[1].tolist()
                raw_edges   = list(zip(src_list, dst_list))
                idx2drug    = graph_data.idx_to_drug
                raw_weights = graph_data.edge_weights or [1] * len(raw_edges)

                paired    = list(zip(raw_edges, raw_weights))
                top_edges = sorted(paired, key=lambda x: -x[1])[:80]

                nodes_used = set()
                for (s, d), _ in top_edges:
                    nodes_used.add(idx2drug.get(s, str(s)))
                    nodes_used.add(idx2drug.get(d, str(d)))

                node_list = list(nodes_used)
                node_idx  = {n: i for i, n in enumerate(node_list)}

                rng = np.random.default_rng(42)
                pos = rng.uniform(0, 1, (len(node_list), 2))

                # Flag nodes that appear in high-ROR signals
                flagged_drugs = set()
                if not ror_df.empty:
                    flagged_drugs = set(ror_df["drug_a"]) | set(ror_df["drug_b"])
                node_colors = [
                    "#00e9ab" if n in flagged_drugs else "#4a6fa5"
                    for n in node_list
                ]
                node_sizes = [
                    12 if n in flagged_drugs else 7
                    for n in node_list
                ]

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
                    line=dict(width=0.8, color="rgba(128,128,128,0.4)"),
                    hoverinfo="none",
                ))
                fig.add_trace(go.Scatter(
                    x=pos[:, 0], y=pos[:, 1],
                    mode="markers+text",
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color="#0a0a0a"),
                    ),
                    text=node_list,
                    textposition="top center",
                    hoverinfo="text",
                    customdata=node_list,
                ))
                fig.update_layout(
                    showlegend=False, height=580,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=0, r=0, t=10, b=0),
                )

                # Render chart and capture click events
                event = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_select="rerun",
                    key="network_chart",
                )

                # Handle node click → find top partner and push to lookup tab
                if event and event.get("selection") and event["selection"].get("points"):
                    clicked_idx = event["selection"]["points"][0].get("point_index")
                    if clicked_idx is not None and clicked_idx < len(node_list):
                        clicked_drug = node_list[clicked_idx]

                        # Find the partner with the highest ROR for this drug
                        if not ror_df.empty:
                            drug_rows = ror_df[
                                (ror_df["drug_a"] == clicked_drug) |
                                (ror_df["drug_b"] == clicked_drug)
                            ].sort_values("ror", ascending=False)

                            if not drug_rows.empty:
                                top_row   = drug_rows.iloc[0]
                                partner   = top_row["drug_b"] if top_row["drug_a"] == clicked_drug else top_row["drug_a"]
                                result    = run_lookup(clicked_drug, partner, ror_df, known_interactions, predictor)
                                st.session_state.lookup_result  = (clicked_drug, partner, *result)
                                st.session_state.network_lookup = clicked_drug

                                # Add to history
                                entry = (clicked_drug, partner)
                                st.session_state.history = [
                                    h for h in st.session_state.history if h != entry
                                ][:4]
                                st.session_state.history.insert(0, entry)

                                st.success(
                                    f"Loaded **{clicked_drug.title()} + {partner.title()}** "
                                    f"→ switch to the **Drug Pair Lookup** tab to see results."
                                )

                # Legend
                st.markdown(
                    "🟢 **Teal nodes** = drugs appearing in at least one flagged signal &nbsp;|&nbsp; "
                    "🔵 **Blue nodes** = no signals found"
                )

            except Exception as e:
                st.error(f"Could not render network: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — Summary
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Pipeline Summary")

        if ror_df.empty:
            st.info("No signals computed yet. Run pipeline.py first.")
        else:
            # Apply same filters as signal table
            summary_df = ror_df[ror_df["ror"] >= min_ror].copy()
            if only_dangerous and "is_dangerous" in summary_df.columns:
                summary_df = summary_df[summary_df["is_dangerous"]]
            if only_novel and known_interactions:
                summary_df = summary_df[
                    summary_df.apply(
                        lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions,
                        axis=1,
                    )
                ]

            n_pairs     = summary_df.groupby(["drug_a", "drug_b"]).ngroups
            n_dang      = int(summary_df["is_dangerous"].sum()) if "is_dangerous" in summary_df.columns else 0
            n_nov       = 0
            if known_interactions:
                n_nov = int(summary_df.apply(
                    lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions,
                    axis=1,
                ).sum())

            # Aggregate metrics
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Filtered signals",         f"{len(summary_df):,}")
            s2.metric("Unique drug pairs",         f"{n_pairs:,}")
            s3.metric("Dangerous reaction signals",f"{n_dang:,}")
            s4.metric("Novel signals",             f"{n_nov:,}")
            st.divider()

            col_left, col_right = st.columns(2)

            # Top 5 most suspicious drugs (most flagged pairs)
            with col_left:
                st.markdown("#### Top 5 most suspicious drugs")
                st.caption("Ranked by number of flagged pairs they appear in.")
                drug_pair_counts = (
                    pd.concat([summary_df["drug_a"], summary_df["drug_b"]])
                    .value_counts()
                    .head(5)
                    .reset_index()
                )
                drug_pair_counts.columns = ["Drug", "Flagged pairs"]
                st.dataframe(drug_pair_counts, use_container_width=True, hide_index=True)

            # Reaction category breakdown
            with col_right:
                st.markdown("#### Reaction breakdown")
                st.caption("Most frequently flagged reaction types.")
                rxn_counts = (
                    summary_df.groupby("reaction")
                    .agg(signals=("ror", "count"), max_ror=("ror", "max"))
                    .sort_values("signals", ascending=False)
                    .head(10)
                    .reset_index()
                )
                rxn_counts.columns = ["Reaction", "# Signals", "Max ROR"]
                st.dataframe(
                    rxn_counts, use_container_width=True, hide_index=True,
                    column_config={
                        "Max ROR": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            st.divider()

            # Dangerous novel signals — the headline result
            st.markdown("#### Dangerous novel signals")
            st.caption(
                "Signals that are both classified as dangerous AND absent from DrugBank — "
                "the most clinically significant findings."
            )
            if known_interactions and "is_dangerous" in ror_df.columns:
                dang_novel = ror_df[ror_df["is_dangerous"]].copy()
                dang_novel = dang_novel[
                    dang_novel.apply(
                        lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions,
                        axis=1,
                    )
                ].sort_values("ror", ascending=False).head(20)

                if dang_novel.empty:
                    st.info("No dangerous novel signals found under current filters.")
                else:
                    dang_novel["is_dangerous"] = dang_novel["is_dangerous"].apply(
                        lambda x: "⚠️ Yes" if x else ""
                    )
                    st.dataframe(
                        dang_novel, use_container_width=True,
                        column_config={
                            "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                            "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                            "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                            "is_dangerous": st.column_config.TextColumn("Dangerous"),
                        },
                    )
                    st.download_button(
                        label="⬇️ Export dangerous novel signals",
                        data=dang_novel.to_csv(index=False).encode("utf-8"),
                        file_name="interx_dangerous_novel.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Load DrugBank data to see novel signal filtering.")


if __name__ == "__main__":
    main()
