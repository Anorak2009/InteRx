"""
InteRx - Personal Drug Interaction Risk Checker
================================================
Enter all your medications and see potential interaction risks.

Uses FAERS data from the Phantom Data folder (FDA adverse event reports).

Run:
    streamlit run InteRx.py
"""

import os
import json
import math
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FAERS_DATA_DIR = Path("Phantom Data")
CACHE_DIR = Path("faers_cache")
REPORTS_FILE = CACHE_DIR / "reports.parquet"
ROR_FILE = CACHE_DIR / "ror_signals.csv"
DRUGBANK_FILE = CACHE_DIR / "drugbank_interactions.csv"

CACHE_DIR.mkdir(exist_ok=True)

DANGEROUS_REACTIONS = {
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}

# ─────────────────────────────────────────────────────────────────────────────
# FAERS ASCII DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_faers_files(data_dir: Path):
    """Find all FAERS ASCII data files in the Phantom Data directory."""
    quarters = []
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("faers_ascii_"):
            ascii_dir = folder / "ASCII"
            if ascii_dir.exists():
                quarters.append(ascii_dir)
    return quarters


def load_faers_quarter(ascii_dir: Path):
    """Load DRUG and REAC files from a single FAERS quarter."""
    # Find the files (they have year/quarter suffix like DEMO23Q1.txt)
    drug_file = reac_file = None
    for f in ascii_dir.iterdir():
        fname = f.name.upper()
        if fname.startswith("DRUG") and fname.endswith(".TXT"):
            drug_file = f
        elif fname.startswith("REAC") and fname.endswith(".TXT"):
            reac_file = f

    if not all([drug_file, reac_file]):
        return None

    # Load DRUG - only columns we need, get suspect drugs (PS=Primary Suspect, SS=Secondary Suspect)
    drug_df = pd.read_csv(
        drug_file, sep="$", dtype=str, low_memory=False,
        usecols=["primaryid", "drugname", "role_cod"]
    )
    drug_df = drug_df[drug_df["role_cod"].isin(["PS", "SS"])]
    drug_df["drugname"] = drug_df["drugname"].str.lower().str.strip()
    drug_df = drug_df[["primaryid", "drugname"]].dropna()

    # Load REAC - only columns we need
    reac_df = pd.read_csv(
        reac_file, sep="$", dtype=str, low_memory=False,
        usecols=["primaryid", "pt"]
    )
    reac_df["pt"] = reac_df["pt"].str.lower().str.strip()
    reac_df = reac_df.dropna()

    # Group drugs and reactions by primaryid
    drugs_grouped = drug_df.groupby("primaryid")["drugname"].apply(list).reset_index()
    drugs_grouped.columns = ["primaryid", "drugs"]

    # Free memory early
    del drug_df

    reacs_grouped = reac_df.groupby("primaryid")["pt"].apply(list).reset_index()
    reacs_grouped.columns = ["primaryid", "reactions"]

    # Free memory early
    del reac_df

    # Merge
    result = drugs_grouped.merge(reacs_grouped, on="primaryid", how="inner")

    # Free memory
    del drugs_grouped, reacs_grouped

    # Filter to reports with at least 2 drugs and at least 1 reaction
    result = result[result["drugs"].apply(len) >= 2]
    result = result[result["reactions"].apply(len) >= 1]

    return result


@st.cache_data(show_spinner=False)
def load_faers_from_phantom_data(max_quarters: int = 4):
    """
    Load FAERS data from the Phantom Data directory.

    Args:
        max_quarters: Limit number of quarters to load (default 4 = ~1 year).
                      Set to 0 for all quarters (requires lots of RAM).
    """
    if REPORTS_FILE.exists():
        return pd.read_parquet(REPORTS_FILE)

    quarters = find_faers_files(FAERS_DATA_DIR)
    if not quarters:
        return None

    # Limit quarters to avoid OOM - take most recent ones
    if max_quarters > 0 and len(quarters) > max_quarters:
        quarters = quarters[-max_quarters:]

    all_reports = []
    seen_ids = set()

    for ascii_dir in quarters:
        quarter_name = ascii_dir.parent.name
        df = load_faers_quarter(ascii_dir)
        if df is not None and len(df) > 0:
            # Deduplicate as we go to save memory
            df = df[~df["primaryid"].isin(seen_ids)]
            seen_ids.update(df["primaryid"].tolist())
            all_reports.append(df)

    if not all_reports:
        return None

    combined = pd.concat(all_reports, ignore_index=True)

    # Free memory
    del all_reports
    del seen_ids

    # Cache for future runs
    combined.to_parquet(REPORTS_FILE)

    return combined


@st.cache_data
def load_reports():
    """Load FAERS reports - tries Phantom Data first, then cache."""
    # Try loading from Phantom Data folder
    df = load_faers_from_phantom_data()
    if df is not None:
        return df

    # Fallback to cached reports from phantom_pharmacology.py
    if REPORTS_FILE.exists():
        return pd.read_parquet(REPORTS_FILE)

    return None


@st.cache_data
def load_ror_signals():
    """Load cached ROR signals."""
    if not ROR_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(ROR_FILE)


@st.cache_data
def load_known_interactions():
    """Load known DDI pairs from DrugBank."""
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


@st.cache_data
def get_all_drugs(df):
    """Extract all unique drug names from reports."""
    return sorted({d for row in df["drugs"] for d in row})


# ─────────────────────────────────────────────────────────────────────────────
# GNN PREDICTOR (simplified loader)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def build_predictor(df, all_drugs):
    """Build the GNN predictor for drug pairs."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data
    except ImportError:
        return None

    # Build graph
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    edge_counter = defaultdict(int)

    for _, row in df.iterrows():
        drugs = [drug_to_idx[d] for d in row["drugs"] if d in drug_to_idx]
        for a, b in itertools.combinations(sorted(drugs), 2):
            edge_counter[(a, b)] += 1

    edges = list(edge_counter.keys())
    if not edges:
        return None

    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])

    # Node features: log-frequency
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

    # Define model
    in_ch = x.shape[1]

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

    # Train
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    pos_src = edge_index[0]
    pos_dst = edge_index[1]
    pos_labels = torch.ones(pos_src.shape[0])

    neg_src = torch.randint(0, x.shape[0], (pos_src.shape[0],))
    neg_dst = torch.randint(0, x.shape[0], (pos_src.shape[0],))
    neg_labels = torch.zeros(neg_src.shape[0])

    all_src = torch.cat([pos_src, neg_src])
    all_dst = torch.cat([pos_dst, neg_dst])
    all_labels = torch.cat([pos_labels, neg_labels])

    for _ in range(50):
        model.train()
        opt.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pred = model.decode(z, all_src, all_dst).squeeze()
        loss = criterion(pred, all_labels)
        loss.backward()
        opt.step()

    return {"model": model, "data": data, "drug_to_idx": drug_to_idx}


def predict_risk(predictor, drug_a, drug_b):
    """Predict interaction risk score for a drug pair."""
    if predictor is None:
        return None
    try:
        import torch
    except ImportError:
        return None

    model = predictor["model"]
    data = predictor["data"]
    d2i = predictor["drug_to_idx"]

    if drug_a not in d2i or drug_b not in d2i:
        return None

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        src = torch.tensor([d2i[drug_a]])
        dst = torch.tensor([d2i[drug_b]])
        score = model.decode(z, src, dst).item()
    return round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_drug_list(drugs, ror_df, predictor, known_interactions):
    """Analyze all pairs from a list of drugs."""
    results = []

    for drug_a, drug_b in itertools.combinations(drugs, 2):
        # Get ROR signals for this pair
        pair_signals = ror_df[
            ((ror_df["drug_a"] == drug_a) & (ror_df["drug_b"] == drug_b)) |
            ((ror_df["drug_a"] == drug_b) & (ror_df["drug_b"] == drug_a))
        ]

        # GNN risk score
        gnn_score = predict_risk(predictor, drug_a, drug_b)

        # Check if novel
        is_novel = frozenset([drug_a, drug_b]) not in known_interactions

        # Get max ROR and dangerous reactions
        max_ror = pair_signals["ror"].max() if not pair_signals.empty else 0
        dangerous_reactions = pair_signals[pair_signals["is_dangerous"] == True]["reaction"].tolist() if not pair_signals.empty and "is_dangerous" in pair_signals.columns else []
        all_reactions = pair_signals["reaction"].tolist() if not pair_signals.empty else []

        # Compute risk level
        risk_level = "Low"
        if gnn_score is not None and gnn_score > 0.7:
            risk_level = "High"
        elif gnn_score is not None and gnn_score > 0.4:
            risk_level = "Medium"
        if max_ror > 5:
            risk_level = "High"
        elif max_ror > 2:
            risk_level = max(risk_level, "Medium", key=lambda x: ["Low", "Medium", "High"].index(x))
        if dangerous_reactions:
            risk_level = "High"

        results.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "gnn_score": gnn_score,
            "max_ror": max_ror if max_ror > 0 else None,
            "num_signals": len(pair_signals),
            "dangerous_reactions": dangerous_reactions,
            "all_reactions": all_reactions,
            "is_novel": is_novel,
            "risk_level": risk_level,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="InteRx - Drug Interaction Checker",
        page_icon="💊",
        layout="wide"
    )

    st.title("💊 InteRx")
    st.caption("Check potential interactions between your medications")

    # Load data
    if not REPORTS_FILE.exists() and FAERS_DATA_DIR.exists():
        st.info("First run: Loading FAERS data from Phantom Data folder. This may take a moment...")

    df = load_reports()

    if df is None:
        st.error("No FAERS data found. Please ensure the 'Phantom Data' folder contains FAERS ASCII data files.")
        st.stop()

    st.sidebar.success(f"Loaded {len(df):,} adverse event reports")

    ror_df = load_ror_signals()
    known_interactions = load_known_interactions()
    all_drugs = get_all_drugs(df)

    # Build predictor (cached)
    with st.spinner("Loading interaction model..."):
        predictor = build_predictor(df, all_drugs)

    st.markdown("---")

    # Drug input
    st.subheader("Enter Your Medications")

    selected_drugs = st.multiselect(
        "Select all medications you are currently taking:",
        options=all_drugs,
        help="Start typing to search for your medications"
    )

    # Also allow manual entry for drugs not in the list
    manual_entry = st.text_input(
        "Or type drug names (comma-separated) if not found above:",
        help="Enter drug names separated by commas"
    )

    if manual_entry:
        manual_drugs = [d.strip().lower() for d in manual_entry.split(",") if d.strip()]
        selected_drugs = list(set(selected_drugs + manual_drugs))

    if len(selected_drugs) < 2:
        st.info("Please select at least 2 medications to check for interactions.")
        st.stop()

    st.markdown(f"**Analyzing {len(selected_drugs)} medications:** {', '.join(selected_drugs)}")

    # Analyze
    if st.button("Check Interactions", type="primary"):
        with st.spinner("Analyzing drug interactions..."):
            results = analyze_drug_list(selected_drugs, ror_df, predictor, known_interactions)

        if results.empty:
            st.success("No interaction data found for these drug combinations.")
            st.stop()

        # Summary metrics
        st.markdown("---")
        st.subheader("Risk Summary")

        high_risk = len(results[results["risk_level"] == "High"])
        medium_risk = len(results[results["risk_level"] == "Medium"])
        low_risk = len(results[results["risk_level"] == "Low"])
        total_pairs = len(results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pairs Checked", total_pairs)
        col2.metric("High Risk", high_risk, delta=None if high_risk == 0 else "⚠️")
        col3.metric("Medium Risk", medium_risk)
        col4.metric("Low Risk", low_risk)

        if high_risk > 0:
            st.error(f"⚠️ {high_risk} high-risk interaction(s) detected. Please consult your healthcare provider.")
        elif medium_risk > 0:
            st.warning(f"⚡ {medium_risk} medium-risk interaction(s) found. Consider discussing with your pharmacist.")
        else:
            st.success("✅ No significant interactions detected in our database.")

        # Detailed results
        st.markdown("---")
        st.subheader("Detailed Interaction Analysis")

        # Sort by risk level
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        results["risk_order"] = results["risk_level"].map(risk_order)
        results = results.sort_values("risk_order")

        for _, row in results.iterrows():
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[row["risk_level"]]

            with st.expander(f"{risk_color} {row['drug_a'].title()} + {row['drug_b'].title()} — {row['risk_level']} Risk"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    if row["gnn_score"] is not None:
                        st.metric("GNN Risk Score", f"{row['gnn_score']:.3f}")
                    else:
                        st.metric("GNN Risk Score", "N/A")

                with c2:
                    if row["max_ror"] is not None:
                        st.metric("Max ROR", f"{row['max_ror']:.2f}")
                    else:
                        st.metric("Max ROR", "No signals")

                with c3:
                    st.metric("Known Interaction", "No" if row["is_novel"] else "Yes")

                if row["dangerous_reactions"]:
                    st.error(f"**Dangerous reactions reported:** {', '.join(row['dangerous_reactions'])}")

                if row["all_reactions"] and not row["dangerous_reactions"]:
                    st.warning(f"**Reactions reported:** {', '.join(row['all_reactions'][:5])}")
                    if len(row["all_reactions"]) > 5:
                        st.caption(f"...and {len(row['all_reactions']) - 5} more")

        # Visual risk matrix
        st.markdown("---")
        st.subheader("Interaction Matrix")

        if len(selected_drugs) <= 10:
            # Create a matrix visualization
            matrix_data = []
            for d1 in selected_drugs:
                row_data = []
                for d2 in selected_drugs:
                    if d1 == d2:
                        row_data.append(0)
                    else:
                        pair_result = results[
                            ((results["drug_a"] == d1) & (results["drug_b"] == d2)) |
                            ((results["drug_a"] == d2) & (results["drug_b"] == d1))
                        ]
                        if not pair_result.empty:
                            risk = pair_result.iloc[0]["risk_level"]
                            row_data.append({"High": 3, "Medium": 2, "Low": 1}[risk])
                        else:
                            row_data.append(0)
                matrix_data.append(row_data)

            fig = go.Figure(data=go.Heatmap(
                z=matrix_data,
                x=[d.title() for d in selected_drugs],
                y=[d.title() for d in selected_drugs],
                colorscale=[
                    [0, "#f0f0f0"],      # No interaction (same drug)
                    [0.33, "#90EE90"],   # Low risk - green
                    [0.66, "#FFD700"],   # Medium risk - yellow
                    [1.0, "#FF6B6B"],    # High risk - red
                ],
                showscale=False,
                hovertemplate="<b>%{x}</b> + <b>%{y}</b><extra></extra>",
            ))
            fig.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="",
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk | ⬜ Same Drug")
        else:
            st.info("Matrix visualization available for 10 or fewer medications.")

        # Disclaimer
        st.markdown("---")
        st.caption(
            "**Disclaimer:** This tool is for informational purposes only and does not constitute medical advice. "
            "The risk assessments are based on FDA adverse event reports and machine learning models, which may not "
            "capture all interactions. Always consult your healthcare provider or pharmacist before making decisions "
            "about your medications."
        )


if __name__ == "__main__":
    main()
