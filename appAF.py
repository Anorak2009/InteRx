"""
InteRx - Streamlit UI
====================================
Fast UI that loads precomputed results from pipeline.py

Usage:
    1. First run: python pipeline.py
    2. Then run:  streamlit run app.py

PillScan integration: scan a pill bottle label via your webcam
to automatically add medications to the My Medications tab.
"""

import json
import os
import pickle
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path("faers_cache")
REPORTS_FILE = CACHE_DIR / "reports.parquet"
ROR_FILE     = CACHE_DIR / "ror_signals.parquet"
GRAPH_FILE   = CACHE_DIR / "graph_data.pkl"
MODEL_FILE   = CACHE_DIR / "gnn_model.pkl"
DRUGS_FILE   = CACHE_DIR / "all_drugs.json"
DRUGBANK_FILE = Path("full database.csv")

DANGEROUS_REACTIONS = {
    "rhabdomyolysis", "qt prolongation", "serotonin syndrome",
    "torsade de pointes", "stevens-johnson syndrome", "agranulocytosis",
    "anaphylaxis", "cardiac arrest", "liver failure", "renal failure",
    "haemorrhage", "ventricular fibrillation", "hypoglycaemia",
    "respiratory failure", "pulmonary embolism",
}

LLM_SYSTEM_PROMPT = (
    "You are a careful, knowledgeable clinical pharmacology assistant. "
    "You communicate statistical drug safety signals clearly to patients. "
    "You always remind users to consult their healthcare provider. "
    "You never diagnose or prescribe."
)

PILLSCAN_PROMPT = """You are a pharmaceutical label expert. Carefully examine this pill bottle image and extract every piece of information visible on the label.

Return ONLY a valid JSON object with these exact keys (use null for any field not visible):
{
  "medication_name": "brand name of medication",
  "generic_name": "generic/chemical name if shown",
  "dosage_strength": "strength per unit e.g. 500mg",
  "dosage_form": "tablet / capsule / liquid / etc.",
  "directions": "full directions for use as written on label",
  "frequency": "how often to take extracted from directions",
  "quantity": "number of pills or volume",
  "refills": "refills remaining",
  "expiration_date": "expiration date exactly as printed",
  "rx_number": "prescription number",
  "patient_name": "name of patient",
  "prescriber": "doctor name",
  "pharmacy_name": "pharmacy name",
  "fill_date": "date prescription was filled",
  "controlled_substance": true,
  "warnings": ["list of all warning labels visible"],
  "storage_instructions": "storage requirements if stated",
  "confidence": "high / medium / low based on image clarity"
}
Return ONLY the JSON — no markdown, no explanation."""


# ─────────────────────────────────────────────────────────────────────────────
# PILLSCAN — CLAUDE VISION API
# ─────────────────────────────────────────────────────────────────────────────

def scan_pill_image(image_bytes: bytes, api_key: str) -> dict:
    """Send a captured image to Claude and return structured medication data."""
    import anthropic

    b64 = base64.standard_b64encode(image_bytes).decode()
    client = anthropic.Anthropic(api_key=api_key)

    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {"type": "text", "text": PILLSCAN_PROMPT},
            ],
        }],
    )

    raw = msg.content[0].text.strip()
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)


def resolve_drug_name(scan_data: dict, all_drugs: list) -> str | None:
    """
    Find the best match for a scanned medication in the drug list.
    Checks both brand name and generic name, case-insensitively.
    Returns the matching drug key (lowercase) or None.
    """
    candidates = [
        scan_data.get("medication_name") or "",
        scan_data.get("generic_name") or "",
    ]
    drugs_lower = {d.lower(): d for d in all_drugs}

    for name in candidates:
        name_lower = name.lower().strip()
        if not name_lower:
            continue
        # Exact match
        if name_lower in drugs_lower:
            return drugs_lower[name_lower]
        # Prefix match (e.g. "lisinopril 10mg" → "lisinopril")
        for key in drugs_lower:
            if name_lower.startswith(key) or key.startswith(name_lower):
                return drugs_lower[key]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (all cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_drugs():
    if not DRUGS_FILE.exists():
        return []
    with open(DRUGS_FILE) as f:
        return json.load(f)


@st.cache_data
def load_ror_signals():
    if not ROR_FILE.exists():
        return pd.DataFrame()
    df = pd.read_parquet(ROR_FILE)
    if "ror_ci_lower" in df.columns and "ci_lower" not in df.columns:
        df["ci_lower"] = df["ror_ci_lower"]
    if "ror_ci_upper" in df.columns and "ci_upper" not in df.columns:
        df["ci_upper"] = df["ror_ci_upper"]
    return df


@st.cache_data
def load_known_interactions():
    if not DRUGBANK_FILE.exists():
        return set()
    try:
        db = pd.read_csv(
            DRUGBANK_FILE,
            usecols=["name", "drug-interactions.drug-interaction.name"],
            dtype=str, low_memory=False,
        )
    except (ValueError, KeyError):
        try:
            db = pd.read_csv(DRUGBANK_FILE, dtype=str, low_memory=False)
            if "drugbank-id" in db.columns and "drug-interactions" in db.columns:
                id_to_name = {}
                for _, row in db.iterrows():
                    db_id = str(row.get("drugbank-id", "")).strip()
                    name  = str(row.get("name", "")).lower().strip()
                    if db_id and name and db_id != "nan" and name != "nan":
                        id_to_name[db_id] = name
                known = set()
                for _, row in db.iterrows():
                    drug_name = str(row.get("name", "")).lower().strip()
                    for interact_id in str(row.get("drug-interactions", "")).split():
                        if interact_id in id_to_name:
                            known.add(frozenset([drug_name, id_to_name[interact_id]]))
                return known
        except Exception:
            pass
        return set()

    known = set()
    for _, row in db.iterrows():
        drug_name = str(row.get("name", "")).lower().strip()
        ints_str  = str(row.get("drug-interactions.drug-interaction.name", ""))
        if not drug_name or drug_name == "nan" or not ints_str or ints_str == "nan":
            continue
        for n in ints_str.split("|"):
            n = n.strip().lower()
            if n and n != drug_name:
                known.add(frozenset([drug_name, n]))
    return known


@st.cache_data
def load_interaction_descriptions():
    if not DRUGBANK_FILE.exists():
        return {}
    try:
        db = pd.read_csv(
            DRUGBANK_FILE,
            usecols=[
                "name",
                "drug-interactions.drug-interaction.name",
                "drug-interactions.drug-interaction.description",
            ],
            dtype=str, low_memory=False,
        )
    except (ValueError, KeyError):
        return {}

    descriptions = {}
    for _, row in db.iterrows():
        drug_name = str(row.get("name", "")).lower().strip()
        ints_str  = str(row.get("drug-interactions.drug-interaction.name", ""))
        desc_str  = str(row.get("drug-interactions.drug-interaction.description", ""))
        if not drug_name or drug_name == "nan" or not ints_str or ints_str == "nan":
            continue
        names = [n.strip().lower() for n in ints_str.split("|")]
        descs = [d.strip() for d in desc_str.split("|")] if desc_str != "nan" else []
        for i, n in enumerate(names):
            if not n or n == drug_name:
                continue
            if i < len(descs) and descs[i]:
                descriptions[frozenset([drug_name, n])] = descs[i]
    return descriptions


def get_interaction_description(descriptions, drug_a, drug_b):
    return descriptions.get(frozenset([drug_a.lower().strip(), drug_b.lower().strip()]))


@st.cache_resource
def load_gnn_predictor():
    if not MODEL_FILE.exists() or not GRAPH_FILE.exists():
        return None
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        with open(GRAPH_FILE, "rb") as f:
            graph_data = pickle.load(f)
        with open(MODEL_FILE, "rb") as f:
            model_data = pickle.load(f)

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
        model.load_state_dict(model_data["state_dict"])
        model.eval()
        return {"model": model, "graph": graph_data, "drug_to_idx": model_data["drug_to_idx"]}
    except Exception as e:
        st.warning(f"Could not load GNN model: {e}")
        return None


def predict_gnn_score(predictor, drug_a, drug_b):
    if predictor is None:
        return None
    import torch
    d2i = predictor["drug_to_idx"]
    if drug_a not in d2i or drug_b not in d2i:
        return None
    model, graph = predictor["model"], predictor["graph"]
    with torch.no_grad():
        z   = model.encode(graph.x, graph.edge_index)
        src = torch.tensor([d2i[drug_a]])
        dst = torch.tensor([d2i[drug_b]])
        return round(model.decode(z, src, dst).item(), 4)


# ─────────────────────────────────────────────────────────────────────────────
# LLM BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

def _stream_groq(prompt, model, placeholder, api_key: str = ""):
    try:
        from groq import Groq
    except ImportError:
        st.error("Run `pip install groq` to use Groq.")
        return ""
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        st.error("Groq API key not set — add it in the sidebar or set GROQ_API_KEY.")
        return ""
    client = Groq(api_key=key)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1500, stream=True,
    )
    full = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full += delta
        placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full


def run_llm(prompt, model, placeholder, api_key: str = ""):
    return _stream_groq(prompt, model, placeholder, api_key)


def build_medication_prompt(my_drugs, pair_signals, patient_context):
    if pair_signals.empty:
        signals_text = (
            "No statistically significant drug-drug interaction signals "
            "were found in the FAERS database for any pair in this list."
        )
    else:
        sort_col = "ic025" if "ic025" in pair_signals.columns else "ror"
        top = pair_signals.sort_values(sort_col, ascending=False).head(30) if sort_col in pair_signals.columns else pair_signals.head(30)
        rows = []
        for _, row in top.iterrows():
            ror_val = row.get("ror", "N/A")
            ror_str = f"ROR={ror_val:.2f}" if isinstance(ror_val, (int, float)) else f"ROR={ror_val}"
            danger  = " !! DANGEROUS" if row.get("is_dangerous") else ""
            rows.append(f"  - {row['drug_a']} + {row['drug_b']} -> {row['reaction']} ({ror_str}, n={row.get('n_cases','?')}){danger}")
        signals_text = "\n".join(rows)

    context_block = (
        f"\nPatient context: {patient_context.strip()}"
        if patient_context and patient_context.strip() else ""
    )

    return f"""You are a clinical pharmacology assistant helping a patient understand \
their medication risks based on FDA FAERS adverse event data.

The patient is currently taking: {', '.join(my_drugs)}.{context_block}

The following drug-drug interaction signals were detected in the FAERS database \
(ROR = Reporting Odds Ratio, values > 2 indicate elevated risk):

{signals_text}

Please provide:
1. A plain-language summary of the most important risks for this specific combination
2. Which pairs are most concerning and why (prioritise dangerous reactions)
3. Any drug classes or specific drugs they may want to discuss avoiding with their doctor
4. A clear reminder that this is based on statistical signals from adverse event reports, \
not clinical trial evidence, and they should consult their prescriber before making any changes

Keep the tone clear, calm, and empowering — not alarmist. Use plain English. Avoid jargon."""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_lookup(drug_a, drug_b, ror_df, known_interactions, interaction_descs, predictor):
    pair_signals = ror_df[
        ((ror_df["drug_a"] == drug_a) & (ror_df["drug_b"] == drug_b)) |
        ((ror_df["drug_a"] == drug_b) & (ror_df["drug_b"] == drug_a))
    ].copy()
    gnn_score    = predict_gnn_score(predictor, drug_a, drug_b)
    is_novel     = frozenset([drug_a, drug_b]) not in known_interactions
    db_desc      = get_interaction_description(interaction_descs, drug_a, drug_b)
    return pair_signals, gnn_score, is_novel, db_desc


def render_lookup_results(pair_signals, gnn_score, is_novel, db_description, drug_a, drug_b):
    n_signals   = len(pair_signals)
    n_dangerous = int(pair_signals["is_dangerous"].sum()) if not pair_signals.empty else 0

    if db_description:
        st.markdown(
            f"""<div style="background: linear-gradient(90deg, rgba(0,233,171,0.1) 0%, rgba(0,0,0,0) 100%);
            border-left: 3px solid #00e9ab; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
            <strong style="color: #00e9ab;">DrugBank Interaction</strong><br/>
            <span style="color: #ccc;">{db_description}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    colA, colB, colC = st.columns(3)
    colA.metric("ROR signals found", n_signals,
                help="Number of adverse reactions with elevated ROR > 2.")
    colB.metric("GNN risk score",
                f"{gnn_score:.3f}" if gnn_score is not None else "N/A",
                help="Graph Neural Network predicted interaction risk (0=low, 1=high).")
    colC.metric("Novel (not in DrugBank)",
                "✅ Yes" if is_novel else "❌ No",
                help="Whether this pair is absent from the DrugBank known-interactions database.")

    if n_signals > 0 and "ror" in pair_signals.columns:
        top = pair_signals.sort_values("ror", ascending=False).iloc[0]
        danger_note = (f" **{n_dangerous} of these are classified as dangerous reactions.**"
                       if n_dangerous > 0 else "")
        novel_note  = (" This pair is **not in DrugBank** and may represent a novel signal."
                       if is_novel else " This pair is already documented in DrugBank.")
        ci_str = (f", 95% CI {top['ci_lower']:.1f}–{top['ci_upper']:.1f}"
                  if "ci_lower" in top and "ci_upper" in top else "")
        st.info(
            f"**{drug_a.title()} + {drug_b.title()}** has **{n_signals}** flagged reaction(s)."
            f" The strongest signal is **{top['reaction']}** "
            f"(ROR {top['ror']:.1f}{ci_str}, n={int(top['n_cases'])})."
            f"{danger_note}{novel_note}"
        )
    else:
        st.info(
            f"No significant ROR signals found for **{drug_a.title()} + {drug_b.title()}**. "
            "This may mean the combination is safe, or simply that it is under-reported in FAERS."
        )

    if not pair_signals.empty:
        st.subheader("Reaction signals")
        display_cols = [c for c in
                        ["reaction", "ror", "ci_lower", "ci_upper", "n_cases", "is_dangerous"]
                        if c in pair_signals.columns]
        display_df = pair_signals[display_cols].copy()
        if "ror" in display_df.columns:
            display_df = display_df.sort_values("ror", ascending=False)
        display_df["is_dangerous"] = display_df["is_dangerous"].apply(
            lambda x: "⚠️ Yes" if x else "")
        st.dataframe(
            display_df, use_container_width=True,
            column_config={
                "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                "n_cases":      st.column_config.NumberColumn("Reports"),
                "is_dangerous": st.column_config.TextColumn("Dangerous"),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# PILLSCAN UI COMPONENT
# ─────────────────────────────────────────────────────────────────────────────

def render_pill_scanner(all_drugs: list, api_key: str) -> None:
    """
    Renders the PillScan camera UI and directly mutates
    st.session_state["my_drugs_scanned"] when the user confirms a drug.
    Returns nothing — all state is written before st.rerun() is called.
    """
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, rgba(0,233,171,0.08) 0%, rgba(0,0,0,0) 100%);
            border: 1px solid rgba(0,233,171,0.25);
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 12px;
        ">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-size:1.4rem;">📷</span>
                <strong style="font-size:1.05rem; color:#00e9ab;">PillScan — Scan a Bottle</strong>
            </div>
            <span style="color:#aaa; font-size:0.88rem;">
                Point your camera at a pill bottle label and click <em>Capture photo</em>.
                Claude will read the label and offer to add it to your list.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not api_key:
        st.warning("Enter your **Anthropic API key** in the sidebar to enable PillScan.", icon="🔑")
        return

    # ── Camera capture ────────────────────────────────────────────────────────
    camera_image = st.camera_input(
        label="Align the pill bottle label and capture",
        label_visibility="collapsed",
        key="pill_scanner_camera",
    )

    if camera_image is None:
        st.caption("💡 Tip: hold the bottle label flat, close to the camera, in good light.")
        return

    img_bytes = camera_image.getvalue()

    # ── Analyse button ────────────────────────────────────────────────────────
    if st.button("🔍 Analyse label with Claude", type="primary", key="pillscan_analyse"):
        with st.spinner("Reading label…"):
            try:
                scan_data = scan_pill_image(img_bytes, api_key)
                st.session_state["last_scan"] = scan_data
            except json.JSONDecodeError:
                st.error("Could not parse Claude's response. Try better lighting or a clearer angle.")
                return
            except Exception as e:
                if "auth" in str(e).lower() or "api_key" in str(e).lower():
                    st.error("Invalid Anthropic API key — check the sidebar.")
                else:
                    st.error(f"Scan failed: {e}")
                return

    # ── Display scan results ──────────────────────────────────────────────────
    scan_data = st.session_state.get("last_scan")
    if not scan_data:
        return

    med_name   = scan_data.get("medication_name") or "Unknown"
    gen_name   = scan_data.get("generic_name")
    strength   = scan_data.get("dosage_strength")
    directions = scan_data.get("directions")
    confidence = (scan_data.get("confidence") or "unknown").upper()
    warnings   = scan_data.get("warnings") or []
    conf_color = {"HIGH": "#00e9ab", "MEDIUM": "#60b4ff", "LOW": "#e06060"}.get(confidence, "#888")

    st.markdown(
        f"""
        <div style="border:1px solid {conf_color}44; border-radius:8px;
                    padding:14px 18px; margin:10px 0; background:rgba(0,0,0,0.2);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <div style="font-size:1.1rem; font-weight:700; color:#fff;">{med_name}</div>
                    {"<div style='color:#aaa; font-size:0.85rem;'>" + gen_name + "</div>" if gen_name else ""}
                    {"<div style='color:#ccc; font-size:0.88rem; margin-top:4px;'>Strength: " + strength + "</div>" if strength else ""}
                    {"<div style='color:#bbb; font-size:0.82rem; margin-top:4px;'>" + directions + "</div>" if directions else ""}
                </div>
                <div style="background:{conf_color}22; border:1px solid {conf_color};
                            border-radius:4px; padding:2px 8px; font-size:0.72rem;
                            color:{conf_color}; white-space:nowrap;">
                    {confidence} CONFIDENCE
                </div>
            </div>
            {"<div style='margin-top:8px;color:#e0a060;font-size:0.82rem;'>⚠️ " + " &nbsp;|&nbsp; ".join(warnings[:3]) + "</div>" if warnings else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Match to drug list and add directly to session state ──────────────────
    matched = resolve_drug_name(scan_data, all_drugs)

    if matched:
        st.success(f"Matched **{matched.title()}** in the drug database.")
        col_add, col_skip = st.columns([1, 1])
        with col_add:
            if st.button(f"➕ Add {matched.title()} to my list", key="pillscan_add_matched"):
                # FIX: write directly to session state — never return a value through
                # st.rerun() because rerun raises StopException before any caller
                # can receive the return value.
                if matched not in st.session_state["my_drugs_scanned"]:
                    st.session_state["my_drugs_scanned"].append(matched)
                st.session_state["last_scan"] = None
                st.toast(f"✅ {matched.title()} added!", icon="💊")
                st.rerun()
        with col_skip:
            if st.button("✕ Skip", key="pillscan_skip"):
                st.session_state["last_scan"] = None
                st.rerun()
    else:
        # No exact match — let user type the name manually
        st.warning(
            f"**{med_name}** was not found in the indexed drug list. "
            "You can search for it manually below, or add it by typing."
        )
        manual = st.selectbox(
            "Select from list (or type to search)",
            options=[""] + all_drugs,
            key="pillscan_manual_select",
        )
        col_add, col_skip = st.columns([1, 1])
        with col_add:
            if manual and st.button(f"➕ Add {manual.title()}", key="pillscan_add_manual"):
                # FIX: write directly to session state so rerun does not lose the value
                if manual not in st.session_state["my_drugs_scanned"]:
                    st.session_state["my_drugs_scanned"].append(manual)
                st.session_state["last_scan"] = None
                st.toast(f"✅ {manual.title()} added!", icon="💊")
                st.rerun()
        with col_skip:
            if st.button("✕ Skip", key="pillscan_skip2"):
                st.session_state["last_scan"] = None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="InteRx", page_icon="💊", layout="wide")

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in {
        "lookup_result":        None,
        "history":              [],
        "network_lookup":       None,
        "my_drugs_scanned":     [],
        "last_scan":            None,
        "rxnorm_results":       None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("InteRx")
    st.caption("Uncovering hidden drug-drug interactions from FDA FAERS data")

    if not DRUGS_FILE.exists():
        st.error("No data found. Please run the pipeline first:")
        st.code("python pipeline.py", language="bash")
        st.stop()

    all_drugs          = load_drugs()
    ror_df             = load_ror_signals()
    known_interactions = load_known_interactions()
    interaction_descs  = load_interaction_descriptions()
    predictor          = load_gnn_predictor()

    # ── Dashboard metrics ─────────────────────────────────────────────────────
    n_novel = 0
    if not ror_df.empty and known_interactions:
        n_novel = int(ror_df.apply(
            lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1
        ).sum())
    n_dangerous_signals = (
        int(ror_df["is_dangerous"].sum())
        if not ror_df.empty and "is_dangerous" in ror_df.columns else 0
    )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Drugs indexed",     f"{len(all_drugs):,}")
    h2.metric("ROR signals",       f"{len(ror_df):,}")
    h3.metric("Novel signals",     f"{n_novel:,}",
              help="Signals not found in DrugBank known interactions.")
    h4.metric("Dangerous signals", f"{n_dangerous_signals:,}",
              help="Signals involving a serious reaction category.")
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")

    # Anthropic API key for PillScan
    st.sidebar.subheader("PillScan API Key")
    anthropic_key = st.sidebar.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        placeholder="sk-ant-…",
        help=(
            "Required for the pill bottle scanner in My Medications. "
            "Get yours at console.anthropic.com. "
            "You can also set the ANTHROPIC_API_KEY environment variable."
        ),
    )
    st.sidebar.markdown("---")

    min_ror        = st.sidebar.slider("Minimum ROR", 1.0, 20.0, 2.0, 0.5,
                                       help="Reporting Odds Ratio threshold for signal filtering.")
    only_dangerous = st.sidebar.checkbox("Only dangerous reactions", value=False)
    only_novel     = st.sidebar.checkbox("Only novel (not in DrugBank)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Assistant (Groq)")
    groq_key = st.sidebar.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_…",
        help="Get a free key at console.groq.com. You can also set the GROQ_API_KEY env var.",
    )
    llm_model = st.sidebar.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    )

    st.sidebar.markdown("---")
    pillscan_status = "✅ ready" if anthropic_key else "❌ no API key"
    groq_status     = "✅ ready" if groq_key      else "❌ no API key"
    st.sidebar.caption(f"PillScan:  {pillscan_status}")
    st.sidebar.caption(f"Groq:      {groq_status}")
    st.sidebar.caption(f"GNN model: {'✅ loaded' if predictor else '❌ not found'}")
    st.sidebar.caption(f"DrugBank pairs: {len(known_interactions):,}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Drug Pair Lookup", "Signal Table", "Network", "Summary", "My Medications", "Name Lookup"
    ])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)
        drug_a = col1.selectbox("Drug A", all_drugs, index=0, placeholder="Type to search…")
        drug_b = col2.selectbox("Drug B", all_drugs,
                                index=min(1, len(all_drugs) - 1), placeholder="Type to search…")

        if st.session_state.network_lookup:
            st.info(f"Jumped from Network tab: showing top pair for **{st.session_state.network_lookup}**")

        if st.button("Analyse Interaction", type="primary"):
            result = run_lookup(drug_a, drug_b, ror_df, known_interactions, interaction_descs, predictor)
            st.session_state.lookup_result = (drug_a, drug_b, *result)
            entry = (drug_a, drug_b)
            st.session_state.history = [h for h in st.session_state.history if h != entry][:4]
            st.session_state.history.insert(0, entry)
            st.session_state.network_lookup = None

        if st.session_state.lookup_result:
            da, db, signals, gnn, novel, db_desc = st.session_state.lookup_result
            st.markdown(f"#### Results for **{da.title()}** + **{db.title()}**")
            render_lookup_results(signals, gnn, novel, db_desc, da, db)

        if st.session_state.history:
            st.markdown("---")
            st.markdown("**Recently analysed pairs**")
            hist_cols = st.columns(min(len(st.session_state.history), 5))
            for i, (ha, hb) in enumerate(st.session_state.history):
                if hist_cols[i].button(f"{ha.title()} + {hb.title()}", key=f"hist_{i}"):
                    result = run_lookup(ha, hb, ror_df, known_interactions, interaction_descs, predictor)
                    st.session_state.lookup_result = (ha, hb, *result)
                    st.rerun()

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        if ror_df.empty:
            st.info("No ROR signals found. Run pipeline.py to compute them.")
        else:
            filtered = ror_df[ror_df["ror"] >= min_ror].copy()
            if only_dangerous and "is_dangerous" in filtered.columns:
                filtered = filtered[filtered["is_dangerous"]]
            if only_novel and known_interactions:
                filtered = filtered[filtered.apply(
                    lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1)]

            st.subheader(f"{len(filtered)} signals (ROR ≥ {min_ror})")
            st.download_button("⬇️ Export as CSV",
                               data=filtered.to_csv(index=False).encode("utf-8"),
                               file_name="interx_signals.csv", mime="text/csv")

            display = filtered.copy()
            if "is_dangerous" in display.columns:
                display["is_dangerous"] = display["is_dangerous"].apply(lambda x: "⚠️ Yes" if x else "")
            st.dataframe(display, use_container_width=True,
                         column_config={
                             "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                             "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                             "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                             "is_dangerous": st.column_config.TextColumn("Dangerous"),
                         })

            top10 = filtered.nlargest(10, "ror") if not filtered.empty else pd.DataFrame()
            if not top10.empty:
                fig = go.Figure(go.Bar(
                    x=top10["ror"],
                    y=top10["drug_a"] + " + " + top10["drug_b"] + " → " + top10["reaction"],
                    orientation="h", marker_color="#00e9ab",
                    error_x=dict(type="data",
                                 array=top10["ci_upper"] - top10["ror"],
                                 arrayminus=top10["ror"] - top10["ci_lower"],
                                 visible=True),
                ))
                fig.update_layout(
                    title="Top 10 signals by ROR", xaxis_title="Reporting Odds Ratio",
                    height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#00e9ab"), yaxis=dict(autorange="reversed"),
                    xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
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
                raw_edges   = list(zip(ei[0].tolist(), ei[1].tolist()))
                idx2drug    = graph_data.idx_to_drug
                raw_weights = graph_data.edge_weights or [1] * len(raw_edges)

                top_edges = sorted(zip(raw_edges, raw_weights), key=lambda x: -x[1])[:80]
                nodes_used = set()
                for (s, d), _ in top_edges:
                    nodes_used.add(idx2drug.get(s, str(s)))
                    nodes_used.add(idx2drug.get(d, str(d)))

                node_list = list(nodes_used)
                node_idx  = {n: i for i, n in enumerate(node_list)}
                rng = np.random.default_rng(42)
                pos = rng.uniform(0, 1, (len(node_list), 2))

                flagged_drugs = (set(ror_df["drug_a"]) | set(ror_df["drug_b"])) if not ror_df.empty else set()
                node_colors = ["#00e9ab" if n in flagged_drugs else "#4a6fa5" for n in node_list]
                node_sizes  = [12 if n in flagged_drugs else 7 for n in node_list]

                edge_x, edge_y = [], []
                for (s, d), _ in top_edges:
                    na, nb = idx2drug.get(s, str(s)), idx2drug.get(d, str(d))
                    if na in node_idx and nb in node_idx:
                        xa, ya = pos[node_idx[na]]
                        xb, yb = pos[node_idx[nb]]
                        edge_x += [xa, xb, None]
                        edge_y += [ya, yb, None]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                         line=dict(width=0.8, color="rgba(128,128,128,0.4)"),
                                         hoverinfo="none"))
                fig.add_trace(go.Scatter(
                    x=pos[:, 0], y=pos[:, 1], mode="markers+text",
                    marker=dict(size=node_sizes, color=node_colors,
                                line=dict(width=1, color="#0a0a0a")),
                    text=node_list, textposition="top center",
                    hoverinfo="text", customdata=node_list,
                ))
                fig.update_layout(
                    showlegend=False, height=580,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=0, r=0, t=10, b=0),
                )

                event = st.plotly_chart(fig, use_container_width=True,
                                        on_select="rerun", key="network_chart")

                if event and event.get("selection") and event["selection"].get("points"):
                    clicked_idx = event["selection"]["points"][0].get("point_index")
                    if clicked_idx is not None and clicked_idx < len(node_list):
                        clicked_drug = node_list[clicked_idx]
                        if not ror_df.empty and "ror" in ror_df.columns:
                            drug_rows = ror_df[
                                (ror_df["drug_a"] == clicked_drug) | (ror_df["drug_b"] == clicked_drug)
                            ].sort_values("ror", ascending=False)
                            if not drug_rows.empty:
                                top_row = drug_rows.iloc[0]
                                partner = top_row["drug_b"] if top_row["drug_a"] == clicked_drug else top_row["drug_a"]
                                result  = run_lookup(clicked_drug, partner, ror_df, known_interactions, interaction_descs, predictor)
                                st.session_state.lookup_result  = (clicked_drug, partner, *result)
                                st.session_state.network_lookup = clicked_drug
                                entry = (clicked_drug, partner)
                                st.session_state.history = [h for h in st.session_state.history if h != entry][:4]
                                st.session_state.history.insert(0, entry)
                                st.success(f"Loaded **{clicked_drug.title()} + {partner.title()}** → switch to the **Drug Pair Lookup** tab.")

                st.markdown(
                    "🟢 **Teal nodes** = drugs in flagged signals &nbsp;|&nbsp; "
                    "🔵 **Blue nodes** = no signals found"
                )
            except Exception as e:
                st.error(f"Could not render network: {e}")

    # ── TAB 4 ─────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Pipeline Summary")
        if ror_df.empty:
            st.info("No signals computed yet. Run pipeline.py first.")
        else:
            summary_df = ror_df[ror_df["ror"] >= min_ror].copy()
            if only_dangerous and "is_dangerous" in summary_df.columns:
                summary_df = summary_df[summary_df["is_dangerous"]]
            if only_novel and known_interactions:
                summary_df = summary_df[summary_df.apply(
                    lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1)]

            n_pairs = summary_df.groupby(["drug_a", "drug_b"]).ngroups
            n_dang  = int(summary_df["is_dangerous"].sum()) if "is_dangerous" in summary_df.columns else 0
            n_nov   = int(summary_df.apply(
                lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1
            ).sum()) if known_interactions else 0

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Filtered signals",          f"{len(summary_df):,}")
            s2.metric("Unique drug pairs",          f"{n_pairs:,}")
            s3.metric("Dangerous reaction signals", f"{n_dang:,}")
            s4.metric("Novel signals",              f"{n_nov:,}")
            st.divider()

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Top 5 most suspicious drugs")
                st.caption("Ranked by number of flagged pairs they appear in.")
                drug_pair_counts = (
                    pd.concat([summary_df["drug_a"], summary_df["drug_b"]])
                    .value_counts().head(5).reset_index()
                )
                drug_pair_counts.columns = ["Drug", "Flagged pairs"]
                st.dataframe(drug_pair_counts, use_container_width=True, hide_index=True)

            with col_right:
                st.markdown("#### Reaction breakdown")
                st.caption("Most frequently flagged reaction types.")
                rxn_counts = (
                    summary_df.groupby("reaction")
                    .agg(signals=("ror", "count"), max_ror=("ror", "max"))
                    .sort_values("signals", ascending=False).head(10).reset_index()
                )
                rxn_counts.columns = ["Reaction", "# Signals", "Max ROR"]
                st.dataframe(rxn_counts, use_container_width=True, hide_index=True,
                             column_config={"Max ROR": st.column_config.NumberColumn(format="%.2f")})

            st.divider()
            st.markdown("#### Dangerous novel signals")
            st.caption("Signals that are both classified as dangerous AND absent from DrugBank.")
            if known_interactions and "is_dangerous" in ror_df.columns and "ror" in ror_df.columns:
                dang_novel = ror_df[ror_df["is_dangerous"]].copy()
                dang_novel = dang_novel[dang_novel.apply(
                    lambda r: frozenset([r["drug_a"], r["drug_b"]]) not in known_interactions, axis=1)]
                dang_novel = dang_novel.sort_values("ror", ascending=False).head(20) if "ror" in dang_novel.columns else dang_novel.head(20)

                if dang_novel.empty:
                    st.info("No dangerous novel signals found under current filters.")
                else:
                    dang_novel["is_dangerous"] = dang_novel["is_dangerous"].apply(lambda x: "⚠️ Yes" if x else "")
                    st.dataframe(dang_novel, use_container_width=True,
                                 column_config={
                                     "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                                     "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                                     "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                                     "is_dangerous": st.column_config.TextColumn("Dangerous"),
                                 })
                    st.download_button("⬇️ Export dangerous novel signals",
                                       data=dang_novel.to_csv(index=False).encode("utf-8"),
                                       file_name="interx_dangerous_novel.csv", mime="text/csv")
            else:
                st.info("Load DrugBank data to see novel signal filtering.")

    # ── TAB 5 — My Medications ────────────────────────────────────────────────
    with tab5:
        st.subheader("My Medication Risk Summary")
        st.caption(
            "Select your current medications manually **or scan a pill bottle** "
            "using your camera to add them automatically."
        )

        # ── PillScan expander ─────────────────────────────────────────────────
        # render_pill_scanner writes directly to st.session_state["my_drugs_scanned"]
        # and calls st.rerun() — it never returns a meaningful value.
        with st.expander("📷  Scan a pill bottle to add medication", expanded=False):
            render_pill_scanner(all_drugs, anthropic_key)

        # ── Scanned drugs chips ───────────────────────────────────────────────
        if st.session_state["my_drugs_scanned"]:
            st.markdown("**Scanned medications** (tap × to remove):")
            chip_cols = st.columns(len(st.session_state["my_drugs_scanned"]))
            to_remove = None
            for i, drug in enumerate(st.session_state["my_drugs_scanned"]):
                with chip_cols[i]:
                    if st.button(
                        f"{drug.title()}  ×",
                        key=f"remove_scanned_{i}",
                        help=f"Remove {drug} from list",
                    ):
                        to_remove = drug
            if to_remove:
                st.session_state["my_drugs_scanned"].remove(to_remove)
                st.rerun()

        # ── Manual multiselect — pre-seeded with scanned drugs ────────────────
        default_drugs = [d for d in st.session_state["my_drugs_scanned"] if d in all_drugs]
        my_drugs = st.multiselect(
            "My current medications",
            options=all_drugs,
            default=default_drugs,
            placeholder="Start typing a drug name…",
            help="Select all drugs you are currently taking, including supplements.",
        )

        # Sync manual changes back to scanned list
        st.session_state["my_drugs_scanned"] = my_drugs

        with st.expander("Add personal context (optional)"):
            patient_context = st.text_area(
                "Any relevant medical context",
                placeholder="e.g. 65 year old, kidney disease, recently diagnosed with…",
                height=80,
            )

        if groq_key:
            st.info(f"Using **Groq** with model `{llm_model}`.")
        else:
            st.warning("Groq API key not set — add it in the sidebar to enable AI summaries.")

        analyse_btn = st.button("Analyse My Medications", type="primary", disabled=len(my_drugs) < 1)

        if analyse_btn and my_drugs:
            my_drugs_lower = [d.lower().strip() for d in my_drugs]
            if ror_df.empty:
                pair_signals = pd.DataFrame()
            else:
                mask = pd.Series(False, index=ror_df.index)
                for i, da in enumerate(my_drugs_lower):
                    for db in my_drugs_lower[i + 1:]:
                        mask |= (
                            ((ror_df["drug_a"] == da) & (ror_df["drug_b"] == db)) |
                            ((ror_df["drug_a"] == db) & (ror_df["drug_b"] == da))
                        )
                pair_signals = ror_df[mask].copy()

            n_pairs   = len(my_drugs) * (len(my_drugs) - 1) // 2
            n_signals = len(pair_signals)
            n_danger  = int(pair_signals["is_dangerous"].sum()) if not pair_signals.empty and "is_dangerous" in pair_signals.columns else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Drug pairs checked", n_pairs)
            c2.metric("Signals found", n_signals)
            c3.metric("Dangerous reactions", n_danger,
                      delta="⚠️ Review" if n_danger > 0 else None, delta_color="inverse")

            if not pair_signals.empty:
                with st.expander("View raw signal data"):
                    show_cols = [c for c in
                                 ["drug_a", "drug_b", "reaction", "ror", "ci_lower", "ci_upper",
                                  "n_cases", "is_dangerous"]
                                 if c in pair_signals.columns]
                    display_df = pair_signals[show_cols].copy()
                    if "ror" in display_df.columns:
                        display_df = display_df.sort_values("ror", ascending=False)
                    if "is_dangerous" in display_df.columns:
                        display_df["is_dangerous"] = display_df["is_dangerous"].apply(
                            lambda x: "⚠️ Yes" if x else "")
                    st.dataframe(display_df, use_container_width=True,
                                 column_config={
                                     "ror":          st.column_config.NumberColumn("ROR",      format="%.2f"),
                                     "ci_lower":     st.column_config.NumberColumn("CI Lower", format="%.2f"),
                                     "ci_upper":     st.column_config.NumberColumn("CI Upper", format="%.2f"),
                                     "is_dangerous": st.column_config.TextColumn("Dangerous"),
                                 })
            else:
                st.info("No signals found for any pair. The AI will still provide general guidance.")

            prompt      = build_medication_prompt(my_drugs, pair_signals, patient_context)
            st.divider()
            st.markdown(
                f"#### AI Risk Summary "
                f"<small style='color:#00e9ab'>(Groq · {llm_model})</small>",
                unsafe_allow_html=True,
            )
            placeholder = st.empty()
            with st.spinner("Generating summary..."):
                run_llm(prompt, llm_model, placeholder, groq_key)

            st.divider()
            st.warning(
                "**Medical disclaimer**: This analysis is based on statistical patterns "
                "in FDA adverse event reports. It is not a substitute for professional "
                "medical advice. Always consult your doctor or pharmacist before "
                "changing your medications."
            )


    # ── TAB 6 — Name Lookup ───────────────────────────────────────────────────
    with tab6:
        st.subheader("Drug Name Lookup")
        st.caption(
            "Convert brand names to generic names using the NIH RxNorm database. "
            "Results are fetched live — no pipeline run required. "
            "Found a drug? Add it straight to My Medications."
        )

        search_input = st.text_input(
            "Enter a drug name (brand or generic)",
            placeholder="e.g. Tylenol, Advil, Lipitor, Xarelto…",
            key="rxnorm_search_input",
        )

        col_search, col_clear = st.columns([1, 5])
        do_search = col_search.button("Look up", type="primary", key="rxnorm_go")
        if col_clear.button("Clear", key="rxnorm_clear"):
            st.session_state["rxnorm_results"] = None
            st.rerun()

        if do_search and search_input.strip():
            with st.spinner(f"Searching RxNorm for **{search_input}**…"):
                candidates = rxnorm_search(search_input.strip())
            st.session_state["rxnorm_results"] = (search_input.strip(), candidates)

        if st.session_state["rxnorm_results"]:
            query, candidates = st.session_state["rxnorm_results"]

            if candidates and "error" in candidates[0]:
                st.error(f"RxNorm API error: {candidates[0]['error']}")
            elif not candidates:
                st.warning(f"No results found for **{query}**. Try an alternate spelling.")
            else:
                brand_hits   = [c for c in candidates if c["tty"] in ("BN", "SBD", "BPCK")]
                generic_hits = [c for c in candidates if c["tty"] in ("IN", "MIN", "PIN", "SCD", "GPCK", "PSN")]

                st.success(f"Found **{len(candidates)}** result(s) for **{query}**")

                best = next(
                    (c for c in candidates if c["name"].lower() == query.lower()),
                    brand_hits[0] if brand_hits else (generic_hits[0] if generic_hits else candidates[0]),
                )
                is_brand = best["tty"] in ("BN", "SBD", "BPCK")

                # ── Result card ───────────────────────────────────────────────
                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Name on label", best["name"])
                    c1.caption(f"Type: {TTY_LABELS.get(best['tty'], best['tty'])}")
                    c2.metric("RxNorm ID (RxCUI)", best["rxcui"])
                    c2.caption("Unique FDA concept identifier")

                    resolved_generics = []

                    if is_brand and best["rxcui"]:
                        with st.spinner("Resolving active ingredient(s)…"):
                            ingredients = rxnorm_get_ingredients(best["rxcui"])

                        if ingredients and "error" not in ingredients[0]:
                            resolved_generics = [i["name"].lower() for i in ingredients]
                            c3.metric(
                                "Active ingredient(s)",
                                ", ".join(i["name"] for i in ingredients),
                                help="The official FDA generic name(s) for this drug.",
                            )
                            all_brands = []
                            for ing in ingredients:
                                all_brands.extend(rxnorm_get_brand_names(ing["rxcui"]))
                            other_brands = sorted(set(b for b in all_brands if b.lower() != query.lower()))
                            if other_brands:
                                st.markdown(
                                    "**Other brand names:** "
                                    + ", ".join(f"`{b}`" for b in other_brands[:15])
                                    + (" …" if len(other_brands) > 15 else "")
                                )
                        else:
                            c3.info("Could not resolve active ingredient.")
                    else:
                        resolved_generics = [best["name"].lower()]
                        with st.spinner("Finding brand names…"):
                            brands = rxnorm_get_brand_names(best["rxcui"])
                        if brands:
                            c3.metric(
                                "Known brand names",
                                brands[0] + (f" (+{len(brands)-1} more)" if len(brands) > 1 else ""),
                            )
                            if len(brands) > 1:
                                st.markdown(
                                    "**All brand names:** "
                                    + ", ".join(f"`{b}`" for b in sorted(brands))
                                )
                        else:
                            c3.info("No brand names found.")

                # ── Add to My Medications ─────────────────────────────────────
                st.markdown("---")
                st.markdown("**Add to My Medications**")

                # Find which resolved generics exist in the drug index
                matched_drugs = [g for g in resolved_generics if g in [d.lower() for d in all_drugs]]
                # Also try the searched name itself
                if not matched_drugs and best["name"].lower() in [d.lower() for d in all_drugs]:
                    matched_drugs = [best["name"].lower()]

                if matched_drugs:
                    # Map back to original casing from all_drugs
                    lower_to_drug = {d.lower(): d for d in all_drugs}
                    add_cols = st.columns(min(len(matched_drugs), 4))
                    for i, gname in enumerate(matched_drugs):
                        drug_display = lower_to_drug.get(gname, gname)
                        with add_cols[i]:
                            already_added = gname in [d.lower() for d in st.session_state["my_drugs_scanned"]]
                            if already_added:
                                st.button(f"✓ {drug_display.title()} already added", disabled=True, key=f"rxn_add_{i}")
                            elif st.button(f"➕ Add {drug_display.title()}", key=f"rxn_add_{i}", type="primary"):
                                st.session_state["my_drugs_scanned"].append(drug_display)
                                st.toast(f"✅ {drug_display.title()} added to My Medications!", icon="💊")
                                st.rerun()
                else:
                    st.info(
                        f"**{best['name']}** (or its generic) isn't in the current FAERS drug index. "
                        "Try searching for it directly in Drug Pair Lookup."
                    )

                # ── Full results table ────────────────────────────────────────
                with st.expander(f"Show all {len(candidates)} RxNorm matches"):
                    rows = [
                        {
                            "Name":    c["name"],
                            "Type":    TTY_LABELS.get(c["tty"], c["tty"]),
                            "RxCUI":   c["rxcui"],
                            "Synonym": c.get("synonym", ""),
                        }
                        for c in candidates
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# RXNORM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

TTY_LABELS = {
    "IN":   "Generic ingredient",
    "MIN":  "Multiple-ingredient generic",
    "PIN":  "Precise ingredient",
    "BN":   "Brand name",
    "SBD":  "Branded drug",
    "SCD":  "Clinical drug (generic)",
    "GPCK": "Generic pack",
    "BPCK": "Brand pack",
    "DF":   "Dose form",
    "ET":   "Entry term",
    "PSN":  "Prescribable name",
    "TMSY": "Tall-man lettering synonym",
    "SY":   "Synonym",
}


def rxnorm_search(name: str) -> list[dict]:
    import requests
    try:
        resp = requests.get(f"{RXNORM_BASE}/drugs.json", params={"name": name}, timeout=8)
        resp.raise_for_status()
        candidates = []
        for group in resp.json().get("drugGroup", {}).get("conceptGroup", []):
            tty = group.get("tty", "")
            for concept in group.get("conceptProperties", []):
                candidates.append({
                    "rxcui":   concept.get("rxcui", ""),
                    "name":    concept.get("name", ""),
                    "tty":     tty,
                    "synonym": concept.get("synonym", ""),
                })
        return candidates
    except Exception as e:
        return [{"error": str(e)}]


def rxnorm_get_ingredients(rxcui: str) -> list[dict]:
    import requests

    def _related(cui, tty):
        r = requests.get(f"{RXNORM_BASE}/rxcui/{cui}/related.json", params={"tty": tty}, timeout=10)
        r.raise_for_status()
        out = []
        for g in r.json().get("relatedGroup", {}).get("conceptGroup", []):
            for c in g.get("conceptProperties", []):
                out.append({"rxcui": c.get("rxcui",""), "name": c.get("name",""), "tty": g.get("tty","")})
        return out

    try:
        for tty in ("IN", "MIN"):
            found = _related(rxcui, tty)
            if found:
                return found
        for parent in _related(rxcui, "BN"):
            for tty in ("IN", "MIN"):
                found = _related(parent["rxcui"], tty)
                if found:
                    return found
        # allrelated fallback
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/allrelated.json", timeout=10)
        r.raise_for_status()
        for g in r.json().get("allRelatedGroup", {}).get("conceptGroup", []):
            if g.get("tty") in ("IN", "MIN"):
                results = [{"rxcui": c.get("rxcui",""), "name": c.get("name",""), "tty": g.get("tty","")}
                           for c in g.get("conceptProperties", [])]
                if results:
                    return results
        return []
    except Exception as e:
        return [{"error": str(e)}]


def rxnorm_get_brand_names(rxcui: str) -> list[str]:
    import requests
    try:
        r = requests.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json", params={"tty": "BN"}, timeout=8)
        r.raise_for_status()
        return [c.get("name","") for g in r.json().get("relatedGroup",{}).get("conceptGroup",[])
                for c in g.get("conceptProperties",[])]
    except Exception:
        return []



if __name__ == '__main__':
    main()
