from __future__ import annotations

import pandas as pd
import streamlit as st

from cellscope.data import FEATURES, make_demo_data, validate_upload
from cellscope.model import fit_demo_model, score_uploaded

st.set_page_config(page_title="CellScope", page_icon="🔋", layout="wide")


@st.cache_resource
def load_model():
    return fit_demo_model()


@st.cache_data
def load_demo():
    return make_demo_data()


model, demo_scored = load_model()
st.title("🔋 CellScope")
st.caption("Battery aging analysis · remaining-life estimates · anomaly triage")
st.info(
    "Portfolio demonstration using synthetic training data. Estimates are not calibrated "
    "for real cells and are not battery-safety guidance."
)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Optional cycle-data CSV", type=["csv"])
    st.caption("Required columns are listed in the README.")
    nominal_hint = st.caption("Uploaded rows are scored by a model trained on synthetic examples.")

if uploaded is None:
    scored = demo_scored.copy()
    st.caption("Viewing generated demo cells. All measurements and labels are simulated.")
else:
    try:
        raw = pd.read_csv(uploaded)
        cleaned, issues = validate_upload(raw)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()
    if cleaned.empty:
        st.error("No valid cycle records were found.")
        st.stop()
    scored = score_uploaded(model, cleaned)
    st.caption(f"Scored {len(scored):,} complete rows across {scored['cell_id'].nunique()} cells.")

cells = sorted(scored["cell_id"].astype(str).unique())
selected = st.selectbox("Cell", cells, index=len(cells) - 1)
cell = scored[scored["cell_id"].astype(str) == selected].sort_values("cycle")
latest = cell.iloc[-1]
health = float(latest["soh_pct"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest state of health", f"{health:.1f}%")
m2.metric("Estimated cycles remaining", f"{latest['estimated_remaining_cycles']:.0f}")
m3.metric("Anomaly flagged", "Review" if bool(latest["anomaly_flag"]) else "No")
m4.metric("Cycles observed", f"{int(latest['cycle'])}")

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Capacity health by cycle")
    chart = cell.set_index("cycle")[["soh_pct"]].rename(columns={"soh_pct": "State of health (%)"})
    st.line_chart(chart, height=300)
    st.caption("80% is the illustrative end-of-life threshold used to create synthetic training labels.")
with right:
    st.subheader("Estimated remaining life")
    rul_chart = cell.set_index("cycle")[["estimated_remaining_cycles"]].rename(
        columns={"estimated_remaining_cycles": "Estimated cycles"}
    )
    st.line_chart(rul_chart, height=300)
    st.caption("The model is evaluated by holding out entire cells during fitting.")

st.subheader("Anomaly review")
flagged = cell[cell["anomaly_flag"]].copy()
if flagged.empty:
    st.success("No cycles from this cell were flagged by the anomaly model.")
else:
    st.warning(f"{len(flagged)} cycle(s) flagged for inspection. A flag is not a fault diagnosis.")
    st.dataframe(
        flagged[["cycle", "soh_pct", "resistance_mohm", "temperature_c", "voltage_sag_mv", "charge_time_min", "anomaly_score"]],
        use_container_width=True,
    )

with st.expander("Held-out evaluation and method"):
    ev = model.evaluation
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out RUL MAE", f"{ev['heldout_rul_mae_cycles']:.1f} cycles")
    c2.metric("Anomaly precision", f"{ev['anomaly_precision']:.1%}")
    c3.metric("Anomaly recall", f"{ev['anomaly_recall']:.1%}")
    st.write(
        f"Evaluation withheld {len(model.test_cells)} complete cells; none of their cycles "
        "were used to fit either model."
    )
    st.write("Model features:", ", ".join(FEATURES))
    st.write(
        "Remaining life is a regression estimate trained against simulated cycles until the "
        "80% capacity threshold. The Isolation Forest ranks unusual combinations of measurements."
    )

st.caption(
    "Synthetic data and synthetic validation results only. Real-cell use requires chemistry- and "
    "protocol-specific labeled data, independent validation, and expert review."
)
