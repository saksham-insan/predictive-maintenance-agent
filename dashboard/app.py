"""
Predictive Maintenance Agent — Live Dashboard.

Runs the streaming simulation and visualizes each sensor reading as it
passes through the agent pipeline (monitoring -> diagnosis -> recommendation).
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

import time
import html
from datetime import datetime
import pandas as pd
import streamlit as st
from src.database import SessionLocal
from src.database_models import Run, HighRiskAlert, LowConfidenceAnomaly
from src.user_service import authenticate_user
from agents.orchestrator import run_pipeline
from llm_reasoning import trigger_async_llm_reasoning, get_cached_insight, make_event_key

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")


def md(content: str, target=None):
    """
    Render raw HTML/CSS via st.markdown.
    """
    lines = [line.lstrip() for line in content.strip("\n").split("\n")]
    cleaned = "\n".join(line for line in lines if line != "")
    sink = target if target is not None else st
    sink.markdown(cleaned, unsafe_allow_html=True)

st.set_page_config(page_title="Predictive Maintenance — Telemetry", layout="wide")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title("Predictive Maintenance Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        db = SessionLocal()

        try:
            user = authenticate_user(db, username, password)

            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user.user_id
                st.session_state.username = user.username
                st.rerun()
            else:
                st.error("Invalid username or password.")

        finally:
            db.close()

    st.stop()

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#14171A",
    "panel": "#1B1F23",
    "panel_alt": "#20252A",
    "hairline": "#2C3339",
    "text": "#EDEAE4",
    "text_dim": "#8B939B",
    "text_faint": "#5C6570",
    "ok": "#4F9DDE",
    "watch": "#F5A623",
    "danger": "#E5484D",
    "teal": "#2DD4BF",
}

# ---------------------------------------------------------------------------
# CSS — inject once
# ---------------------------------------------------------------------------
def inject_css():
    md(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
        --bg: {COLORS['bg']};
        --panel: {COLORS['panel']};
        --panel-alt: {COLORS['panel_alt']};
        --hairline: {COLORS['hairline']};
        --text: {COLORS['text']};
        --text-dim: {COLORS['text_dim']};
        --text-faint: {COLORS['text_faint']};
        --ok: {COLORS['ok']};
        --watch: {COLORS['watch']};
        --danger: {COLORS['danger']};
        --teal: {COLORS['teal']};
    }}

    .stApp {{ background: var(--bg); color: var(--text); }}
    section[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--hairline); }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}
    body, p, div, span {{ font-family: 'Space Grotesk', sans-serif; }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: var(--teal);
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .pulse-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 0 rgba(45,212,191,0.6);
        animation: pulse 1.8s infinite;
    }}
    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(45,212,191,0.55); }}
        70%  {{ box-shadow: 0 0 0 8px rgba(45,212,191,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(45,212,191,0); }}
    }}
    .dash-title {{ font-size: 1.9rem; font-weight: 700; margin: 0 0 2px 0; color: var(--text); }}
    .dash-sub {{ color: var(--text-dim); font-size: 0.92rem; margin-bottom: 1.4rem; }}
    .dash-sub .arrow {{ color: var(--text-faint); margin: 0 6px; }}

    .strip-wrap {{
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        margin-bottom: 18px;
    }}
    .strip-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-faint);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .strip {{
        display: flex;
        align-items: flex-end;
        gap: 3px;
        height: 46px;
        overflow: hidden;
    }}
    .bar {{
        flex: 1 1 auto;
        min-width: 3px;
        border-radius: 2px 2px 0 0;
        transition: height 0.15s ease;
    }}
    .strip-empty {{
        color: var(--text-faint);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        padding: 12px 0;
    }}

    .dial-row {{ display: flex; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }}
    .dial-card {{
        flex: 1 1 220px;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .dial {{
        width: 64px; height: 64px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .dial-inner {{
        width: 48px; height: 48px; border-radius: 50%;
        background: var(--panel);
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; font-weight: 600;
        color: var(--text);
    }}
    .dial-meta .dial-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem; font-weight: 600; color: var(--text); line-height: 1;
    }}
    .dial-meta .dial-label {{
        font-size: 0.76rem; color: var(--text-dim); margin-top: 4px;
    }}

    .status-panel {{
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-left: 4px solid var(--ok);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 18px;
    }}
    .status-panel.watch {{ border-left-color: var(--watch); }}
    .status-panel.danger {{ border-left-color: var(--danger); }}
    .status-head {{
        display: flex; justify-content: space-between; align-items: baseline;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        color: var(--text-faint); margin-bottom: 6px;
    }}
    .status-tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
        padding: 2px 8px; border-radius: 4px;
    }}
    .status-tag.ok {{ color: var(--ok); background: rgba(79,157,222,0.12); }}
    .status-tag.watch {{ color: var(--watch); background: rgba(245,166,35,0.12); }}
    .status-tag.danger {{ color: var(--danger); background: rgba(229,72,77,0.12); }}
    .status-body {{ font-size: 0.95rem; color: var(--text); line-height: 1.5; }}
    .status-why {{ color: var(--text-dim); font-size: 0.85rem; margin-top: 6px; }}

    .log-row {{
        display: grid;
        grid-template-columns: 60px 90px 1fr 1.3fr 1.3fr;
        gap: 12px;
        align-items: center;
        padding: 10px 14px;
        border-bottom: 1px solid var(--hairline);
        font-size: 0.83rem;
    }}
    .log-row:last-child {{ border-bottom: none; }}
    .log-row.head {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--text-faint);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .log-time {{ font-family: 'IBM Plex Mono', monospace; color: var(--text-dim); }}
    .log-conf {{ font-family: 'IBM Plex Mono', monospace; color: var(--danger); font-weight: 600; }}
    .log-conf.watch {{ color: var(--watch); }}
    .log-action {{ color: var(--text); font-weight: 500; }}
    .log-reason {{ color: var(--text-dim); }}
    .log-ai {{ color: var(--teal); }}
    .log-wrap {{
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 18px;
    }}

    .side-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        color: var(--text-faint);
        text-transform: uppercase;
        margin-bottom: 6px;
    }}
    </style>
    """)


inject_css()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
md("""
<div class="eyebrow"><span class="pulse-dot"></span>LIVE TELEMETRY</div>
<div class="dash-title">Predictive Maintenance</div>
<div class="dash-sub">Monitoring<span class="arrow">→</span>Diagnosis<span class="arrow">→</span>Recommendation</div>
""")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="side-label">Simulation</div>', unsafe_allow_html=True)
max_rows = st.sidebar.slider("Rows to stream", 10, 500, 100)
delay = st.sidebar.slider("Delay between readings (sec)", 0.0, 2.0, 0.3)
start_button = st.sidebar.button("▶  Start simulation", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="side-label">Custom Input</div>', unsafe_allow_html=True)

input_mode = st.sidebar.radio(
    "Data source",
    ["Simulated stream", "Upload CSV", "Manual entry"],
    label_visibility="collapsed"
)

uploaded_file = None
manual_row = None
manual_submit = False

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload sensor data (CSV)",
        type=["csv"],
        help="Required columns: Type, Air temperature [K], Process temperature [K], "
             "Rotational speed [rpm], Torque [Nm], Tool wear [min]"
    )

elif input_mode == "Manual entry":
    with st.sidebar.form("manual_input_form"):
        m_type = st.selectbox("Type", ["L", "M", "H"])
        m_air_temp = st.number_input("Air temperature [K]", value=298.1, step=0.1)
        m_process_temp = st.number_input("Process temperature [K]", value=308.6, step=0.1)
        m_rpm = st.number_input("Rotational speed [rpm]", value=1500, step=10)
        m_torque = st.number_input("Torque [Nm]", value=40.0, step=0.1)
        m_tool_wear = st.number_input("Tool wear [min]", value=0, step=1)
        manual_submit = st.form_submit_button("Run diagnosis")

        if manual_submit:
            manual_row = {
                "Type": m_type,
                "Air temperature [K]": m_air_temp,
                "Process temperature [K]": m_process_temp,
                "Rotational speed [rpm]": m_rpm,
                "Torque [Nm]": m_torque,
                "Tool wear [min]": m_tool_wear
            }
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="side-label">Legend</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="font-size:0.82rem; line-height:2; color:{COLORS['text_dim']}">
<span style="color:{COLORS['ok']}">●</span> Normal reading<br>
<span style="color:{COLORS['watch']}">●</span> Anomaly, low confidence<br>
<span style="color:{COLORS['danger']}">●</span> High-risk, high confidence
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "total_scanned" not in st.session_state:
    st.session_state.total_scanned = 0
    st.session_state.total_anomalies = 0
    st.session_state.total_high_confidence = 0
    st.session_state.pulse_history = []
    st.session_state.log_rows = []
    st.session_state.low_conf_rows = []
    st.session_state.last_csv_name = None

# ---------------------------------------------------------------------------
# Layout placeholders
# ---------------------------------------------------------------------------
strip_placeholder = st.empty()
dial_placeholder = st.empty()
status_placeholder = st.empty()
results_placeholder = st.container()


def render_strip():
    history = st.session_state.pulse_history[-60:]
    if not history:
        bars_html = '<div class="strip-empty">— waiting for readings —</div>'
    else:
        bars = "".join(
            f'<div class="bar" style="height:{h}%;background:{c};"></div>'
            for c, h in history
        )
        bars_html = f'<div class="strip">{bars}</div>'
    md(f"""
    <div class="strip-wrap">
        <div class="strip-label">Reading confidence — last {len(history)} of {len(st.session_state.pulse_history)}</div>
        {bars_html}
    </div>
    """, target=strip_placeholder)


def dial(value, label, pct, color):
    pct = max(0, min(100, pct))
    return f"""
    <div class="dial-card">
        <div class="dial" style="background:conic-gradient({color} 0% {pct}%, {COLORS['hairline']} {pct}% 100%);">
            <div class="dial-inner">{pct:.0f}%</div>
        </div>
        <div class="dial-meta">
            <div class="dial-value">{value}</div>
            <div class="dial-label">{label}</div>
        </div>
    </div>
    """


def render_dials():
    scanned = st.session_state.total_scanned
    anomalies = st.session_state.total_anomalies
    high_conf = st.session_state.total_high_confidence

    anomaly_pct = (anomalies / scanned * 100) if scanned else 0
    high_pct = (high_conf / scanned * 100) if scanned else 0

    md(f"""
    <div class="dial-row">
        {dial(scanned, "Readings scanned", 100, COLORS['teal'])}
        {dial(anomalies, "Anomalies flagged", anomaly_pct, COLORS['watch'])}
        {dial(high_conf, "High-risk alerts", high_pct, COLORS['danger'])}
    </div>
    """, target=dial_placeholder)


def render_status(kind, tag_text, body, why=None):
    css_class = {"ok": "", "watch": "watch", "danger": "danger"}[kind]
    why_html = f'<div class="status-why">{html.escape(why)}</div>' if why else ""
    md(f"""
    <div class="status-panel {css_class}">
        <div class="status-head">
            <span>CURRENT READING</span>
            <span class="status-tag {kind}">{tag_text}</span>
        </div>
        <div class="status-body">{html.escape(body)}</div>
        {why_html}
    </div>
    """, target=status_placeholder)


def render_alert_table(rows, title, conf_class="danger"):
    """
    Renders a table with Time, Confidence, Action/Prediction, Reason (SHAP),
    AND AI Insight as a column -- so the AI-generated text is shown inline
    per row instead of a separate panel.
    """
    if not rows:
        return
    rows_html = "".join(f"""
        <div class="log-row">
            <div class="log-time mono">t={r['Time']}</div>
            <div class="log-conf {conf_class if conf_class == 'watch' else ''}">{r['Confidence']}</div>
            <div class="log-action">{html.escape(r.get('Action') or r.get('Prediction', ''))}</div>
            <div class="log-reason">{html.escape(r['Reason'])}</div>
            <div class="log-ai">{html.escape(r.get('AI_Insight', r['Reason']))}</div>
        </div>
    """ for r in rows)

    md(f"""
    <div style="margin-top:6px; margin-bottom:8px; font-weight:600; font-size:1.05rem;">{title}</div>
    <div class="log-wrap">
        <div class="log-row head">
            <div>Time</div><div>Confidence</div><div>Action</div><div>Reason</div><div>AI Insight</div>
        </div>
        {rows_html}
    </div>
    """)


def sync_cached_insights():
    """
    Updates session state log rows and low-confidence rows with any
    completed Gemini insights that have finished in the background.
    """
    for r in st.session_state.log_rows:
        key = r.get("_key")
        if key:
            cached = get_cached_insight(key)
            if cached:
                r["AI_Insight"] = cached

    for r in st.session_state.low_conf_rows:
        key = r.get("_key")
        if key:
            cached = get_cached_insight(key)
            if cached:
                r["AI_Insight"] = cached


def render_results():
    """
    Renders BOTH result tables (each with an AI Insight column) and their
    download buttons, always reading from session_state so they persist
    across reruns (e.g. after a download button click) until a new run
    explicitly resets the state.
    """
    sync_cached_insights()

    log_rows = st.session_state.log_rows
    low_conf_rows = st.session_state.low_conf_rows

    if not log_rows and not low_conf_rows:
        return

    with results_placeholder:
        if log_rows:
            render_alert_table(log_rows, "High-risk events", conf_class="danger")
            high_export = [
                {
                    "Time": r["Time"],
                    "Confidence": r["Confidence"],
                    "Action": r.get("Action", ""),
                    "Reason": r["Reason"],
                    "AI_Insight": r.get("AI_Insight", r["Reason"])
                }
                for r in log_rows
            ]
            high_df = pd.DataFrame(high_export)
            st.download_button(
                "⬇ Download high-risk alerts (CSV)",
                data=high_df.to_csv(index=False).encode("utf-8"),
                file_name="high_risk_alerts.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_high"
            )

        if low_conf_rows:
            render_alert_table(low_conf_rows, "Low-confidence anomalies", conf_class="watch")
            low_export = [
                {
                    "Time": r["Time"],
                    "Confidence": r["Confidence"],
                    "Prediction": r.get("Prediction", ""),
                    "Reason": r["Reason"],
                    "AI_Insight": r.get("AI_Insight", r["Reason"])
                }
                for r in low_conf_rows
            ]
            low_df = pd.DataFrame(low_export)
            st.download_button(
                "⬇ Download low-confidence anomalies (CSV)",
                data=low_df.to_csv(index=False).encode("utf-8"),
                file_name="low_confidence_anomalies.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_low"
            )


def row_to_dict(row: pd.Series) -> dict:
    return {
        "Type": row["Type"],
        "Air temperature [K]": row["Air temperature [K]"],
        "Process temperature [K]": row["Process temperature [K]"],
        "Rotational speed [rpm]": row["Rotational speed [rpm]"],
        "Torque [Nm]": row["Torque [Nm]"],
        "Tool wear [min]": row["Tool wear [min]"]
    }
def create_database_run(user_id, run_type, source_filename=None):
    """Create a PostgreSQL record for one prediction run."""
    db = SessionLocal()

    try:
        run = Run(
            user_id=user_id,
            run_type=run_type,
            source_filename=source_filename,
            threshold_used=0.4,
            contamination_used=0.15,
        )

        db.add(run)
        db.commit()
        db.refresh(run)

        return run.run_id

    finally:
        db.close()
def update_database_run(
    run_id,
    total_scanned,
    total_anomalies,
    total_high_confidence,
):
    """Update PostgreSQL with the final results of a prediction run."""
    db = SessionLocal()

    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()

        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.total_scanned = total_scanned
        run.total_anomalies = total_anomalies
        run.total_high_confidence = total_high_confidence

        db.commit()

    finally:
        db.close()
def save_high_risk_alert(
    run_id,
    confidence,
    action,
    reason,
):
    """Save one high-risk alert to PostgreSQL."""
    db = SessionLocal()

    try:
        alert = HighRiskAlert(
            run_id=run_id,
            row_time=datetime.now(),
            confidence=confidence,
            action=action,
            reason=reason,
        )

        db.add(alert)
        db.commit()
        db.refresh(alert)

        return alert.alert_id

    finally:
        db.close()
def save_low_confidence_anomaly(
    run_id,
    confidence,
    prediction,
    reason,
):
    """Save one low-confidence anomaly to PostgreSQL."""
    db = SessionLocal()

    try:
        anomaly = LowConfidenceAnomaly(
            run_id=run_id,
            row_time=datetime.now(),
            confidence=confidence,
            prediction=prediction,
            reason=reason,
        )

        db.add(anomaly)
        db.commit()
        db.refresh(anomaly)

        return anomaly.anomaly_id

    finally:
        db.close()


def process_row(i, sensor_row, label_prefix="t", run_id=None):
    """
    Runs one row through the pipeline, updates counters/history, appends
    to session_state result lists (with an AI_Insight field), triggers the
    background LLM call, and renders the current status panel.
    """
    result = run_pipeline(sensor_row)
    st.session_state.total_scanned += 1

    if result["status"] == "normal":
        st.session_state.pulse_history.append((COLORS["ok"], 35))
        render_status("ok", "NORMAL", f"[{label_prefix}={i}] Reading OK — no anomaly detected.")
    else:
        st.session_state.total_anomalies += 1
        diagnosis = result["diagnosis"]
        recommendation = result["recommendation"]
        conf_pct = diagnosis["confidence"] * 100
        bar_height = max(15, min(100, conf_pct))
        shap_reason = diagnosis.get("plain_explanation") or diagnosis.get("explanation", "")
        event_label = f"{label_prefix}={i}"
        event_key = make_event_key(event_label, diagnosis, recommendation)

        if diagnosis["confidence"] >= 0.70 and diagnosis["prediction"] == 1:
            st.session_state.total_high_confidence += 1
            st.session_state.pulse_history.append((COLORS["danger"], bar_height))
            render_status(
                "danger", f"{conf_pct:.0f}% CONFIDENCE",
                f"[{label_prefix}={i}] {recommendation['action']}",
                why=shap_reason,
            )
            high_row = {
                "Time": i,
                "Confidence": f"{diagnosis['confidence']:.0%}",
                "Action": recommendation["action"],
                "Reason": shap_reason,
                "AI_Insight": shap_reason,
                "_key": event_key,
            }
            st.session_state.log_rows.append(high_row)
            if run_id is not None:
                save_high_risk_alert(
                    run_id=run_id,
                    confidence=diagnosis["confidence"],
                    action=recommendation["action"],
                    reason=shap_reason,
                )

            def _on_high_ready(eid, ekey, text):
                high_row["AI_Insight"] = text

            initial_insight = trigger_async_llm_reasoning(
                event_id=event_label,
                diagnosis=diagnosis,
                recommendation=recommendation,
                callback=_on_high_ready
            )
            if initial_insight and initial_insight != shap_reason:
                high_row["AI_Insight"] = initial_insight
        else:
            st.session_state.pulse_history.append((COLORS["watch"], bar_height))
            render_status(
                "watch", f"{conf_pct:.0f}% CONFIDENCE",
                f"[{label_prefix}={i}] Anomaly flagged — confidence too low to act on.",
                why=shap_reason,
            )
            low_row = {
                "Time": i,
                "Confidence": f"{diagnosis['confidence']:.0%}",
                "Prediction": "Failure" if diagnosis["prediction"] == 1 else "No failure",
                "Reason": shap_reason,
                "AI_Insight": shap_reason,
                "_key": event_key,
            }
            st.session_state.low_conf_rows.append(low_row)

            if run_id is not None:
                save_low_confidence_anomaly(
                    run_id=run_id,
                    confidence=diagnosis["confidence"],
                    prediction=diagnosis["prediction"],
                    reason=shap_reason,
                )

            def _on_low_ready(eid, ekey, text):
                low_row["AI_Insight"] = text

            # Only call the LLM for low-confidence rows if you want AI text
            # there too. Comment this out if you'd rather save API quota
            # and only get AI insights on high-risk rows.
            initial_insight = trigger_async_llm_reasoning(
                event_id=event_label,
                diagnosis=diagnosis,
                recommendation=recommendation,
                callback=_on_low_ready
            )
            if initial_insight and initial_insight != shap_reason:
                low_row["AI_Insight"] = initial_insight

render_strip()
render_dials()


render_strip()
render_dials()

# ---------------------------------------------------------------------------
# Simulation loop — starting a NEW run resets session_state result lists
# ---------------------------------------------------------------------------
if start_button:
    run_id = create_database_run(
        st.session_state.user_id,
        "simulation",
    )    
    st.session_state.total_scanned = 0
    st.session_state.total_anomalies = 0
    st.session_state.total_high_confidence = 0
    st.session_state.pulse_history = []
    st.session_state.log_rows = []
    st.session_state.low_conf_rows = []

    df = pd.read_csv(DATA_PATH).head(max_rows)
    for i, row in df.iterrows():
        process_row(i, row_to_dict(row), label_prefix="t", run_id=run_id)
        time.sleep(delay)
    update_database_run(
    run_id,
    st.session_state.total_scanned,
    st.session_state.total_anomalies,
    st.session_state.total_high_confidence,
)

    sync_cached_insights()
    st.success("Simulation complete.")

# ---------------------------------------------------------------------------
# CSV Upload handling — only reprocess when a NEW file is uploaded
# ---------------------------------------------------------------------------
if uploaded_file is not None and uploaded_file.name != st.session_state.last_csv_name:
    required_cols = ["Type", "Air temperature [K]", "Process temperature [K]",
                      "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing = [c for c in required_cols if c not in uploaded_df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {', '.join(missing)}")
        else:
            st.session_state.total_scanned = 0
            st.session_state.total_anomalies = 0
            st.session_state.total_high_confidence = 0
            st.session_state.pulse_history = []
            st.session_state.log_rows = []
            st.session_state.low_conf_rows = []
            st.session_state.last_csv_name = uploaded_file.name

            run_id = create_database_run(
                st.session_state.user_id,
                "csv_upload",
                uploaded_file.name,
            )

            st.success(f"Loaded {len(uploaded_df)} rows. Running through the pipeline...")

            for i, row in uploaded_df.iterrows():
                process_row(
                    i,
                    row_to_dict(row),
                    label_prefix="row",
                    run_id=run_id,
                )

            update_database_run(
                run_id,
                st.session_state.total_scanned,
                st.session_state.total_anomalies,
                st.session_state.total_high_confidence,
            )

            sync_cached_insights()
    except Exception as e:
        st.error(f"Couldn't process the file: {e}")

# ---------------------------------------------------------------------------
# Manual entry handling — each submission adds to the running results
# ---------------------------------------------------------------------------
if manual_row is not None:
    process_row("manual", manual_row, label_prefix="entry")
    sync_cached_insights()

# ---------------------------------------------------------------------------
# Always render results (persists across reruns, e.g. after download click)
# ---------------------------------------------------------------------------
render_results()