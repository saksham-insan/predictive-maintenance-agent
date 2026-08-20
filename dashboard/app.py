"""
Predictive Maintenance Agent — Live Telemetry & Agentic AI Dashboard.

Features:
- Dual Pipeline Modes: Rule-Based (Baseline) vs Agentic AI (LLM + ReAct + RAG + Memory)
- Live Step-by-Step ReAct Reasoning Chain Visualizer
- Machine Temporal Degradation Trajectory Monitor
- Real-time Tool Audit Trail (tickets, inventory checks, escalations)
- Streaming Simulation, Custom CSV Upload, and Manual Sensor Parameter Testing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import json
import html
import pandas as pd
import streamlit as st

from agents.orchestrator import run_pipeline as run_pipeline_rule_based
from agents.agentic_pipeline import run_pipeline_agentic
from memory import MachineMemory
from llm_client import OllamaClient, MockLLMClient, GroqClient
from human_readable import translate_to_human_readable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "ai4i2020.csv")
TOOL_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "agent_logs", "tool_calls.jsonl")


def md(content: str, target=None):
    """
    Render raw HTML/CSS via st.markdown.
    """
    lines = [line.lstrip() for line in content.strip("\n").split("\n")]
    cleaned = "\n".join(line for line in lines if line != "")
    sink = target if target is not None else st
    sink.markdown(cleaned, unsafe_allow_html=True)


st.set_page_config(
    page_title="Predictive Maintenance — Autonomous Agent Telemetry",
    page_icon="⚡",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#121518",
    "panel": "#1A1E23",
    "panel_alt": "#22272E",
    "hairline": "#2D333B",
    "text": "#F0F3F6",
    "text_dim": "#9DA7B3",
    "text_faint": "#636E7B",
    "ok": "#38BDF8",
    "watch": "#FBBF24",
    "danger": "#F87171",
    "teal": "#2DD4BF",
    "purple": "#A78BFA",
    "accent_bg": "rgba(45, 212, 191, 0.08)"
}


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
        --purple: {COLORS['purple']};
    }}

    .stApp {{ background: var(--bg); color: var(--text); }}
    section[data-testid="stSidebar"] {{ background: var(--panel); border-right: 1px solid var(--hairline); }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}
    body, p, div, span {{ font-family: 'Space Grotesk', sans-serif; }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.18em;
        color: var(--teal);
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }}
    .pulse-dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 0 rgba(45,212,191,0.6);
        animation: pulse 1.8s infinite;
    }}
    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(45,212,191,0.55); }}
        70%  {{ box-shadow: 0 0 0 8px rgba(45,212,191,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(45,212,191,0); }}
    }}
    .dash-title {{ font-size: 2.1rem; font-weight: 700; margin: 0 0 2px 0; color: var(--text); }}
    .dash-sub {{ color: var(--text-dim); font-size: 0.94rem; margin-bottom: 1.2rem; }}
    .dash-sub .arrow {{ color: var(--text-faint); margin: 0 6px; }}

    .mode-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        margin-bottom: 12px;
    }}
    .mode-badge.agentic {{
        background: rgba(45,212,191,0.12);
        color: var(--teal);
        border: 1px solid rgba(45,212,191,0.3);
    }}
    .mode-badge.rule {{
        background: rgba(157,167,179,0.12);
        color: var(--text-dim);
        border: 1px solid var(--hairline);
    }}

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
        flex: 1 1 200px;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 16px 18px;
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .dial {{
        width: 60px; height: 60px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .dial-inner {{
        width: 44px; height: 44px; border-radius: 50%;
        background: var(--panel);
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; font-weight: 600;
        color: var(--text);
    }}
    .dial-meta .dial-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.45rem; font-weight: 600; color: var(--text); line-height: 1;
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
    .status-tag.ok {{ color: var(--ok); background: rgba(56,189,248,0.12); }}
    .status-tag.watch {{ color: var(--watch); background: rgba(251,191,36,0.12); }}
    .status-tag.danger {{ color: var(--danger); background: rgba(248,113,113,0.12); }}
    .status-body {{ font-size: 0.95rem; color: var(--text); line-height: 1.5; }}
    .status-why {{ color: var(--text-dim); font-size: 0.85rem; margin-top: 6px; }}

    /* Agentic ReAct Cards */
    .agent-box {{
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }}
    .agent-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--hairline);
        padding-bottom: 10px;
        margin-bottom: 14px;
    }}
    .agent-step-card {{
        background: var(--panel-alt);
        border: 1px solid var(--hairline);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }}
    .step-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--teal);
        background: rgba(45,212,191,0.12);
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 8px;
    }}
    .tool-badge {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--purple);
        background: rgba(167,139,250,0.12);
        border: 1px solid rgba(167,139,250,0.3);
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 6px;
    }}
    .tool-output-box {{
        background: #0E1114;
        border: 1px solid #232830;
        border-radius: 6px;
        padding: 8px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: #8FD5B9;
        margin-top: 6px;
        overflow-x: auto;
    }}
    .grounding-chip {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--ok);
        background: rgba(56,189,248,0.1);
        border: 1px solid rgba(56,189,248,0.25);
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-top: 4px;
    }}

    .log-row {{
        display: grid;
        grid-template-columns: 70px 110px 1fr 2fr;
        gap: 14px;
        align-items: center;
        padding: 10px 14px;
        border-bottom: 1px solid var(--hairline);
        font-size: 0.85rem;
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
# Sidebar controls & Mode selection
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="side-label">Pipeline Architecture</div>', unsafe_allow_html=True)
pipeline_mode = st.sidebar.radio(
    "Architecture",
    ["Agentic AI (LLM + ReAct + RAG)", "Rule-based (Baseline if/else)"],
    index=0
)

llm_backend = "mock"
if "Agentic" in pipeline_mode:
    st.sidebar.markdown('<div class="side-label">LLM Engine</div>', unsafe_allow_html=True)
    backend_choice = st.sidebar.selectbox(
        "Backend Provider",
        ["Ollama (Local llama3.2)", "Mock Agent (Deterministic / Offline)", "Groq (Cloud API)"],
        index=0
    )
    if "Ollama" in backend_choice:
        llm_backend = "ollama"
    elif "Groq" in backend_choice:
        llm_backend = "groq"
    else:
        llm_backend = "mock"

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="side-label">Simulation</div>', unsafe_allow_html=True)
max_rows = st.sidebar.slider("Rows to stream", 10, 500, 100)
delay = st.sidebar.slider("Delay between readings (sec)", 0.0, 2.0, 0.2)
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
        m_air_temp = st.number_input("Air temperature [K]", value=300.5, step=0.1)
        m_process_temp = st.number_input("Process temperature [K]", value=311.2, step=0.1)
        m_rpm = st.number_input("Rotational speed [rpm]", value=1350, step=10)
        m_torque = st.number_input("Torque [Nm]", value=65.0, step=0.1)
        m_tool_wear = st.number_input("Tool wear [min]", value=220, step=1)
        manual_submit = st.form_submit_button("Run Autonomous Diagnosis")

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
<span style="color:{COLORS['watch']}">●</span> Anomaly / Low urgency<br>
<span style="color:{COLORS['danger']}">●</span> High-risk / Critical action<br>
<span style="color:{COLORS['teal']}">●</span> Agent Tool Call Executed
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
    st.session_state.human_history = []
    st.session_state.last_csv_name = None
    st.session_state.last_agent_result = None
    st.session_state.last_sensor_row = None
    st.session_state.temporal_memory = MachineMemory(maxlen=20)

# Instantiate LLM client
active_llm_client = None
if "Agentic" in pipeline_mode:
    try:
        if llm_backend == "ollama":
            active_llm_client = OllamaClient()
        elif llm_backend == "groq":
            active_llm_client = GroqClient()
        else:
            active_llm_client = MockLLMClient()
    except Exception as e:
        st.sidebar.warning(f"Backend '{llm_backend}' initialization note: {e}. Defaulting to Mock agent.")
        active_llm_client = MockLLMClient()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
is_agentic = "Agentic" in pipeline_mode
badge_class = "agentic" if is_agentic else "rule"
badge_text = "⚡ AGENTIC AI LAYER (RE-ACT + RAG + MEMORY)" if is_agentic else "📐 RULE-BASED BASELINE (IF/ELSE)"

md(f"""
<div class="eyebrow"><span class="pulse-dot"></span>LIVE INDUSTRIAL TELEMETRY</div>
<div class="dash-title">Predictive Maintenance Autonomous Agent</div>
<div class="mode-badge {badge_class}">{badge_text}</div>
<div class="dash-sub">Isolation Forest Anomaly Monitor <span class="arrow">→</span> Tuned SHAP Diagnosis <span class="arrow">→</span> Grounded Physics RAG <span class="arrow">→</span> ReAct Action Loop</div>
""")

# ---------------------------------------------------------------------------
# Dual-Section Tab Navigation
# ---------------------------------------------------------------------------
tab_tech, tab_human = st.tabs([
    "🔧 Engineering & Diagnostic Telemetry",
    "📋 Plain-English Executive & Operator View"
])

with tab_tech:
    strip_placeholder = st.empty()
    dial_placeholder = st.empty()
    status_placeholder = st.empty()
    agentic_trace_placeholder = st.container()
    results_placeholder = st.container()

with tab_human:
    human_overview_placeholder = st.empty()
    human_explanation_placeholder = st.empty()
    human_checklist_placeholder = st.empty()
    human_impact_placeholder = st.empty()
    human_history_placeholder = st.container()


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
        <div class="strip-label">Telemetry pulse & confidence stream — last {len(history)} readings</div>
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
        {dial(anomalies, "Anomalies detected", anomaly_pct, COLORS['watch'])}
        {dial(high_conf, "Maintenance actions", high_pct, COLORS['danger'])}
    </div>
    """, target=dial_placeholder)


def render_status(kind, tag_text, body, why=None):
    css_class = {"ok": "", "watch": "watch", "danger": "danger"}[kind]
    why_html = f'<div class="status-why">{html.escape(why)}</div>' if why else ""
    md(f"""
    <div class="status-panel {css_class}">
        <div class="status-head">
            <span>CURRENT STATUS</span>
            <span class="status-tag {kind}">{tag_text}</span>
        </div>
        <div class="status-body">{html.escape(body)}</div>
        {why_html}
    </div>
    """, target=status_placeholder)


def render_human_view(sensor_row, status, diagnosis=None, trend=None, final=None, tool_calls=None):
    """
    Renders the plain-English executive and operator section.
    """
    hr = translate_to_human_readable(
        telemetry=sensor_row,
        status=status,
        diagnosis=diagnosis,
        trend_snapshot=trend,
        final_answer=final,
        tool_calls=tool_calls
    )

    badge_bg = hr["health_badge_color"]
    with human_overview_placeholder.container():
        md(f"""
        <div style="background:var(--panel); border:1px solid var(--hairline); border-left:6px solid {badge_bg}; border-radius:10px; padding:20px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:700; color:{badge_bg}; background:rgba(255,255,255,0.06); padding:4px 10px; border-radius:4px; text-transform:uppercase; letter-spacing:0.08em;">
                        {hr['health_status']}
                    </span>
                    <h3 style="margin:10px 0 4px 0; font-size:1.35rem; color:var(--text);">{hr['headline']}</h3>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:var(--text-dim); text-transform:uppercase;">Overall Machine Reliability</div>
                    <div style="font-size:2.2rem; font-weight:700; color:{badge_bg}; font-family:'Space Grotesk',sans-serif;">{hr['reliability_score']}</div>
                </div>
            </div>
        </div>
        """)

    with human_explanation_placeholder.container():
        md(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px;">
            <div style="background:var(--panel-alt); border:1px solid var(--hairline); border-radius:8px; padding:16px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:var(--teal); font-weight:600; text-transform:uppercase; margin-bottom:6px;">
                    🔍 What the Machine is Experiencing
                </div>
                <div style="font-size:0.92rem; line-height:1.6; color:var(--text);">
                    {html.escape(hr['what_happened'])}
                </div>
            </div>
            <div style="background:var(--panel-alt); border:1px solid var(--hairline); border-radius:8px; padding:16px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:var(--watch); font-weight:600; text-transform:uppercase; margin-bottom:6px;">
                    ⚙️ Plain-English Root Cause Explanation
                </div>
                <div style="font-size:0.92rem; line-height:1.6; color:var(--text-dim);">
                    {html.escape(hr['root_cause_plain'])}
                </div>
            </div>
        </div>
        """)

    with human_checklist_placeholder.container():
        tasks_html = ""
        for item in hr['operator_checklist']:
            icon = "✅" if item.get("done") else "◻️"
            task_text = html.escape(item['task'])
            tasks_html += f"""
            <div style="display:flex; align-items:flex-start; gap:10px; padding:10px 14px; background:var(--panel); border:1px solid var(--hairline); border-radius:6px; margin-bottom:8px;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div style="font-size:0.9rem; color:var(--text);">
                    <strong style="color:var(--teal);">Step {item['step']}:</strong> {task_text}
                </div>
            </div>
            """
        md(f"""
        <div style="margin-bottom:18px;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:var(--teal); font-weight:600; text-transform:uppercase; margin-bottom:8px;">
                🛠️ Shop Floor Action Checklist for Technicians
            </div>
            {tasks_html}
        </div>
        """)

    with human_impact_placeholder.container():
        imp = hr["impact_cards"]
        md(f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:14px; margin-bottom:18px;">
            <div style="background:var(--panel); border:1px solid var(--hairline); border-radius:8px; padding:14px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--text-dim); text-transform:uppercase;">⏱ Time Until Breakdown</div>
                <div style="font-size:0.92rem; font-weight:600; color:var(--text); margin-top:4px;">{html.escape(imp['rul'])}</div>
            </div>
            <div style="background:var(--panel); border:1px solid var(--hairline); border-radius:8px; padding:14px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--text-dim); text-transform:uppercase;">📦 Spare Parts Status</div>
                <div style="font-size:0.92rem; font-weight:600; color:var(--text); margin-top:4px;">{html.escape(imp['parts_status'])}</div>
            </div>
            <div style="background:var(--panel); border:1px solid var(--hairline); border-radius:8px; padding:14px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--text-dim); text-transform:uppercase;">💰 Downtime Prevention Risk</div>
                <div style="font-size:0.92rem; font-weight:600; color:var(--text); margin-top:4px;">{html.escape(imp['downtime_risk'])}</div>
            </div>
            <div style="background:var(--panel); border:1px solid var(--hairline); border-radius:8px; padding:14px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:var(--text-dim); text-transform:uppercase;">👷 Assigned Role</div>
                <div style="font-size:0.92rem; font-weight:600; color:var(--text); margin-top:4px;">{html.escape(imp['assigned_role'])}</div>
            </div>
        </div>
        """)


def render_agentic_reasoning(agent_res: dict):
    """
    Renders the live ReAct reasoning trace, tool calls, and grounded knowledge badges.
    """
    if not agent_res or agent_res.get("status") != "anomaly_detected":
        return

    with agentic_trace_placeholder:
        trace = agent_res.get("reasoning_trace", [])
        grounding = agent_res.get("grounding_sources", [])
        trend = agent_res.get("trend_snapshot")
        final = agent_res.get("final_answer", {})

        # Knowledge Grounding Chips
        grounding_html = "".join(f'<span class="grounding-chip">📖 {g}</span>' for g in grounding)

        # Trajectory Details
        trajectory_html = ""
        if trend:
            twf_txt = f"{trend.est_readings_to_twf_band:.0f} readings" if trend.est_readings_to_twf_band is not None else "Stable / >20 steps"
            trajectory_html = f"""
            <div style="font-size:0.84rem; color:var(--text-dim); margin-top:8px; padding:8px 12px; background:var(--panel-alt); border-radius:6px; border:1px solid var(--hairline);">
                <strong>Temporal Trajectory:</strong> Tool Wear Slope: <code>{trend.tool_wear_slope:+.2f} min/reading</code> | 
                Torque Slope: <code>{trend.torque_slope:+.2f} Nm/reading</code> | 
                Est. to 200 min TWF: <code>{twf_txt}</code>
            </div>
            """

        # Steps HTML
        steps_html = []
        for s in trace:
            step_num = s.get("iteration", 1)
            reasoning = html.escape(s.get("reasoning", ""))
            tool_name = s.get("tool_called")
            tool_args = s.get("tool_args")
            tool_res = s.get("tool_result")

            tool_block = ""
            if tool_name:
                args_str = html.escape(json.dumps(tool_args))
                res_str = html.escape(json.dumps(tool_res, indent=2))
                tool_block = f"""
                <div class="tool-badge">🔧 Tool Invocated: {tool_name}</div>
                <div style="font-size:0.78rem; color:var(--text-dim); margin-top:4px;">Args: <code>{args_str}</code></div>
                <div class="tool-output-box">{res_str}</div>
                """

            steps_html.append(f"""
            <div class="agent-step-card">
                <div><span class="step-badge">STEP {step_num}</span> <span style="font-size:0.9rem; color:var(--text);">{reasoning}</span></div>
                {tool_block}
            </div>
            """)

        steps_rendered = "".join(steps_html)

        md(f"""
        <div class="agent-box">
            <div class="agent-header">
                <div>
                    <span style="font-weight:700; font-size:1.05rem; color:var(--teal);">🧠 Autonomous ReAct Reasoning Trace</span>
                </div>
                <div>{grounding_html}</div>
            </div>
            {trajectory_html}
            <div style="margin-top:12px;">{steps_rendered}</div>
            <div style="margin-top:14px; padding:12px 14px; background:rgba(45,212,191,0.06); border:1px solid rgba(45,212,191,0.25); border-radius:8px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:var(--teal); font-weight:600; text-transform:uppercase;">Final Directive & Synthesis</div>
                <div style="font-size:0.95rem; color:var(--text); margin-top:4px; font-weight:500;">{html.escape(final.get('summary', ''))}</div>
            </div>
        </div>
        """)


def render_alert_table(rows, title, conf_class="danger"):
    if not rows:
        return
    rows_html = "".join(f"""
        <div class="log-row">
            <div class="log-time mono">t={r['Time']}</div>
            <div class="log-conf {conf_class if conf_class == 'watch' else ''}">{r['Confidence']}</div>
            <div class="log-action">{html.escape(r.get('Action') or r.get('Prediction', ''))}</div>
            <div class="log-reason">{html.escape(r['Reason'])}</div>
        </div>
    """ for r in rows)

    md(f"""
    <div style="margin-top:6px; margin-bottom:8px; font-weight:600; font-size:1.05rem;">{title}</div>
    <div class="log-wrap">
        <div class="log-row head">
            <div>Time</div><div>Confidence</div><div>Action</div><div>Reason</div>
        </div>
        {rows_html}
    </div>
    """)


def render_tool_audit_log():
    """Renders recent tool calls from the JSONL log file."""
    if not os.path.exists(TOOL_LOG_PATH):
        return

    try:
        with open(TOOL_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return

        records = [json.loads(line) for line in lines[-15:]][::-1]
        if not records:
            return

        rows = []
        for r in records:
            ts = r.get("timestamp", "")[:19].replace("T", " ")
            tool = r.get("tool", "")
            args = json.dumps(r.get("args", {}))
            res = r.get("result", {})
            status = res.get("status", "success")
            detail = res.get("ticket_id") or res.get("escalation_id") or (res.get("details", {}).get("part_name")) or str(res)[:60]
            rows.append({
                "Timestamp (UTC)": ts,
                "Tool Executed": tool,
                "Parameters": args,
                "Result / Output": str(detail)
            })

        st.markdown("### 📋 Tool Audit Trail & Automated Actions Log")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    except Exception as e:
        pass


def render_results():
    log_rows = st.session_state.log_rows
    low_conf_rows = st.session_state.low_conf_rows

    with results_placeholder:
        if log_rows:
            render_alert_table(log_rows, "High-Risk & Maintenance Action Events", conf_class="danger")
            high_df = pd.DataFrame(log_rows)
            st.download_button(
                "⬇ Download high-risk alerts (CSV)",
                data=high_df.to_csv(index=False).encode("utf-8"),
                file_name="high_risk_alerts.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_high"
            )

        if low_conf_rows:
            render_alert_table(low_conf_rows, "Low-Confidence / Transient Anomalies", conf_class="watch")
            low_df = pd.DataFrame(low_conf_rows)
            st.download_button(
                "⬇ Download low-confidence anomalies (CSV)",
                data=low_df.to_csv(index=False).encode("utf-8"),
                file_name="low_confidence_anomalies.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_low"
            )

        render_tool_audit_log()

    with human_history_placeholder:
        if st.session_state.human_history:
            st.markdown("### 📋 Operator Activity Log (Plain English)")
            st.dataframe(pd.DataFrame(st.session_state.human_history), use_container_width=True)


def row_to_dict(row: pd.Series) -> dict:
    return {
        "Type": row["Type"],
        "Air temperature [K]": float(row["Air temperature [K]"]),
        "Process temperature [K]": float(row["Process temperature [K]"]),
        "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
        "Torque [Nm]": float(row["Torque [Nm]"]),
        "Tool wear [min]": float(row["Tool wear [min]"])
    }


def process_row(i, sensor_row, label_prefix="t"):
    """
    Processes one sensor row according to the selected pipeline mode.
    """
    st.session_state.total_scanned += 1
    st.session_state.last_sensor_row = sensor_row

    if "Agentic" in pipeline_mode:
        result = run_pipeline_agentic(
            sensor_row,
            st.session_state.temporal_memory,
            llm_client=active_llm_client
        )
        st.session_state.last_agent_result = result

        if result["status"] == "normal":
            st.session_state.pulse_history.append((COLORS["ok"], 35))
            render_status("ok", "NORMAL TELEMETRY", f"[{label_prefix}={i}] Reading nominal. No anomaly flagged.")
            render_human_view(sensor_row, status="normal")
        else:
            st.session_state.total_anomalies += 1
            diagnosis = result["diagnosis"]
            final = result["final_answer"]
            conf_pct = diagnosis["confidence"] * 100
            bar_height = max(15, min(100, conf_pct))

            is_high_risk = (final.get("urgency") in ["High", "Critical"] or diagnosis["confidence"] >= 0.70)
            if is_high_risk:
                st.session_state.total_high_confidence += 1
                st.session_state.pulse_history.append((COLORS["danger"], bar_height))
                render_status(
                    "danger", f"{conf_pct:.0f}% CONFIDENCE | {final.get('urgency', 'High').upper()} URGENCY",
                    f"[{label_prefix}={i}] {final.get('summary', '')}",
                    why=f"SHAP Attribution: {diagnosis['explanation']} | Agent Rationale: {final.get('reasoning', '')}"
                )
                st.session_state.log_rows.append({
                    "Time": i,
                    "Confidence": f"{diagnosis['confidence']:.0%}",
                    "Action": final.get("action_taken", "Maintenance Order Dispatched"),
                    "Reason": final.get("summary", diagnosis["explanation"])
                })
            else:
                st.session_state.pulse_history.append((COLORS["watch"], bar_height))
                render_status(
                    "watch", f"{conf_pct:.0f}% CONFIDENCE | {final.get('urgency', 'Low').upper()}",
                    f"[{label_prefix}={i}] {final.get('summary', 'Anomaly flagged with low urgency.')}",
                    why=diagnosis["explanation"]
                )
                st.session_state.low_conf_rows.append({
                    "Time": i,
                    "Confidence": f"{diagnosis['confidence']:.0%}",
                    "Prediction": "Failure" if diagnosis["prediction"] == 1 else "No failure",
                    "Reason": final.get("summary", diagnosis["explanation"])
                })

            # Render both Technical ReAct trace and Human-Readable View
            render_agentic_reasoning(result)
            render_human_view(
                sensor_row=sensor_row,
                status=result["status"],
                diagnosis=diagnosis,
                trend=result.get("trend_snapshot"),
                final=final,
                tool_calls=result.get("tool_calls_made")
            )

            # Record human-readable entry in log
            hr_record = translate_to_human_readable(
                telemetry=sensor_row,
                status=result["status"],
                diagnosis=diagnosis,
                final_answer=final,
                tool_calls=result.get("tool_calls_made")
            )
            st.session_state.human_history.append({
                "Cycle": f"t={i}",
                "Status": hr_record["health_status"],
                "Condition": hr_record["mode_title"],
                "Summary": hr_record["what_happened"],
                "Technician Action": hr_record["operator_checklist"][0]["task"] if hr_record["operator_checklist"] else "Monitor"
            })

    else:
        # Rule-based pipeline mode
        result = run_pipeline_rule_based(sensor_row)
        if result["status"] == "normal":
            st.session_state.pulse_history.append((COLORS["ok"], 35))
            render_status("ok", "NORMAL", f"[{label_prefix}={i}] Reading OK — no anomaly detected.")
            render_human_view(sensor_row, status="normal")
        else:
            st.session_state.total_anomalies += 1
            diagnosis = result["diagnosis"]
            recommendation = result["recommendation"]
            conf_pct = diagnosis["confidence"] * 100
            bar_height = max(15, min(100, conf_pct))

            if diagnosis["confidence"] >= 0.70 and diagnosis["prediction"] == 1:
                st.session_state.total_high_confidence += 1
                st.session_state.pulse_history.append((COLORS["danger"], bar_height))
                render_status(
                    "danger", f"{conf_pct:.0f}% CONFIDENCE",
                    f"[{label_prefix}={i}] {recommendation['action']}",
                    why=diagnosis["explanation"],
                )
                st.session_state.log_rows.append({
                    "Time": i,
                    "Confidence": f"{diagnosis['confidence']:.0%}",
                    "Action": recommendation["action"],
                    "Reason": diagnosis["explanation"],
                })
            else:
                st.session_state.pulse_history.append((COLORS["watch"], bar_height))
                render_status(
                    "watch", f"{conf_pct:.0f}% CONFIDENCE",
                    f"[{label_prefix}={i}] Anomaly flagged — confidence too low to act on.",
                )
                st.session_state.low_conf_rows.append({
                    "Time": i,
                    "Confidence": f"{diagnosis['confidence']:.0%}",
                    "Prediction": "Failure" if diagnosis["prediction"] == 1 else "No failure",
                    "Reason": diagnosis["explanation"],
                })

            render_human_view(
                sensor_row=sensor_row,
                status=result["status"],
                diagnosis=diagnosis,
                final={"summary": recommendation["action"], "urgency": recommendation["urgency"]}
            )

    render_strip()
    render_dials()


# Initial render on page load
default_nominal_row = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 10
}
render_strip()
render_dials()
render_human_view(st.session_state.last_sensor_row or default_nominal_row, status="normal")

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
if start_button:
    st.session_state.total_scanned = 0
    st.session_state.total_anomalies = 0
    st.session_state.total_high_confidence = 0
    st.session_state.pulse_history = []
    st.session_state.log_rows = []
    st.session_state.low_conf_rows = []
    st.session_state.human_history = []
    st.session_state.temporal_memory = MachineMemory(maxlen=20)

    df = pd.read_csv(DATA_PATH).head(max_rows)
    for i, row in df.iterrows():
        process_row(i, row_to_dict(row), label_prefix="t")
        time.sleep(delay)

    st.success("Simulation stream completed successfully.")

# ---------------------------------------------------------------------------
# CSV Upload handling
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
            st.session_state.human_history = []
            st.session_state.last_csv_name = uploaded_file.name
            st.session_state.temporal_memory = MachineMemory(maxlen=20)

            st.success(f"Loaded {len(uploaded_df)} rows. Processing telemetry...")
            for i, row in uploaded_df.iterrows():
                process_row(i, row_to_dict(row), label_prefix="row")
    except Exception as e:
        st.error(f"Couldn't process uploaded file: {e}")

# ---------------------------------------------------------------------------
# Manual entry handling
# ---------------------------------------------------------------------------
if manual_row is not None:
    process_row("manual", manual_row, label_prefix="entry")

# ---------------------------------------------------------------------------
# Always render results & audit log
# ---------------------------------------------------------------------------
render_results()