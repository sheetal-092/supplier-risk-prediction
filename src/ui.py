"""Presentation helpers for the dark enterprise theme.

CSS, header, KPI cards, badges, alert feed and Plotly chart factories.  All
colours come from :mod:`src.config` so the palette is defined in one place.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.schema import feature_label, feature_unit

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
C = config

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---------- base ---------- */
html, body, .stApp, [class*="css"] {{ font-family: {FONT}; color: {C.TEXT}; }}
.stApp {{ background: {C.PAGE_BG}; }}
.stApp > header {{ background: transparent !important; }}
[data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
[data-testid="stToolbar"], [data-testid="stToolbarActions"], [data-testid="stHeaderActionElements"], .stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {{ display: none !important; }}
[class*="viewerBadge"], a[href*="streamlit.io/cloud"], [data-testid="stAppViewBlockContainer"] + div a[href*="streamlit.io"] {{ display: none !important; }}
.block-container {{ padding: 2.4rem 1.6rem 2rem 1.6rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ color: {C.TEXT}; font-weight: 600; letter-spacing: -0.01em; }}
p, li, label, .stMarkdown {{ color: {C.TEXT}; }}
a {{ color: {C.PRIMARY_LIGHT}; }}
hr {{ border-color: {C.BORDER}; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: {C.BORDER}; border-radius: 6px; }}
::-webkit-scrollbar-track {{ background: {C.PAGE_BG}; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{ background: {C.SIDEBAR_BG}; border-right: 1px solid {C.BORDER}; width: 330px !important; min-width: 330px !important; }}
section[data-testid="stSidebar"] .block-container {{ padding: 1rem 1rem 2rem 1rem; }}
.srp-side-label {{ font-size: 11px; font-weight: 700; letter-spacing: .1em; color: {C.MUTED}; text-transform: uppercase; margin: 12px 0 6px 0; }}

/* ---------- header ---------- */
.srp-header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding: 12px 18px; margin-bottom: 10px;
  background: linear-gradient(90deg, {C.CARD_BG} 0%, {C.CARD_BG_2} 100%); border: 1px solid {C.BORDER}; border-radius: 16px; flex-wrap: wrap; }}
.srp-header .left {{ display:flex; align-items:center; gap:14px; }}
.srp-header .mark {{ width: 44px; height: 44px; border-radius: 12px; background: {C.PRIMARY_SOFT}; border: 1px solid {C.BORDER}; color: {C.PRIMARY_LIGHT}; display:flex; align-items:center; justify-content:center; font-size: 22px; font-weight: 700; }}
.srp-header .t {{ font-weight: 800; font-size: 18px; letter-spacing: .08em; color: {C.TEXT}; line-height: 1.1; }}
.srp-header .s {{ font-size: 12px; color: {C.MUTED}; margin-top: 3px; }}
.srp-header .right {{ display:flex; align-items:center; gap:10px; flex-wrap: wrap; }}
.srp-chip {{ display:inline-flex; align-items:center; gap:6px; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: .06em; border: 1px solid {C.BORDER}; background: {C.CARD_BG_2}; color: {C.MUTED}; }}
.srp-chip.dataset {{ background: {C.PRIMARY_SOFT}; color: {C.PRIMARY_LIGHT}; border-color: rgba(96,165,250,0.45); }}
.srp-chip.live {{ background: {C.RISK_SOFT['HIGH']}; color: #FCA5A5; border-color: rgba(239,68,68,0.45); }}
.srp-chip.online {{ color: #86EFAC; }}
.srp-chip .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {C.RISK_COLORS['LOW']}; box-shadow: 0 0 0 3px rgba(34,197,94,0.2); }}
.srp-chip .dot.red {{ background: {C.RISK_COLORS['HIGH']}; box-shadow: 0 0 0 3px rgba(239,68,68,0.25); }}
.srp-chip .dot.grey {{ background: {C.MUTED}; box-shadow: none; }}

/* ---------- navigation & data source (segmented controls) ---------- */
[data-testid="stButtonGroup"] [role="radiogroup"] {{ gap: 6px; flex-wrap: wrap; }}
button[data-variant="segmented_control"] {{ border-radius: 10px !important; padding: 7px 16px !important; border: 1px solid {C.BORDER} !important; background: {C.CARD_BG} !important; color: {C.MUTED} !important; }}
button[data-variant="segmented_control"] p {{ font-size: 13px !important; font-weight: 600; color: {C.MUTED}; }}
button[data-variant="segmented_control"]:hover {{ border-color: {C.PRIMARY} !important; }}
button[data-variant="segmented_control"]:hover p {{ color: {C.TEXT}; }}
button[data-variant="segmented_control"][aria-checked="true"] {{ background: {C.PRIMARY_SOFT} !important; border-color: {C.PRIMARY} !important; box-shadow: inset 0 -2px 0 {C.PRIMARY}; }}
button[data-variant="segmented_control"][aria-checked="true"] p {{ color: {C.TEXT}; }}
.st-key-datasource button[data-variant="segmented_control"][aria-checked="true"] {{ background: {C.PRIMARY} !important; box-shadow: none; }}
.st-key-datasource button[data-variant="segmented_control"][aria-checked="true"] p {{ color: #fff; }}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {C.CARD_BG}; border: 1px solid {C.BORDER} !important; border-radius: 14px; box-shadow: 0 4px 16px rgba(2,6,23,0.35); }}
.kpi-card {{ background: {C.CARD_BG}; border: 1px solid {C.BORDER}; border-radius: 14px; padding: 14px 16px 12px 16px; box-shadow: 0 4px 16px rgba(2,6,23,0.35); height: 100%; position: relative; overflow: hidden; }}
.kpi-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width: 4px; background: var(--accent, {C.PRIMARY}); }}
.kpi-label {{ font-size: 11px; color: {C.MUTED}; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {C.TEXT}; margin-top: 4px; line-height: 1.15; }}
.kpi-sub {{ font-size: 12px; color: {C.MUTED}; margin-top: 4px; }}
.srp-card {{ background: {C.CARD_BG}; border: 1px solid {C.BORDER}; border-radius: 14px; padding: 16px 18px; box-shadow: 0 4px 16px rgba(2,6,23,0.35); margin-bottom: 12px; }}
.srp-card h4 {{ margin: 0 0 6px 0; font-size: 14px; font-weight: 600; color: {C.TEXT}; }}
.srp-page-title {{ display:flex; align-items:baseline; gap: 12px; margin: 4px 0 2px 0; }}
.srp-page-title h1 {{ font-size: 22px; margin: 0; letter-spacing: .04em; }}
.srp-page-title .crumb {{ color: {C.MUTED}; font-size: 13px; }}
.srp-subtitle {{ color: {C.MUTED}; font-size: 13px; margin-bottom: 12px; }}
.srp-section {{ font-size: 12px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {C.TEXT}; margin: 18px 0 6px 0; display:flex; align-items:center; gap:8px; }}
.srp-section::before {{ content:""; width: 4px; height: 14px; border-radius: 2px; background: {C.PRIMARY}; display:inline-block; }}
.srp-section-sub {{ color: {C.MUTED}; font-size: 12px; margin: -2px 0 10px 12px; }}
.srp-badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: .05em; }}
.srp-badge.LOW {{ background: {C.RISK_SOFT['LOW']}; color: #86EFAC; }}
.srp-badge.MEDIUM {{ background: {C.RISK_SOFT['MEDIUM']}; color: #FCD34D; }}
.srp-badge.HIGH {{ background: {C.RISK_SOFT['HIGH']}; color: #FCA5A5; }}
.srp-badge.neutral {{ background: rgba(148,163,184,0.16); color: {C.MUTED}; }}
.srp-badge.blue {{ background: {C.PRIMARY_SOFT}; color: {C.PRIMARY_LIGHT}; }}
.srp-badge.red {{ background: {C.RISK_SOFT['HIGH']}; color: #FCA5A5; }}
.srp-badge.green {{ background: {C.RISK_SOFT['LOW']}; color: #86EFAC; }}
.srp-badge.orange {{ background: {C.RISK_SOFT['MEDIUM']}; color: #FCD34D; }}
.srp-note {{ border-radius: 10px; padding: 9px 14px; font-size: 12.5px; border: 1px solid; margin-bottom: 10px; }}
.srp-note.sim {{ background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.45); color: #FCD34D; }}
.srp-note.info {{ background: {C.PRIMARY_SOFT}; border-color: rgba(96,165,250,0.4); color: #BFDBFE; }}
.feature-card {{ background: {C.CARD_BG_2}; border: 1px solid {C.BORDER}; border-radius: 12px; padding: 10px 12px; text-align: center; }}
.feature-card .fl {{ font-size: 11px; color: {C.MUTED}; font-weight: 600; }}
.feature-card .fv {{ font-size: 18px; font-weight: 700; color: {C.TEXT}; margin-top: 2px; }}
.feature-card .fu {{ font-size: 11px; color: {C.MUTED}; }}
.alert-row {{ display:flex; gap: 12px; align-items:flex-start; padding: 9px 6px; border-bottom: 1px solid {C.BORDER}; font-size: 13px; }}
.alert-row:last-child {{ border-bottom: none; }}
.alert-row .ic {{ font-size: 15px; width: 20px; }}
.alert-row .who {{ font-weight: 700; min-width: 72px; color: {C.TEXT}; }}
.alert-row .what {{ font-weight: 600; min-width: 150px; color: {C.TEXT}; }}
.alert-row .why {{ color: {C.MUTED}; flex: 1; }}
.alert-row .when {{ color: {C.MUTED}; font-variant-numeric: tabular-nums; min-width: 64px; text-align:right; }}
.big-score {{ font-size: 46px; font-weight: 800; line-height: 1; }}
.big-score-label {{ font-size: 11px; color: {C.MUTED}; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
.stat-strip {{ display:flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 8px 0; }}
.stat-strip .st {{ background: {C.CARD_BG}; border: 1px solid {C.BORDER}; border-radius: 10px; padding: 6px 12px; font-size: 12px; color: {C.MUTED}; }}
.stat-strip .st b {{ color: {C.TEXT}; font-size: 13px; margin-left: 4px; }}
.health-bar {{ position: relative; height: 10px; border-radius: 6px; margin: 12px 0 6px 0;
  background: linear-gradient(90deg, {C.RISK_COLORS['LOW']} 0%, {C.RISK_COLORS['LOW']} 33%, {C.RISK_COLORS['MEDIUM']} 33%, {C.RISK_COLORS['MEDIUM']} 66%, {C.RISK_COLORS['HIGH']} 66%, {C.RISK_COLORS['HIGH']} 100%); opacity: .9; }}
.health-marker {{ position: absolute; top: -5px; width: 4px; height: 20px; background: {C.TEXT}; border-radius: 2px; box-shadow: 0 0 0 2px {C.CARD_BG}; }}
.warn-card {{ background: {C.CARD_BG_2}; border: 1px solid rgba(245,158,11,0.45); border-left: 4px solid {C.RISK_COLORS['MEDIUM']}; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; }}
.warn-card .row {{ display:flex; justify-content:space-between; align-items:center; gap: 10px; flex-wrap: wrap; }}
.warn-card .id {{ font-weight: 700; font-size: 14px; }}
.warn-card .nums {{ display:flex; gap: 18px; font-size: 12px; color: {C.MUTED}; }}
.warn-card .nums b {{ color: {C.TEXT}; font-size: 15px; display:block; }}
.warn-card .reasons {{ color: {C.MUTED}; font-size: 12px; margin-top: 6px; }}
.upload-card {{ text-align:center; padding: 22px 16px 6px 16px; }}
.upload-card .ic {{ font-size: 34px; color: {C.PRIMARY_LIGHT}; }}
.upload-card .h {{ font-size: 16px; font-weight: 700; margin-top: 6px; }}
.upload-card .p {{ font-size: 12.5px; color: {C.MUTED}; margin-top: 4px; }}

/* ---------- streamlit widgets ---------- */
div[data-testid="stMetric"] {{ background: {C.CARD_BG}; border: 1px solid {C.BORDER}; border-radius: 12px; padding: 10px 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {C.BORDER}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 8px 14px; font-weight: 600; color: {C.MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {C.TEXT} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {C.PRIMARY}; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {C.BORDER}; border-radius: 12px; overflow: hidden; }}
.stButton>button, .stDownloadButton>button {{ border-radius: 10px; font-weight: 600; border: 1px solid {C.BORDER}; background: {C.CARD_BG}; color: {C.TEXT}; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color: {C.PRIMARY}; color: {C.PRIMARY_LIGHT}; }}
.stButton>button[kind="primary"] {{ background: {C.PRIMARY}; border-color: {C.PRIMARY}; color: #fff; }}
.stButton>button[kind="primary"]:hover {{ background: {C.PRIMARY_DARK}; color: #fff; }}
.stButton>button:disabled {{ opacity: .45; }}
section[data-testid="stSidebar"] .stButton>button {{ padding: 0.3rem 0.45rem; font-size: 13px; }}
section[data-testid="stSidebar"] .stButton>button p {{ font-size: 13px; white-space: nowrap; }}
[data-testid="stFileUploaderDropzone"] {{ background: {C.CARD_BG_2}; border: 1px dashed {C.BORDER}; border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {C.PRIMARY}; }}
[data-testid="stExpander"] {{ border: 1px solid {C.BORDER}; border-radius: 12px; background: {C.CARD_BG}; }}
[data-testid="stExpander"] summary {{ font-weight: 600; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stTextInput input, .stNumberInput input {{ background: {C.CARD_BG_2} !important; border-color: {C.BORDER} !important; color: {C.TEXT}; border-radius: 10px; }}
div[data-testid="stAlert"] {{ border-radius: 12px; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {C.MUTED} !important; }}
[data-testid="stWidgetLabel"] p {{ color: {C.MUTED}; font-size: 12px; font-weight: 600; }}
@media (max-width: 900px) {{ .kpi-value {{ font-size: 22px; }} .srp-header .t {{ font-size: 15px; }} .block-container {{ padding: 0.6rem 0.8rem 2rem 0.8rem; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header / text blocks
# --------------------------------------------------------------------------- #
def app_header(mode: str, running: bool = False) -> None:
    live = mode == "Live Data"
    mode_chip = ('<span class="srp-chip live"><span class="dot red"></span>🔴 LIVE DATA</span>' if live
                 else '<span class="srp-chip dataset">📁 DATASET</span>')
    sim_chip = ""
    if live:
        sim_chip = (f'<span class="srp-chip"><span class="dot {"red" if running else "grey"}"></span>'
                    f'{"SIMULATION RUNNING" if running else "SIMULATION PAUSED"}</span>')
    st.markdown(
        f'<div class="srp-header"><div class="left"><div class="mark">◈</div><div>'
        f'<div class="t">SUPPLIER RISK PREDICTION</div><div class="s">Intelligent Supplier Risk Analytics</div></div></div>'
        f'<div class="right">{mode_chip}{sim_chip}<span class="srp-chip online"><span class="dot"></span>SYSTEM ONLINE</span></div></div>',
        unsafe_allow_html=True,
    )


def page_title(title: str, crumb: str = "", subtitle: str = "") -> None:
    st.markdown(
        f'<div class="srp-page-title"><h1>{title}</h1><span class="crumb">{crumb}</span></div>'
        + (f'<div class="srp-subtitle">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="srp-section">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="srp-section-sub">{sub}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="srp-badge {kind}">{text}</span>'


def note(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="srp-note {kind}">{text}</div>', unsafe_allow_html=True)


def simulated_notice() -> None:
    note("<b>SIMULATED / ARTIFICIAL DATA</b> · records in this mode are generated by the built-in supplier "
         "simulator for demonstration and are not real supplier records.", "sim")


def card():
    """Bordered dark container to group widgets/charts."""
    return st.container(border=True)


def fmt_num(v, digits: int = 1) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(v):
        return "-"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs(v) >= 10_000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.{max(digits - 1, 0)}f}"
    return f"{v:,.{digits}f}"


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def kpi_row(items: Sequence[Dict[str, object]]) -> None:
    """Render KPI cards. item keys: label, value, sub (optional), color (optional)."""
    items = [i for i in items if i]
    if not items:
        return
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        accent = item.get("color", C.PRIMARY)
        sub = item.get("sub", "")
        col.markdown(
            f'<div class="kpi-card" style="--accent:{accent}"><div class="kpi-label">{item["label"]}</div>'
            f'<div class="kpi-value">{item["value"]}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def stat_strip(items: Sequence[tuple]) -> None:
    html = "".join(f'<span class="st">{label}<b>{value}</b></span>' for label, value in items)
    st.markdown(f'<div class="stat-strip">{html}</div>', unsafe_allow_html=True)


def feature_cards(values: Dict[str, float], per_row: int = 7) -> None:
    items = list(values.items())
    if not items:
        st.caption("No feature values available.")
        return
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, (feat, val) in zip(cols, chunk):
            col.markdown(
                f'<div class="feature-card"><div class="fl">{feature_label(feat)}</div>'
                f'<div class="fv">{fmt_num(val, 2)}</div><div class="fu">{feature_unit(feat)}</div></div>',
                unsafe_allow_html=True,
            )


def risk_health_card(score: float, n_high: int, n_total: int) -> None:
    """Portfolio risk health: average supplier risk (0-100) with category."""
    if score < C.RISK_LOW_MAX * 100:
        cat, kind, word = "HEALTHY", "LOW", "low"
    elif score < C.RISK_MEDIUM_MAX * 100:
        cat, kind, word = "MODERATE", "MEDIUM", "moderate"
    else:
        cat, kind, word = "CRITICAL", "HIGH", "high"
    pct = min(max(score, 0), 100)
    st.markdown(
        f'<div class="srp-card" style="height:100%"><div class="kpi-label">Supplier Risk Health</div>'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-top:6px"><span class="big-score" style="color:{C.RISK_COLORS[kind]}">{score:.0f}</span>'
        f'<span style="color:{C.MUTED};font-size:16px">/ 100</span></div>'
        f'<div style="margin-top:8px">{badge(cat, kind)}</div>'
        f'<div class="health-bar"><div class="health-marker" style="left:calc({pct:.1f}% - 2px)"></div></div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:{C.MUTED}"><span>0</span><span>33</span><span>66</span><span>100</span></div>'
        f'<div style="font-size:12px;color:{C.MUTED};margin-top:10px">Overall risk is <b style="color:{C.TEXT}">{word}</b>: average supplier risk score across the portfolio. '
        f'<b style="color:{C.TEXT}">{n_high}</b> of {n_total} suppliers ({n_high / max(n_total, 1):.0%}) are in the HIGH band.</div></div>',
        unsafe_allow_html=True,
    )


def alert_feed(alerts: pd.DataFrame, limit: int = 12, empty_text: str = "No alerts yet.") -> None:
    if alerts is None or alerts.empty:
        st.markdown(f'<div class="srp-card" style="color:{C.MUTED}">{empty_text}</div>', unsafe_allow_html=True)
        return
    rows = []
    for _, a in alerts.head(limit).iterrows():
        lvl = str(a.get("Risk_Level", "") or "")
        rows.append(
            f'<div class="alert-row"><span class="ic">{a.get("Icon", "")}</span>'
            f'<span class="who">{a.get("Supplier_ID", "")}</span>'
            f'<span class="what">{a.get("Alert_Type", "")} {badge(lvl, lvl) if lvl in C.RISK_ORDER else ""}</span>'
            f'<span class="why">{a.get("Reason", "")}</span>'
            f'<span class="when">{a.get("Timestamp", "")}</span></div>'
        )
    st.markdown('<div class="srp-card">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def warning_card(row: pd.Series) -> None:
    st.markdown(
        f'<div class="warn-card"><div class="row"><div><span class="id">{row["Supplier_ID"]}</span> '
        f'<span style="color:{C.MUTED};font-size:12px">{row.get("Supplier_Name", "")}</span> {badge("⚠️ RISK INCREASING", "orange")}</div>'
        f'<div class="nums"><span>Previous<b>{row["Previous_Risk"]:.0f}%</b></span><span>Current<b>{row["Current_Risk"]:.0f}%</b></span>'
        f'<span>Change<b style="color:{C.RISK_COLORS["HIGH"]}">{row["Change"]:+.0f}%</b></span></div></div>'
        f'<div class="reasons">Reason: {row.get("Reasons", "") or "risk score rising"}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Plotly (dark)
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 360, legend_top: bool = True, margin_t: int = 48) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        font=dict(family=FONT, size=12, color=C.TEXT),
        title=dict(font=dict(size=14, color=C.TEXT), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=margin_t, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11, color=C.MUTED), bgcolor="rgba(0,0,0,0)") if legend_top else dict(font=dict(color=C.MUTED), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C.CARD_BG, bordercolor=C.BORDER, font_size=12, font_family=FONT, font_color=C.TEXT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C.BORDER, tickfont=dict(color=C.MUTED), title_font=dict(color=C.MUTED))
    fig.update_yaxes(gridcolor=C.GRID, zeroline=False, linecolor=C.BORDER, tickfont=dict(color=C.MUTED), title_font=dict(color=C.MUTED))
    fig.update_layout(legend_title_text="")
    return fig


def show(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def donut(counts: Dict[str, int], title: str = "Risk Distribution", height: int = 320) -> go.Figure:
    labels = [lvl for lvl in C.RISK_ORDER if counts.get(lvl, 0) > 0]
    values = [counts.get(lvl, 0) for lvl in labels]
    fig = go.Figure(go.Pie(labels=[f"{l.title()} Risk" for l in labels], values=values, hole=0.66,
                           marker=dict(colors=[C.RISK_COLORS[l] for l in labels], line=dict(color=C.CARD_BG, width=3)),
                           textinfo="percent", textfont=dict(color="#0B1220", size=11), hovertemplate="%{label}: %{value} suppliers (%{percent})<extra></extra>", sort=False))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px;color:{C.MUTED}'>suppliers</span>", showarrow=False, font=dict(size=22, color=C.TEXT))
    fig.update_layout(title=title)
    fig = style(fig, height)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    return fig


def histogram(scores: pd.Series, title: str = "Risk Score Distribution", height: int = 320) -> go.Figure:
    fig = go.Figure()
    for lo, hi, lvl in [(0, C.RISK_LOW_MAX * 100, "LOW"), (C.RISK_LOW_MAX * 100, C.RISK_MEDIUM_MAX * 100, "MEDIUM"), (C.RISK_MEDIUM_MAX * 100, 100.01, "HIGH")]:
        part = scores[(scores >= lo) & (scores < hi)]
        fig.add_trace(go.Histogram(x=part, xbins=dict(start=0, end=100, size=5), name=f"{lvl.title()} risk",
                                   marker_color=C.RISK_COLORS[lvl], opacity=0.9,
                                   hovertemplate="Score %{x}: %{y} records<extra></extra>"))
    fig.update_layout(barmode="stack", title=title, xaxis_title="Risk score (0-100)", yaxis_title="Records", bargap=0.08)
    fig.add_vline(x=C.RISK_LOW_MAX * 100, line_dash="dot", line_color=C.MUTED)
    fig.add_vline(x=C.RISK_MEDIUM_MAX * 100, line_dash="dot", line_color=C.MUTED)
    fig = style(fig, height)
    fig.update_layout(showlegend=False)
    return fig


def bar(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, orientation: str = "v",
        color_map: Optional[Dict[str, str]] = None, height: int = 360, text: bool = False, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> go.Figure:
    if color and color_map:
        fig = px.bar(df, x=x, y=y, color=color, color_discrete_map=color_map, orientation=orientation, text_auto=".1f" if text else False)
    elif color:
        fig = px.bar(df, x=x, y=y, color=color, color_continuous_scale=["#1E40AF", C.PRIMARY, "#93C5FD"], orientation=orientation, text_auto=".1f" if text else False)
    else:
        fig = px.bar(df, x=x, y=y, orientation=orientation, text_auto=".1f" if text else False)
        fig.update_traces(marker_color=C.PRIMARY)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title=title, xaxis_title=feature_label(x) if xlabel is None else xlabel,
                      yaxis_title=feature_label(y) if ylabel is None else ylabel, coloraxis_showscale=False)
    return style(fig, height)


def line(df: pd.DataFrame, x: str, y, title: str, colors: Optional[List[str]] = None, height: int = 340,
         ylabel: str = "", xlabel: str = "", markers: bool = True, ylim: Optional[tuple] = None, fill: bool = False) -> go.Figure:
    ys = y if isinstance(y, (list, tuple)) else [y]
    fig = go.Figure()
    palette = colors or C.CATEGORICAL
    for i, col in enumerate(ys):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers" if markers else "lines",
                                 name=feature_label(col) if col in C.FEATURE_LABELS else str(col).replace("_", " "),
                                 line=dict(color=color, width=2.4), marker=dict(size=5),
                                 fill="tozeroy" if fill and i == 0 else None, fillcolor="rgba(59,130,246,0.12)" if fill else None,
                                 hovertemplate="%{x}<br>%{y:.2f}<extra>" + str(col) + "</extra>"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel or (feature_label(ys[0]) if len(ys) == 1 else ""))
    if ylim:
        fig.update_yaxes(range=list(ylim))
    return style(fig, height)


def risk_bands(fig: go.Figure) -> go.Figure:
    fig.add_hrect(y0=0, y1=C.RISK_LOW_MAX * 100, fillcolor=C.RISK_COLORS["LOW"], opacity=0.07, line_width=0)
    fig.add_hrect(y0=C.RISK_LOW_MAX * 100, y1=C.RISK_MEDIUM_MAX * 100, fillcolor=C.RISK_COLORS["MEDIUM"], opacity=0.07, line_width=0)
    fig.add_hrect(y0=C.RISK_MEDIUM_MAX * 100, y1=100, fillcolor=C.RISK_COLORS["HIGH"], opacity=0.08, line_width=0)
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str, title: str, hover: Optional[List[str]] = None,
            color_map: Optional[Dict[str, str]] = None, size: Optional[str] = None, height: int = 400, legend_bottom: bool = False) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover or [], color_discrete_map=color_map,
                     color_discrete_sequence=C.CATEGORICAL, size=size, size_max=18, opacity=0.85)
    fig.update_traces(marker=dict(line=dict(width=0.5, color=C.PAGE_BG)))
    fig.update_layout(title=title, xaxis_title=feature_label(x), yaxis_title=feature_label(y))
    fig = style(fig, height)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
    return fig


def heatmap(corr: pd.DataFrame, title: str = "Feature Correlation") -> go.Figure:
    labels = [feature_label(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
                               colorscale=[[0, C.RISK_COLORS["HIGH"]], [0.5, C.CARD_BG], [1, C.PRIMARY_LIGHT]],
                               text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=10, color=C.TEXT),
                               hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(tickangle=-35)
    return style(fig, 460, legend_top=False)


def contribution_chart(contrib: pd.DataFrame, title: str, method: str) -> go.Figure:
    d = contrib.head(10).iloc[::-1]
    labels = [feature_label(f) for f in d["Feature"]]
    colors = [C.RISK_COLORS["HIGH"] if v > 0 else C.RISK_COLORS["LOW"] for v in d["Contribution"]]
    if method == "importance":
        colors = [C.PRIMARY] * len(d)
    fig = go.Figure(go.Bar(x=d["Contribution"], y=labels, orientation="h", marker_color=colors, marker_line_width=0,
                           customdata=d["Value"], hovertemplate="%{y}<br>contribution %{x:.3f}<br>value %{customdata:.3f}<extra></extra>",
                           text=[f"{v:+.2f}" for v in d["Contribution"]], textposition="outside", textfont=dict(color=C.TEXT)))
    xlabel = "SHAP value (impact on model log-odds; red raises risk, green lowers it)" if method == "shap" else "Model feature importance"
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="")
    fig.add_vline(x=0, line_color=C.MUTED, line_width=1)
    return style(fig, 380, legend_top=False)


def gauge(score: float, title: str = "Risk Score") -> go.Figure:
    color = C.RISK_COLORS["LOW" if score < C.RISK_LOW_MAX * 100 else "MEDIUM" if score < C.RISK_MEDIUM_MAX * 100 else "HIGH"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix="%", font=dict(size=34, color=C.TEXT)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 33, 66, 100], ticktext=["0", "33", "66", "100"], tickfont=dict(size=11, color=C.MUTED)),
                   bar=dict(color=color, thickness=0.3), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, C.RISK_LOW_MAX * 100], color="rgba(34,197,94,0.18)"),
                          dict(range=[C.RISK_LOW_MAX * 100, C.RISK_MEDIUM_MAX * 100], color="rgba(245,158,11,0.18)"),
                          dict(range=[C.RISK_MEDIUM_MAX * 100, 100], color="rgba(239,68,68,0.18)")]),
        title=dict(text=title, font=dict(size=13, color=C.MUTED)),
    ))
    fig = style(fig, 230, legend_top=False, margin_t=30)
    fig.update_layout(margin=dict(l=30, r=30, t=30, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def risk_column_config(df: pd.DataFrame) -> Dict[str, object]:
    cfg: Dict[str, object] = {}
    if "Risk_Score" in df.columns:
        cfg["Risk_Score"] = st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f")
    for c in ("Avg_Risk_Score", "Max_Risk_Score", "Previous_Risk", "Current_Risk", "Change"):
        if c in df.columns:
            cfg[c] = st.column_config.NumberColumn(c.replace("_", " "), format="%.1f")
    if "Last_Seen" in df.columns:
        cfg["Last_Seen"] = st.column_config.DatetimeColumn("Last Seen", format="YYYY-MM-DD HH:mm")
    if "Date" in df.columns:
        cfg["Date"] = st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm:ss")
    for c in df.columns:
        if c in C.FEATURE_LABELS and c not in cfg:
            lab, unit = C.FEATURE_LABELS[c]
            cfg[c] = st.column_config.NumberColumn(f"{lab}" + (f" ({unit})" if unit else ""), format="%.2f")
    for c in ("Supplier_ID", "Supplier_Name", "Risk_Level", "Risk_Status", "Cluster_Label", "Alert_Type", "Severity", "Normal_Range", "Detected", "Status", "Timestamp", "Reason", "Reasons"):
        if c in df.columns and c not in cfg:
            cfg[c] = st.column_config.TextColumn(c.replace("_", " "))
    return cfg


def dataframe(df: pd.DataFrame, height: Optional[int] = None, hide_index: bool = True, key: Optional[str] = None) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(df, width="stretch", hide_index=hide_index, column_config=risk_column_config(df), key=key, **kwargs)
