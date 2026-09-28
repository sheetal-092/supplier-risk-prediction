"""Presentation helpers: CSS, KPI cards, badges and Plotly chart factories."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.schema import feature_label, feature_unit

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; color: {config.TEXT}; }}
.stApp {{ background: {config.PAGE_BG}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {config.TEXT}; font-weight: 600; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {config.BORDER}; width: 330px !important; min-width: 330px !important; }}
section[data-testid="stSidebar"] .stButton>button {{ padding: 0.3rem 0.45rem; font-size: 13px; }}
section[data-testid="stSidebar"] .stButton>button p {{ font-size: 13px; white-space: nowrap; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
.srp-brand {{ display:flex; align-items:center; gap:10px; padding: 4px 0 10px 0; border-bottom: 1px solid {config.BORDER}; margin-bottom: 8px; }}
.srp-brand .logo {{ width: 36px; height: 36px; border-radius: 10px; background: {config.PRIMARY}; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; font-size: 15px; }}
.srp-brand .title {{ font-weight: 700; font-size: 14px; letter-spacing: .06em; color: {config.TEXT}; }}
.srp-brand .sub {{ font-size: 11px; color: {config.MUTED}; }}
.srp-side-label {{ font-size: 11px; font-weight: 600; letter-spacing: .08em; color: {config.MUTED}; text-transform: uppercase; margin: 10px 0 2px 0; }}
.srp-mode {{ border-radius: 10px; padding: 10px 12px; font-weight: 600; font-size: 13px; margin: 6px 0 4px 0; border: 1px solid {config.BORDER}; }}
.srp-mode.dataset {{ background: {config.PRIMARY_LIGHT}; color: {config.PRIMARY_DARK}; border-color: #BFDBFE; }}
.srp-mode.live {{ background: #FEE2E2; color: #991B1B; border-color: #FECACA; }}
.srp-mode small {{ display:block; font-weight:500; font-size: 11px; opacity: .85; margin-top: 2px; }}
.srp-page-title {{ display:flex; align-items:baseline; gap: 12px; margin-bottom: 2px; }}
.srp-page-title h1 {{ font-size: 26px; margin: 0; }}
.srp-page-title .crumb {{ color: {config.MUTED}; font-size: 13px; }}
.srp-subtitle {{ color: {config.MUTED}; font-size: 14px; margin-bottom: 14px; }}
.kpi-card {{ background: {config.CARD_BG}; border: 1px solid {config.BORDER}; border-radius: 14px; padding: 14px 16px 12px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 100%; position: relative; overflow: hidden; }}
.kpi-card::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width: 4px; background: var(--accent, {config.PRIMARY}); }}
.kpi-label {{ font-size: 12px; color: {config.MUTED}; font-weight: 500; letter-spacing: .02em; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {config.TEXT}; margin-top: 2px; line-height: 1.15; }}
.kpi-sub {{ font-size: 12px; color: {config.MUTED}; margin-top: 4px; }}
.srp-card {{ background: {config.CARD_BG}; border: 1px solid {config.BORDER}; border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04); margin-bottom: 12px; }}
.srp-card h4 {{ margin: 0 0 6px 0; font-size: 14px; font-weight: 600; color: {config.TEXT}; }}
.srp-section {{ font-size: 16px; font-weight: 600; color: {config.TEXT}; margin: 18px 0 6px 0; display:flex; align-items:center; gap:8px; }}
.srp-section::before {{ content:""; width: 4px; height: 16px; border-radius: 2px; background: {config.PRIMARY}; display:inline-block; }}
.srp-section-sub {{ color: {config.MUTED}; font-size: 13px; margin: -2px 0 10px 14px; }}
.srp-badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: .03em; }}
.srp-badge.LOW {{ background: {config.RISK_SOFT['LOW']}; color: #166534; }}
.srp-badge.MEDIUM {{ background: {config.RISK_SOFT['MEDIUM']}; color: #92400E; }}
.srp-badge.HIGH {{ background: {config.RISK_SOFT['HIGH']}; color: #991B1B; }}
.srp-badge.neutral {{ background: #E2E8F0; color: #334155; }}
.srp-badge.blue {{ background: {config.PRIMARY_LIGHT}; color: {config.PRIMARY_DARK}; }}
.srp-badge.red {{ background: #FEE2E2; color: #991B1B; }}
.srp-badge.green {{ background: #DCFCE7; color: #166534; }}
.srp-note {{ border-radius: 10px; padding: 10px 14px; font-size: 13px; border: 1px solid; margin-bottom: 10px; }}
.srp-note.sim {{ background: #FFF7ED; border-color: #FDBA74; color: #9A3412; }}
.srp-note.info {{ background: {config.PRIMARY_LIGHT}; border-color: #BFDBFE; color: {config.PRIMARY_DARK}; }}
.feature-card {{ background: #F8FAFC; border: 1px solid {config.BORDER}; border-radius: 12px; padding: 10px 12px; text-align: center; }}
.feature-card .fl {{ font-size: 11px; color: {config.MUTED}; font-weight: 500; }}
.feature-card .fv {{ font-size: 18px; font-weight: 700; color: {config.TEXT}; margin-top: 2px; }}
.feature-card .fu {{ font-size: 11px; color: {config.MUTED}; }}
.alert-row {{ display:flex; gap: 12px; align-items:flex-start; padding: 9px 6px; border-bottom: 1px solid {config.BORDER}; font-size: 13px; }}
.alert-row:last-child {{ border-bottom: none; }}
.alert-row .ic {{ font-size: 15px; width: 20px; }}
.alert-row .who {{ font-weight: 600; min-width: 70px; }}
.alert-row .what {{ font-weight: 600; min-width: 140px; }}
.alert-row .why {{ color: {config.MUTED}; flex: 1; }}
.alert-row .when {{ color: {config.MUTED}; font-variant-numeric: tabular-nums; min-width: 64px; text-align:right; }}
.big-score {{ font-size: 44px; font-weight: 700; line-height: 1; }}
.big-score-label {{ font-size: 12px; color: {config.MUTED}; font-weight: 500; }}
div[data-testid="stMetric"] {{ background: {config.CARD_BG}; border: 1px solid {config.BORDER}; border-radius: 12px; padding: 10px 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: 8px 14px; font-weight: 500; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {config.BORDER}; border-radius: 12px; overflow: hidden; }}
.stButton>button {{ border-radius: 9px; font-weight: 600; }}
footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Text blocks
# --------------------------------------------------------------------------- #
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
    note("<b>SIMULATED / ARTIFICIAL DATA</b> - records on this page are generated by the built-in supplier "
         "simulator for demonstration. They are not real supplier records.", "sim")


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
        accent = item.get("color", config.PRIMARY)
        sub = item.get("sub", "")
        col.markdown(
            f'<div class="kpi-card" style="--accent:{accent}"><div class="kpi-label">{item["label"]}</div>'
            f'<div class="kpi-value">{item["value"]}</div>'
            + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
            + "</div>",
            unsafe_allow_html=True,
        )


def feature_cards(values: Dict[str, float], per_row: int = 6) -> None:
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


def alert_feed(alerts: pd.DataFrame, limit: int = 12) -> None:
    if alerts is None or alerts.empty:
        st.markdown('<div class="srp-card" style="color:#64748B">No alerts yet.</div>', unsafe_allow_html=True)
        return
    rows = []
    for _, a in alerts.head(limit).iterrows():
        rows.append(
            f'<div class="alert-row"><span class="ic">{a.get("Icon", "")}</span>'
            f'<span class="who">{a.get("Supplier_ID", "")}</span>'
            f'<span class="what">{a.get("Alert_Type", "")} {badge(str(a.get("Risk_Level", "")), str(a.get("Risk_Level", "neutral")))}</span>'
            f'<span class="why">{a.get("Reason", "")}</span>'
            f'<span class="when">{a.get("Timestamp", "")}</span></div>'
        )
    st.markdown('<div class="srp-card">' + "".join(rows) + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Plotly
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 360, legend_top: bool = True, margin_t: int = 48) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family=FONT, size=12, color=config.TEXT),
        title=dict(font=dict(size=15, color=config.TEXT), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=margin_t, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)) if legend_top else {},
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=config.BORDER)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


def show(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def donut(counts: Dict[str, int], title: str = "Risk Distribution") -> go.Figure:
    labels = [lvl for lvl in config.RISK_ORDER if counts.get(lvl, 0) > 0]
    values = [counts.get(lvl, 0) for lvl in labels]
    fig = go.Figure(go.Pie(labels=[f"{l.title()} Risk" for l in labels], values=values, hole=0.62,
                           marker=dict(colors=[config.RISK_COLORS[l] for l in labels], line=dict(color="white", width=2)),
                           textinfo="percent", hovertemplate="%{label}: %{value} suppliers (%{percent})<extra></extra>", sort=False))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total}</b><br><span style='font-size:11px;color:{config.MUTED}'>suppliers</span>", showarrow=False, font=dict(size=20))
    fig.update_layout(title=title)
    fig = style(fig, 340)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    return fig


def histogram(scores: pd.Series, title: str = "Risk Score Distribution") -> go.Figure:
    fig = go.Figure()
    bins = list(range(0, 101, 5))
    for lo, hi, lvl in [(0, config.RISK_LOW_MAX * 100, "LOW"), (config.RISK_LOW_MAX * 100, config.RISK_MEDIUM_MAX * 100, "MEDIUM"), (config.RISK_MEDIUM_MAX * 100, 100.01, "HIGH")]:
        part = scores[(scores >= lo) & (scores < hi)]
        fig.add_trace(go.Histogram(x=part, xbins=dict(start=0, end=100, size=5), name=f"{lvl.title()} risk",
                                   marker_color=config.RISK_COLORS[lvl], opacity=0.9,
                                   hovertemplate="Score %{x}: %{y} records<extra></extra>"))
    fig.update_layout(barmode="stack", title=title, xaxis_title="Risk score (0-100)", yaxis_title="Records", bargap=0.06)
    fig.add_vline(x=config.RISK_LOW_MAX * 100, line_dash="dot", line_color=config.MUTED)
    fig.add_vline(x=config.RISK_MEDIUM_MAX * 100, line_dash="dot", line_color=config.MUTED)
    fig = style(fig, 340)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5))
    return fig


def bar(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None, orientation: str = "v",
        color_map: Optional[Dict[str, str]] = None, height: int = 360, text: bool = False, xlabel: str = "", ylabel: str = "") -> go.Figure:
    if color and color_map:
        fig = px.bar(df, x=x, y=y, color=color, color_discrete_map=color_map, orientation=orientation, text_auto=".1f" if text else False)
    elif color:
        fig = px.bar(df, x=x, y=y, color=color, color_continuous_scale=["#93C5FD", config.PRIMARY, "#1E3A8A"], orientation=orientation, text_auto=".1f" if text else False)
    else:
        fig = px.bar(df, x=x, y=y, orientation=orientation, text_auto=".1f" if text else False)
        fig.update_traces(marker_color=config.PRIMARY)
    fig.update_layout(title=title, xaxis_title=xlabel or feature_label(x), yaxis_title=ylabel or feature_label(y), coloraxis_showscale=False)
    return style(fig, height)


def line(df: pd.DataFrame, x: str, y, title: str, colors: Optional[List[str]] = None, height: int = 340,
         ylabel: str = "", xlabel: str = "", markers: bool = True, ylim: Optional[tuple] = None) -> go.Figure:
    ys = y if isinstance(y, (list, tuple)) else [y]
    fig = go.Figure()
    palette = colors or config.CATEGORICAL
    for i, col in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers" if markers else "lines", name=feature_label(col) if col in config.FEATURE_LABELS else col.replace("_", " "),
                                 line=dict(color=palette[i % len(palette)], width=2.2), marker=dict(size=5),
                                 hovertemplate="%{x}<br>%{y:.2f}<extra>" + str(col) + "</extra>"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel or (feature_label(ys[0]) if len(ys) == 1 else ""))
    if ylim:
        fig.update_yaxes(range=list(ylim))
    return style(fig, height)


def risk_bands(fig: go.Figure) -> go.Figure:
    fig.add_hrect(y0=0, y1=config.RISK_LOW_MAX * 100, fillcolor=config.RISK_COLORS["LOW"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=config.RISK_LOW_MAX * 100, y1=config.RISK_MEDIUM_MAX * 100, fillcolor=config.RISK_COLORS["MEDIUM"], opacity=0.06, line_width=0)
    fig.add_hrect(y0=config.RISK_MEDIUM_MAX * 100, y1=100, fillcolor=config.RISK_COLORS["HIGH"], opacity=0.06, line_width=0)
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str, title: str, hover: Optional[List[str]] = None,
            color_map: Optional[Dict[str, str]] = None, size: Optional[str] = None, height: int = 400) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover or [], color_discrete_map=color_map,
                     color_discrete_sequence=config.CATEGORICAL, size=size, size_max=18, opacity=0.85)
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig.update_layout(title=title, xaxis_title=feature_label(x), yaxis_title=feature_label(y))
    return style(fig, height)


def heatmap(corr: pd.DataFrame, title: str = "Feature Correlation") -> go.Figure:
    labels = [feature_label(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
                               colorscale=[[0, "#DC2626"], [0.5, "#FFFFFF"], [1, config.PRIMARY]],
                               text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=10),
                               hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title=title)
    fig.update_xaxes(tickangle=-35)
    return style(fig, 460, legend_top=False)


def contribution_chart(contrib: pd.DataFrame, title: str, method: str) -> go.Figure:
    d = contrib.head(10).iloc[::-1]
    labels = [feature_label(f) for f in d["Feature"]]
    colors = [config.RISK_COLORS["HIGH"] if v > 0 else config.RISK_COLORS["LOW"] for v in d["Contribution"]]
    if method == "importance":
        colors = [config.PRIMARY] * len(d)
    fig = go.Figure(go.Bar(x=d["Contribution"], y=labels, orientation="h", marker_color=colors,
                           customdata=d["Value"], hovertemplate="%{y}<br>contribution %{x:.3f}<br>value %{customdata:.3f}<extra></extra>",
                           text=[f"{v:+.2f}" for v in d["Contribution"]], textposition="outside"))
    xlabel = "SHAP value (impact on model log-odds; red raises risk, green lowers it)" if method == "shap" else "Model feature importance"
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="")
    fig.add_vline(x=0, line_color=config.MUTED, line_width=1)
    return style(fig, 380, legend_top=False)


def gauge(score: float, title: str = "Risk Score") -> go.Figure:
    color = config.RISK_COLORS["LOW" if score < config.RISK_LOW_MAX * 100 else "MEDIUM" if score < config.RISK_MEDIUM_MAX * 100 else "HIGH"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix="%", font=dict(size=34)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 33, 66, 100], ticktext=["0", "33", "66", "100"], tickfont=dict(size=11, color=config.MUTED)), bar=dict(color=color, thickness=0.3),
                   bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, config.RISK_LOW_MAX * 100], color=config.RISK_SOFT["LOW"]),
                          dict(range=[config.RISK_LOW_MAX * 100, config.RISK_MEDIUM_MAX * 100], color=config.RISK_SOFT["MEDIUM"]),
                          dict(range=[config.RISK_MEDIUM_MAX * 100, 100], color=config.RISK_SOFT["HIGH"])]),
        title=dict(text=title, font=dict(size=13, color=config.MUTED)),
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
    for c in ("Avg_Risk_Score", "Max_Risk_Score"):
        if c in df.columns:
            cfg[c] = st.column_config.NumberColumn(c.replace("_", " "), format="%.1f")
    if "Last_Seen" in df.columns:
        cfg["Last_Seen"] = st.column_config.DatetimeColumn("Last Seen", format="YYYY-MM-DD HH:mm")
    if "Date" in df.columns:
        cfg["Date"] = st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm:ss")
    for c in df.columns:
        if c in config.FEATURE_LABELS and c not in cfg:
            lab, unit = config.FEATURE_LABELS[c]
            cfg[c] = st.column_config.NumberColumn(f"{lab}" + (f" ({unit})" if unit else ""), format="%.2f")
    for c in ("Supplier_ID", "Supplier_Name", "Risk_Level", "Risk_Status", "Cluster_Label", "Alert_Type", "Severity", "Normal_Range", "Detected", "Status"):
        if c in df.columns and c not in cfg:
            cfg[c] = st.column_config.TextColumn(c.replace("_", " "))
    return cfg


def dataframe(df: pd.DataFrame, height: Optional[int] = None, hide_index: bool = True, key: Optional[str] = None) -> None:
    kwargs = {"height": height} if height else {}
    st.dataframe(df, width="stretch", hide_index=hide_index, column_config=risk_column_config(df), key=key, **kwargs)
