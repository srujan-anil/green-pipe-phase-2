CLEAN, MODERATE, DIRTY, ACCENT = "#22c55e", "#f59e0b", "#ef4444", "#3b82f6"
BG, CARD, BORDER, TEXT, DIM = "#f5f6f8", "#ffffff", "#e5e9f0", "#1e293b", "#64748b"

STATUS_COLOR = {"CLEAN": CLEAN, "MODERATE": MODERATE, "DIRTY": DIRTY, "UNKNOWN": DIM}
DECISION_COLOR = {"TRAIN": CLEAN, "WAIT": MODERATE, "PAUSE": DIRTY}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }}
.stApp {{ background: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; }}
h1,h2,h3,h4 {{ font-family:'Inter',sans-serif; font-weight:700; color:{TEXT}; }}
.mono {{ font-family:'JetBrains Mono', monospace; }}

[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] div[role="radiogroup"] label p {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] div[role="radiogroup"] label {{ padding: 4px 0; }}
[data-testid="stSidebar"] .stCaption p {{ color: {DIM} !important; }}
[data-testid="stSidebar"] [data-testid="stImage"] {{ display: flex; justify-content: center; width: 100%; }}
[data-testid="stSidebar"] [data-testid="stImage"] img {{ margin: 0 auto; display: block; }}
[data-testid="stSidebar"] [data-testid="stImage"] {{ display: flex; justify-content: center; width: 100%; }}
[data-testid="stSidebar"] [data-testid="stImage"] img {{ margin: 0 auto; display: block; }}
[data-testid="stSidebar"] .brand-name, [data-testid="stSidebar"] .brand-sub {{ text-align: center; }}

.big-metric, .big-metric-label {{ text-align: center; }}
.region-mini {{ justify-content: center; gap: 24px; }}
.stat-row {{ justify-content: center; gap: 20px; }}
.brand {{ text-align:center; padding: 20px 0 14px 0; }}
.brand-icon {{ width:36px; height:36px; border-radius:9px; background:{ACCENT}; display:inline-flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1.1em; }}
.brand-name {{ font-size:1.1em; font-weight:800; color:{TEXT}; margin-top:8px; }}
.brand-sub {{ font-size:0.7em; color:{DIM}; }}

.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:16px 18px; height:100%; }}
.card-title {{ font-size:0.85em; font-weight:700; color:{TEXT}; margin-bottom:12px; display:flex; align-items:center; gap:7px; }}
.card-title .dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.card-source {{ font-size:0.62em; color:{DIM}; font-family:'JetBrains Mono',monospace; margin-top:8px; }}
.card {{ text-align: center; }}
.card-title {{ justify-content: center; }}
.region-mini {{ justify-content: center; gap: 24px; }}
.stat-row {{ justify-content: center; gap: 20px; }}

.stat-row {{ display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid {BORDER}; font-size:0.82em; }}
.stat-row:last-child {{ border-bottom:none; }}
.stat-label {{ color:{DIM}; }}
.stat-value {{ font-weight:700; color:{TEXT}; font-family:'JetBrains Mono',monospace; }}

.big-metric {{ font-size:1.9em; font-weight:800; font-family:'JetBrains Mono',monospace; color:{TEXT}; line-height:1.1; }}
.big-metric-label {{ font-size:0.72em; color:{DIM}; margin-top:4px; }}

.region-mini {{ display:flex; justify-content:space-between; align-items:center; padding:9px 12px; background:{BG}; border-radius:8px; margin-bottom:7px; font-size:0.83em; }}
.region-mini.best {{ border:1px solid {CLEAN}; }}
.region-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:8px; }}

.badge {{ display:inline-block; font-size:0.68em; font-weight:700; padding:3px 10px; border-radius:20px; }}

.terminal {{ background:#0f172a; color:#94e2a4; border-radius:10px; padding:14px 16px; font-family:'JetBrains Mono',monospace; font-size:0.75em; line-height:1.7; max-height:280px; overflow-y:auto; }}
.terminal .line-dim {{ color:#64748b; }}

.rule-note {{ font-size:0.72em; color:{DIM}; margin-top:8px; font-style:italic; }}
.warn-box {{ background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 16px; font-size:0.82em; color:#991b1b; }}

table.gp-table {{ width:100%; border-collapse:collapse; font-size:0.8em; }}
table.gp-table th {{ text-align:left; font-family:'JetBrains Mono',monospace; font-size:0.65em; color:{DIM}; text-transform:uppercase; border-bottom:1px solid {BORDER}; padding:8px 6px; }}
table.gp-table td {{ border-bottom:1px solid {BORDER}; padding:8px 6px; font-family:'JetBrains Mono',monospace; }}
table.gp-table tr.excluded td {{ color:{DIM}; opacity:0.7; }}

::-webkit-scrollbar {{ width:8px; height:8px; }} ::-webkit-scrollbar-thumb {{ background:#cbd5e1; border-radius:4px; }}
</style>
"""

def badge(text, color):
    return f"<span class='badge' style='color:{color};background:{color}18;'>{text}</span>"

def gauge_svg(pct, color, label):
    pct = max(0, min(100, pct))
    r = 34
    c = 2 * 3.14159 * r
    offset = c * (1 - pct / 100)
    return f"""<svg width="88" height="88" viewBox="0 0 90 90">
        <circle cx="45" cy="45" r="{r}" fill="none" stroke="#e5e9f0" stroke-width="8"/>
        <circle cx="45" cy="45" r="{r}" fill="none" stroke="{color}" stroke-width="8"
            stroke-dasharray="{c:.1f}" stroke-dashoffset="{offset:.1f}"
            stroke-linecap="round" transform="rotate(-90 45 45)"/>
        <text x="45" y="42" text-anchor="middle" font-size="15" font-weight="800" font-family="JetBrains Mono" fill="#1e293b">{pct:.0f}%</text>
        <text x="45" y="58" text-anchor="middle" font-size="9" font-family="Inter" fill="#64748b">{label}</text>
    </svg>"""

def apply_style(st, title):
    st.set_page_config(page_title=f"Green-Pipe — {title}", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)