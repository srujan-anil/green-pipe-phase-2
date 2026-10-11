import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from dashboard.style import apply_style, badge, gauge_svg, STATUS_COLOR, DECISION_COLOR, CLEAN, MODERATE, DIRTY, ACCENT, DIM
from dashboard.data_loader import (
    get_all_regions_status, get_live_green_pipe_decision, get_carbon_history,
    get_thresholds, get_forecasting_comparison, get_strategy_summary,
    list_experiment_runs, load_run, get_checkpoint_sweep_results,
    get_dirty_event_results, get_dirty_event_timeline, get_hardware_usage,
)
from scheduler import config

apply_style(st, "Dashboard")

# ══════════════════════════ SIDEBAR ══════════════════════════
with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    lcol1, lcol2, lcol3 = st.columns([1, 2, 1])
    with lcol2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=80)
        else:
            st.markdown("<div class='brand-icon'>GP</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><div class='brand-name'>Green-Pipe</div><div class='brand-sub'>Carbon-Aware ML Orchestrator</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    nav = st.radio("Navigation", ["Dashboard", "Dirty-Event Evidence", "Full Results"], label_visibility="collapsed")
    st.markdown("---")
    base_zone = st.selectbox("Base region", config.ZONES, index=0)

    selected_run = None
    runs = list_experiment_runs()
    if nav == "Dashboard" and runs:
        labels = [f"{r['strategy']} · seed {r['seed']}" for r in runs]
        run_idx = st.selectbox("Inspect run", range(len(runs)), format_func=lambda i: labels[i])
        selected_run = load_run(runs[run_idx]["path"])

    st.markdown("---")
    st.caption("Model: SimpleCNN · Dataset: MNIST")
    st.caption("Hardware: CPU-only, shared machine")

gp = get_live_green_pipe_decision(base_zone)
regions = get_all_regions_status()
hw = get_hardware_usage()

# ══════════════════════════ VIEW: DASHBOARD ══════════════════════════
if nav == "Dashboard":

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='card'><div class='card-title'><span class='dot' style='background:{ACCENT};'></span>Carbon — decision basis</div>", unsafe_allow_html=True)
        eff_val = f"{gp['effective_carbon']:.1f}" if gp["effective_carbon"] is not None else "—"
        st.markdown(f"<div class='big-metric'>{eff_val}</div><div class='big-metric-label'>gCO2/kWh, {gp['effective_zone']} — this is what the decision below is based on</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><div class='card-title'><span class='dot' style='background:{MODERATE};'></span>Forecast (+1h)</div>", unsafe_allow_html=True)
        fc_val = f"{gp['forecast'][0]:.1f}" if gp["forecast"] else "unavailable"
        st.markdown(f"<div class='big-metric'>{fc_val}</div><div class='big-metric-label'>{(gp['forecaster_model'] or 'n/a').upper()}, for {gp['effective_zone']}</div></div>", unsafe_allow_html=True)
    with c3:
        dcolor = DECISION_COLOR.get(gp["decision"], DIM)
        st.markdown(f"<div class='card'><div class='card-title'><span class='dot' style='background:{dcolor};'></span>Decision — real decide() call</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric' style='color:{dcolor};'>{gp['decision']}</div><div class='big-metric-label'>batch size {gp['batch_size']}</div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='reason-strip'>{gp['reason']}</div>", unsafe_allow_html=True)
    base_val = f"{gp['base_carbon']:.1f}" if gp["base_carbon"] is not None else "—"
    if gp["migrated"]:
        st.markdown(f"<div class='rule-note' style='margin-top:6px;'>Your selected base region ({gp['base_zone']}) reads {base_val} gCO2/kWh. Green-Pipe found {gp['effective_zone']} cheaper and evaluated the decision there instead — this is the region-comparison logic</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='rule-note' style='margin-top:6px;'>Your selected base region ({gp['base_zone']}) is already the cheapest available — no migration needed.</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='card'><div class='card-title'><span class='dot' style='background:{CLEAN};'></span>All three regions right now</div>", unsafe_allow_html=True)
    for r in regions:
        dot_color = CLEAN if r["is_best"] else "#cbd5e1"
        cls = "region-mini best" if r["is_best"] else "region-mini"
        val = f"{r['carbon']:.0f} gCO2/kWh" if r["carbon"] is not None else "—"
        st.markdown(f"<div class='{cls}'><span><span class='region-dot' style='background:{dot_color};'></span>{r['zone']}</span><span class='stat-value'>{val}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='card-source'></div></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='card'><div class='card-title'><span class='dot' style='background:{DIM};'></span>Historical carbon intensity (last 24h of replay dataset)</div>", unsafe_allow_html=True)
    hist_frames = []
    for z in config.ZONES:
        h = get_carbon_history(z, hours=24)
        if h is not None and not h.empty:
            hist_frames.append(h.set_index("timestamp")["carbon_intensity"].rename(z))
    if hist_frames:
        st.line_chart(pd.concat(hist_frames, axis=1), height=220)
    else:
        st.markdown("<div class='warn-box'>No historical data found.</div>", unsafe_allow_html=True)
    st.markdown("<div class='rule-note'>This is the fixed historical dataset every experiment on this project replays — not live, and not today's date.</div></div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"<div class='card'><div class='card-title'>Training information</div>", unsafe_allow_html=True)
        if selected_run:
            migrated = "Yes" if selected_run.get("migrated_during_run") else "No"
            regions_seen = selected_run.get("regions_seen", [selected_run.get("region")])
            migration_detail = f"{selected_run.get('region')} → {regions_seen[-1]}" if selected_run.get("migrated_during_run") else "none"
            epoch_str = f"{len(selected_run.get('per_epoch', []))} / {selected_run['epochs']}"
            st.markdown(f"""
            <div class='stat-row'><span class='stat-label'>Training state</span><span class='stat-value'>{selected_run['completion_status']} (final)</span></div>
            <div class='stat-row'><span class='stat-label'>Strategy</span><span class='stat-value'>{selected_run['strategy']}</span></div>
            <div class='stat-row'><span class='stat-label'>Effective zone</span><span class='stat-value'>{regions_seen[-1]}</span></div>
            <div class='stat-row'><span class='stat-label'>Epoch progress</span><span class='stat-value'>{epoch_str}</span></div>
            <div class='stat-row'><span class='stat-label'>Elapsed time</span><span class='stat-value'>{selected_run['runtime_seconds']:.1f}s</span></div>
            <div class='stat-row'><span class='stat-label'>Migration decided</span><span class='stat-value'>{migrated}</span></div>
            <div class='stat-row'><span class='stat-label'>Migration path</span><span class='stat-value'>{migration_detail}</span></div>
            """, unsafe_allow_html=True)
            if selected_run["strategy"] not in ("multiregion", "green_pipe"):
                st.markdown("<div class='rule-note'>Only multiregion and green_pipe compare/select regions — this strategy never does, by design.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>No run selected in sidebar.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-source'></div></div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div class='card'><div class='card-title'>Carbon and power impact</div>", unsafe_allow_html=True)
        if selected_run:
            m1, m2 = st.columns(2)
            m1.markdown(f"<div class='big-metric'>{selected_run['co2_kg']*1000:.3f}</div><div class='big-metric-label'>g CO2 emitted — this run</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='big-metric'>{selected_run['energy_kwh']*1000:.4f}</div><div class='big-metric-label'>Wh consumed — this run</div>", unsafe_allow_html=True)
            st.markdown("<div class='rule-note'>CO2 = real measured energy × the carbon intensity recorded at decision time / 1000.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>No run selected in sidebar.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-source'>CodeCarbon</div></div>", unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"<div class='card'><div class='card-title'>Hardware usage</div>", unsafe_allow_html=True)
        if hw["available"]:
            st.markdown(f"""
            <div class='stat-row'><span class='stat-label'>CPU used</span><span class='stat-value'>{hw['cpu_percent']:.1f}%</span></div>
            <div class='stat-row'><span class='stat-label'>Memory used</span><span class='stat-value'>{hw['memory_percent']:.1f}%</span></div>
            <div class='stat-row'><span class='stat-label'>GPU</span><span class='stat-value'>not applicable — CPU-only</span></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>psutil not installed.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-source'>psutil</div></div>", unsafe_allow_html=True)

    with col6:
        st.markdown(f"<div class='card'><div class='card-title'>Resource gauges</div>", unsafe_allow_html=True)
        if hw["available"]:
            g1, g2 = st.columns(2)
            g1.markdown(gauge_svg(hw["cpu_percent"], ACCENT, "CPU"), unsafe_allow_html=True)
            g2.markdown(gauge_svg(hw["memory_percent"], CLEAN, "Memory"), unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>psutil not installed.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col7, col8 = st.columns([1.4, 1])
    with col7:
        st.markdown(f"<div class='card'><div class='card-title'>Carbon emissions over time — this run</div>", unsafe_allow_html=True)
        if selected_run:
            epoch_data = [{"epoch": e["epoch"], "co2_g": e.get("co2_kg", 0) * 1000}
                          for e in selected_run.get("per_epoch", []) if e.get("energy_kwh", 0) > 0]
            if epoch_data:
                df = pd.DataFrame(epoch_data).groupby("epoch").sum().cumsum()
                st.area_chart(df.rename(columns={"co2_g": "cumulative CO2 (g)"}), height=190)
                st.markdown("<div class='rule-note'>Cumulative real CO2 measured per trained epoch of the selected run.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warn-box'>No trained epochs in this run.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='warn-box'>No run selected in sidebar.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col8:
        st.markdown(f"<div class='card'><div class='card-title'>Decision log</div>", unsafe_allow_html=True)
        show_log = st.checkbox("Show reasoning log for selected run", value=False)
        if show_log and selected_run:
            lines = []
            for e in selected_run.get("per_epoch", []):
                ts = e.get("real_historical_timestamp") or e.get("timestamp", "")
                lines.append(f"<div>[{ts}] {e.get('reason', '')}</div>")
            st.markdown(f"<div class='terminal'>{''.join(lines) if lines else 'no entries'}</div>", unsafe_allow_html=True)
        else:
            st.caption("Every real reasoning string decide() produced during the selected run, epoch by epoch.")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════ VIEW: DIRTY-EVENT EVIDENCE ══════════════════════════
elif nav == "Dirty-Event Evidence":
    st.markdown("### Dirty-Event Evidence")
    st.caption("Real historical high-carbon event, DE, Dec 2 2025.")

    gp_df, base_trained, gp_trained, gp_paused = get_dirty_event_timeline(42)
    st.markdown("<div class='rule-note'>Shown once — the carbon sequence and TRAIN/WAIT/PAUSE decisions depend only on replayed carbon and forecast values, not on the random seed, so this timeline is identical across all 9 seeds. Only real energy/CO2 measurements (table below) vary by seed.</div>", unsafe_allow_html=True)
    if gp_df is not None and not gp_df.empty:
        st.line_chart(gp_df.set_index("sim_hour")[["carbon"]].rename(columns={"carbon": "gCO2/kWh"}), height=220)
        c1, c2 = st.columns(2)
        c1.markdown(f"<div class='card'><div class='card-title'>Baseline trained at</div><div class='mono' style='color:{DIRTY};'>{base_trained}</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'><div class='card-title'>Green-Pipe trained at</div><div class='mono' style='color:{CLEAN};'>{gp_trained}</div></div>", unsafe_allow_html=True)

    dirty_rows, dirty_stats = get_dirty_event_results()
    rows_html = ""
    for r in dirty_rows:
        cls = "excluded" if r["anomaly"] else ""
        pct = f"{r['pct_change']:+.1f}%" if r["pct_change"] is not None else "—"
        color = DIM if r["anomaly"] else (DIRTY if (r["pct_change"] or 0) > 0 else CLEAN)
        tag = "  [EXCLUDED]" if r["anomaly"] else ""
        rows_html += f"<tr class='{cls}'><td>{r['seed']}</td><td>{r['baseline_co2_kg']:.6f}</td><td>{r['green_pipe_co2_kg']:.6f}</td><td style='color:{color};font-weight:700;'>{pct}{tag}</td></tr>"
    st.markdown(f"<table class='gp-table'><tr><th>Seed</th><th>Baseline CO2</th><th>Green-Pipe CO2</th><th>Change</th></tr>{rows_html}</table>", unsafe_allow_html=True)

    st.markdown(f"""<div class='card' style='margin-top:14px;'>
    <b>Mechanism verified correct in {dirty_stats['n_total']}/{dirty_stats['n_total']} trials.</b><br>
    Aggregate CO2 change: <b>{dirty_stats['mean_pct_change']:+.1f}%</b>, std {dirty_stats['std_pct_change']:.1f}%, n={dirty_stats['n_used']}/{dirty_stats['n_total']} — statistically inconclusive, 95% CI crosses zero. {dirty_stats['n_excluded']} trials excluded via the ≥3× run-median timing-anomaly rule.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════ VIEW: FULL RESULTS ══════════════════════════
elif nav == "Full Results":
    st.markdown("### Full Results")
    summary = get_strategy_summary()
    if summary:
        order = ["baseline", "reactive", "predictive", "adaptive", "multiregion", "green_pipe"]
        strategies = sorted(summary.keys(), key=lambda s: order.index(s) if s in order else 99)
        rows_html = ""
        chart_rows = []
        for s in strategies:
            m = summary[s]["metrics"]
            co2 = f"{m['co2_kg']['mean']:.6f} ± {m['co2_kg']['std']:.6f}" if m["co2_kg"]["mean"] is not None else "—"
            pct = m["co2_kg"]["pct_change_vs_baseline"]
            pct_str = f"{pct:+.1f}%" if pct is not None else "—"
            color = DIRTY if (pct or 0) > 0 else (CLEAN if (pct or 0) < 0 else DIM)
            acc = f"{m['accuracy']['mean']*100:.2f}%" if m["accuracy"]["mean"] is not None else "—"
            rows_html += f"<tr><td style='font-weight:700;'>{s}</td><td>{co2}</td><td style='color:{color};font-weight:700;'>{pct_str}</td><td>{acc}</td></tr>"
            if pct is not None:
                chart_rows.append({"Strategy": s, "CO2 change (%)": pct})
        st.markdown(f"<table class='gp-table'><tr><th>Strategy</th><th>CO2 (kg)</th><th>Change vs Baseline</th><th>Accuracy</th></tr>{rows_html}</table>", unsafe_allow_html=True)
        if chart_rows:
            st.bar_chart(pd.DataFrame(chart_rows).set_index("Strategy"), height=220)
    else:
        st.markdown("<div class='warn-box'>No strategy summary found.</div>", unsafe_allow_html=True)

    ckpt = get_checkpoint_sweep_results()
    if ckpt:
        st.markdown("#### Checkpoint overhead by model size")
        df = pd.DataFrame(ckpt).set_index("tier")[["mean_overhead_pct"]].rename(columns={"mean_overhead_pct": "overhead % of epoch"})
        st.bar_chart(df, height=200)

    st.markdown("#### Forecasting — ARIMA vs LSTM")
    fc = get_forecasting_comparison()
    fc_rows = "".join(
        f"<tr><td style='font-weight:700;'>{r['zone']}</td><td>{r['arima_mape'] or '—'}</td><td>{r['lstm_mape'] or '—'}</td><td style='color:{ACCENT};font-weight:700;'>{(r['selected_model'] or '—').upper()}</td></tr>"
        for r in fc
    )
    st.markdown(f"<table class='gp-table'><tr><th>Region</th><th>ARIMA MAPE</th><th>LSTM MAPE</th><th>Auto-Selected</th></tr>{fc_rows}</table>", unsafe_allow_html=True)