"""Dashboard theme: BI-style CSS layered on top of the app's premium dark theme.

The main app injects its global premium stylesheet; these dashboard-specific
styles add the professional BI toolbar, section headers, KPI chips, chart tiles
and insight cards without duplicating the global chrome polish.
"""

from __future__ import annotations

_DASHBOARD_CSS = """
<style>
/* ============================================================
   AI ANALYTICS DASHBOARD — BI WORKSPACE STYLES
   ============================================================ */

.ada-bi-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 8px;
}

.ada-bi-title {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.ada-bi-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #93C5FD;
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 999px;
    padding: 4px 12px;
}

.ada-bi-subtitle {
    font-size: 14px;
    color: #9CA3AF;
    margin: 4px 0 22px 0;
}

.ada-bi-section {
    font-size: 16px;
    font-weight: 700;
    color: #E5E7EB;
    letter-spacing: 0.02em;
    margin: 26px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.ada-bi-section::after {
    content: "";
    flex: 1;
    height: 1px;
    background: #1F2937;
}

/* ---------- Filter bar ---------- */

.ada-bi-filters {
    background: #0F172A;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 16px 18px 6px 18px;
    margin-bottom: 8px;
}

.ada-bi-filter-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 999px;
    padding: 4px 12px;
    color: #9CA3AF;
    margin: 0 6px 6px 0;
}

.ada-bi-filter-chip strong {
    color: #E5E7EB;
    font-weight: 600;
}

/* ---------- KPI grid ---------- */

.ada-bi-kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.ada-bi-kpi-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 16px 18px;
    transition: transform 0.18s ease, border-color 0.18s ease;
    animation: adaFadeUp 0.4s ease both;
}

.ada-bi-kpi-card:hover {
    transform: translateY(-2px);
    border-color: #374151;
}

.ada-bi-kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.ada-bi-kpi-value {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.2;
    margin: 4px 0 2px 0;
}

.ada-bi-kpi-value.tone-success { color: #22C55E; }
.ada-bi-kpi-value.tone-danger  { color: #EF4444; }
.ada-bi-kpi-value.tone-info    { color: #3B82F6; }
.ada-bi-kpi-value.tone-warning { color: #F59E0B; }
.ada-bi-kpi-value.tone-neutral { color: #E5E7EB; }

.ada-bi-kpi-delta {
    font-size: 12px;
    font-weight: 600;
}

.ada-bi-kpi-delta.delta-up   { color: #22C55E; }
.ada-bi-kpi-delta.delta-down { color: #EF4444; }
.ada-bi-kpi-delta.delta-flat { color: #9CA3AF; }

/* ---------- Chart tiles ---------- */

.ada-bi-chart-tile {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 18px 20px 10px 20px;
    margin-bottom: 18px;
    animation: adaFadeUp 0.4s ease both;
}

.ada-bi-chart-title {
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 2px;
}

.ada-bi-chart-subtitle {
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 8px;
}

/* ---------- Insight cards ---------- */

.ada-bi-insight {
    background: #111827;
    border: 1px solid #1F2937;
    border-left: 3px solid #3B82F6;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    animation: adaFadeUp 0.4s ease both;
}

.ada-bi-insight .ada-bi-insight-title {
    font-size: 12px;
    font-weight: 700;
    color: #93C5FD;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.ada-bi-insight .ada-bi-insight-body {
    font-size: 14px;
    color: #D1D5DB;
    line-height: 1.6;
}

/* ---------- Empty / error states ---------- */

.ada-bi-empty {
    background: #0F172A;
    border: 1px dashed #1F2937;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    color: #9CA3AF;
    font-size: 14px;
}

@media (max-width: 900px) {
    .ada-bi-kpi-grid { grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
}
</style>
"""


def inject_dashboard_styles() -> None:
    """Inject the BI-style dashboard stylesheet into the Streamlit app."""
    import streamlit as st

    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)