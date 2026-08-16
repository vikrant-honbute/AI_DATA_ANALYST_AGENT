"""Deterministic dashboard layout builder.

Turns a DataFrame plus its DataProfile into a complete dashboard spec:
KPI row with period-over-period deltas, a curated set of charts with embedded
aggregated data, and fact-based insight bullets. No LLM required.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.formatting import (
        classify_trend,
        format_compact,
        format_kpi_value,
        looks_like_money,
        prettify_name,
    )
    from dashboard.profiler import (
        DataProfile,
        aggregate_for_metric,
        parse_time_column,
)
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.formatting import (
        classify_trend,
        format_compact,
        format_kpi_value,
        looks_like_money,
        prettify_name,
    )
    from project.dashboard.profiler import (
        DataProfile,
        aggregate_for_metric,
        parse_time_column,
)

MAX_TOP_CATEGORIES = 10
MAX_DONUT_CATEGORIES = 7
MAX_CHART_SERIES_POINTS = 200
MAX_HIST_VALUES = 500
GRANULARITY_LABELS = {"D": "day", "W": "week", "M": "month"}


def _rotated_metrics(profile: DataProfile, focus_offset: int) -> list[str]:
    """Rotate metric candidates so retry attempts emphasize different metrics."""
    candidates = list(profile.metric_candidates)
    if not candidates or focus_offset <= 0:
        return candidates
    offset = focus_offset % len(candidates)
    return candidates[offset:] + candidates[:offset]


def _pick_breakdown_category(profile: DataProfile) -> str | None:
    """Pick the most dashboard-friendly categorical column (2-20 uniques)."""
    candidates = [
        column
        for column in profile.categorical_columns
        if 2 <= profile.categorical_unique_counts.get(column, 0) <= 20
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: (abs(profile.categorical_unique_counts[c] - 5), c))


def _round_number(value: Any, digits: int = 2) -> Any:
    """Round floats for embedding in chart data records."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(number):
        return None
    return round(number, digits)


def _period_delta(trend_values: list[float], granularity: str) -> tuple[str, str] | None:
    """Compute a period-over-period delta label and tone for the last two points."""
    if len(trend_values) < 2:
        return None
    previous, last = trend_values[-2], trend_values[-1]
    if previous == 0:
        return None
    pct = (last - previous) / abs(previous) * 100
    label = GRANULARITY_LABELS.get(granularity, "period")
    if abs(pct) < 0.05:
        return f"→ flat vs prev {label}", "neutral"
    if pct > 0:
        return f"▲ +{pct:.1f}% vs prev {label}", "success"
    return f"▼ {pct:.1f}% vs prev {label}", "danger"


def _build_kpis(
    df: pd.DataFrame,
    profile: DataProfile,
    metric: str | None,
    trend_values: list[float] | None,
    category_share: tuple[str, float] | None,
) -> list[dict[str, Any]]:
    """Build the KPI card row for the dashboard header."""
    kpis: list[dict[str, Any]] = [
        {
            "label": "Records",
            "icon": "📊",
            "value": f"{profile.rows:,}",
            "sub": f"{len(profile.columns)} columns",
            "tone": "neutral",
            "delta": None,
        }
    ]

    if metric is None:
        if category_share is not None:
            kpis.append(
                {
                    "label": "Top Segment",
                    "icon": "🏆",
                    "value": category_share[0],
                    "sub": f"{category_share[1]:.0f}% share",
                    "tone": "success",
                    "delta": None,
                }
            )
        return kpis

    metric_label = prettify_name(metric)
    money = looks_like_money(metric_label)
    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    if series.empty:
        return kpis

    delta_info = _period_delta(trend_values, profile.time_granularity) if trend_values else None
    delta_text = delta_info[0] if delta_info else None
    delta_tone = delta_info[1] if delta_info else "neutral"

    agg = aggregate_for_metric(metric)
    if agg == "sum":
        kpis.append(
            {
                "label": f"Total {metric_label}",
                "icon": "🏆",
                "value": format_kpi_value(float(series.sum()), money),
                "sub": delta_text or "Across full period",
                "tone": delta_tone if delta_text else "success",
                "delta": delta_text,
            }
        )
    kpis.append(
        {
            "label": f"Average {metric_label}",
            "icon": "📈",
            "value": format_kpi_value(float(series.mean()), money),
            "sub": delta_text or "Per record",
            "tone": delta_tone if delta_text else "info",
            "delta": delta_text,
        }
    )

    if category_share is not None:
        kpis.append(
            {
                "label": "Top Segment Share",
                "icon": "🎯",
                "value": f"{category_share[1]:.0f}%",
                "sub": category_share[0],
                "tone": "warning",
                "delta": None,
            }
        )
    elif len(profile.metric_candidates) >= 2:
        second = profile.metric_candidates[1]
        second_series = pd.to_numeric(df[second], errors="coerce").dropna()
        if not second_series.empty:
            second_label = prettify_name(second)
            second_money = looks_like_money(second_label)
            second_agg = aggregate_for_metric(second)
            value = float(second_series.mean() if second_agg == "mean" else second_series.sum())
            kpis.append(
                {
                    "label": f"{'Average' if second_agg == 'mean' else 'Total'} {second_label}",
                    "icon": "🧮",
                    "value": format_kpi_value(value, second_money),
                    "sub": "Secondary metric",
                    "tone": "neutral",
                    "delta": None,
                }
            )

    return kpis


def _trend_chart(
    df: pd.DataFrame,
    profile: DataProfile,
    metric: str,
    time_series: pd.Series,
) -> tuple[dict[str, Any] | None, list[float], pd.DataFrame | None]:
    """Build the time-trend line chart and return (chart, values, aggregated frame)."""
    granularity = profile.time_granularity if profile.time_granularity in {"D", "W", "M"} else "M"
    working = pd.DataFrame(
        {
            "_period": time_series.dt.to_period(granularity),
            "_metric": pd.to_numeric(df[metric], errors="coerce"),
        }
    ).dropna()
    if working.empty:
        return None, [], None

    agg = aggregate_for_metric(metric)
    grouped = working.groupby("_period")["_metric"].agg(agg).sort_index()
    values = [float(v) for v in grouped.values]
    records = [
        {"period": str(period), "value": _round_number(value)}
        for period, value in grouped.items()
    ][:MAX_CHART_SERIES_POINTS]
    if not records:
        return None, [], None

    metric_label = prettify_name(metric)
    period_label = GRANULARITY_LABELS[granularity]
    chart = {
        "chart_id": "trend",
        "chart_type": "line",
        "title": f"{metric_label} over time",
        "subtitle": f"{'Sum' if agg == 'sum' else 'Average'} per {period_label} across the analyzed period.",
        "x": "period",
        "y": "value",
        "data": records,
        "data_note": "",
    }
    return chart, values, grouped


def _category_chart(
    df: pd.DataFrame,
    profile: DataProfile,
    metric: str,
    category: str,
) -> tuple[dict[str, Any] | None, tuple[str, float] | None]:
    """Build the top-category bar/donut chart and return (chart, top share)."""
    grouped = (
        pd.to_numeric(df[metric], errors="coerce")
        .groupby(df[category].astype(str), sort=False)
        .sum()
        .dropna()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return None, None

    total = float(grouped.sum())
    unique_count = int(len(grouped))
    top10 = grouped.head(MAX_TOP_CATEGORIES)
    records = [
        {"category": str(index), "value": _round_number(value)}
        for index, value in top10.items()
    ]
    if not records:
        return None, None

    metric_label = prettify_name(metric)
    category_label = prettify_name(category)
    if unique_count <= MAX_DONUT_CATEGORIES:
        chart_type = "donut"
        subtitle = f"Share of {metric_label.lower()} contributed by each {category_label.lower()}."
    else:
        chart_type = "bar"
        subtitle = f"Top {len(records)} of {unique_count} {category_label.lower()} segments by {metric_label.lower()}."

    top_name = str(grouped.index[0])
    top_share = (float(grouped.iloc[0]) / total * 100) if total else 0.0
    chart = {
        "chart_id": "categories",
        "chart_type": chart_type,
        "title": f"{metric_label} by {category_label}",
        "subtitle": subtitle,
        "x": "category",
        "y": "value",
        "data": records,
        "data_note": f"Top {len(records)} of {unique_count} categories" if unique_count > len(records) else "",
    }
    return chart, (top_name, top_share)


def _distribution_chart(df: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    """Build the histogram chart for the primary metric distribution."""
    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    if series.empty or int(series.nunique()) < 2:
        return None
    values = [
        _round_number(value)
        for value in series.head(MAX_HIST_VALUES).to_list()
    ]
    metric_label = prettify_name(metric)
    return {
        "chart_id": "distribution",
        "chart_type": "hist",
        "title": f"Distribution of {metric_label}",
        "subtitle": f"Spread of {metric_label.lower()} values across all records.",
        "x": "value",
        "y": "count",
        "data": [{"value": value} for value in values if value is not None],
        "data_note": "",
    }


def _correlation_chart(df: pd.DataFrame, profile: DataProfile) -> dict[str, Any] | None:
    """Build the correlation heatmap when enough numeric columns exist."""
    numeric = [c for c in profile.numeric_columns][:6]
    if len(numeric) < 3:
        return None
    try:
        matrix = df[numeric].corr(numeric_only=True)
    except Exception:
        return None
    if matrix.empty:
        return None

    cells: list[dict[str, Any]] = []
    for row in matrix.index:
        for col in matrix.columns:
            value = matrix.loc[row, col]
            if pd.isna(value):
                continue
            cells.append(
                {"x": str(row), "y": str(col), "z": _round_number(float(value), 3)}
            )
    if not cells:
        return None

    return {
        "chart_id": "correlation",
        "chart_type": "heatmap",
        "title": "Metric correlation heatmap",
        "subtitle": "Pearson correlation between numeric metrics.",
        "x": "x",
        "y": "y",
        "data": cells,
        "data_note": "",
    }


def _count_chart(df: pd.DataFrame, category: str) -> dict[str, Any] | None:
    """Build a frequency bar chart (used when no time column exists)."""
    counts = df[category].astype(str).value_counts().head(MAX_TOP_CATEGORIES)
    if counts.empty:
        return None
    records = [{"category": str(index), "value": int(value)} for index, value in counts.items()]
    category_label = prettify_name(category)
    return {
        "chart_id": "frequencies",
        "chart_type": "bar",
        "title": f"Frequency of {category_label}",
        "subtitle": f"Record counts per {category_label.lower()} value.",
        "x": "category",
        "y": "value",
        "data": records,
        "data_note": "",
    }


def _build_insights(
    df: pd.DataFrame,
    profile: DataProfile,
    metric: str | None,
    trend_grouped: pd.DataFrame | None,
    category_info: tuple[str, str, float] | None,
    correlation_pair: tuple[str, str, float] | None,
) -> list[str]:
    """Compose fact-based insight bullets from computed aggregates only."""
    facts: list[str] = []
    metric_label = prettify_name(metric) if metric else "primary metric"
    money = looks_like_money(metric_label) if metric else False

    if trend_grouped is not None and not trend_grouped.empty:
        period_label = GRANULARITY_LABELS.get(profile.time_granularity, "period")
        peak_index = trend_grouped.idxmax()
        trough_index = trend_grouped.idxmin()
        peak_value = float(trend_grouped.max())
        trend = classify_trend(trend_grouped)
        trend_text = {
            "rising": "is trending upward across the observed period",
            "declining": "is trending downward across the observed period",
            "mixed": "shows no consistent direction across the observed period",
            "stable": "is broadly stable across the observed period",
        }.get(trend, "fluctuates across the observed period")
        facts.append(
            f"{metric_label} {trend_text}: peaked in {peak_index} at "
            f"{'$' if money else ''}{format_compact(peak_value)} and bottomed in {trough_index}. "
            f"(values are per {period_label})"
        )

    if category_info is not None:
        top_name, category_label, share = category_info
        facts.append(
            f"'{top_name}' is the leading {category_label.lower()} segment, contributing "
            f"approximately {share:.0f}% of total {metric_label.lower()}."
        )

    if metric is not None:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not series.empty:
            mean_value = float(series.mean())
            median_value = float(series.median())
            if median_value > 0 and abs(mean_value - median_value) / median_value > 0.25:
                direction = "right-skewed" if mean_value > median_value else "left-skewed"
                facts.append(
                    f"{metric_label} is {direction}: average {format_compact(mean_value)} vs "
                    f"median {format_compact(median_value)}, so a small number of large values "
                    f"pulls the mean."
                )

    if correlation_pair is not None:
        first, second, strength = correlation_pair
        facts.append(
            f"'{prettify_name(first)}' and '{prettify_name(second)}' are the most correlated "
            f"metrics (r = {strength:.2f})."
        )

    missing_columns = [
        column for column, ratio in profile.missing_ratio.items() if ratio >= 0.1
    ][:2]
    if missing_columns:
        facts.append(
            "Missing values are concentrated in: "
            + ", ".join(f"'{c}' ({profile.missing_ratio[c] * 100:.0f}%)" for c in missing_columns)
            + ". Figures may understate true totals."
        )

    return facts[:5]


def _strongest_correlation(df: pd.DataFrame, profile: DataProfile) -> tuple[str, str, float] | None:
    """Find the strongest off-diagonal correlation pair among numeric columns."""
    numeric = [c for c in profile.numeric_columns][:6]
    if len(numeric) < 2:
        return None
    try:
        matrix = df[numeric].corr(numeric_only=True)
    except Exception:
        return None
    if matrix.empty:
        return None

    best: tuple[str, str, float] | None = None
    columns = list(matrix.columns)
    for i, row in enumerate(columns):
        for col in columns[i + 1:]:
            value = matrix.loc[row, col]
            if pd.isna(value):
                continue
            if best is None or abs(float(value)) > abs(best[2]):
                best = (str(row), str(col), float(value))
    if best is not None and abs(best[2]) >= 0.5:
        return best
    return None


def build_dashboard_spec(
    df: pd.DataFrame,
    profile: DataProfile,
    query: str = "",
    focus_offset: int = 0,
) -> dict[str, Any]:
    """Assemble a complete deterministic dashboard spec from a DataFrame.

    Args:
        df: Source data.
        profile: DataProfile produced by ``profile_dataframe``.
        query: Original user query (reserved for future context use).
        focus_offset: Rotates primary metric selection (used on retries).

    Returns:
        A JSON-safe dashboard spec dict (not yet LLM-refined).
    """
    metrics = _rotated_metrics(profile, focus_offset)
    metric = metrics[0] if metrics else None

    charts: list[dict[str, Any]] = []
    trend_values: list[float] = []
    trend_grouped: pd.DataFrame | None = None
    category_info: tuple[str, str, float] | None = None
    category_share: tuple[str, float] | None = None

    time_series = (
        parse_time_column(df, profile.time_column) if profile.time_column else None
    )

    if metric is not None and time_series is not None:
        trend_chart, trend_values, trend_grouped = _trend_chart(df, profile, metric, time_series)
        if trend_chart is not None:
            charts.append(trend_chart)

    breakdown_category = _pick_breakdown_category(profile)
    if metric is not None and breakdown_category is not None:
        category_chart, category_share = _category_chart(df, profile, metric, breakdown_category)
        if category_chart is not None:
            charts.append(category_chart)
        if category_share is not None:
            category_info = (
                category_share[0],
                prettify_name(breakdown_category),
                category_share[1],
            )

    if metric is not None:
        distribution = _distribution_chart(df, metric)
        if distribution is not None:
            charts.append(distribution)

    correlation_pair = _strongest_correlation(df, profile)
    correlation_chart = _correlation_chart(df, profile)
    if correlation_chart is not None:
        charts.append(correlation_chart)

    if not time_series and profile.categorical_columns:
        count_chart = _count_chart(df, profile.categorical_columns[0])
        if count_chart is not None:
            charts.append(count_chart)

    kpis = _build_kpis(df, profile, metric, trend_values or None, category_share)
    insights = _build_insights(
        df, profile, metric, trend_grouped, category_info, correlation_pair
    )

    metric_label = prettify_name(metric) if metric else None
    title = (
        f"{metric_label} Performance Dashboard"
        if metric_label
        else "Data Overview Dashboard"
    )
    subtitle_bits = [f"{profile.rows:,} records", f"{len(profile.columns)} columns"]
    if profile.time_start and profile.time_end:
        subtitle_bits.append(f"{profile.time_start} to {profile.time_end}")
    subtitle = " · ".join(subtitle_bits)

    summary_bits = [f"Analyzed {profile.rows:,} records across {len(profile.columns)} columns."]
    if metric is not None:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if not series.empty:
            agg = aggregate_for_metric(metric)
            headline = (
                f"{metric_label} averaged {format_compact(float(series.mean()))} per record "
                f"(total {format_compact(float(series.sum()))})."
                if agg == "sum"
                else f"{metric_label} averaged {format_compact(float(series.mean()))} per record."
            )
            summary_bits.append(headline)
    if category_info is not None:
        summary_bits.append(
            f"The leading segment is '{category_info[0]}' "
            f"({category_info[1].lower()})."
        )
    if not insights:
        summary_bits.append(
            "The dataset did not expose strong numeric patterns; review the raw table for "
            "structural issues before drawing conclusions."
        )

    recommendations = [
        f"Validate the definition and coverage of '{metric_label}' before acting on "
        f"segment-level differences." if metric else "Review field definitions and coverage before acting on segment differences.",
        "Compare period-over-period changes against the business calendar (promotions, seasonality) before attributing causes.",
    ]
    if category_info is not None:
        recommendations.append(
            f"Drill into the '{category_info[0]}' segment to understand what drives its share."
        )

    return {
        "title": title,
        "subtitle": subtitle,
        "data_source": "",
        "generated_at": "",
        "row_count": profile.rows,
        "column_count": len(profile.columns),
        "time_range": (
            {"start": profile.time_start, "end": profile.time_end}
            if profile.time_start and profile.time_end
            else None
        ),
        "kpis": kpis,
        "charts": charts,
        "executive_summary": " ".join(summary_bits),
        "insights": insights,
        "recommendations": recommendations[:3],
    }
