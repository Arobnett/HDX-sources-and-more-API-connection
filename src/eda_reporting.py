"""Generate public-feature EDA reports, visuals, and a concise findings summary."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd


KEY_COLUMNS = ["iso3", "country", "year", "month"]
MIN_CORRELATION_OVERLAP = 100
MEANINGFUL_CORRELATION_THRESHOLD = 0.30


def _save_figure(path: Path) -> None:
    """Save the current matplotlib figure with consistent export settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def generate_eda_outputs(
    base_features: pd.DataFrame,
    reports_dir: Path,
    visuals_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create reusable EDA tables and PNG visuals from the validated Gold table."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [column for column in base_features.columns if column not in KEY_COLUMNS]
    duplicate_keys = int(base_features.duplicated(KEY_COLUMNS).sum())
    if duplicate_keys:
        raise ValueError(f"Gold table contains {duplicate_keys} duplicate modeling keys.")

    # 1. Gold integrity and feature inventory.
    integrity_summary = pd.DataFrame(
        {
            "metric": [
                "Rows",
                "Columns",
                "Predictor features",
                "Countries / ISO3 areas",
                "First year",
                "Last year",
                "Duplicate modeling keys",
            ],
            "value": [
                len(base_features),
                len(base_features.columns),
                len(feature_columns),
                base_features["iso3"].nunique(),
                int(base_features["year"].min()),
                int(base_features["year"].max()),
                duplicate_keys,
            ],
        }
    )
    integrity_summary.to_csv(reports_dir / "base_feature_gold_integrity.csv", index=False)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Gold Feature Table Integrity & Inventory", fontsize=16, y=0.96)
    fig.text(0.08, 0.78, f"{len(base_features):,}", fontsize=28, weight="bold")
    fig.text(0.08, 0.71, "Country-month rows", fontsize=11)
    fig.text(0.38, 0.78, f"{len(feature_columns):,}", fontsize=28, weight="bold")
    fig.text(0.38, 0.71, "Public predictor features", fontsize=11)
    fig.text(0.68, 0.78, f"{base_features['iso3'].nunique():,}", fontsize=28, weight="bold")
    fig.text(0.68, 0.71, "Countries / ISO3 areas", fontsize=11)
    fig.text(0.08, 0.48, f"{int(base_features['year'].min())}–{int(base_features['year'].max())}", fontsize=26, weight="bold")
    fig.text(0.08, 0.41, "Temporal span", fontsize=11)
    fig.text(0.48, 0.48, f"{duplicate_keys:,}", fontsize=26, weight="bold")
    fig.text(0.48, 0.41, "Duplicate modeling keys", fontsize=11)
    fig.text(0.08, 0.18, "Modeling grain: ISO3 + Country + Year + Month", fontsize=12)
    fig.text(0.08, 0.11, "EDA scope: validated public Gold predictors only", fontsize=12)
    plt.axis("off")
    _save_figure(visuals_dir / "gold_integrity_summary.png")

    # 2. Missingness analysis.
    missingness = pd.DataFrame(
        {
            "column": feature_columns,
            "missing_rows": [int(base_features[column].isna().sum()) for column in feature_columns],
            "missing_pct": [float(base_features[column].isna().mean() * 100) for column in feature_columns],
        }
    ).sort_values("missing_pct", ascending=False)
    missingness.to_csv(reports_dir / "base_feature_missingness.csv", index=False)

    plot_missingness = missingness.head(20).sort_values("missing_pct")
    plt.figure(figsize=(11, 8))
    plt.barh(plot_missingness["column"], plot_missingness["missing_pct"])
    plt.xlabel("Missing observations (%)")
    plt.title("Top 20 Public Features by Missingness")
    _save_figure(visuals_dir / "feature_missingness.png")

    # 3. Temporal coverage analysis.
    year_coverage = base_features.groupby("year")[feature_columns].agg(lambda series: series.notna().mean() * 100).reset_index()
    year_coverage.to_csv(reports_dir / "base_feature_year_coverage.csv", index=False)

    year_matrix = year_coverage.set_index("year").T
    plt.figure(figsize=(14, max(6, len(feature_columns) * 0.38)))
    plt.imshow(year_matrix.values, aspect="auto", interpolation="nearest", vmin=0, vmax=100)
    plt.colorbar(label="Non-null coverage (%)")
    plt.yticks(range(len(year_matrix.index)), year_matrix.index)
    plt.xticks(range(len(year_matrix.columns)), year_matrix.columns, rotation=90)
    plt.xlabel("Year")
    plt.title("Temporal Coverage by Public Feature")
    _save_figure(visuals_dir / "temporal_coverage.png")

    # 4. Geographic coverage analysis.
    country_coverage = base_features.groupby(["iso3", "country"])[feature_columns].agg(lambda series: series.notna().mean() * 100).reset_index()
    country_coverage["overall_coverage_pct"] = country_coverage[feature_columns].mean(axis=1)
    country_coverage.to_csv(reports_dir / "base_feature_country_coverage.csv", index=False)

    lowest_country_coverage = country_coverage.nsmallest(30, "overall_coverage_pct").sort_values("overall_coverage_pct")
    plt.figure(figsize=(11, 9))
    plt.barh(lowest_country_coverage["country"], lowest_country_coverage["overall_coverage_pct"])
    plt.xlabel("Mean predictor coverage (%)")
    plt.title("Lowest 30 Countries by Public-Feature Coverage")
    _save_figure(visuals_dir / "geographic_coverage.png")

    # 5. Numeric distribution analysis.
    numeric_features = base_features[feature_columns].select_dtypes(include="number")
    numeric_summary = numeric_features.describe().T
    numeric_summary["missing_pct"] = numeric_features.isna().mean() * 100
    numeric_summary["skew"] = numeric_features.skew(numeric_only=True)
    numeric_summary["abs_skew"] = numeric_summary["skew"].abs()
    numeric_summary.to_csv(reports_dir / "base_feature_numeric_summary.csv")

    # Plot actual distributions rather than presenting skewness itself as a distribution chart.
    distribution_features = numeric_summary.sort_values(["missing_pct", "abs_skew"]).head(6).index.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for axis, feature in zip(axes.flat, distribution_features):
        values = numeric_features[feature].dropna()
        if values.empty:
            axis.set_visible(False)
            continue
        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
        visible_values = values[(values >= lower) & (values <= upper)]
        axis.hist(visible_values, bins=40)
        axis.set_title(feature, fontsize=9)
        axis.set_ylabel("Observations")
    for axis in axes.flat[len(distribution_features):]:
        axis.set_visible(False)
    fig.suptitle("Selected Numeric Public-Feature Distributions", fontsize=16)
    fig.text(0.5, 0.01, "Display range clipped to each feature's 1st–99th percentiles for readability; source values are unchanged.", ha="center", fontsize=9)
    _save_figure(visuals_dir / "numeric_distributions.png")

    # 6. Pairwise predictor correlation analysis with explicit overlap safeguards.
    correlation_rows: list[dict[str, object]] = []
    numeric_columns = list(numeric_features.columns)
    for left_index, left_feature in enumerate(numeric_columns):
        for right_feature in numeric_columns[left_index + 1:]:
            pair = numeric_features[[left_feature, right_feature]].dropna()
            overlap_rows = len(pair)
            if overlap_rows < MIN_CORRELATION_OVERLAP:
                continue
            correlation = pair[left_feature].corr(pair[right_feature])
            if pd.notna(correlation):
                correlation_rows.append(
                    {
                        "feature_1": left_feature,
                        "feature_2": right_feature,
                        "overlap_rows": int(overlap_rows),
                        "correlation": float(correlation),
                        "abs_correlation": float(abs(correlation)),
                    }
                )

    meaningful_correlations = pd.DataFrame(
        correlation_rows,
        columns=["feature_1", "feature_2", "overlap_rows", "correlation", "abs_correlation"],
    )
    if not meaningful_correlations.empty:
        meaningful_correlations = meaningful_correlations.loc[
            meaningful_correlations["abs_correlation"] >= MEANINGFUL_CORRELATION_THRESHOLD
        ].sort_values(["abs_correlation", "overlap_rows"], ascending=[False, False]).reset_index(drop=True)
    meaningful_correlations.to_csv(reports_dir / "base_feature_meaningful_correlations.csv", index=False)

    top_correlations = meaningful_correlations.head(20).sort_values("correlation")
    plt.figure(figsize=(13, 9))
    if not top_correlations.empty:
        pair_labels = (
            top_correlations["feature_1"]
            + " | "
            + top_correlations["feature_2"]
            + " (n="
            + top_correlations["overlap_rows"].astype(str)
            + ")"
        )
        plt.barh(pair_labels, top_correlations["correlation"])
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Pearson correlation")
    plt.title(f"Top Pairwise Public Feature Correlations (minimum overlap n={MIN_CORRELATION_OVERLAP})")
    _save_figure(visuals_dir / "pairwise_feature_correlations.png")

    # 7. Concise findings summary and modeling implications.
    highest_missing = missingness.iloc[0] if not missingness.empty else None
    lowest_country = country_coverage.sort_values("overall_coverage_pct").iloc[0] if not country_coverage.empty else None
    most_skewed = numeric_summary.sort_values("abs_skew", ascending=False).iloc[0] if not numeric_summary.empty else None
    strongest_pair = meaningful_correlations.iloc[0] if not meaningful_correlations.empty else None

    findings = [
        {
            "analysis": "Gold integrity",
            "finding": f"{len(base_features):,} rows, {len(feature_columns):,} public predictors, {base_features['iso3'].nunique():,} countries/areas, {duplicate_keys} duplicate keys.",
            "modeling_implication": "Gold structure is suitable for downstream target integration once the restricted target grain is reconciled.",
        },
        {
            "analysis": "Missingness",
            "finding": f"Highest missingness: {highest_missing['column']} ({highest_missing['missing_pct']:.1f}%)." if highest_missing is not None else "No predictor missingness available.",
            "modeling_implication": "Treat missingness as source coverage first; choose imputation or exclusion after target join and modeling-window selection.",
        },
        {
            "analysis": "Temporal coverage",
            "finding": f"Public Gold spans {int(base_features['year'].min())}–{int(base_features['year'].max())}, with source-specific coverage varying by year.",
            "modeling_implication": "Use chronological train/validation/test boundaries and prevent future-period leakage in feature construction.",
        },
        {
            "analysis": "Geographic coverage",
            "finding": f"Lowest mean public-feature coverage: {lowest_country['country']} ({lowest_country['overall_coverage_pct']:.1f}%)." if lowest_country is not None else "No geographic coverage available.",
            "modeling_implication": "Evaluate geographic representativeness after target join and consider coverage flags or scope restrictions.",
        },
        {
            "analysis": "Numeric distributions",
            "finding": f"Most skewed numeric feature: {most_skewed.name} (|skew|={most_skewed['abs_skew']:.2f})." if most_skewed is not None else "No numeric predictor summary available.",
            "modeling_implication": "Test log or robust transformations and scaling inside training-only preprocessing pipelines where appropriate.",
        },
        {
            "analysis": "Pairwise correlations",
            "finding": (
                f"Strongest eligible pair: {strongest_pair['feature_1']} ↔ {strongest_pair['feature_2']} "
                f"(r={strongest_pair['correlation']:.2f}, n={int(strongest_pair['overlap_rows']):,})."
                if strongest_pair is not None
                else f"No pair with at least {MIN_CORRELATION_OVERLAP} overlapping observations met |r| ≥ {MEANINGFUL_CORRELATION_THRESHOLD:.2f}."
            ),
            "modeling_implication": "Review correlated predictors for redundancy, regularization, or combined representations; correlation is not causality.",
        },
    ]

    findings_summary = pd.DataFrame(findings)
    findings_summary.to_csv(reports_dir / "eda_findings_summary.csv", index=False)

    # Wrap long text before rendering so the exported summary is readable without clipping.
    display_findings = findings_summary.copy()
    display_findings["analysis"] = display_findings["analysis"].map(lambda value: fill(str(value), width=20))
    display_findings["finding"] = display_findings["finding"].map(lambda value: fill(str(value), width=58))
    display_findings["modeling_implication"] = display_findings["modeling_implication"].map(lambda value: fill(str(value), width=68))

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis("off")
    ax.set_title("EDA Findings Summary → Modeling Implications", fontsize=16, pad=20)
    table = ax.table(
        cellText=display_findings.values,
        colLabels=display_findings.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.14, 0.38, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.3)
    _save_figure(visuals_dir / "eda_findings_summary.png")

    return {
        "integrity_summary": integrity_summary,
        "missingness": missingness,
        "year_coverage": year_coverage,
        "country_coverage": country_coverage,
        "numeric_summary": numeric_summary,
        "meaningful_correlations": meaningful_correlations,
        "findings_summary": findings_summary,
    }
