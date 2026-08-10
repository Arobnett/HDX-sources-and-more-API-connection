"""Source-specific cleaning for Silver country-month-year inputs."""  # Describe the module purpose.

from __future__ import annotations  # Keep type hints forward-compatible in Colab.

import re  # Normalize generated feature names.
from pathlib import Path  # Represent file paths consistently.
from typing import Callable  # Type callable source handlers.

import pandas as pd  # Read, transform, and write tabular source files.

from paths import CLEAN_DIR, CLEAN_REPORTS_DIR, SILVER_DIR, SOURCE_CLEANING_RULES_PATH, ensure_output_directories  # Import canonical paths.


KEY_COLUMNS = ["iso3", "country", "year", "month"]  # Define the target modeling grain.
MONTHS = list(range(1, 13))  # Expand annual sources to every month explicitly.
METADATA_ONLY_SOURCES = {  # List source files that must never become model features.
    "hdx_colab_download_manifest__country_month_year.csv",
    "worldriskindex_meta__country_month_year.csv",
}


def _clean_feature_name(value: object) -> str:  # Convert source labels into stable column names.
    """Return a lowercase snake_case feature label."""  # Document the helper behavior.
    text = "missing" if pd.isna(value) else str(value)  # Replace missing labels with a deterministic token.
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()  # Remove punctuation and whitespace variability.
    return text or "missing"  # Ensure the caller always receives a usable label.


def _read_csv(path: Path) -> pd.DataFrame:  # Centralize CSV loading.
    """Read a Silver CSV with pandas' default inference."""  # Keep IO behavior explicit.
    return pd.read_csv(path, low_memory=False)  # Avoid mixed-type chunk warnings on wide files.


def _write_output(frame: pd.DataFrame, output_name: str) -> Path:  # Centralize cleaned writes.
    """Write one cleaned output CSV and return its path."""  # Describe the write contract.
    output_path = CLEAN_DIR / output_name  # Build the cleaned output path.
    frame.to_csv(output_path, index=False)  # Write without pandas row indexes.
    return output_path  # Return the path for validation reporting.


def _validate_key(frame: pd.DataFrame) -> dict[str, object]:  # Validate the target grain.
    """Summarize whether a cleaned feature table has one row per country-month."""  # Explain validation intent.
    missing_keys = [column for column in KEY_COLUMNS if column not in frame.columns]  # Find absent key columns.
    if missing_keys:  # Handle sources that failed to produce the modeling key.
        return {"key_unique": None, "duplicate_key_rows": None, "missing_key_columns": ";".join(missing_keys)}  # Report incomplete validation.
    duplicate_rows = int(frame.duplicated(KEY_COLUMNS).sum())  # Count repeated country-month keys.
    return {"key_unique": duplicate_rows == 0, "duplicate_key_rows": duplicate_rows, "missing_key_columns": ""}  # Return validation fields.


def _split_model_ready_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Separate usable rows from rejects.
    """Exclude rows without ISO3 from model-ready output and retain them for reporting."""  # Preserve rejected rows safely.
    if "iso3" not in frame.columns:  # Avoid failing on unexpected source schemas.
        return frame.copy(), frame.iloc[0:0].copy()  # Return no rejects when the expected key is absent.
    reject_mask = frame["iso3"].isna() | frame["iso3"].astype(str).str.strip().eq("")  # Identify missing ISO3 keys.
    return frame.loc[~reject_mask].copy(), frame.loc[reject_mask].copy()  # Return accepted rows and rejected rows.


def _aggregate_event_source(frame: pd.DataFrame, value_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:  # Aggregate event/admin sources.
    """Aggregate lower-geography event rows to one country-month row."""  # Explain source-specific grain change.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows that cannot join to country-level features.
    available_values = [column for column in value_columns if column in accepted.columns]  # Keep only present numeric fields.
    group_columns = [column for column in KEY_COLUMNS if column in accepted.columns]  # Use available target key columns.
    output = accepted.groupby(group_columns, dropna=False)[available_values].sum().reset_index()  # Sum event metrics to country-month.
    return output, rejected  # Return cleaned output and rejects.


def _clean_gdacs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean GDACS event rows.
    """Summarize GDACS disaster events to country-month features."""  # Explain GDACS-specific output.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows without ISO3 from the feature output.
    group_columns = [column for column in KEY_COLUMNS if column in accepted.columns]  # Define the country-month key.
    output = accepted.groupby(group_columns, dropna=False).agg(  # Aggregate event severity at country-month grain.
        gdacs_event_count=("id", "count"),  # Count feed records as disaster event observations.
        gdacs_max_severity_value=("severity_value", "max"),  # Preserve the strongest observed severity.
        gdacs_mean_severity_value=("severity_value", "mean"),  # Preserve average observed severity.
    ).reset_index()  # Convert grouped index back to columns.
    return output, rejected  # Return cleaned output and rejects.


def _clean_covid(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean WHO daily COVID rows.
    """Roll daily WHO COVID rows to country-month features."""  # Explain COVID-specific aggregation.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows that cannot join to country-level features.
    for column in ["new_cases", "new_deaths", "cumulative_cases", "cumulative_deaths"]:  # Iterate over expected numeric COVID fields.
        if column in accepted.columns:  # Coerce present values only.
            accepted[column] = pd.to_numeric(accepted[column], errors="coerce")  # Make aggregations numeric and tolerate blanks.
    group_columns = [column for column in KEY_COLUMNS if column in accepted.columns]  # Define the country-month key.
    output = accepted.groupby(group_columns, dropna=False).agg(  # Roll daily observations to month.
        covid_new_cases=("new_cases", "sum"),  # Sum daily new cases within month.
        covid_new_deaths=("new_deaths", "sum"),  # Sum daily new deaths within month.
        covid_cumulative_cases=("cumulative_cases", "max"),  # Use the month-end cumulative maximum.
        covid_cumulative_deaths=("cumulative_deaths", "max"),  # Use the month-end cumulative maximum.
    ).reset_index()  # Convert grouped index back to columns.
    return output, rejected  # Return cleaned output and rejects.


def _clean_views(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean VIEWS forecast rows.
    """Pass through already-unique VIEWS country-month forecasts with feature prefixes."""  # Explain pass-through behavior.
    accepted, rejected = _split_model_ready_rows(frame)  # Keep the reject path consistent.
    keep_columns = KEY_COLUMNS + ["country_id", "month_id", "main_mean", "main_dich", "main_mean_ln"]  # Retain key and forecast fields.
    present_columns = [column for column in keep_columns if column in accepted.columns]  # Avoid hard failure on optional IDs.
    output = accepted[present_columns].copy()  # Select stable columns only.
    output = output.rename(columns={  # Prefix forecast variables for downstream feature clarity.
        "main_mean": "views_main_mean",
        "main_dich": "views_main_dich",
        "main_mean_ln": "views_main_mean_ln",
    })
    return output, rejected  # Return cleaned output and rejects.


def _expand_annual_rows(frame: pd.DataFrame) -> pd.DataFrame:  # Expand annual rows to monthly rows.
    """Repeat each annual country-year row across months with provenance flags."""  # Make annual carry-forward explicit.
    expanded = frame.loc[frame.index.repeat(len(MONTHS))].copy()  # Repeat each row twelve times.
    expanded["month"] = MONTHS * len(frame)  # Assign calendar months to repeated rows.
    expanded["annual_carried_monthly"] = True  # Flag annual values copied to monthly grain.
    return expanded  # Return expanded rows.


def _clean_inform(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean INFORM annual indicator rows.
    """Pivot INFORM indicators wide and expand annual values to months."""  # Explain annual index handling.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows without ISO3.
    accepted["indicator_feature"] = "inform_" + accepted["indicatorid"].map(_clean_feature_name)  # Build stable indicator names.
    pivot = accepted.pivot_table(  # Pivot long indicators into numeric feature columns.
        index=["iso3", "country", "year"],
        columns="indicator_feature",
        values="indicatorscore",
        aggfunc="mean",
    ).reset_index()  # Return keys to ordinary columns.
    pivot.columns.name = None  # Remove the pandas pivot column index name.
    return _expand_annual_rows(pivot), rejected  # Return monthly-expanded output and rejects.


def _clean_whs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean WHS annual indicator rows.
    """Pivot WHS numeric indicator rows wide and expand annual values to months."""  # Explain WHS handling.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows without ISO3.
    accepted = accepted.copy()  # Avoid chained assignment on caller data.
    accepted["year"] = pd.to_numeric(accepted["year"], errors="coerce").astype("Int64")  # Normalize object years to integers.
    accepted["numericvalue"] = pd.to_numeric(accepted["numericvalue"], errors="coerce")  # Coerce numeric feature values.
    accepted = accepted.dropna(subset=["year", "numericvalue"])  # Keep only rows with usable annual numeric values.
    accepted["feature_name"] = (  # Build stable feature names from indicator and location dimensions.
        "whs_"
        + accepted["indicatorcode"].map(_clean_feature_name)
        + "_"
        + accepted["locationcode"].map(_clean_feature_name)
    )
    pivot = accepted.pivot_table(  # Pivot selected numeric indicators into wide annual feature columns.
        index=["iso3", "country", "year"],
        columns="feature_name",
        values="numericvalue",
        aggfunc="mean",
    ).reset_index()  # Return keys to ordinary columns.
    pivot.columns.name = None  # Remove the pandas pivot column index name.
    return _expand_annual_rows(pivot), rejected  # Return monthly-expanded output and rejects.


def _clean_worldriskindex(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Clean WorldRiskIndex trend rows.
    """Prefix numeric WorldRiskIndex annual fields and expand them to months."""  # Explain WRI handling.
    accepted, rejected = _split_model_ready_rows(frame)  # Remove rows without ISO3.
    identity_columns = {"source_dataset", "source_file", "country", "iso3", "year", "wri_country", "iso3_code"}  # Define non-feature columns.
    numeric_columns = [column for column in accepted.columns if column not in identity_columns and pd.api.types.is_numeric_dtype(accepted[column])]  # Keep numeric WRI fields.
    output = accepted[["iso3", "country", "year"] + numeric_columns].copy()  # Keep keys and numeric annual fields.
    output = output.rename(columns={column: f"wri_{column}" for column in numeric_columns})  # Prefix WRI feature fields.
    return _expand_annual_rows(output), rejected  # Return monthly-expanded output and rejects.


HANDLERS: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = {  # Map Silver filenames to cleaner functions.
    "civilian_targeting_events_and_fatalities__country_month_year.csv": lambda frame: _aggregate_event_source(frame, ["events", "fatalities"]),
    "demonstration_events__country_month_year.csv": lambda frame: _aggregate_event_source(frame, ["events"]),
    "political_violence_events_and_fatalities__country_month_year.csv": lambda frame: _aggregate_event_source(frame, ["events", "fatalities"]),
    "who_covid_19_global_daily_data__country_month_year.csv": _clean_covid,
    "gdacs_rss_information__country_month_year.csv": _clean_gdacs,
    "views_conflict_forecasts_country_month__country_month_year.csv": _clean_views,
    "inform_risk_index_trends__country_month_year.csv": _clean_inform,
    "whs2026_datadownload_as_of_2026_08_02__country_month_year.csv": _clean_whs,
    "worldriskindex_trend__country_month_year.csv": _clean_worldriskindex,
}


def load_cleaning_rules(rules_path: Path = SOURCE_CLEANING_RULES_PATH) -> pd.DataFrame:  # Load the committed 03b rules.
    """Read the source cleaning rules CSV."""  # Describe IO behavior.
    return pd.read_csv(rules_path)  # Return the rule table to callers.


def clean_silver_directory(silver_dir: Path = SILVER_DIR, rules_path: Path = SOURCE_CLEANING_RULES_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:  # Main public entry point.
    """Clean all rule-covered Silver feature sources and write reports."""  # Explain the function contract.
    ensure_output_directories()  # Create generated output folders only.
    rules = load_cleaning_rules(rules_path)  # Load source-specific rule metadata.
    summary_rows: list[dict[str, object]] = []  # Collect one validation summary per source.
    reject_rows: list[pd.DataFrame] = []  # Collect rejected rows with source labels.
    for rule in rules.to_dict("records"):  # Iterate over committed rules in file order.
        source_file = str(rule["source_file"])  # Extract the source filename.
        source_path = silver_dir / source_file  # Resolve the Silver input path.
        output_name = str(rule["output_name"])  # Extract the configured output filename.
        if source_file in METADATA_ONLY_SOURCES:  # Skip metadata-only files for feature output.
            summary_rows.append({"source_file": source_file, "output_name": output_name, "status": "metadata_skipped"})  # Report intentional skip.
            continue  # Move to the next source.
        if source_file not in HANDLERS:  # Guard against missing source-specific logic.
            summary_rows.append({"source_file": source_file, "output_name": output_name, "status": "missing_handler"})  # Report missing handler.
            continue  # Move to the next source.
        if not source_path.exists():  # Guard against missing Silver files.
            summary_rows.append({"source_file": source_file, "output_name": output_name, "status": "missing_source"})  # Report missing input.
            continue  # Move to the next source.
        try:  # Keep one failed source from stopping the entire cleaning run.
            raw = _read_csv(source_path)  # Load the Silver source.
            cleaned, rejected = HANDLERS[source_file](raw)  # Apply source-specific cleaning.
            output_path = _write_output(cleaned, output_name)  # Write the cleaned feature table.
            validation = _validate_key(cleaned)  # Validate country-month uniqueness.
            summary_rows.append({  # Store run metadata and validation results.
                "source_file": source_file,
                "output_name": output_name,
                "status": "ok",
                "input_rows": len(raw),
                "output_rows": len(cleaned),
                "rejected_rows": len(rejected),
                "output_bytes": output_path.stat().st_size,
                **validation,
            })
            if not rejected.empty:  # Preserve rejected rows for transparent review.
                rejected = rejected.copy()  # Avoid mutating handler-owned data.
                rejected["source_file"] = source_file  # Label the reject source without duplicating existing columns.
                reject_rows.append(rejected)  # Add to the combined reject report.
        except Exception as exc:  # Record source-level failures without hiding them.
            summary_rows.append({"source_file": source_file, "output_name": output_name, "status": "error", "error": str(exc)})  # Store error text.
    summary = pd.DataFrame(summary_rows)  # Build the validation summary.
    rejects = pd.concat(reject_rows, ignore_index=True, sort=False) if reject_rows else pd.DataFrame(columns=["source_file"])  # Build rejects report.
    summary.to_csv(CLEAN_REPORTS_DIR / "cleaning_validation_report.csv", index=False)  # Write validation report.
    rejects.to_csv(CLEAN_REPORTS_DIR / "cleaning_reject_rows.csv", index=False)  # Write rejected row report.
    return summary, rejects  # Return reports for notebooks and tests.
