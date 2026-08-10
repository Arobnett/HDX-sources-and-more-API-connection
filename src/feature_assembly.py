"""Assemble cleaned country-month-year sources into one modeling feature table."""  # Describe the module purpose.

from __future__ import annotations  # Keep type hints forward-compatible in Colab.

import re  # Normalize source-derived prefixes.
from functools import reduce  # Combine many source tables through repeated merges.
from pathlib import Path  # Represent filesystem paths consistently.

import pandas as pd  # Read, join, validate, and write tabular data.

from paths import CLEAN_DIR, FEATURE_REPORTS_DIR, MODEL_FEATURES_DIR, ensure_output_directories  # Import canonical output paths.


KEY_COLUMNS = ["iso3", "country", "year", "month"]  # Define the target modeling grain.
ASSEMBLED_OUTPUT_NAME = "model_feature_table_country_month_year.csv"  # Name the assembled modeling table.
ASSEMBLY_REPORT_NAME = "feature_assembly_report.csv"  # Name the source-level assembly report.
COLUMN_CATALOG_NAME = "feature_column_catalog.csv"  # Name the feature provenance catalog.
NON_FEATURE_COLUMNS = {"country_id", "month_id"}  # Keep optional reference IDs out of model feature columns.


def _clean_prefix(value: str) -> str:  # Build a safe prefix from a cleaned source filename.
    """Return a stable snake_case prefix for source-derived feature names."""  # Document prefix behavior.
    stem = Path(value).stem  # Remove the CSV suffix from the source name.
    stem = re.sub(r"^clean_", "", stem)  # Remove the shared cleaned-output prefix.
    stem = re.sub(r"_country_month$", "", stem)  # Remove the shared country-month suffix.
    stem = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").lower()  # Normalize punctuation and case.
    return stem or "source"  # Guarantee a non-empty prefix.


def _read_clean_file(path: Path) -> pd.DataFrame:  # Centralize clean-source loading.
    """Read one Step 03c cleaned CSV."""  # Explain IO behavior.
    return pd.read_csv(path, low_memory=False)  # Avoid mixed-type chunk warnings on wide feature files.


def _validate_key(frame: pd.DataFrame) -> dict[str, object]:  # Validate a country-month table key.
    """Return key validation fields for a country-month-year table."""  # Explain validation intent.
    missing_keys = [column for column in KEY_COLUMNS if column not in frame.columns]  # Detect absent key columns.
    if missing_keys:  # Handle tables that cannot be validated at the target grain.
        return {"key_unique": None, "duplicate_key_rows": None, "missing_key_columns": ";".join(missing_keys)}  # Report incomplete validation.
    duplicate_rows = int(frame.duplicated(KEY_COLUMNS).sum())  # Count duplicate country-month keys.
    return {"key_unique": duplicate_rows == 0, "duplicate_key_rows": duplicate_rows, "missing_key_columns": ""}  # Return validation fields.


def _prepare_source_frame(path: Path) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:  # Prepare one cleaned table for joining.
    """Validate and rename one cleaned feature table before assembly."""  # Explain the function contract.
    raw = _read_clean_file(path)  # Load the cleaned source table.
    validation = _validate_key(raw)  # Validate source-level country-month uniqueness.
    prefix = _clean_prefix(path.name)  # Build a source-specific prefix.
    feature_columns = [column for column in raw.columns if column not in KEY_COLUMNS and column not in NON_FEATURE_COLUMNS]  # Select joinable feature columns.
    rename_map = {column: column if column.startswith(f"{prefix}_") else f"{prefix}_{column}" for column in feature_columns}  # Prevent feature-name collisions.
    prepared_columns = KEY_COLUMNS + feature_columns  # Keep only keys plus feature columns.
    prepared = raw[prepared_columns].rename(columns=rename_map).copy()  # Rename features and preserve the key columns.
    catalog = pd.DataFrame(  # Build feature provenance rows for this source.
        [{"source_file": path.name, "original_column": original, "assembled_column": renamed} for original, renamed in rename_map.items()]  # Map source columns to final columns.
    )
    report = {  # Build source-level assembly metadata.
        "source_file": path.name,  # Record the cleaned source filename.
        "input_rows": len(raw),  # Record source row count.
        "feature_columns": len(feature_columns),  # Record feature column count from this source.
        **validation,  # Include source-level key validation.
    }
    return prepared, report, catalog  # Return the prepared frame, source report, and column catalog.


def _outer_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:  # Merge two prepared feature frames.
    """Outer-join two country-month feature tables on the modeling key."""  # Explain merge behavior.
    return left.merge(right, on=KEY_COLUMNS, how="outer")  # Preserve partial source coverage across countries and months.


def assemble_feature_table(clean_dir: Path = CLEAN_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # Main public entry point.
    """Assemble all cleaned source outputs into one wide modeling feature table."""  # Explain function contract.
    ensure_output_directories()  # Create generated output folders only.
    clean_paths = sorted(clean_dir.glob("*.csv"))  # Find all cleaned feature outputs from Step 03c.
    if not clean_paths:  # Stop clearly when 03c has not been run.
        raise FileNotFoundError(f"No cleaned CSV files found in {clean_dir}")  # Raise a visible path-specific error.
    prepared_frames: list[pd.DataFrame] = []  # Collect source frames ready for joining.
    report_rows: list[dict[str, object]] = []  # Collect source-level report records.
    catalog_frames: list[pd.DataFrame] = []  # Collect feature provenance catalog frames.
    for path in clean_paths:  # Process each cleaned source file in stable filename order.
        prepared, report, catalog = _prepare_source_frame(path)  # Validate and rename one source table.
        prepared_frames.append(prepared)  # Keep the prepared frame for assembly.
        report_rows.append(report)  # Keep source-level metadata for reporting.
        catalog_frames.append(catalog)  # Keep source-column provenance for reporting.
    assembled = reduce(_outer_join, prepared_frames)  # Combine all sources into one wide table.
    assembled = assembled.sort_values(KEY_COLUMNS).reset_index(drop=True)  # Make output row order deterministic.
    assembled_validation = _validate_key(assembled)  # Validate final assembled key uniqueness.
    assembled_path = MODEL_FEATURES_DIR / ASSEMBLED_OUTPUT_NAME  # Build the assembled output path.
    assembled.to_csv(assembled_path, index=False)  # Write the modeling feature table.
    report = pd.DataFrame(report_rows)  # Build the source-level assembly report.
    report["assembled_output_name"] = ASSEMBLED_OUTPUT_NAME  # Record the final output table name.
    report["assembled_rows"] = len(assembled)  # Record final assembled row count.
    report["assembled_columns"] = len(assembled.columns)  # Record final assembled column count.
    report["assembled_key_unique"] = assembled_validation["key_unique"]  # Record final key uniqueness.
    report["assembled_duplicate_key_rows"] = assembled_validation["duplicate_key_rows"]  # Record final duplicate key rows.
    report["assembled_missing_key_columns"] = assembled_validation["missing_key_columns"]  # Record final missing key columns.
    catalog = pd.concat(catalog_frames, ignore_index=True, sort=False) if catalog_frames else pd.DataFrame(columns=["source_file", "original_column", "assembled_column"])  # Build feature catalog.
    report.to_csv(FEATURE_REPORTS_DIR / ASSEMBLY_REPORT_NAME, index=False)  # Write source-level assembly report.
    catalog.to_csv(FEATURE_REPORTS_DIR / COLUMN_CATALOG_NAME, index=False)  # Write feature provenance catalog.
    return assembled, report, catalog  # Return outputs for notebooks and tests.
