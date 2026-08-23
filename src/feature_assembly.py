"""Assemble cleaned country-month-year sources into one modeling feature table."""  # Describe the module purpose.

from __future__ import annotations  # Keep type hints forward-compatible in Colab.

import re  # Normalize source-derived prefixes.
from functools import reduce  # Combine many source tables through repeated merges.
from pathlib import Path  # Represent filesystem paths consistently.

import pandas as pd  # Read, join, validate, and write tabular data.

from paths import CLEAN_DIR, FEATURE_REPORTS_DIR, MODEL_FEATURES_DIR, ensure_output_directories  # Import canonical output paths.


KEY_COLUMNS = ["iso3", "country", "year", "month"]  # Define the target modeling grain.
BASE_OUTPUT_NAME = "base_feature_table_country_month_year.csv"  # Name the memory-safe base feature table.
ASSEMBLY_REPORT_NAME = "feature_assembly_report.csv"  # Name the source-level assembly report.
COLUMN_CATALOG_NAME = "feature_column_catalog.csv"  # Name the feature provenance catalog.
NON_FEATURE_COLUMNS = {  # Keep identifiers and lineage fields out of model features.
    "country_id",
    "month_id",
    "country_original",
    "iso3_original",
    "annual_carried_monthly",
    "source_dataset",
    "source_file",
    "source_sheet_name",
}
WIDE_SOURCE_NAMES = {  # Keep very wide sources out of the base table.
    "clean_whs2026_country_month.csv",
    "clean_worldriskindex_country_month.csv",
}


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
        "input_rows": len(raw),  # Record source row count after geographic validation.
        "feature_columns": len(feature_columns),  # Record feature column count from this source.
        **validation,  # Include source-level key validation.
    }
    return prepared, report, catalog  # Return the prepared frame, source report, and column catalog.


def _outer_join(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:  # Merge two prepared feature frames.
    """Outer-join two country-month feature tables on the modeling key."""  # Explain merge behavior.
    return left.merge(right, on=KEY_COLUMNS, how="outer", validate="one_to_one")  # Preserve partial coverage and prevent duplicate-key blowups.


def assemble_feature_table(clean_dir: Path = CLEAN_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # Main public entry point.
    """Assemble normal-width cleaned outputs into a memory-safe base feature table."""  # Explain function contract.
    ensure_output_directories()  # Create generated output folders only.
    clean_paths = sorted(clean_dir.glob("*.csv"))  # Find all cleaned feature outputs from Step 03c/03b.
    if not clean_paths:  # Stop clearly when cleaning has not been run.
        raise FileNotFoundError(f"No cleaned CSV files found in {clean_dir}")  # Raise a visible path-specific error.

    base_paths = [path for path in clean_paths if path.name not in WIDE_SOURCE_NAMES]  # Select normal-width sources.
    wide_paths = [path for path in clean_paths if path.name in WIDE_SOURCE_NAMES]  # Defer wide sources.

    prepared_frames: list[pd.DataFrame] = []  # Collect non-empty source frames ready for joining.
    report_rows: list[dict[str, object]] = []  # Collect source-level report records.
    catalog_frames: list[pd.DataFrame] = []  # Collect feature provenance catalog frames.

    for path in base_paths:  # Process normal-width sources only.
        prepared, report, catalog = _prepare_source_frame(path)  # Validate and rename one source table.
        if prepared.empty:  # Keep zero-coverage sources from creating all-null Gold columns.
            report_rows.append({**report, "assembly_role": "zero_coverage_excluded", "included_in_base_table": False})  # Record explicit exclusion.
            continue  # Move to the next source without adding its columns to Gold.
        prepared_frames.append(prepared)  # Keep the prepared frame for assembly.
        report_rows.append({**report, "assembly_role": "base_feature_table", "included_in_base_table": True})  # Mark as included.
        catalog_frames.append(catalog)  # Keep source-column provenance only for assembled features.

    if not prepared_frames:  # Stop clearly if every source was deferred or empty.
        raise FileNotFoundError(f"No non-empty base feature CSV files found in {clean_dir}")  # Raise a visible path-specific error.

    assembled = reduce(_outer_join, prepared_frames)  # Combine normal-width sources into one base table.
    assembled = assembled.sort_values(KEY_COLUMNS).reset_index(drop=True)  # Make output row order deterministic.
    assembled_validation = _validate_key(assembled)  # Validate final assembled key uniqueness.
    if assembled_validation["key_unique"] is not True:  # Block writes when Gold does not preserve one-to-one keys.
        raise ValueError(f"Gold assembly key validation failed: {assembled_validation}")  # Stop instead of writing ambiguous output.

    assembled_path = MODEL_FEATURES_DIR / BASE_OUTPUT_NAME  # Build the base output path.
    assembled.to_csv(assembled_path, index=False)  # Write the base feature table.

    for path in wide_paths:  # Record deferred wide sources without loading them into memory.
        header = pd.read_csv(path, nrows=0).columns.tolist()  # Read only the header to avoid memory pressure.
        report_rows.append({  # Store deferred-source metadata.
            "source_file": path.name,
            "input_rows": None,
            "feature_columns": max(len([column for column in header if column not in KEY_COLUMNS and column not in NON_FEATURE_COLUMNS]), 0),
            "key_unique": None,
            "duplicate_key_rows": None,
            "missing_key_columns": "",
            "assembly_role": "wide_feature_block_later",
            "included_in_base_table": False,
        })

    report = pd.DataFrame(report_rows)  # Build the source-level assembly report.
    report["assembled_output_name"] = BASE_OUTPUT_NAME  # Record the final output table name.
    report["assembled_rows"] = len(assembled)  # Record final assembled row count.
    report["assembled_columns"] = len(assembled.columns)  # Record final assembled column count.
    report["assembled_key_unique"] = assembled_validation["key_unique"]  # Record final key uniqueness.
    report["assembled_duplicate_key_rows"] = assembled_validation["duplicate_key_rows"]  # Record final duplicate key rows.
    report["assembled_missing_key_columns"] = assembled_validation["missing_key_columns"]  # Record final missing key columns.

    catalog = pd.concat(catalog_frames, ignore_index=True, sort=False) if catalog_frames else pd.DataFrame(columns=["source_file", "original_column", "assembled_column"])  # Build feature catalog.
    report.to_csv(FEATURE_REPORTS_DIR / ASSEMBLY_REPORT_NAME, index=False)  # Write source-level assembly report.
    catalog.to_csv(FEATURE_REPORTS_DIR / COLUMN_CATALOG_NAME, index=False)  # Write feature provenance catalog.

    print(f"Base sources merged: {len(prepared_frames)}")  # Show non-empty base source count.
    print(f"Zero-coverage sources excluded: {[row['source_file'] for row in report_rows if row.get('assembly_role') == 'zero_coverage_excluded']}")  # Show empty-source exclusions.
    print(f"Wide sources deferred: {[path.name for path in wide_paths]}")  # Show deferred source names.
    print(f"Base feature rows: {len(assembled):,}")  # Show assembled row count.
    print(f"Base feature columns: {len(assembled.columns):,}")  # Show assembled column count.
    print(f"Wrote: {assembled_path}")  # Show written output path.

    return assembled, report, catalog  # Return outputs for notebooks and tests.
