"""Read-only, chunked inventory checks for existing Silver CSV files."""  # Describe this module's role.

from __future__ import annotations  # Permit modern type annotations consistently across Colab runtimes.

import csv  # Read CSV headers without loading full datasets into memory.
from pathlib import Path  # Represent input and output paths safely.
from typing import Iterable  # Describe functions that accept collections of column names.

import pandas as pd  # Read large CSV files in manageable chunks and create reports.

LFS_SIGNATURE = "version https://git-lfs.github.com/spec/v1"  # Identify a Git LFS pointer instead of real CSV content.
METADATA_STEMS = {"hdx_colab_download_manifest__country_month_year", "worldriskindex_meta__country_month_year"}  # Keep known metadata out of feature processing.
COUNTRY_CANDIDATES = ("iso3", "iso_code", "country_code", "country", "country_name", "location")  # List common country-key column names in priority order.
YEAR_CANDIDATES = ("year", "observation_year", "reference_year", "target_year")  # List common year column names in priority order.
MONTH_CANDIDATES = ("month", "month_number", "observation_month", "reference_month", "target_month")  # List common month column names in priority order.
DATE_CANDIDATES = ("date", "event_date", "observation_date", "reference_date", "target_date", "forecast_target_month")  # List common date column names in priority order.


def classify_dataset(file_path: Path) -> str:  # Separate feature candidates from known metadata files.
    """Return the initial downstream role assigned to a Silver CSV."""  # Define the classification result.
    return "metadata" if file_path.stem in METADATA_STEMS else "feature_candidate"  # Apply the explicit registry rule.


def is_lfs_pointer(file_path: Path) -> bool:  # Detect incomplete Git LFS downloads before pandas runs.
    """Return True when a path contains a Git LFS pointer rather than CSV bytes."""  # Define the boolean check.
    with file_path.open("r", encoding="utf-8", errors="replace") as file_handle:  # Read a tiny text prefix safely.
        return file_handle.readline().strip() == LFS_SIGNATURE  # Compare the first line with the documented pointer signature.


def read_header(file_path: Path) -> list[str]:  # Read only the CSV header row.
    """Return normalized column names without reading data rows."""  # Define the lightweight schema read.
    with file_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file_handle:  # Open common UTF-8 CSV variants safely.
        return [column.strip() for column in next(csv.reader(file_handle))]  # Remove accidental whitespace from header names.


def choose_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:  # Match likely semantic columns without case sensitivity.
    """Return the first candidate column found, preserving its original spelling."""  # Define deterministic candidate matching.
    normalized_columns = {column.strip().lower(): column for column in columns}  # Map normalized names back to source names.
    return next((normalized_columns[candidate] for candidate in candidates if candidate in normalized_columns), None)  # Choose the first available candidate.


def _non_null_minimum(current_value: object, new_values: pd.Series) -> object:  # Update a running minimum while ignoring missing values.
    """Combine a stored minimum with one chunk's non-null values."""  # Define the aggregation helper.
    clean_values = new_values.dropna()  # Exclude missing values from minimum calculations.
    if clean_values.empty:  # Handle chunks with no usable values.
        return current_value  # Preserve the previously observed minimum.
    chunk_minimum = clean_values.min()  # Calculate the current chunk's minimum.
    return chunk_minimum if current_value is None else min(current_value, chunk_minimum)  # Combine chunk and running minima.


def _non_null_maximum(current_value: object, new_values: pd.Series) -> object:  # Update a running maximum while ignoring missing values.
    """Combine a stored maximum with one chunk's non-null values."""  # Define the aggregation helper.
    clean_values = new_values.dropna()  # Exclude missing values from maximum calculations.
    if clean_values.empty:  # Handle chunks with no usable values.
        return current_value  # Preserve the previously observed maximum.
    chunk_maximum = clean_values.max()  # Calculate the current chunk's maximum.
    return chunk_maximum if current_value is None else max(current_value, chunk_maximum)  # Combine chunk and running maxima.


def inspect_csv(file_path: Path, chunk_size: int = 100_000) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:  # Inspect one CSV with bounded memory use.
    """Return file, column, and finding records for one Silver CSV."""  # Define the three report outputs.
    file_record: dict[str, object] = {"file_name": file_path.name, "dataset_role": classify_dataset(file_path), "file_bytes": file_path.stat().st_size}  # Start the file-level report.
    column_records: list[dict[str, object]] = []  # Collect one summary record per column.
    findings: list[dict[str, object]] = []  # Collect human-readable quality conditions.
    if is_lfs_pointer(file_path):  # Stop early when the repository clone lacks real LFS objects.
        file_record.update({"read_status": "lfs_pointer", "row_count": None, "column_count": None, "key_unique": None})  # Record the blocked read clearly.
        findings.append({"file_name": file_path.name, "severity": "error", "check": "git_lfs", "message": "Git LFS pointer found; run git lfs pull before inventory."})  # Explain the recovery action.
        return file_record, column_records, findings  # Avoid sending pointer text to pandas.
    columns = read_header(file_path)  # Read the schema before streaming rows.
    country_column = choose_column(columns, COUNTRY_CANDIDATES)  # Infer the most likely country field.
    year_column = choose_column(columns, YEAR_CANDIDATES)  # Infer the most likely year field.
    month_column = choose_column(columns, MONTH_CANDIDATES)  # Infer the most likely month field.
    date_column = choose_column(columns, DATE_CANDIDATES)  # Infer a fallback date field.
    key_columns = [column for column in (country_column, year_column, month_column) if column is not None]  # Build the explicit country-year-month key when available.
    total_rows = 0  # Count all data rows across chunks.
    null_counts = {column: 0 for column in columns}  # Accumulate missing values per column.
    observed_dtypes = {column: set() for column in columns}  # Capture pandas data types seen across chunks.
    duplicate_key_rows = 0  # Count repeated key rows conservatively across the full file.
    seen_keys: set[tuple[object, ...]] = set()  # Track keys so duplicates spanning chunks are detected.
    minimum_year = None  # Store the earliest numeric year.
    maximum_year = None  # Store the latest numeric year.
    minimum_date = None  # Store the earliest parsed date.
    maximum_date = None  # Store the latest parsed date.
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):  # Stream the dataset to limit peak memory use.
        chunk.columns = [str(column).strip() for column in chunk.columns]  # Normalize the loaded headers consistently.
        total_rows += len(chunk)  # Add this chunk's rows to the file total.
        for column in columns:  # Update statistics for every source column.
            null_counts[column] += int(chunk[column].isna().sum())  # Add this chunk's missing-value count.
            observed_dtypes[column].add(str(chunk[column].dtype))  # Record the inferred type observed in this chunk.
        if len(key_columns) == 3:  # Test uniqueness only when country, year, and month all exist.
            for key in chunk[key_columns].itertuples(index=False, name=None):  # Stream keys without creating a second large DataFrame.
                normalized_key = tuple(None if pd.isna(value) else value for value in key)  # Make missing key components hashable and consistent.
                duplicate_key_rows += int(normalized_key in seen_keys)  # Count each repeated row after its first occurrence.
                seen_keys.add(normalized_key)  # Retain the key for later chunks.
        if year_column is not None:  # Calculate coverage from an explicit year field when available.
            numeric_years = pd.to_numeric(chunk[year_column], errors="coerce")  # Convert invalid year values to missing values.
            minimum_year = _non_null_minimum(minimum_year, numeric_years)  # Update the earliest observed year.
            maximum_year = _non_null_maximum(maximum_year, numeric_years)  # Update the latest observed year.
        if date_column is not None:  # Calculate coverage from a date field when available.
            parsed_dates = pd.to_datetime(chunk[date_column], errors="coerce", utc=True)  # Parse mixed date values consistently.
            minimum_date = _non_null_minimum(minimum_date, parsed_dates)  # Update the earliest observed date.
            maximum_date = _non_null_maximum(maximum_date, parsed_dates)  # Update the latest observed date.
    file_record.update({"read_status": "ok", "row_count": total_rows, "column_count": len(columns), "country_column": country_column, "year_column": year_column, "month_column": month_column, "date_column": date_column, "minimum_year": minimum_year, "maximum_year": maximum_year, "minimum_date": minimum_date, "maximum_date": maximum_date, "duplicate_key_rows": duplicate_key_rows if len(key_columns) == 3 else None, "key_unique": duplicate_key_rows == 0 if len(key_columns) == 3 else None})  # Complete the file summary.
    for column in columns:  # Convert accumulated column statistics into report rows.
        missing_count = null_counts[column]  # Retrieve the final missing-value count.
        column_records.append({"file_name": file_path.name, "column_name": column, "observed_dtypes": "|".join(sorted(observed_dtypes[column])), "missing_count": missing_count, "missing_percent": round((missing_count / total_rows) * 100, 4) if total_rows else None})  # Store a concise column profile.
    if file_record["dataset_role"] == "feature_candidate" and country_column is None:  # Flag feature tables with no inferred country field.
        findings.append({"file_name": file_path.name, "severity": "error", "check": "country_key", "message": "No recognized country column was found."})  # Record the missing key condition.
    if file_record["dataset_role"] == "feature_candidate" and year_column is None and date_column is None:  # Flag feature tables with no usable time field.
        findings.append({"file_name": file_path.name, "severity": "error", "check": "time_key", "message": "No recognized year or date column was found."})  # Record the missing time condition.
    if file_record["dataset_role"] == "feature_candidate" and len(key_columns) != 3:  # Explain why grain uniqueness could not be tested.
        findings.append({"file_name": file_path.name, "severity": "warning", "check": "country_month_year_grain", "message": "Country-year-month key was not fully inferred; inspect source-specific time columns."})  # Request source-specific review.
    if duplicate_key_rows > 0:  # Flag datasets that violate one-row-per-key expectations.
        findings.append({"file_name": file_path.name, "severity": "warning", "check": "country_month_year_grain", "message": f"Found {duplicate_key_rows:,} repeated country-year-month key rows; aggregation or additional dimensions may be required."})  # Quantify the grain problem.
    return file_record, column_records, findings  # Return the complete inspection results.


def inventory_silver_directory(silver_dir: Path, report_dir: Path, chunk_size: int = 100_000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:  # Inspect every Silver CSV and write three reports.
    """Inventory all Silver CSV files without modifying the source directory."""  # Define the stage-level operation.
    csv_files = sorted(silver_dir.glob("*.csv"))  # Discover CSV inputs in a reproducible order.
    if not silver_dir.exists():  # Fail clearly when the expected repository path is absent.
        raise FileNotFoundError(f"Silver directory does not exist: {silver_dir}")  # Provide the exact missing path.
    if not csv_files:  # Fail clearly when the folder exists but contains no CSV inputs.
        raise FileNotFoundError(f"No CSV files found in: {silver_dir}")  # Distinguish an empty folder from a missing folder.
    report_dir.mkdir(parents=True, exist_ok=True)  # Create only the generated report destination.
    file_records: list[dict[str, object]] = []  # Collect file-level results across inputs.
    column_records: list[dict[str, object]] = []  # Collect column-level results across inputs.
    finding_records: list[dict[str, object]] = []  # Collect quality findings across inputs.
    for file_path in csv_files:  # Inspect each source independently for easier troubleshooting.
        try:  # Preserve other inventory results if one source is unreadable.
            file_record, file_columns, file_findings = inspect_csv(file_path=file_path, chunk_size=chunk_size)  # Run the bounded-memory inspection.
        except Exception as error:  # Convert a source-specific read failure into a reportable finding.
            file_record = {"file_name": file_path.name, "dataset_role": classify_dataset(file_path), "file_bytes": file_path.stat().st_size, "read_status": "error", "row_count": None, "column_count": None, "key_unique": None}  # Retain basic file facts despite failure.
            file_columns = []  # Record no column details when parsing fails.
            file_findings = [{"file_name": file_path.name, "severity": "error", "check": "csv_read", "message": f"{type(error).__name__}: {error}"}]  # Preserve the exception type and message.
        file_records.append(file_record)  # Add the file summary to the combined report.
        column_records.extend(file_columns)  # Add the column summaries to the combined report.
        finding_records.extend(file_findings)  # Add the source findings to the combined report.
    files_frame = pd.DataFrame(file_records)  # Build the file-level inventory table.
    columns_frame = pd.DataFrame(column_records)  # Build the column-level inventory table.
    findings_frame = pd.DataFrame(finding_records, columns=["file_name", "severity", "check", "message"])  # Build a consistently shaped findings table even when empty.
    files_frame.to_csv(report_dir / "silver_inventory.csv", index=False)  # Save file-level structure, coverage, and grain results.
    columns_frame.to_csv(report_dir / "silver_column_inventory.csv", index=False)  # Save types and missingness by column.
    findings_frame.to_csv(report_dir / "silver_quality_findings.csv", index=False)  # Save actionable warnings and errors.
    return files_frame, columns_frame, findings_frame  # Return reports for immediate notebook inspection.
