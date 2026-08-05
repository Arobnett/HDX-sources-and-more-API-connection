from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def clean_name(value: str) -> str:
    # Normalize column and file labels into stable lowercase snake_case names.
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip())).strip("_").lower()


def file_sha256(path: Path) -> str:
    # Hash file content so Bronze metadata can detect changed source artifacts.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_month_column(values: pd.Series) -> pd.DataFrame:
    # Convert text, integer, and date-like month values with vectorized pandas operations.
    raw = values.copy()

    # Preserve numeric month values when they are in calendar range.
    numeric_month = pd.to_numeric(raw, errors="coerce")
    valid_numeric = numeric_month.between(1, 12)

    # Parse only values that look date-like so text month names avoid noisy warnings.
    date_like = raw.astype("string").str.contains(r"[-/]", na=False) & ~valid_numeric
    parsed_date = pd.to_datetime(raw.where(date_like), errors="coerce")
    date_month = parsed_date.dt.month

    # Match month strings such as January, Jan, or JAN.
    month_key = raw.astype("string").str.strip().str.lower()
    text_month = month_key.map(MONTH_LOOKUP)
    text_month = text_month.fillna(month_key.str[:3].map(MONTH_LOOKUP))

    # Prefer numeric values, then dates, then text names.
    month_number = numeric_month.where(valid_numeric).fillna(date_month).fillna(text_month).astype("Int64")

    # Create readable month names for review and Tableau labels.
    month_name_map = {number: datetime(2000, number, 1).strftime("%B") for number in range(1, 13)}
    month_name = month_number.map(month_name_map).astype("string")

    # Return both normalized fields.
    return pd.DataFrame({"month": month_number, "month_name": month_name})


def ensure_country_month_year(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize all column names before source-specific mappings are applied.
    df = df.copy()
    df.columns = [clean_name(column) for column in df.columns]

    # Map common country columns into one canonical country field.
    if "country" not in df.columns:
        for candidate in ["countryname", "wri_country", "name", "location"]:
            if candidate in df.columns:
                df["country"] = df[candidate]
                break

    # Map common ISO3 columns into one canonical iso3 field.
    if "iso3" not in df.columns:
        for candidate in ["iso3_code", "isoab", "locationcode", "country_code", "iso"]:
            if candidate in df.columns:
                df["iso3"] = df[candidate]
                break

    # Map common year columns into one canonical year field.
    if "year" not in df.columns:
        for candidate in ["gnayear"]:
            if candidate in df.columns:
                df["year"] = df[candidate]
                break

    # Derive year and month from date fields when explicit fields are absent.
    for date_column in ["date_reported", "from_date", "to_date"]:
        if date_column in df.columns and ("year" not in df.columns or "month" not in df.columns):
            parsed = pd.to_datetime(df[date_column], errors="coerce", utc=True)
            if "year" not in df.columns:
                df["year"] = parsed.dt.year
            if "month" not in df.columns:
                df["month"] = parsed.dt.month
            break

    # Derive month name and month number from existing month values.
    if "month" in df.columns:
        month_parts = normalize_month_column(df["month"])
        df["month"] = month_parts["month"]
        df["month_name"] = month_parts["month_name"]

    # Create a first-day month date for time-series joins.
    if "year" in df.columns and "month" in df.columns:
        year_numeric = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        month_numeric = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        df["country_month"] = pd.to_datetime(
            {
                "year": year_numeric,
                "month": month_numeric,
                "day": 1,
            },
            errors="coerce",
        ).dt.date

    # Keep country-month-year columns first when present.
    preferred = [
        "country",
        "iso3",
        "year",
        "month",
        "month_name",
        "country_month",
        "admin1",
        "admin2",
        "admin1_pcode",
        "admin2_pcode",
        "events",
        "fatalities",
    ]
    ordered = [column for column in preferred if column in df.columns]
    ordered += [column for column in df.columns if column not in ordered]
    return df[ordered]


def normalize_acled_workbook(path: Path, raw_csv_dir: Path, silver_dir: Path) -> list[dict]:
    # Convert each ACLED workbook sheet to raw CSV and append data sheets into Silver.
    rows = []
    combined = []
    workbook_id = clean_name(path.stem)
    excel_file = pd.ExcelFile(path)

    for sheet_name in excel_file.sheet_names:
        sheet_id = clean_name(sheet_name)
        df = pd.read_excel(path, sheet_name=sheet_name)
        if df.empty:
            continue

        # Save the untouched sheet payload as a raw converted CSV.
        raw_csv_path = raw_csv_dir / f"{workbook_id}__{sheet_id}.csv"
        df.to_csv(raw_csv_path, index=False)

        # Skip terms/license sheets during country-month-year normalization.
        if sheet_id in {"tou", "terms", "licensing"}:
            rows.append({"source_file": path.name, "sheet_name": sheet_name, "raw_csv": str(raw_csv_path), "silver_csv": None, "rows": len(df), "status": "raw_only"})
            continue

        # Normalize schema and preserve source sheet context.
        normalized = ensure_country_month_year(df)
        normalized.insert(0, "source_sheet_name", sheet_name)
        normalized.insert(0, "source_file", path.name)
        normalized.insert(0, "source_dataset", workbook_id)

        # Convert ACLED HRP split sheets into modeling-friendly status fields.
        if sheet_id == "non_hrp":
            normalized.insert(3, "hrp_status", "non_hrp")
        elif sheet_id.startswith("hrp_"):
            normalized.insert(3, "hrp_status", "hrp")
        else:
            normalized.insert(3, "hrp_status", pd.NA)

        combined.append(normalized)
        rows.append({"source_file": path.name, "sheet_name": sheet_name, "raw_csv": str(raw_csv_path), "silver_csv": None, "rows": len(df), "status": "normalized_pending"})

    if combined:
        silver = pd.concat(combined, ignore_index=True)
        silver_csv_path = silver_dir / f"{workbook_id}__country_month_year.csv"
        silver.to_csv(silver_csv_path, index=False)
        for row in rows:
            if row["status"] == "normalized_pending":
                row["silver_csv"] = str(silver_csv_path)
                row["status"] = "normalized"

    return rows


def normalize_flat_file(path: Path, raw_csv_dir: Path, silver_dir: Path) -> dict:
    # Copy CSV sources into raw CSV and create a country-month-year normalized variant.
    dataset_id = clean_name(path.stem)
    raw_csv_path = raw_csv_dir / f"{dataset_id}.csv"
    df = pd.read_csv(path)
    df.to_csv(raw_csv_path, index=False)

    # Normalize known country/time columns for modeling joins.
    normalized = ensure_country_month_year(df)
    normalized.insert(0, "source_file", path.name)
    normalized.insert(0, "source_dataset", dataset_id)

    # Write the normalized Silver CSV.
    silver_csv_path = silver_dir / f"{dataset_id}__country_month_year.csv"
    normalized.to_csv(silver_csv_path, index=False)

    return {"source_file": path.name, "sheet_name": None, "raw_csv": str(raw_csv_path), "silver_csv": str(silver_csv_path), "rows": len(df), "status": "normalized"}


def main() -> None:
    # Read command-line folders for Colab, Databricks, or local execution.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="upload")
    parser.add_argument("--output-dir", default="outputs/country_month_year")
    args = parser.parse_args()

    # Create output folders for raw converted CSVs, Silver CSVs, and manifests.
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_csv_dir = output_dir / "raw_converted_csv"
    silver_dir = output_dir / "silver_country_month_year"
    raw_csv_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Track every processed file for auditability.
    manifest_rows = []

    # Process Excel workbooks first so each sheet becomes its own raw CSV.
    for path in sorted(input_dir.glob("*.xlsx")):
        try:
            workbook_rows = normalize_acled_workbook(path, raw_csv_dir, silver_dir)
            manifest_rows.extend(workbook_rows)
        except Exception as error:
            manifest_rows.append({"source_file": path.name, "sheet_name": None, "raw_csv": None, "silver_csv": None, "rows": None, "status": f"failed: {error}"})

    # Process existing CSVs into normalized Silver outputs.
    for path in sorted(input_dir.glob("*.csv")):
        try:
            manifest_rows.append(normalize_flat_file(path, raw_csv_dir, silver_dir))
        except Exception as error:
            manifest_rows.append({"source_file": path.name, "sheet_name": None, "raw_csv": None, "silver_csv": None, "rows": None, "status": f"failed: {error}"})

    # Build a source artifact manifest with hashes for converted files.
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = output_dir / "conversion_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Save lightweight file-level checksums for downstream Bronze metadata.
    checksum_rows = []
    for output_path in sorted(output_dir.rglob("*.csv")):
        checksum_rows.append({"path": str(output_path), "filename": output_path.name, "byte_count": output_path.stat().st_size, "content_sha256": file_sha256(output_path)})
    checksum_path = output_dir / "output_checksums.json"
    checksum_path.write_text(json.dumps(checksum_rows, indent=2), encoding="utf-8")

    # Print concise run summary for notebooks and jobs.
    print(f"Input folder: {input_dir}")
    print(f"Raw converted CSV folder: {raw_csv_dir}")
    print(f"Silver country-month-year folder: {silver_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Files written: {len(checksum_rows)}")


if __name__ == "__main__":
    main()
