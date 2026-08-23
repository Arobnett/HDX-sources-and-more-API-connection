"""Strict geographic-key validation for cleaned Silver country-month files."""  # Describe module purpose.

from __future__ import annotations  # Keep type hints forward-compatible.

from pathlib import Path  # Work with repository paths.

import pandas as pd  # Read and validate tabular Silver outputs.

try:  # Use the ISO reference package already used by the public pipeline.
    import pycountry  # Resolve canonical ISO country codes and names.
except ImportError:  # Fail clearly when strict validation cannot be performed.
    pycountry = None  # Record the unavailable dependency.


APPROVED_ISO3_EXCEPTIONS = {"XKX": "Kosovo"}  # Preserve explicitly approved non-ISO exceptions.
KEY_COLUMNS = ["iso3", "country", "year", "month"]  # Define the modeling join grain.


def _country_record(iso3: str):  # Resolve one ISO3 code against the reference table.
    """Return a pycountry record for a normalized ISO3 code."""  # Document helper behavior.
    if pycountry is None:  # Require the reference library for strict validation.
        raise ImportError("pycountry is required for strict geographic-key validation.")  # Stop rather than guessing.
    return pycountry.countries.get(alpha_3=iso3)  # Return the ISO record or None.


def canonical_country_name(iso3: str) -> str | None:  # Resolve the canonical country label.
    """Return the canonical country name for a valid ISO3 code or approved exception."""  # Explain output.
    if iso3 in APPROVED_ISO3_EXCEPTIONS:  # Check auditable non-ISO exceptions first.
        return APPROVED_ISO3_EXCEPTIONS[iso3]  # Return the approved label.
    record = _country_record(iso3)  # Look up the ISO3 code.
    return record.name if record else None  # Return canonical name only for recognized codes.


def validate_geographic_keys(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # Validate one cleaned Silver table.
    """Canonicalize valid country keys and separate invalid rows for audit."""  # Describe validation contract.
    missing_columns = [column for column in KEY_COLUMNS if column not in frame.columns]  # Identify missing join-key fields.
    if missing_columns:  # Stop when the table cannot be validated at the modeling grain.
        raise ValueError(f"Missing geographic/modeling key columns: {missing_columns}")  # Report the schema problem.

    checked = frame.copy()  # Avoid mutating the caller-owned dataframe.
    checked["country_original"] = checked["country"]  # Preserve source country labels for lineage.
    checked["iso3"] = checked["iso3"].astype("string").str.strip().str.upper()  # Normalize ISO3 formatting.
    checked["country"] = checked["country"].astype("string").str.strip()  # Normalize country-label spacing.

    checked["canonical_country"] = checked["iso3"].map(  # Resolve every code to a canonical country label.
        lambda value: canonical_country_name(value) if isinstance(value, str) and len(value) == 3 else None
    )
    checked["valid_iso3"] = checked["canonical_country"].notna()  # Mark recognized ISO3 codes and approved exceptions.

    normalized_source_name = checked["country"].fillna("").str.casefold()  # Normalize source names for comparison.
    normalized_canonical_name = checked["canonical_country"].fillna("").str.casefold()  # Normalize canonical names for comparison.
    checked["country_name_matches_iso3"] = (  # Record whether a populated source label agrees with the code.
        checked["country"].isna()
        | checked["country"].eq("")
        | normalized_source_name.eq(normalized_canonical_name)
    )

    invalid_iso = ~checked["valid_iso3"]  # Reject unrecognized three-character codes.
    mismatch = checked["valid_iso3"] & ~checked["country_name_matches_iso3"]  # Reject code/name conflicts for audit.
    reject_mask = invalid_iso | mismatch  # Combine strict geographic-key failures.

    rejected = checked.loc[reject_mask].copy()  # Preserve failed rows separately.
    rejected["reject_reason"] = ""  # Initialize transparent reject reasons.
    rejected.loc[invalid_iso.loc[reject_mask].to_numpy(), "reject_reason"] = "invalid_iso3"  # Label invalid codes.
    rejected.loc[mismatch.loc[reject_mask].to_numpy(), "reject_reason"] = "country_iso3_mismatch"  # Label code/name conflicts.

    accepted = checked.loc[~reject_mask].copy()  # Keep only model-ready geographic keys.
    accepted["country"] = accepted["canonical_country"]  # Standardize accepted rows to canonical country names.
    accepted = accepted.drop(columns=["canonical_country", "valid_iso3", "country_name_matches_iso3"])  # Remove validation-only fields.

    duplicate_rows = int(accepted.duplicated(KEY_COLUMNS).sum())  # Check uniqueness after canonicalization.
    if duplicate_rows:  # Prevent canonicalization from creating ambiguous Gold joins.
        raise ValueError(f"Canonical geographic cleanup created {duplicate_rows} duplicate modeling keys.")  # Stop for review.

    return accepted, rejected  # Return model-ready rows and auditable rejects.


def validate_clean_directory(clean_dir: Path, report_dir: Path) -> pd.DataFrame:  # Validate every cleaned Silver CSV.
    """Rewrite cleaned Silver files with canonical keys and write geographic validation reports."""  # Explain directory runner.
    report_dir.mkdir(parents=True, exist_ok=True)  # Ensure the validation report directory exists.
    summary_rows: list[dict[str, object]] = []  # Collect one validation summary per file.
    reject_frames: list[pd.DataFrame] = []  # Collect rejected rows across sources.

    for path in sorted(clean_dir.glob("*.csv")):  # Process each cleaned feature source deterministically.
        frame = pd.read_csv(path, low_memory=False)  # Load one cleaned Silver output.
        accepted, rejected = validate_geographic_keys(frame)  # Apply strict geographic validation.
        accepted.to_csv(path, index=False)  # Replace the cleaned file with canonical model-ready keys.
        summary_rows.append({  # Record validation results for downstream review.
            "source_file": path.name,
            "input_rows": len(frame),
            "accepted_rows": len(accepted),
            "rejected_rows": len(rejected),
            "key_unique": not accepted.duplicated(KEY_COLUMNS).any(),
        })
        if not rejected.empty:  # Preserve rejected rows with their source file.
            rejected = rejected.copy()  # Avoid modifying the validator-owned frame.
            rejected.insert(0, "source_file", path.name)  # Add source lineage.
            reject_frames.append(rejected)  # Add rows to the combined reject report.

    summary = pd.DataFrame(summary_rows)  # Build the validation summary table.
    summary.to_csv(report_dir / "geographic_key_validation_summary.csv", index=False)  # Persist summary evidence.
    combined_rejects = pd.concat(reject_frames, ignore_index=True) if reject_frames else pd.DataFrame()  # Combine rejects safely.
    combined_rejects.to_csv(report_dir / "geographic_key_rejects.csv", index=False)  # Persist rejected rows for audit.
    return summary  # Return the summary for notebook display.
