"""Central repository paths for the public-feature pipeline."""  # Describe this module's single responsibility.

from pathlib import Path  # Use operating-system-independent filesystem paths.

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Resolve the repository root from the src folder.
SILVER_DIR = PROJECT_ROOT / "country_month_year_outputs" / "silver_country_month_year"  # Point to the existing read-only Silver inputs.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Point to the root for generated pipeline products.
EDA_REPORTS_DIR = OUTPUTS_DIR / "eda_reports"  # Store inventory and later EDA quality reports together.


def ensure_output_directories() -> None:  # Create only writable output folders needed by this stage.
    """Create generated-output directories without modifying Silver inputs."""  # Explain the function's safety boundary.
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the report folder when it does not already exist.
