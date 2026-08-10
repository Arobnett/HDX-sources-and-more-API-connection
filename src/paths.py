"""Central repository paths for the public-feature pipeline."""  # Describe this module's single responsibility.

from pathlib import Path  # Use operating-system-independent filesystem paths.

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Resolve the repository root from the src folder.
SILVER_DIR = PROJECT_ROOT / "country_month_year_outputs" / "silver_country_month_year"  # Point to the existing read-only Silver inputs.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"  # Point to the root for generated pipeline products.
EDA_REPORTS_DIR = OUTPUTS_DIR / "eda_reports"  # Store inventory and later EDA quality reports together.
CLEAN_DIR = OUTPUTS_DIR / "clean_country_month_year"  # Store model-ready country-month-year feature outputs.
CLEAN_REPORTS_DIR = OUTPUTS_DIR / "cleaning_reports"  # Store cleaning validation and reject reports.
MODEL_FEATURES_DIR = OUTPUTS_DIR / "model_features"  # Store assembled wide modeling feature tables.
FEATURE_REPORTS_DIR = OUTPUTS_DIR / "feature_reports"  # Store feature assembly validation and column catalog reports.
CONFIG_DIR = PROJECT_ROOT / "config"  # Store small version-controlled rule files.
SOURCE_CLEANING_RULES_PATH = CONFIG_DIR / "source_cleaning_rules.csv"  # Locate the 03b source-specific rules.


def ensure_output_directories() -> None:  # Create only writable output folders needed by pipeline stages.
    """Create generated-output directories without modifying Silver inputs."""  # Explain the function's safety boundary.
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the report folder when it does not already exist.
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)  # Create the cleaned feature output folder when needed.
    CLEAN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the cleaning report folder when needed.
    MODEL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)  # Create the assembled feature output folder when needed.
    FEATURE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Create the feature assembly report folder when needed.
