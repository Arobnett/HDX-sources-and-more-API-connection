# HDX-sources-and-more-API-connections

## Run Latest Public Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arobnett/HDX-sources-and-more-API-connection/blob/main/notebooks/00_main_run_all_open_source.ipynb)

Use this Colab notebook to regenerate the latest public-source pipeline outputs and download them as a timestamped ZIP. The run includes standardized public source outputs, cleaned country-month files, the base public Gold feature table, feature assembly reports, and EDA reports.

This public workflow does not use restricted target variables or internal operational datasets. Please do not edit notebooks directly in GitHub; proposed changes should go through a [pull request](https://docs.github.com/en/pull-requests).


## Downloaded ZIP Structure

The Colab runner keeps the repo's internal paths stable, but the downloaded ZIP uses clearer layer names:

| Repo source folder | ZIP folder |
| --- | --- |
| `country_month_year_outputs/` | `01_bronze_standardized_public_sources/` |
| `outputs/clean_country_month_year/` | `02_silver_clean_country_month_year/` |
| `outputs/model_features/` | `03_gold_base_feature_table/` |
| `outputs/feature_reports/` | `04_validation_reports/` |
| `outputs/eda_reports/` | `05_eda_reports/` |
| `outputs/eda_visuals/` | `06_eda_visuals/` |

## Machine Learning Life-cycle Workflow

Reference: [GeeksforGeeks Machine Learning Lifecycle](https://www.geeksforgeeks.org/machine-learning/machine-learning-lifecycle/)

1. Problem Definition
2. Data Collection
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Engineering and Selection
6. Model Selection
7. Model Training
8. Model Evaluation and Tuning
9. Model Deployment
10. Model Monitoring and Maintenance

## Repository Structure Reference

[src layout vs flat layout - Python Packaging User Guide](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout)
