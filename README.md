# Dynamic Business Performance Forecaster: ML-Driven Predictions, Factor Prioritization, and Interactive Real-Time Scenario Planning with Natural Language Explanations

Dynamic Business Performance Forecaster is an interactive Streamlit application for business-focused time series forecasting, feature importance analysis, scenario planning, and natural-language explanation of results. The project is designed to help small and medium-sized enterprises explore predictive analytics without requiring deep machine learning expertise.

## Overview

This system allows a user to:

- Upload business datasets in CSV or Excel format
- Automatically preprocess date fields and numeric-like text fields
- Train an XGBoost-based forecasting model
- View forecast accuracy metrics and future predictions
- Inspect feature importance to understand model drivers
- Run what-if scenario analysis on selected numeric drivers
- Generate natural-language explanations of forecasts and scenarios

The application emphasizes usability, interpretability, and flexibility across different business datasets.

## Features

### 1. Dataset Upload and Preprocessing
Users can upload `.csv`, `.xlsx`, or `.xls` files directly into the dashboard.

The preprocessing pipeline supports:
- Automatic detection and conversion of date columns
- Conversion of numeric-like text values such as currency strings
- Basic dataset validation and data quality warnings

### 2. Forecasting
The forecasting engine is built around XGBoost, with a fallback to `GradientBoostingRegressor` if XGBoost is unavailable.

Users can configure:
- Target variable
- Predictor variables
- Test set size
- Forecast horizon
- Number of lag features
- Whether to include time-based features
- Whether to scale features

### 3. Forecast Interpretation
After training, the system displays:
- RMSE
- MAE
- R²
- MAPE
- Historical predictions vs. actual values
- Future forecast tables and charts
- Feature importance rankings

### 4. Scenario Analysis
The scenario module allows users to test alternative futures by changing selected numeric drivers.

Scenario input supports:
- Multiplier-based changes
- Direct-value changes

This makes it possible to compare a base forecast with user-defined business assumptions.

### 5. Natural Language Explanations
The system includes an optional LLM-based explanation layer.

If an OpenAI API key is available, the app can generate AI-written forecast explanations.
If no key is available, the app falls back to a built-in deterministic explanation.

## Project Structure

```text
juniorIS/
├── src/
│   └── business_forecaster/
│       ├── app.py
│       ├── dataloader.py
│       ├── forecaster.py
│       ├── llm_explainer.py
│       ├── scenario_simulator.py
│       ├── test_forecaster.py
│       └── __init__.py
├── main.tex
├── bibliography.bib
├── Chocolate Sales.csv
└── README.md



## Feature Calendar

| **Issue** | **Due date** | |
| --------- | ------------ | -- |
| [Example issue description with link](https://github.com/hmm34/example-annotated-bibliography/issues/1) | 9/1/25 | |
|Load and process business datasets(https://github.com/AndreEbu-git/juniorIS/issues/1)| 16/2/26 ||
|Design user inputs for forecasting variables(https://github.com/AndreEbu-git/juniorIS/issues/2)| 23/2/26 ||
|Implement a basic ML forecasting model(https://github.com/AndreEbu-git/juniorIS/issues/3)| 2/3/26 ||
|Integrate feature importance calculation(https://github.com/AndreEbu-git/juniorIS/issues/3)| 7/3/26 ||
|Build interactive dashboard interface(https://github.com/AndreEbu-git/juniorIS/issues/5)| 14/3/26 ||
|Add real time data pulling(https://github.com/AndreEbu-git/juniorIS/issues/6)| 21/3/26 ||
|Generate scenario simulations(https://github.com/AndreEbu-git/juniorIS/issues/7)| 28/3/26 ||
|Incorporate GenAI for explanations(https://github.com/AndreEbu-git/juniorIS/issues/8)| 4/4/26 ||
|Display results with visualizations(https://github.com/AndreEbu-git/juniorIS/issues/9)| 11/4/26 ||
|Validate model accuracy(https://github.com/AndreEbu-git/juniorIS/issues/10)| 14/3/26 ||

