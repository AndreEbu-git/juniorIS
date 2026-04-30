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

## Installation
### 1. Clone the repository
`git clone <your-repository-url>
cd juniorIS`
### 2. Create and activate a virtual environment
`python -m venv .venv`

Windows PowerShell:

`.venv\Scripts\Activate.ps1`
### 3. Install dependencies
`pip install streamlit pandas numpy scikit-learn xgboost plotly openai python-dotenv`

Running the Application
From the project root, run:

`streamlit run src/business_forecaster/app.py`

Then open the local Streamlit URL shown in the terminal.

Optional OpenAI Setup
To enable AI-generated natural-language explanations, create a `.env` file in:

`src/business_forecaster/.env`
Add:

`OPENAI_API_KEY=your_api_key_here`

If the API key is missing, the explanation feature will still work using the built-in fallback explanation.

### How to Use
Launch the app with Streamlit.

Upload a business dataset.

Review validation feedback and preprocessing notes.

Go to the Forecasting tab.

Select:
a target variable
one or more predictor variables
your preferred test set size
forecast periods
Train the model.
Review forecast metrics, charts, and feature importance.
Use the Scenario Analysis panel to test alternative assumptions.
Generate an AI explanation if OpenAI access is configured.


### Known Limitations
Forecast quality depends heavily on the structure and quality of the uploaded dataset.

The model currently works best with tabular time-series business data.

Scenario analysis is limited to selected numeric driver columns.

AI explanations require valid API access and available quota when using OpenAI.

There is no live external API data ingestion yet.

### Future Work
Potential future improvements include:

Live business or market data ingestion.

Additional forecasting model options.

More advanced explanation methods such as SHAP.

Formal usability testing with SME users.

Improved export and reporting features.

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

