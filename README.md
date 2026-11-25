# Electricity Consumption Forecasting Using Machine Learning

An interactive Machine Learning web application built with **Streamlit**, **Scikit-learn**, **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn** to analyze and forecast household appliance electricity consumption (in Wh) using the **UCI Appliances Energy Prediction** dataset (Dataset ID: 374).

## 1. Project Objectives

- Analyze historical 10-minute household electricity consumption and environmental sensor data.
- Visualize consumption distributions, hourly load curves, feature correlations, and actual vs. predicted forecasts.
- Train and compare **Linear Regression**, **Decision Tree Regression**, and **Gradient Boosting Regression** using a strict **chronological train-test split** with zero data leakage.
- Evaluate models fairly using **MAE**, **RMSE**, and **R²**.
- Provide a real-time interactive prediction system for appliance energy consumption in Watt-hours (Wh).

## 2. Dataset Information

- **Dataset:** Appliances Energy Prediction
- **Source:** [UCI Machine Learning Repository (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- **Records:** 19,735 observations recorded at 10-minute intervals (Jan 11, 2016 – May 27, 2016)
- **Target Variable:** `Appliances` (energy consumption in Wh)
- **Key Features:** Lighting consumption (`lights`), indoor temperatures & humidities (`T1–T9`, `RH_1–RH_9`), outdoor weather (`T_out`, `RH_out`, `Windspeed`, `Press_mm_hg`, `Visibility`, `Tdewpoint`), and temporal features (`hour`, `dayofweek`).

## 3. Application Pages

1. **Home Dashboard** — Project introduction, dataset KPIs, chronological split overview, and model performance summary.
2. **Dataset Explorer** — Interactive preview, dimensions, missing value audit, schema descriptions, and statistical summary with CSV exports.
3. **Data Visualization** — Dynamic electricity distribution, hourly usage patterns (weekday vs. weekend), interactive correlation heatmaps, and actual vs. predicted time-series overlays.
4. **Machine Learning Models** — Train and tune Linear Regression, Decision Tree Regression, and Gradient Boosting Regression with chronological validation, MAE/RMSE/R² comparison, best-model spotlight, and feature drivers.
5. **Prediction System** — Interactive scenario presets and input form for temperature, humidity, lighting, wind speed, hour of day, and day of week with live Wh forecasts across all models.
6. **About Project** — Problem statement, objectives, dataset details, ML workflow, and conclusions/limitations.

## 4. Project Structure

```text
Electricity-Consumption-Forecasting/
│
├── app.py                                    # Multi-page Streamlit dashboard
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
├── Electricity_Consumption_Forecasting.ipynb # Original Colab research notebook
├── .streamlit/
│   └── config.toml                           # Dark navy & electric blue theme config
├── graphs/
│   ├── consumption_distribution.png
│   ├── hourly_consumption.png
│   ├── correlation_heatmap.png
│   └── actual_vs_predicted.png
└── src/
    ├── __init__.py
    └── model.py                              # Data loading, leakage-free pipelines & evaluation
```

## 5. How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/mayuresh486/Electricity-Consumption-Forecasting.git
cd Electricity-Consumption-Forecasting

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py
```

## 6. Deploying to Streamlit Community Cloud

1. Push this repository (including `app.py`, `src/model.py`, `requirements.txt`, `.streamlit/config.toml`, and `graphs/`) to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select `mayuresh486/Electricity-Consumption-Forecasting`, branch `main`, and main file path `app.py`.
4. Click **Deploy**. Streamlit Community Cloud will automatically install `requirements.txt` and launch the live app.
