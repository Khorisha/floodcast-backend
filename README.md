# FloodCast Backend

Flask REST API that serves flood predictions for Port Louis, Mauritius. It runs a TFLite GRU model trained on ERA5 reanalysis data (2010-2024), fetches live weather from Open-Meteo, and applies GIS-based zone risk multipliers.

---

## Live Deployment

The backend is hosted on Render and is called automatically by the frontend. No manual setup is needed to use the app. If you only want to browse the app, visit:

    https://khorisha.github.io/floodcast-frontend/

The instructions below are for running the backend locally or redeploying it yourself.

---

## Requirements

- Python 3.10 or 3.11
- pip

---

## Local Setup

1. Navigate to the backend folder:

        cd backend

2. Create and activate a virtual environment:

        python -m venv venv

        # Windows
        venv\Scripts\activate

        # Mac / Linux
        source venv/bin/activate

3. Install dependencies:

        pip install -r requirements.txt

4. Start the server:

        python app.py

   The API will be available at http://localhost:5000

---

## Environment Variables

No environment variables are required for local use. On Render, the following are set through the dashboard:

| Variable | Purpose |
|---|---|
| PORT | Port the server listens on (Render sets this automatically) |

---

## Project Structure

```
backend/
    app.py                  Main Flask application and all API routes
    requirements.txt        Python dependencies
    models/
        final_model.tflite  Trained GRU flood prediction model
        scaler.pkl           Feature scaler fitted on ERA5 training data
        gru_iso.pkl          Isotonic calibrator for probability outputs
        shap_values_flood_test.npy   Pre-computed SHAP feature importances
        zone_risk_lookup.csv         Per-district risk multipliers
    data/
        era5_raw.csv        ERA5 reanalysis data 2010-2024 (131,496 records)
    utils/
        predictor.py        Model inference, feature engineering, SHAP driver detection
        weather_api.py      ERA5 CSV loader and Open-Meteo API wrappers
        gis_fusion.py       GIS zone risk scoring and alert levels
        shap_service.py     SHAP feature importance helpers
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /api/predict/now | Current flood prediction for Port Louis |
| GET | /api/predict/date/YYYY-MM-DD | Prediction for a specific date |
| GET | /api/predict/date/YYYY-MM-DD/hours | Hourly predictions for a past date |
| GET | /api/predict/hour/N | Prediction offset N hours from now |
| GET | /api/forecast/7day | 7-day daily flood risk forecast |
| GET | /api/forecast/7day?from=YYYY-MM-DD | 7-day forecast starting from a given date |
| GET | /api/gis/zones | GIS zone metadata and risk levels |
| POST | /api/shap/features | SHAP feature importance scores |

---

## Data Sources

- Historical weather (2010-2024): ERA5 reanalysis CSV stored locally in data/era5_raw.csv
- Recent and forecast weather (2025 onwards): Open-Meteo forecast API
- Archive weather (2025+, older than 8 days): Open-Meteo archive API

Weather data is fetched at request time. No database is used.

---

## Deploying to Render

1. Push the backend folder to a GitHub repository
2. Create a new Web Service on Render and connect the repository
3. Set the build command to:

        pip install -r requirements.txt

4. Set the start command to:

        gunicorn app:app

5. Render will assign a URL automatically. Set that URL as VITE_BACKEND_URL in the frontend build settings.
