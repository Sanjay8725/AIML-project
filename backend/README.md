# Heavy Metal Removal Predictor Backend

## Run backend

1. Open terminal in this folder
2. Run:

```powershell
./run_backend.ps1
```

Backend URL: http://127.0.0.1:8000

Frontend dev URL: http://127.0.0.1:5173

The frontend calls `/api/predict` and Vite proxies it to the backend.

## API

### Health
- GET /health

### Predict
- POST /predict

Request JSON:

```json
{
  "Adsorbent": "Activated Carbon",
  "Metal": "Pb",
  "Dosage (g/L)": 1.5,
  "Temp (°C)": 30,
  "pH": 6,
  "Time (min)": 90,
  "RPM": 150,
  "C0 (mg/L)": 50
}
```

Response JSON:

```json
{
  "predicted_removal_percentage": 78.4,
  "predicted_removal_percentage (%)": 78.4,
  "model": "Linear Regression"
}
```
