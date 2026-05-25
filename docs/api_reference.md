# API Reference — House Price Predictor

Base URL: `http://localhost:5000`

---

## Endpoints

### POST /api/predict

Predict the market price of a property.

**Headers:** `Content-Type: application/json`

**Request Fields:**

| Field | Type | Required | Valid Range |
|---|---|---|---|
| area_sqft | float | ✅ | 100–10,000 |
| bedrooms | int | ✅ | 1–10 |
| bathrooms | int | ✅ | 1–10 |
| age_years | int | ✅ | 0–100 |
| floor | int | ✅ | 0–50 |
| parking_spaces | int | ✅ | 0–5 |
| location | string | ✅ | City Center, Suburbs, Rural, Near School, Near Mall |
| property_type | string | ✅ | Apartment, Villa, Independent House, Studio |

**Success Response — 200:**
```json
{
  "success": true,
  "prediction": 12450000,
  "prediction_formatted": "₹1,24,50,000",
  "lower_bound": 11205000,
  "lower_formatted": "₹1,12,05,000",
  "upper_bound": 13695000,
  "upper_formatted": "₹1,36,95,000"
}
```

**Error Response — 422:**
```json
{
  "success": false,
  "errors": [
    "Area must be between 100 and 10,000 sqft",
    "Location must be one of: City Center, Suburbs, Rural, Near School, Near Mall"
  ]
}
```

**Server Error — 500:**
```json
{
  "success": false,
  "errors": ["Server error: <detail>"]
}
```

---

### GET /api/metadata

Returns model training information and performance metrics.

**Response — 200:**
```json
{
  "best_model": "Gradient Boosting",
  "trained_at": "2024-05-21T12:00:00",
  "train_size": 235,
  "test_size": 59,
  "features": ["area_sqft", "bedrooms", ...],
  "all_results": {
    "Gradient Boosting": {"R2": 0.9498, "MAE": 841839, "MAPE": 8.15, "CV_R2_mean": 0.9265}
  },
  "feature_importance": {
    "area_sqft": 0.394,
    "property_type_Studio": 0.214
  }
}
```

---

### GET /api/health

Health check — confirms the model is loaded and the service is running.

**Response — 200:**
```json
{"status": "ok", "model": "loaded"}
```

**Response — 500 (model not loaded):**
```json
{"status": "error", "message": "Model not found at models/best_model.pkl"}
```

---

## Example — cURL

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 8,
    "floor": 5,
    "parking_spaces": 1,
    "location": "Near Mall",
    "property_type": "Apartment"
  }'
```

## Example — Python

```python
import requests

payload = {
    "area_sqft": 1500, "bedrooms": 3, "bathrooms": 2,
    "age_years": 8, "floor": 5, "parking_spaces": 1,
    "location": "Near Mall", "property_type": "Apartment"
}
r = requests.post("http://localhost:5000/api/predict", json=payload)
data = r.json()
print(f"Predicted price: {data['prediction_formatted']}")
print(f"Range: {data['lower_formatted']} – {data['upper_formatted']}")
```
