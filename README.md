# 🏠 House Price Prediction System
### Month 4 Internship Project — Machine Learning Fundamentals

An end-to-end machine learning system that predicts residential property prices using Gradient Boosting, achieving **R² = 0.95** and **MAPE ≈ 8%**. Includes a complete data pipeline, model comparison, feature importance analysis, and a Flask web application.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Data Dictionary](#data-dictionary)
5. [Methodology](#methodology)
6. [Model Performance](#model-performance)
7. [Feature Importance](#feature-importance)
8. [API Reference](#api-reference)
9. [Business Insights](#business-insights)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Business Problem:** Real estate agents and buyers lack an objective, data-driven way to assess fair property prices. This system provides instant valuations based on key property attributes.

**Solution:** A supervised regression pipeline that:
- Cleans and engineers features from raw property data
- Trains and compares 3 ML algorithms (Ridge, Random Forest, Gradient Boosting)
- Selects the best model via cross-validated R² score
- Serves predictions through a REST API and web interface

---

## Quick Start

```bash
# 1. Clone / enter the project
cd house_price_ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset (already included in data/)
python scripts/generate_data.py

# 4. Train models
python src/model_training.py

# 5. Run web app
python app/web_app.py
# → Open http://localhost:5000
```

---

## Project Structure

```
house_price_ml/
├── data/
│   └── house_prices.csv          # 300-row property dataset
├── notebooks/
│   └── 01_eda_and_modeling.py    # EDA, visualisations, model training
├── src/
│   ├── data_preprocessing.py     # Cleaning + feature engineering
│   ├── model_training.py         # Train & compare 3 algorithms
│   └── model_inference.py        # Prediction + input validation
├── models/
│   ├── best_model.pkl            # Serialised best model
│   └── metadata.json             # Metrics & feature importances
├── tests/
│   └── test_ml_pipeline.py       # 26 unit & integration tests
├── app/
│   ├── web_app.py                # Flask application
│   └── templates/index.html      # Web UI
├── docs/
│   ├── 01_price_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_price_by_category.png
│   ├── 04_area_vs_price.png
│   ├── 05_model_comparison.png
│   └── 06_feature_importance.png
├── config/
│   └── config.yaml               # All tunable parameters
├── scripts/
│   └── generate_data.py          # Dataset generation
└── requirements.txt
```

---

## Data Dictionary

| Column | Type | Range | Description |
|---|---|---|---|
| `area_sqft` | int | 100–10,000 | Property area in square feet |
| `bedrooms` | int | 1–10 | Number of bedrooms |
| `bathrooms` | int | 1–10 | Number of bathrooms |
| `age_years` | int | 0–100 | Age of property in years |
| `floor` | int | 0–50 | Floor number (0 = ground) |
| `parking_spaces` | int | 0–5 | Number of parking slots |
| `location` | str | 5 values | City Center / Suburbs / Rural / Near School / Near Mall |
| `property_type` | str | 4 values | Apartment / Villa / Independent House / Studio |
| `price` | int | ~25L–3Cr | Target: property price in ₹ |

**Engineered Features (created during training):**

| Feature | Description |
|---|---|
| `total_rooms` | bedrooms + bathrooms |
| `bath_bed_ratio` | bathrooms ÷ bedrooms |
| `age_category` | New (0–5yr) / Mid (6–15yr) / Old (>15yr) |
| `area_category` | Small (<800) / Medium (800–1800) / Large (>1800) |
| `is_high_floor` | 1 if floor ≥ 10, else 0 |

---

## Methodology

### Week-by-Week Approach

**Week 1 — Problem Definition & Data Preparation**
- Defined the regression target (property price)
- Synthesised a realistic 300-row dataset with location and type premiums
- Performed EDA: distribution plots, correlation matrix, category comparisons

**Week 2 — Feature Engineering & Model Selection**
- Created 5 derived features to capture domain knowledge
- Selected Ridge, Random Forest, and Gradient Boosting as algorithm candidates
- Established baseline metrics

**Week 3 — Model Training & Cross-Validation**
- Trained all 3 models inside sklearn `Pipeline` objects (prevents data leakage)
- 5-fold cross-validation for reliable R² estimates
- Compared MAE, RMSE, R², and MAPE

**Week 4 — Interpretation & Optimisation**
- Extracted Gradient Boosting feature importances
- Identified top predictors: area, property type, location
- Generated visualisation suite (6 charts)

**Week 5 — Deployment Preparation**
- Built Flask REST API with `/api/predict`, `/api/metadata`, `/api/health`
- Serialised model with `joblib`
- Added strict input validation

**Week 6 — Production Integration**
- 26-test pytest suite (unit + integration)
- Comprehensive error handling in all modules
- Full documentation and config file

---

## Model Performance

| Model | R² | MAE | MAPE | CV R² |
|---|---|---|---|---|
| Ridge Regression | 0.684 | ₹24,07,747 | 24.6% | 0.636 ± 0.038 |
| Random Forest | 0.876 | ₹14,74,247 | 15.3% | 0.852 ± 0.022 |
| **Gradient Boosting** ✅ | **0.950** | **₹8,41,839** | **8.2%** | **0.927 ± 0.025** |

**Best model:** Gradient Boosting — selected automatically based on highest R².

---

## Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | area_sqft | 39.4% |
| 2 | property_type_Studio | 21.4% |
| 3 | location_Rural | 7.2% |
| 4 | property_type_Villa | 6.2% |
| 5 | location_City Center | 5.6% |

Area is by far the dominant predictor. Property type and location together account for ~40% of the model's decisions.

---

## API Reference

### `POST /api/predict`

**Request body (JSON):**
```json
{
  "area_sqft": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "age_years": 5,
  "floor": 4,
  "parking_spaces": 1,
  "location": "City Center",
  "property_type": "Apartment"
}
```

**Response (200 OK):**
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

**Response (422 — validation error):**
```json
{
  "success": false,
  "errors": ["Area must be between 100 and 10,000 sqft"]
}
```

### `GET /api/metadata`
Returns model performance metrics and feature importances.

### `GET /api/health`
Returns `{"status": "ok", "model": "loaded"}` when app is healthy.

---

## Business Insights

1. **Area is king** — Square footage alone explains ~39% of price variation. Each 100 sqft adds roughly ₹3–4 lakh.
2. **Location premium** — City Center properties are priced ~40% higher than Rural equivalents for the same specifications.
3. **Property type multiplier** — Villas command a 1.5× premium over Apartments; Studios sell at a 0.7× discount.
4. **Age depreciation** — Properties lose ~₹50,000 in value per year of age.
5. **Parking value** — Each parking space adds approximately ₹2 lakh.
6. **Model accuracy** — An 8.2% mean error means most valuations are within ₹70,000–₹2,00,000 of actual market price.

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Expected: 26 passed
```

Test classes:
- `TestDataPreprocessing` — 9 tests (cleaning, feature engineering)
- `TestInputValidation` — 8 tests (boundary conditions, missing fields)
- `TestModelInference` — 9 tests (prediction logic, ordering, formatting)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: best_model.pkl` | Run `python src/model_training.py` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Flask port 5000 in use | Change port in `config/config.yaml` → `app.port` |
| Prediction returns validation error | Check input ranges in the Data Dictionary above |

---

*Project completed as part of Month 4 Machine Learning Fundamentals — Internship Program*
