# Technical Documentation — House Price Predictor

## Architecture Overview

```
[Raw CSV Data]
      │
      ▼
[data_preprocessing.py]
  • load_data()          → validates file, loads DataFrame
  • clean_data()         → dedup, fill nulls, remove outliers (IQR)
  • engineer_features()  → creates 5 new derived columns
  • build_preprocessor() → sklearn ColumnTransformer
      │
      ▼
[model_training.py]
  • Pipeline: preprocessor + model (Ridge / RF / GBM)
  • 5-fold cross-validation
  • Selects best model by R²
  • Saves best_model.pkl + metadata.json
      │
      ▼
[model_inference.py]
  • validate_input()     → range checks + allowed values
  • predict()            → loads model, engineers features, returns price + CI
      │
      ▼
[web_app.py — Flask]
  • GET  /              → renders HTML dashboard
  • POST /api/predict   → JSON prediction endpoint
  • GET  /api/metadata  → model info
  • GET  /api/health    → status check
```

---

## Data Pipeline Details

### Cleaning Steps
1. **Deduplication** — `df.drop_duplicates()`
2. **Null imputation** — numeric cols → median; categorical → mode
3. **Outlier removal** — prices below 1st percentile or above 99th percentile are dropped

### Feature Engineering
| Feature | Formula | Rationale |
|---|---|---|
| `total_rooms` | bedrooms + bathrooms | proxy for house size |
| `bath_bed_ratio` | bathrooms / bedrooms | quality indicator |
| `age_category` | binned: New/Mid/Old | captures non-linear depreciation |
| `area_category` | binned: Small/Medium/Large | captures size segment |
| `is_high_floor` | floor >= 10 | city view premium flag |

### Preprocessing (in sklearn Pipeline)
- **Numeric** features → `StandardScaler` (zero mean, unit variance)
- **Categorical** features → `OneHotEncoder(handle_unknown='ignore')`
- Combined via `ColumnTransformer`

---

## Model Details

### Ridge Regression
- Linear model with L2 regularisation
- Fast, interpretable, but cannot capture feature interactions
- Hyperparameter: `alpha=100`

### Random Forest
- Ensemble of 200 decision trees with bagging
- Captures non-linear patterns; resistant to outliers
- Hyperparameters: `n_estimators=200, max_depth=10, min_samples_split=5`

### Gradient Boosting (Best)
- Sequential boosting: each tree corrects previous residuals
- Best accuracy but slower to train
- Hyperparameters: `n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8`

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | mean(|y - ŷ|) | Average prediction error in ₹ |
| RMSE | √mean((y - ŷ)²) | Penalises large errors more |
| R² | 1 - SS_res/SS_tot | % variance explained (1.0 = perfect) |
| MAPE | mean(|y - ŷ| / y) × 100 | % error relative to true price |
| CV R² | mean R² across 5 folds | Generalisation estimate |

---

## Model Persistence

Models are saved using `joblib.dump()` (more efficient than pickle for numpy arrays).

```python
# Save
joblib.dump(pipeline, 'models/best_model.pkl')

# Load
pipeline = joblib.load('models/best_model.pkl')
```

Metadata is saved as JSON alongside the model, tracking training date, metrics, and feature importances for model versioning.

---

## Input Validation Rules

```
area_sqft      : 100 ≤ x ≤ 10,000
bedrooms       : 1 ≤ x ≤ 10
bathrooms      : 1 ≤ x ≤ 10
age_years      : 0 ≤ x ≤ 100
floor          : 0 ≤ x ≤ 50
parking_spaces : 0 ≤ x ≤ 5
location       : must be in allowed set
property_type  : must be in allowed set
```

All validation is done before feature engineering to give clear, user-friendly error messages.

---

## Confidence Interval

The system returns a ±10% confidence band around each prediction as a proxy interval. For production use, this should be replaced with:
- **Quantile Regression Forests** for true quantile estimates
- **Conformal Prediction** for distribution-free coverage guarantees
