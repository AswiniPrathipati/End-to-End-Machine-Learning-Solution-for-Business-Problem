"""
Model Training & Evaluation
Trains and compares multiple ML algorithms with comprehensive metrics
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_data, clean_data, engineer_features,
    build_preprocessor, get_feature_columns
)


def get_models():
    """Return dictionary of models to train and compare."""
    return {
        'Ridge Regression': Ridge(alpha=100),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    }


def evaluate_model(model, X_test, y_test):
    """Compute comprehensive evaluation metrics."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
        'predictions': y_pred.tolist()
    }


def get_feature_importance(pipeline, numeric_features, categorical_features):
    """Extract feature importances from tree-based models."""
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        if not hasattr(model, 'feature_importances_'):
            return {}

        # Get encoded feature names
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
        all_feature_names = numeric_features + cat_feature_names

        importances = model.feature_importances_
        importance_dict = dict(zip(all_feature_names, importances))

        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return importance_dict
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")
        return {}


def train_all_models(data_path: str):
    """
    Full training pipeline:
    1. Load & clean data
    2. Feature engineering
    3. Train 3 models
    4. Evaluate all
    5. Save best model
    """
    # Load & preprocess
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)

    numeric_features, categorical_features = get_feature_columns()
    feature_cols = numeric_features + categorical_features
    target_col = 'price'

    X = df[feature_cols]
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Train and evaluate each model
    models = get_models()
    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        logger.info(f"Training: {name}")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
        metrics['CV_R2_mean'] = round(cv_scores.mean(), 4)
        metrics['CV_R2_std'] = round(cv_scores.std(), 4)

        results[name] = metrics
        trained_pipelines[name] = pipeline

        logger.info(f"  R²={metrics['R2']:.4f} | MAE=₹{metrics['MAE']:,.0f} | MAPE={metrics['MAPE']:.1f}%")

    # Select best model (highest R2)
    best_name = max(results, key=lambda k: results[k]['R2'])
    best_pipeline = trained_pipelines[best_name]
    logger.info(f"Best model: {best_name} (R²={results[best_name]['R2']})")

    # Feature importance
    importance = get_feature_importance(best_pipeline, numeric_features, categorical_features)

    # Save model
    os.makedirs('/home/claude/house_price_ml/models', exist_ok=True)
    model_path = '/home/claude/house_price_ml/models/best_model.pkl'
    joblib.dump(best_pipeline, model_path)

    # Save metadata
    metadata = {
        'best_model': best_name,
        'trained_at': datetime.now().isoformat(),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': feature_cols,
        'all_results': {k: {m: v for m, v in metrics.items() if m != 'predictions'}
                        for k, metrics in results.items()},
        'feature_importance': {k: round(float(v), 5) for k, v in list(importance.items())[:10]},
        'model_path': model_path
    }

    with open('/home/claude/house_price_ml/models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {model_path}")

    # Print report
    print_training_report(metadata, results)

    return best_pipeline, metadata, results


def print_training_report(metadata, results):
    """Print a formatted training report."""
    print("\n" + "="*60)
    print("  HOUSE PRICE PREDICTION SYSTEM - TRAINING REPORT")
    print("="*60)

    print(f"\n📊 DATA:")
    print(f"   Training samples : {metadata['train_size']}")
    print(f"   Test samples     : {metadata['test_size']}")
    print(f"   Features used    : {len(metadata['features'])}")

    print(f"\n🤖 MODEL COMPARISON:")
    print(f"   {'Model':<25} {'R²':>8} {'MAE':>15} {'MAPE':>8} {'CV R²':>10}")
    print(f"   {'-'*68}")
    for name, m in results.items():
        marker = " ✅" if name == metadata['best_model'] else "   "
        print(f"   {marker}{name:<23} {m['R2']:>8.4f} ₹{m['MAE']:>13,.0f} {m['MAPE']:>7.1f}% {m['CV_R2_mean']:>8.4f}±{m['CV_R2_std']:.4f}")

    best = results[metadata['best_model']]
    print(f"\n🏆 BEST MODEL: {metadata['best_model']}")
    print(f"   R² Score       : {best['R2']:.4f}")
    print(f"   MAE            : ₹{best['MAE']:,.0f}")
    print(f"   MAPE           : {best['MAPE']:.2f}%")
    print(f"   CV R²          : {best['CV_R2_mean']:.4f} ± {best['CV_R2_std']:.4f}")

    if metadata['feature_importance']:
        print(f"\n🔍 TOP FEATURE IMPORTANCES:")
        for i, (feat, imp) in enumerate(list(metadata['feature_importance'].items())[:5], 1):
            bar = '█' * int(imp * 100)
            print(f"   {i}. {feat:<30} {imp*100:5.1f}% {bar}")

    print("\n✅ Model saved successfully!")
    print("="*60 + "\n")


if __name__ == '__main__':
    train_all_models('/home/claude/house_price_ml/data/house_prices.csv')
