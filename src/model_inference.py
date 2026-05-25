"""
Model Inference
Loads trained model and handles predictions with validation
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import engineer_features, get_feature_columns

MODEL_PATH = '/home/claude/house_price_ml/models/best_model.pkl'
METADATA_PATH = '/home/claude/house_price_ml/models/metadata.json'


def load_model():
    """Load the saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train first.")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
    return model


def load_metadata():
    """Load model metadata."""
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# Input validation rules
VALIDATION_RULES = {
    'area_sqft': (100, 10000, 'Area must be between 100 and 10,000 sqft'),
    'bedrooms': (1, 10, 'Bedrooms must be between 1 and 10'),
    'bathrooms': (1, 10, 'Bathrooms must be between 1 and 10'),
    'age_years': (0, 100, 'Age must be between 0 and 100 years'),
    'floor': (0, 50, 'Floor must be between 0 and 50'),
    'parking_spaces': (0, 5, 'Parking spaces must be between 0 and 5'),
}

VALID_LOCATIONS = ['City Center', 'Suburbs', 'Rural', 'Near School', 'Near Mall']
VALID_PROPERTY_TYPES = ['Apartment', 'Villa', 'Independent House', 'Studio']


def validate_input(input_data: dict) -> tuple:
    """
    Validate input data. Returns (is_valid, errors_list).
    """
    errors = []

    for field, (min_val, max_val, msg) in VALIDATION_RULES.items():
        if field in input_data:
            try:
                val = float(input_data[field])
                if not (min_val <= val <= max_val):
                    errors.append(msg)
            except (ValueError, TypeError):
                errors.append(f"'{field}' must be a valid number")
        else:
            errors.append(f"Missing required field: '{field}'")

    if input_data.get('location') not in VALID_LOCATIONS:
        errors.append(f"Location must be one of: {', '.join(VALID_LOCATIONS)}")

    if input_data.get('property_type') not in VALID_PROPERTY_TYPES:
        errors.append(f"Property type must be one of: {', '.join(VALID_PROPERTY_TYPES)}")

    return len(errors) == 0, errors


def predict(input_data: dict, model=None) -> dict:
    """
    Make a prediction for a single input.
    Returns prediction with confidence interval.
    """
    # Validate
    is_valid, errors = validate_input(input_data)
    if not is_valid:
        return {'success': False, 'errors': errors}

    # Load model if not provided
    if model is None:
        model = load_model()

    try:
        # Build dataframe
        df = pd.DataFrame([{
            'area_sqft': float(input_data['area_sqft']),
            'bedrooms': int(input_data['bedrooms']),
            'bathrooms': int(input_data['bathrooms']),
            'age_years': int(input_data['age_years']),
            'floor': int(input_data['floor']),
            'parking_spaces': int(input_data['parking_spaces']),
            'location': input_data['location'],
            'property_type': input_data['property_type'],
        }])

        # Engineer features (same as training)
        df = engineer_features(df)

        numeric_features, categorical_features = get_feature_columns()
        feature_cols = numeric_features + categorical_features
        df = df[feature_cols]

        # Predict
        prediction = model.predict(df)[0]

        # Confidence interval (±10% as proxy)
        lower = prediction * 0.90
        upper = prediction * 1.10

        return {
            'success': True,
            'prediction': round(prediction),
            'lower_bound': round(lower),
            'upper_bound': round(upper),
            'prediction_formatted': f"₹{prediction:,.0f}",
            'lower_formatted': f"₹{lower:,.0f}",
            'upper_formatted': f"₹{upper:,.0f}",
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'success': False, 'errors': [f"Prediction failed: {str(e)}"]}


def batch_predict(records: list, model=None) -> list:
    """Make predictions for a list of records."""
    if model is None:
        model = load_model()
    return [predict(record, model) for record in records]


if __name__ == '__main__':
    sample = {
        'area_sqft': 1200,
        'bedrooms': 3,
        'bathrooms': 2,
        'age_years': 5,
        'floor': 4,
        'parking_spaces': 1,
        'location': 'City Center',
        'property_type': 'Apartment'
    }
    result = predict(sample)
    print("Sample Prediction:")
    print(f"  Price     : {result['prediction_formatted']}")
    print(f"  Range     : {result['lower_formatted']} – {result['upper_formatted']}")
