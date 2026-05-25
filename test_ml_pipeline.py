"""
Test Suite - Unit & Integration Tests
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import pytest
import pandas as pd
import numpy as np

from data_preprocessing import (
    clean_data, engineer_features, build_preprocessor,
    get_feature_columns, get_data_summary
)
from model_inference import validate_input, predict, load_model


# ─── FIXTURES ──────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'area_sqft': [1200, 2500, 800],
        'bedrooms': [3, 4, 1],
        'bathrooms': [2, 3, 1],
        'age_years': [5, 15, 2],
        'floor': [4, 10, 0],
        'parking_spaces': [1, 2, 0],
        'location': ['City Center', 'Suburbs', 'Rural'],
        'property_type': ['Apartment', 'Villa', 'Studio'],
        'price': [12000000, 25000000, 3500000]
    })


@pytest.fixture
def valid_input():
    return {
        'area_sqft': 1200,
        'bedrooms': 3,
        'bathrooms': 2,
        'age_years': 5,
        'floor': 4,
        'parking_spaces': 1,
        'location': 'City Center',
        'property_type': 'Apartment'
    }


# ─── DATA PREPROCESSING TESTS ──────────────────────────────

class TestDataPreprocessing:
    def test_clean_data_removes_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df]).reset_index(drop=True)
        cleaned = clean_data(df_dup)
        assert cleaned.duplicated().sum() == 0

    def test_clean_data_handles_missing_values(self, sample_df):
        df_missing = sample_df.copy()
        df_missing.loc[0, 'area_sqft'] = np.nan
        cleaned = clean_data(df_missing)
        assert cleaned['area_sqft'].isnull().sum() == 0

    def test_engineer_features_creates_expected_columns(self, sample_df):
        df_eng = engineer_features(sample_df)
        assert 'total_rooms' in df_eng.columns
        assert 'bath_bed_ratio' in df_eng.columns
        assert 'age_category' in df_eng.columns
        assert 'area_category' in df_eng.columns
        assert 'is_high_floor' in df_eng.columns

    def test_total_rooms_calculation(self, sample_df):
        df_eng = engineer_features(sample_df)
        for _, row in df_eng.iterrows():
            assert row['total_rooms'] == row['bedrooms'] + row['bathrooms']

    def test_age_category_values(self, sample_df):
        df_eng = engineer_features(sample_df)
        valid_categories = {'New', 'Mid', 'Old'}
        assert set(df_eng['age_category'].unique()).issubset(valid_categories)

    def test_area_category_values(self, sample_df):
        df_eng = engineer_features(sample_df)
        valid_cats = {'Small', 'Medium', 'Large'}
        assert set(df_eng['area_category'].unique()).issubset(valid_cats)

    def test_is_high_floor_binary(self, sample_df):
        df_eng = engineer_features(sample_df)
        assert set(df_eng['is_high_floor'].unique()).issubset({0, 1})

    def test_get_feature_columns_returns_lists(self):
        num, cat = get_feature_columns()
        assert isinstance(num, list)
        assert isinstance(cat, list)
        assert len(num) > 0
        assert len(cat) > 0

    def test_data_summary_keys(self, sample_df):
        summary = get_data_summary(sample_df)
        assert 'total_rows' in summary
        assert 'total_columns' in summary


# ─── INPUT VALIDATION TESTS ─────────────────────────────────

class TestInputValidation:
    def test_valid_input_passes(self, valid_input):
        is_valid, errors = validate_input(valid_input)
        assert is_valid
        assert errors == []

    def test_invalid_area_too_small(self, valid_input):
        valid_input['area_sqft'] = 50
        is_valid, errors = validate_input(valid_input)
        assert not is_valid
        assert any('Area' in e for e in errors)

    def test_invalid_area_too_large(self, valid_input):
        valid_input['area_sqft'] = 50000
        is_valid, errors = validate_input(valid_input)
        assert not is_valid

    def test_invalid_location(self, valid_input):
        valid_input['location'] = 'Mars'
        is_valid, errors = validate_input(valid_input)
        assert not is_valid
        assert any('Location' in e for e in errors)

    def test_invalid_property_type(self, valid_input):
        valid_input['property_type'] = 'Castle'
        is_valid, errors = validate_input(valid_input)
        assert not is_valid

    def test_missing_field_raises_error(self, valid_input):
        del valid_input['area_sqft']
        is_valid, errors = validate_input(valid_input)
        assert not is_valid
        assert any('area_sqft' in e for e in errors)

    def test_zero_bedrooms_invalid(self, valid_input):
        valid_input['bedrooms'] = 0
        is_valid, errors = validate_input(valid_input)
        assert not is_valid

    def test_negative_age_invalid(self, valid_input):
        valid_input['age_years'] = -1
        is_valid, errors = validate_input(valid_input)
        assert not is_valid


# ─── MODEL INFERENCE TESTS ──────────────────────────────────

class TestModelInference:
    def test_model_loads_successfully(self):
        model = load_model()
        assert model is not None

    def test_prediction_returns_success(self, valid_input):
        result = predict(valid_input)
        assert result['success'] is True
        assert 'prediction' in result

    def test_prediction_is_positive(self, valid_input):
        result = predict(valid_input)
        assert result['prediction'] > 0

    def test_confidence_interval_ordering(self, valid_input):
        result = predict(valid_input)
        assert result['lower_bound'] < result['prediction'] < result['upper_bound']

    def test_invalid_input_returns_failure(self, valid_input):
        valid_input['area_sqft'] = -100
        result = predict(valid_input)
        assert result['success'] is False
        assert 'errors' in result

    def test_prediction_varies_with_area(self, valid_input):
        result_small = predict({**valid_input, 'area_sqft': 500})
        result_large = predict({**valid_input, 'area_sqft': 3000})
        assert result_large['prediction'] > result_small['prediction']

    def test_prediction_varies_with_location(self, valid_input):
        result_city = predict({**valid_input, 'location': 'City Center'})
        result_rural = predict({**valid_input, 'location': 'Rural'})
        assert result_city['prediction'] > result_rural['prediction']

    def test_prediction_formatted_includes_rupee_sign(self, valid_input):
        result = predict(valid_input)
        assert '₹' in result['prediction_formatted']

    def test_villa_more_expensive_than_studio(self, valid_input):
        result_villa = predict({**valid_input, 'property_type': 'Villa'})
        result_studio = predict({**valid_input, 'property_type': 'Studio'})
        assert result_villa['prediction'] > result_studio['prediction']


# ─── RUN ─────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
