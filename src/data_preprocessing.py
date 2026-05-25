"""
Data Preprocessing Pipeline
Handles cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and perform initial validation on dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Remove duplicates
    - Handle missing values
    - Fix outliers
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

    # Fill missing numeric values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"Filled {missing} missing values in '{col}' with median")

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            logger.info(f"Filled {missing} missing values in '{col}' with mode")

    # Remove extreme outliers using IQR for price
    if 'price' in df.columns:
        Q1 = df['price'].quantile(0.01)
        Q3 = df['price'].quantile(0.99)
        before = len(df)
        df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
        logger.info(f"Removed {before - len(df)} outlier rows from price")

    logger.info(f"Final cleaned dataset: {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new engineered features:
    - price_per_sqft (if price is in df, else it's for inference)
    - room_ratio
    - property_age_category
    - total_rooms
    """
    df = df.copy()

    # Feature: total rooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Feature: bathroom to bedroom ratio
    df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)

    # Feature: age category
    def age_category(age):
        if age <= 5:
            return 'New'
        elif age <= 15:
            return 'Mid'
        else:
            return 'Old'

    df['age_category'] = df['age_years'].apply(age_category)

    # Feature: area category
    def area_category(area):
        if area <= 800:
            return 'Small'
        elif area <= 1800:
            return 'Medium'
        else:
            return 'Large'

    df['area_category'] = df['area_sqft'].apply(area_category)

    # Feature: is high floor
    df['is_high_floor'] = (df['floor'] >= 10).astype(int)

    logger.info(f"Engineered features: total_rooms, bath_bed_ratio, age_category, area_category, is_high_floor")
    return df


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor


def get_feature_columns():
    """Return the lists of feature columns used in the model."""
    numeric_features = [
        'area_sqft', 'bedrooms', 'bathrooms', 'age_years',
        'floor', 'parking_spaces', 'total_rooms', 'bath_bed_ratio', 'is_high_floor'
    ]
    categorical_features = [
        'location', 'property_type', 'age_category', 'area_category'
    ]
    return numeric_features, categorical_features


def get_data_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary summarizing the dataset."""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
