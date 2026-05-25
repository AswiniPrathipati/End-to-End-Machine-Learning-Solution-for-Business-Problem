"""Generate synthetic house price dataset for training."""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300

locations = ['City Center', 'Suburbs', 'Rural', 'Near School', 'Near Mall']
property_types = ['Apartment', 'Villa', 'Independent House', 'Studio']

location_premium = {
    'City Center': 1.4, 'Near Mall': 1.2, 'Near School': 1.15,
    'Suburbs': 1.0, 'Rural': 0.75
}
type_premium = {
    'Villa': 1.5, 'Independent House': 1.3, 'Apartment': 1.0, 'Studio': 0.7
}

area = np.random.randint(400, 3500, n)
bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
bathrooms = np.clip(bedrooms - np.random.randint(0, 2, n), 1, 5)
age = np.random.randint(0, 30, n)
floor = np.random.randint(0, 20, n)
parking = np.random.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
location = np.random.choice(locations, n)
property_type = np.random.choice(property_types, n)

base_price = 3000
price = (
    area * base_price
    + bedrooms * 800000
    + bathrooms * 300000
    - age * 50000
    + floor * 20000
    + parking * 200000
)
price = price * np.array([location_premium[l] for l in location])
price = price * np.array([type_premium[t] for t in property_type])
noise = np.random.normal(1.0, 0.05, n)
price = (price * noise).astype(int)

df = pd.DataFrame({
    'area_sqft': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': age,
    'floor': floor,
    'parking_spaces': parking,
    'location': location,
    'property_type': property_type,
    'price': price
})

df.to_csv('/home/claude/house_price_ml/data/house_prices.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
print(df.describe())
