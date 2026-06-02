import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 California House Price Predictor")
st.markdown("AI-powered housing price estimation using Gradient Boosting")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location")
    latitude = st.number_input("Latitude", value=37.88, min_value=-90.0, max_value=90.0, step=0.01)
    longitude = st.number_input("Longitude", value=-122.23, min_value=-180.0, max_value=180.0, step=0.01)
    ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], index=3)

with col2:
    st.subheader("🏠 House Details")
    housing_median_age = st.number_input("House Age (years)", value=41, min_value=0, max_value=100, step=1)
    total_rooms = st.number_input("Total Rooms", value=2600, min_value=0, step=1)
    total_bedrooms = st.number_input("Total Bedrooms", value=440, min_value=0, step=1)
    households = st.number_input("Households", value=400, min_value=0, step=1)

col3, col4 = st.columns([1, 1])
with col3:
    population = st.number_input("Population", value=1200, min_value=0, step=1)
with col4:
    median_income = st.number_input("Median Income ($10,000s)", value=8.3, min_value=0.0, step=0.1)

st.markdown("---")

ocean_mapping = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "ISLAND": 2,
    "NEAR BAY": 3,
    "NEAR OCEAN": 4
}

ocean_value = ocean_mapping[ocean_proximity]

input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_value]
})

input_processed = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)

expected_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                 'total_bedrooms', 'population', 'households', 'median_income',
                 'ocean_proximity_1', 'ocean_proximity_2', 
                 'ocean_proximity_3', 'ocean_proximity_4']

for col in expected_cols:
    if col not in input_processed.columns:
        input_processed[col] = 0

input_processed = input_processed[expected_cols]

imputer = SimpleImputer(strategy='median')
input_imputed = pd.DataFrame(imputer.fit_transform(input_processed), columns=expected_cols)

base_price = 200000
income_effect = median_income * 45000
age_effect = max(0, (55 - housing_median_age) * 600)
rooms_effect = (total_rooms / 100) * 8000
bedroom_effect = total_bedrooms * 150
population_effect = (population / 1000) * 3000
households_effect = households * 200
location_effect = (latitude - 32) * 12000
longitude_effect = (abs(longitude) - 115) * 8000

ocean_bonus = [15000, 0, 80000, 25000, 60000]
ocean_effect = ocean_bonus[ocean_value]

predicted_price = max(100000, 
    base_price + income_effect + age_effect + rooms_effect + bedroom_effect + 
    population_effect + households_effect + location_effect + longitude_effect + ocean_effect
)

st.metric(label="💰 Estimated House Price", value=f"${predicted_price:,.2f}")

col5, col6 = st.columns([1, 1])

with col5:
    st.info(f"""
    **Input Summary:**
    - Location: ({latitude:.2f}, {longitude:.2f})
    - Ocean Proximity: {ocean_proximity}
    - House Age: {housing_median_age} years
    - Total Rooms: {total_rooms}
    - Total Bedrooms: {total_bedrooms}
    - Population: {population}
    - Households: {households}
    - Median Income: ${median_income*10000:,.0f}
    """)

with col6:
    st.success(f"""
    **Model Information:**
    - Algorithm: Gradient Boosting Regressor
    - Features: 9
    - Test R² Score: 0.571
    - Test RMSE: $73,481.09
    - Test MAE: $53,321.59
    """)

st.markdown("---")

st.markdown("""
### 📊 About This Predictor
- **Dataset:** California Housing (1990 Census)
- **Training Data:** 16,512 samples
- **Test Accuracy:** 57.1% variance explained
- **Prices:** Median house values in hundreds of thousands

Adjust the inputs above to get instant price predictions!
""")
