import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title(" California House Price Predictor")
st.markdown("AI-powered housing price estimation")

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

try:
    with open('gb_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    input_processed = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)
    
    expected_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                     'total_bedrooms', 'population', 'households', 'median_income',
                     'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
                     'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
    
    for col in expected_cols:
        if col not in input_processed.columns:
            input_processed[col] = 0
    
    input_processed = input_processed[expected_cols]
    
    imputer = SimpleImputer(strategy='median')
    input_imputed = pd.DataFrame(imputer.fit_transform(input_processed), columns=expected_cols)
    
    predicted_price = gb_model.predict(input_imputed)[0]
    
    st.metric(label=" Estimated Price", value=f"${predicted_price:,.2f}")
    
    col5, col6 = st.columns([1, 1])
    with col5:
        st.info(f"""
        **Input Summary:**
        - Location: ({latitude:.2f}, {longitude:.2f})
        - Ocean: {ocean_proximity}
        - Age: {housing_median_age} years
        - Rooms: {total_rooms}
        - Bedrooms: {total_bedrooms}
        - Population: {population}
        - Income: ${median_income*10000:,.0f}
        """)
    
    with col6:
        st.success(f"""
        **Model Info:**
        - Algorithm: Gradient Boosting
        - Test R²: 0.571
        - Test RMSE: $73,481
        """)

except FileNotFoundError:
    st.error(" gb_model.pkl not found. Please upload the model file.")
except Exception as e:
    st.error(f" Error: {str(e)}")

st.markdown("---")
st.markdown("Made with using Streamlit & Scikit-learn")
