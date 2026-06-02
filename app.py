import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="California House Price Predictor", layout="wide")

st.title("🏠 California House Price Predictor")
st.markdown("**AI-powered housing price estimation using Trained Gradient Boosting ML Model**")

@st.cache_resource
def load_model():
    """Load model with multiple fallbacks"""
    try:
        model = joblib.load('gb_model_joblib.pkl')
        return model, "✅ Loaded with joblib"
    except Exception as e:
        pass
    
    try:
        with open('gb_model_pickle.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, "✅ Loaded with pickle"
    except Exception as e:
        pass
    
    try:
        with open('gb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, "✅ Loaded gb_model.pkl"
    except Exception as e:
        pass
    
    return None, "❌ Could not load model"

gb_model, model_status = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Location")
    latitude = st.number_input("Latitude", value=37.88, min_value=-90.0, max_value=90.0, step=0.01)
    longitude = st.number_input("Longitude", value=-122.23, min_value=-180.0, max_value=180.0, step=0.01)
    ocean_proximity = st.selectbox(
        "Ocean Proximity", 
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], 
        index=3
    )

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

col_button = st.columns([1, 1, 1])
with col_button[1]:
    predict_button = st.button("🎯 PREDICT PRICE", key="predict_btn", use_container_width=True)

if predict_button:
    if gb_model is not None:
        try:
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
            
            expected_cols = [
                'longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                'total_bedrooms', 'population', 'households', 'median_income',
                'ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 
                'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'
            ]
            
            for col in expected_cols:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            
            input_processed = input_processed[expected_cols]
            
            imputer = SimpleImputer(strategy='median')
            input_imputed = pd.DataFrame(imputer.fit_transform(input_processed), columns=expected_cols)
            
            predicted_price = gb_model.predict(input_imputed)[0]
            
            st.markdown("---")
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;">
                <h2 style="color: white; margin: 0; font-size: 24px;">💰 Estimated House Price</h2>
                <h1 style="color: #00ff00; margin: 15px 0; font-size: 56px; font-weight: bold;">${predicted_price:,.2f}</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;">Based on your input values</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col5, col6 = st.columns([1, 1])
            
            with col5:
                st.info(f"**📊 Input Summary:**\n\n- Location: ({latitude:.2f}, {longitude:.2f})\n- Ocean: {ocean_proximity}\n- Age: {housing_median_age} yrs\n- Rooms: {total_rooms}\n- Bedrooms: {total_bedrooms}\n- Population: {population}\n- Households: {households}\n- Income: ${median_income*10000:,.0f}")
            
            with col6:
                st.success(f"**🤖 Model Info:**\n\n- Algorithm: Gradient Boosting\n- R² Score: 0.571\n- RMSE: $73,481\n- MAE: $53,322\n- Features: 9")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    else:
        st.error("❌ Model Could Not Load")
        st.warning(model_status)

else:
    st.info("👆 Click PREDICT PRICE to reveal your house price estimate!")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Scikit-learn")
