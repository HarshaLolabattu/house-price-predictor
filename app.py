import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="California House Price Predictor", layout="wide")

st.title(" California House Price Predictor")
st.markdown("**AI-powered housing price estimation using Trained Gradient Boosting Machine Learning Model**")

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    """Try loading model with joblib first, then pickle"""
    try:
        model = joblib.load('gb_model_joblib.pkl')
        return model, " Model loaded with joblib"
    except Exception as e1:
        try:
            with open('gb_model_pickle.pkl', 'rb') as f:
                model = pickle.load(f)
            return model, "Model loaded with pickle"
        except Exception as e2:
            return None, f" Failed to load model: {str(e2)}"

gb_model, model_status = load_model()

# ============ INPUT SECTION ============
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
    st.subheader(" House Details")
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

# ============ BUTTON & PREDICTION ============
col_button = st.columns([1, 1, 1])
with col_button[1]:
    predict_button = st.button(
        "🎯 PREDICT PRICE", 
        key="predict_btn",
        use_container_width=True
    )

# Show prediction only when button is clicked
if predict_button:
    if gb_model is not None:
        try:
            # Prepare input data
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
            
            # One-hot encode ocean_proximity
            input_processed = pd.get_dummies(input_data, columns=['ocean_proximity'], drop_first=True)
            
            # Ensure all expected columns exist
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
            
            # Impute missing values
            imputer = SimpleImputer(strategy='median')
            input_imputed = pd.DataFrame(
                imputer.fit_transform(input_processed), 
                columns=expected_cols
            )
            
            # Make prediction
            predicted_price = gb_model.predict(input_imputed)[0]
            
            # ============ DISPLAY RESULTS ============
            st.markdown("---")
            
            # Big price display
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #1f77b4; border-radius: 10px;">
                <h2 style="color: white; margin: 0;"> Estimated House Price</h2>
                <h1 style="color: #00ff00; margin: 10px 0; font-size: 60px;">${predicted_price:,.2f}</h1>
                <p style="color: white; margin: 0;">Based on your inputs</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Details
            col5, col6 = st.columns([1, 1])
            
            with col5:
                st.info(f"""
                **📊 Your Input Summary:**
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
                **🤖 Model Performance:**
                - Algorithm: Gradient Boosting Regressor
                - Features: 9
                - Test R² Score: 0.571
                - Test RMSE: $73,481
                - Test MAE: $53,322
                - Dataset: California Housing
                """)

        except Exception as e:
            st.error(f" Error making prediction: {str(e)}")
            st.info("This usually means there's a compatibility issue with the model file.")

    else:
        st.error("Model Could Not Load")
        st.warning(model_status)
        st.info("""
        **Troubleshooting:**
        - Make sure both model files are in the repository
        - Try re-saving the model in Colab
        """)

else:
    # Show this before button is clicked
    st.info("""
    ** Click the "PREDICT PRICE" button above to get your house price estimate!**
    
    Adjust the input values on the left side, then click the button to reveal the predicted price.
    """)

st.markdown("---")

st.markdown("""
###  About This Application
- **Dataset:** California Housing Dataset (1990 Census)
- **Training Samples:** 16,512 houses
- **Features:** 9 (Location, House details, Demographics)
- **Algorithm:** Gradient Boosting Regressor
- **Accuracy:** R² = 0.571 (explains 57.1% of price variance)
- **Prediction Unit:** Median house value in hundreds of thousands

**How to Use:**
1. Enter the house details using the input fields on the left
2. Click the **"PREDICT PRICE"** button
3. See the estimated price revealed!
4. Adjust inputs and click again for a new prediction
""")
