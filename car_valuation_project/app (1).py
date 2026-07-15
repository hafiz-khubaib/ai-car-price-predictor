import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="AI Car Value Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR: Model Trust & Stats ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("AI Engine Specs")
    st.write("This application is powered by a trained Machine Learning model.")
    st.markdown("---")
    
    # Metrics in sidebar
    st.metric(label="Model Accuracy (R²)", value="86.62%")
    st.metric(label="Average Margin of Error", value="±0.99 Lakhs")
    st.caption("Algorithm: K-Nearest Neighbors (K=4)")
    st.caption("Distance Metric: Manhattan")
    st.markdown("---")
    st.write("Developed by: **AI Engineer** 🚀")

# --- MAIN PAGE ---
st.title("🚗 Intelligent Car Resale Value Predictor")
st.write("Enter your vehicle's specifications below. Our AI model will estimate the best resale price based on market trends and historical data.")
st.markdown("---")

# Model and Scaler Loading (Using joblib to load .pkl files)
@st.cache_resource
def load_artifacts():
    model = joblib.load('knn_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# Main Layout split into 2 Columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📋 Basic Vehicle Details")
    year = st.slider("Model Year", 2003, 2018, 2015, help="Select the manufacturing year of the vehicle.")
    present_price = st.number_input(
        "Current Showroom Price (in Lakhs)", 
        min_value=0.1, 
        max_value=95.0, 
        value=6.0, 
        step=0.1,
        help="What is the current showroom price of this vehicle model today?"
    )
    kms_driven = st.number_input(
        "Total Kilometers Driven", 
        min_value=100, 
        max_value=500000, 
        value=30000, 
        step=1000,
        help="How many kilometers has the vehicle been driven so far?"
    )

with col2:
    st.subheader("⚙️ Technical Specs & Ownership")
    owner = st.selectbox("Previous Owners", [0, 1, 3], help="How many times has this vehicle been resold previously?")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

st.markdown("---")

# Predict Button centered
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    predict_btn = st.button("💰 Calculate Resale Value", use_container_width=True)

# Prediction Logic & Beautiful Output
if predict_btn:
    with st.spinner("AI Engine is calculating the pricing... Please wait!"):
        # Preprocessing input
        input_data = {
            'Year': year,
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Fuel_Type_CNG': 1 if fuel == "CNG" else 0,
            'Fuel_Type_Diesel': 1 if fuel == "Diesel" else 0,
            'Fuel_Type_Petrol': 1 if fuel == "Petrol" else 0,
            'Seller_Type_Dealer': 1 if seller == "Dealer" else 0,
            'Seller_Type_Individual': 1 if seller == "Individual" else 0,
            'Transmission_Automatic': 1 if transmission == "Automatic" else 0,
            'Transmission_Manual': 1 if transmission == "Manual" else 0
        }

        # Conversion to DataFrame & Order matching
        input_df = pd.DataFrame([input_data])
        columns_order = [
            'Year', 'Present_Price', 'Kms_Driven', 'Owner',
            'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
            'Seller_Type_Dealer', 'Seller_Type_Individual',
            'Transmission_Automatic', 'Transmission_Manual'
        ]
        input_df = input_df[columns_order]

        # Scaling and Prediction
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        # Calculate Range based on MAE (0.99 Lakhs)
        mae = 0.99
        lower_limit = max(0.1, prediction - mae)
        upper_limit = prediction + mae

        # Display Results in a beautiful container
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("### 🎉 AI Valuation Complete!")
        
        out_col1, out_col2 = st.columns(2)
        with out_col1:
            st.metric(
                label="Estimated Selling Price", 
                value=f"{prediction:.2f} Lakhs",
                delta=f"Based on current market trends"
            )
        with out_col2:
            st.metric(
                label="Realistic Price Range (±0.99L MAE)", 
                value=f"{lower_limit:.2f}L - {upper_limit:.2f}L",
                delta="Expected variation limit"
            )
        
        st.info("💡 **Pro Tip:** If the vehicle's body condition is pristine and it has a complete service history, you can confidently ask for the upper price range!")
