import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏠")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stNumberInput input {
            border-radius: 5px;
        }
        .stSelectbox div {
            border-radius: 5px;
        }
        h1 {
            color: #2C3E50;
            font-size: 36px;
        }
        .stSuccess {
            background-color: #D4EDDA;
        }
        .stInfo {
            background-color: #D1ECF1;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏠 House Price Predictor")
st.caption("🔍 Get quick estimates of house prices based on your input 📊")

# Input Section
st.header("📥 Enter Property Details")
area = st.number_input("📏 Area (sq ft)", value=100, step=10)
bedrooms = st.number_input("🛏️ Bedrooms", value=1, step=1)
bathrooms = st.number_input("🛁 Bathrooms", value=1, step=1)
location = st.selectbox("📍 Location", ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore', 'Pune', 'Kolkata'])

# Prediction
if st.button("Predict 💰"):
    locs = ['Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore', 'Pune', 'Kolkata']
    location_encoding = [1 if location == loc else 0 for loc in locs]

    input_data = [area, bedrooms, bathrooms] + location_encoding
    input_df = pd.DataFrame([input_data], columns=['Area', 'Bedrooms', 'Bathrooms'] + locs)

    prediction = model.predict(input_df)[0]
    emi = prediction * 0.0075

    st.success(f"💵 Estimated House Price: ₹ {prediction:,.2f}")
    st.info(f"📆 Estimated Monthly EMI: ₹ {emi:,.2f}")

    # Feature Importance Chart
    st.subheader("📊 Feature Importance (Sample)")
    features = ['Area', 'Bedrooms', 'Bathrooms']
    importance = [0.6, 0.2, 0.2]
    fig, ax = plt.subplots()
    ax.bar(features, importance, color='skyblue')
    ax.set_ylabel("Importance Score")
    ax.set_title("Feature Impact on Price")
    st.pyplot(fig)
