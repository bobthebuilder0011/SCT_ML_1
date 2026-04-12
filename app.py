import streamlit as st
import pandas as pd
import numpy as np
import os
from model import HousePriceModel

st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏠")

# --- Model Loading with Caching ---
@st.cache_resource
def load_hpm():
    hpm = HousePriceModel()
    if not hpm.load():
        # If model doesn't exist, try to train it
        if os.path.exists('train.csv'):
            with st.spinner('Training model for the first time...'):
                df = hpm.load_raw_data()
                data = hpm.preprocess(df)
                hpm.train(data)
                hpm.save()
        else:
            return None, "Error: 'train.csv' not found. Please provide data to train the model."
    return hpm, None

# --- Main App Interface ---
st.title("🏠 House Price Predictor")
st.markdown("""
Predict the value of your property using our Linear Regression model.
This tool analyzes square footage, bedrooms, and bathrooms to give you an estimate.
""")

model, error_msg = load_hpm()

if error_msg:
    st.error(error_msg)
    st.info("Ensure `train.csv` is present in the project directory.")
else:
    st.success("Model loaded and ready for predictions!")
    
    with st.container():
        st.write("### Property Details")
        col1, col2 = st.columns(2)
        
        with col1:
            sqft = st.number_input("Square Footage (sqft)", min_value=100, max_value=20000, value=1500, step=50)
            beds = st.slider("Number of Bedrooms", 1, 10, 3)
            
        with col2:
            baths = st.slider("Number of Bathrooms", 1.0, 10.0, 2.0, 0.5)
            
        submit = st.button("Calculate Estimated Price", type="primary")

    if submit:
        with st.spinner('Predicting...'):
            try:
                prediction = model.predict(sqft, beds, baths)
                
                # Format price for display
                formatted_price = f"${prediction:,.2f}"
                
                st.divider()
                st.markdown(f"### Estimated Market Value")
                st.markdown(f"<h1 style='color: #2e7d32;'>{formatted_price}</h1>", unsafe_allow_html=True)
                
                # Display inputs back for confirmation
                st.info(f"Details: {sqft} sqft, {beds} beds, {baths} baths")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.divider()
st.caption("Built with ❤️ using Python, Scikit-Learn, and Streamlit.")
