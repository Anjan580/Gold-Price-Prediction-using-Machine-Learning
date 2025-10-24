import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.joblib")  # Make sure this file exists in the same folder

# App title
st.title("🏆 Gold Price Prediction App")
st.write("Predict the Gold (GLD) price based on SPX, USO, SLV, and EUR/USD values.")

# Input fields
spx = st.number_input("Enter SPX value:", value=1447.16)
uso = st.number_input("Enter USO (Oil Price) value:", value=78.47)
slv = st.number_input("Enter SLV (Silver Price) value:", value=15.18)
eurusd = st.number_input("Enter EUR/USD exchange rate:", value=1.471692)

# Predict button
if st.button("Predict Gold Price"):
    # Prepare input data
    new_data = np.array([[spx, uso, slv, eurusd]])
    
    # Predict using the loaded model
    predicted_price = model.predict(new_data)[0]
    
    # Display result
    st.success(f"💰 Predicted Gold Price (GLD): {predicted_price:.2f}")

# The model.joblib file should be created by training a model in the Gold-Price-Prediction.ipynb notebook.
