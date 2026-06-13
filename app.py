import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("SHIP_GASTURBINE_FUELFLOW")
st.subheader("Gas Turbine Performance Analysis using ML & AI")

st.write("Enter the input values:")

lever_position = st.number_input("Lever Position")
ship_speed = st.number_input("Ship Speed")
gt_shaft = st.number_input("GT Shaft Torque")
gt_rate = st.number_input("GT Rate of Revolutions")

# Prediction
if st.button("Predict Fuel Flow"):
    input_data = [[lever_position, ship_speed, gt_shaft, gt_rate]]
    prediction = model.predict(input_data)

    st.success(f"Predicted Fuel Flow: {prediction[0]}")
