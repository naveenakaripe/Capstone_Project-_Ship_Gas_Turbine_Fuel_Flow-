import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("SHIP_GASTURBINE_FUELFLOW")
st.subheader("Gas Turbine Performance Analysis using ML & AI")

# Input fields
lever_position = st.number_input("Lever Position")
ship_speed = st.number_input("Ship Speed")
gt_shaft_torque = st.number_input("GT Shaft Torque")
gt_rate_revolutions = st.number_input("GT Rate of Revolutions")

if st.button("Predict Fuel Flow"):
    input_data = np.array([[lever_position,
                            ship_speed,
                            gt_shaft_torque,
                            gt_rate_revolutions]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Fuel Flow: {prediction[0]}")
