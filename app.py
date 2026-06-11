import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
# Page Title
st.title("CAPSTONE_SHIP_GAS_TURBINE_FUEL_FLOW")

# Load Dataset
df = pd.read_csv("/content/data.csv")
# Remove index column if present
if 'index' in df.columns:
    df.drop(columns=['index'], inplace=True)
  # Remove extra spaces in column names
df.columns = df.columns.str.strip()
# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())
# Dataset Information
st.subheader("Dataset Information")
st.write("Shape:", df.shape)
# Missing Values
st.subheader("Missing Values")
# Correlation Heatmap
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)
# Fuel Flow Trend
st.subheader("Fuel Flow Trend")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df['Fuel flow (mf) [kg/s]'])
ax.set_title("Fuel Flow Trend")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Fuel Flow")
st.pyplot(fig)
# Speed vs Fuel Flow
st.subheader("Ship Speed vs Fuel Flow")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(
    df['Ship speed (v)'],
    df['Fuel flow (mf) [kg/s]']
)
ax.set_xlabel("Ship Speed")
ax.set_ylabel("Fuel Flow")
ax.set_title("Ship Speed vs Fuel Flow")
st.pyplot(fig)
# Fuel Flow Distribution
st.subheader("Fuel Flow Distribution")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x=df['Fuel flow (mf) [kg/s]'], ax=ax)
st.pyplot(fig)
# Machine Learning
st.subheader("Fuel Flow Prediction")

X = df.drop(columns=['Fuel flow (mf) [kg/s]'])
y = df['Fuel flow (mf) [kg/s]']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)

st.success(f"Model Accuracy (R² Score): {accuracy:.4f}")
# Prediction Plot
y_pred = rf.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Fuel Flow")
ax.set_ylabel("Predicted Fuel Flow")
ax.set_title("Actual vs Predicted Fuel Flow")
st.pyplot(fig)
# Anomaly Detection
st.subheader("Engine Health Monitoring (Anomaly Detection)")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

df['Anomaly'] = iso.fit_predict(df)

fig, ax = plt.subplots(figsize=(8, 4))
scatter = ax.scatter(
    df.index,
    df['Fuel flow (mf) [kg/s]'],
    c=df['Anomaly'],
    cmap='coolwarm'
)

ax.set_title("Anomaly Detection in Fuel Flow")
ax.set_xlabel("Index")
ax.set_ylabel("Fuel Flow")

st.pyplot(fig)
# Anomaly Summary
st.subheader("Anomaly Summary")
st.write(df['Anomaly'].value_counts())
%%writefile app.py
%%writefile requirements.txt
streamlit
pandas
matplotlib
seaborn
scikit-learn
st.write(df.isnull().sum())
