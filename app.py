import streamlit as st
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load("scaler.pkl")
model = joblib.load("knn_model.pkl")
expected_columns = joblib.load("columns.pkl")
print("Scaler expects:", scaler.feature_names_in_)
st.title(f"Earthquake Alert Prediction 🌍")

magnitude = st.number_input("Magnitude", 6.0, 8.6, 7.0)
depth = st.number_input("Depth (km)", 2.0, 670.0, 50.0)
cdi = st.number_input("CDI", 0, 9, 7)
mmi = st.number_input("MMI", 1, 9, 7)
sig = st.number_input("Significance", 0, 255, 130)

if st.button("Predict"):
    depth_log = np.log1p(depth)
    input_df = pd.DataFrame([[depth, cdi, mmi, sig, magnitude, depth_log]],
                            columns=['depth', 'cdi', 'mmi', 'sig', 'magnitude', 'depth_log'])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    if prediction.lower() == "green":
        st.success(f"{prediction} alert")
    elif prediction.lower() == "red":
        st.error(f"{prediction} alert")
    else:  # any other color like orange, yellow, blue
        st.warning(f"{prediction.upper()} alert")


