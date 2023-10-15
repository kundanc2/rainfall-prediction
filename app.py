import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib



# Load dataset
df = pd.read_csv("rainfall.csv")
df = df.fillna(df.mean(numeric_only=True))

subdivision_values = df['SUBDIVISION'].unique().tolist()

# Define the month mapping
Month_map = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9,
    'OCT': 10, 'NOV': 11, 'DEC': 12
}

#load models 
models = {}
for month in Month_map.values():
    model_filename = f"./models/{month}_model.pkl"
    model = joblib.load(model_filename)
    models[month] = model
    

#Defining 
label_encoder = LabelEncoder()
label_encoder.fit(df['SUBDIVISION'])


st.title("Rainfall Prediction")
st.write("This app predicts monthly rainfall for a given year, month, and subdivision.")

# Collect user input
year_input = st.number_input("Enter the year:", step=1, format="%d")
month_input = st.selectbox("Select the month:", list(Month_map.keys()))
subdivision_input = st.selectbox("Enter the subdivision:",list(subdivision_values))

if st.button("Predict Rainfall"):
    # Preprocess input
    month_num_input = Month_map.get(month_input, None)
    if month_num_input is None:
        st.error("Invalid month input.")
    else:
        subdivision_encoded = label_encoder.transform([subdivision_input])
        input_features = np.array([[year_input, subdivision_encoded[0]]])
        # Get the corresponding model for the input month
        model = models.get(month_num_input, None)
        if model:
            # Make predictions using the selected model
            predicted_rainfall = model.predict(input_features)
            st.success(f"Predicted rainfall for {month_input} {year_input} in subdivision {subdivision_input} is {predicted_rainfall[0]:.2f} mm")
        else:
            st.error("No model available for the specified month and year.")
        

