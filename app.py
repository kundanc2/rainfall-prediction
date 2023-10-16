import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
import matplotlib.pyplot as plt


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
        
if st.checkbox("Plot Yearly Rainfall Prediction"):
    month_num_input = Month_map.get(month_input, None)
    if month_num_input is None:
        st.error("Invalid month input.")
    else:
        subdivision_encoded = label_encoder.transform([subdivision_input])

        # Initialize lists to store monthly predictions
        yearly_predictions = []
        for month in range(1, 13):
            input_features = np.array([[year_input, subdivision_encoded[0]]])
            model = models.get(month, None)
            if model:
                predicted_rainfall = model.predict(input_features)
                yearly_predictions.append(predicted_rainfall[0])
            else:
                yearly_predictions.append(0.0)  # Handle the case when the model is not available

        # Plot the yearly prediction
        fig, ax = plt.subplots()
        ax.plot(list(Month_map.keys()), yearly_predictions, marker='o')
        ax.set(xlabel='Month', ylabel='Rainfall (mm)', title=f"Yearly Rainfall Prediction for {year_input} in {subdivision_input}")
        st.pyplot(fig)

if st.checkbox("Plot years data of Subdivison"):
    subdivision_encoded = label_encoder.transform([subdivision_input])
    
    # Initialize lists to store yearly predictions
    yearly_predictions = []
    years = range(year_input - 10, year_input)  # Get the past 5 years
    for year in years:
        # Initialize a list to store monthly predictions for the current year
        monthly_predictions = []
        for month in range(1, 13):
            input_features = np.array([[year, subdivision_encoded[0]]])
            model = models.get(month, None)
            if model:
                predicted_rainfall = model.predict(input_features)
                monthly_predictions.append(predicted_rainfall[0])
                
            else:
                monthly_predictions.append(0.0) 
                # Handle the case when the model is not available
        
        # Calculate the yearly average for the current year
        yearly_avg = sum(monthly_predictions) / 12
        
        yearly_predictions.append(yearly_avg)

    # Plot the yearly prediction
    fig, ax = plt.subplots()
    ax.plot(years, yearly_predictions, marker='o')
    ax.set(xlabel='Year', ylabel='Yearly Rainfall (mm)', title=f"Yearly Rainfall Prediction for {subdivision_input} (Last 5 Years)")
    st.pyplot(fig)
