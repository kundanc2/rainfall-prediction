import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import os
import joblib

# Load your dataset, replace 'rainfall.csv' with your actual file path
df = pd.read_csv("rainfall.csv")
df = df.fillna(df.mean(numeric_only=True))


# Define the month mapping
Month_map = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9,
    'OCT': 10, 'NOV': 11, 'DEC': 12
}


# Group the data by 'SUBDIVISION'
subdivision_groups = df.groupby('SUBDIVISION')




# Initialize an empty DataFrame to store the results
result_df = pd.DataFrame()




# Iterate through each subdivision group
for subdivision, group_data in subdivision_groups:
    # Create a new DataFrame to store the results for this subdivision
    subdivision_data = pd.DataFrame()




    # Iterate through the unique years
    for year in group_data['YEAR'].unique():
        # Calculate the monthly average rainfall for this year
        year_data = group_data[group_data['YEAR'] == year]
        monthly_avg_rainfall = []
        for month_abbr, month_num in Month_map.items():
            avg_rainfall = year_data[month_abbr].mean()
            monthly_avg_rainfall.append({'Year': year, 'Month': month_num, 'Avg_Rainfall': avg_rainfall})




        # Create a DataFrame for this year and concatenate it with the subdivision_data
        year_df = pd.DataFrame(monthly_avg_rainfall)
        subdivision_data = pd.concat([subdivision_data, year_df], ignore_index=True)




    # Add the 'SUBDIVISION' column with the subdivision name to the subdivision_data
    subdivision_data['SUBDIVISION'] = subdivision




    # Concatenate the subdivision_data with the result_df
    result_df = pd.concat([result_df, subdivision_data], ignore_index=True)




# Sort the result DataFrame by 'SUBDIVISION', 'year', and 'month'
result_df.sort_values(by=['SUBDIVISION', 'Year', 'Month'], inplace=True)




# Reset the index
result_df.reset_index(drop=True, inplace=True)




label_encoder = LabelEncoder()
# Encode the 'SUBDIVISION' column
result_df['SUBDIVISION'] = label_encoder.fit_transform(result_df['SUBDIVISION'])
# Initialize dictionaries to store models for each month
models = {}
month_groups = result_df.groupby('Month')


for month, group_data in month_groups:
    # Create a new DataFrame to store the results for this month
    month_data = group_data.copy()
    # Extract features and target
    X = month_data[['Year', 'SUBDIVISION']].values.astype(int)
    y = month_data['Avg_Rainfall'].values.astype(int)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


    # Initialize and train a model for the current month
    model = XGBRegressor()
    model.fit(X_train, y_train)


    # Make predictions for the test set
    y_pred = model.predict(X_test)


    # Calculate and print MAE for the current month
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE for {month}: {mae:.2f} mm")


    # Store the model for the current month
    models[month] = model


# Save the trained models to a file

# Define the folder name
folder_name = "models"

# Check if the folder exists, and if not, create it
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for month, model in models.items():
    model_filename = os.path.join(folder_name, f"{month}_model.pkl")
    joblib.dump(model, model_filename)
