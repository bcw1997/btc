#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:43:30 2023

@author: bradleywest
"""

from os import chdir, getcwd
wd=getcwd()
chdir(wd)

import pandas as pd

##### Data Preprocessing #####

# read init csv in 
df = pd.read_csv("dataset.csv")

# Convert timestamp from milliseconds to seconds
df['timestamp'] = df['timestamp'] / 1000

# Convert the Unix timestamp to a human-readable datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# clean up the datetime variable and make it more readable
df['formatted_datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Create Test/Training sets for data. 
# Do this based on a Temportal Dependency (timestamp or datetime)
# That way I can predict future BTC fee rate

column_names = df.columns.tolist()

print(column_names)

##### Feature Selection, Data Splitting, Feature Scaling #####

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# Create X/Ys & Target column naming

X = df[['height', 'size', 'tx_count', 'difficulty', 
        'total_fees', 'fee_range_min', 'fee_range_max']]

y = df['avg_fee_rate']  # wish to predict the average fee rate

# Create the time series object
n_splits = 5
tscv = TimeSeriesSplit(n_splits)

# init scaler 
scaler = StandardScaler()


# Iterate through the splits 
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
##### Model Selection #####

### Linear Regression ###

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# init Linear Regression 
linear_reg_model = LinearRegression()

# Fit the model 
linear_reg_model.fit(X_train_scaled, y_train)

## Predictions and Evaluations 

# Predict on the scaled test data
y_pred = linear_reg_model.predict(X_test_scaled)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

coef = linear_reg_model.coef_
intercept = linear_reg_model.intercept_

# Understanding the outcome 

import matplotlib.pyplot as plt

# Create a DataFrame to store feature names and their corresponding coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coef})

# Sort the DataFrame by coefficient magnitude
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

# Plot the coefficients
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Linear Regression Coefficients')
plt.show()

# doesnt look good at all. 

### Gradient Boosting ###

from sklearn.ensemble import GradientBoostingRegressor
# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Initialize lists to store evaluation metrics
mae_scores = []
mse_scores = []
r2_scores = []

# Iterate through the splits
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Gradient Boosting model on the scaled training data
    gb_model.fit(X_train_scaled, y_train)
    
    # Predict the target variable for the test set
    y_pred = gb_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Append scores to lists
    mae_scores.append(mae)
    mse_scores.append(mse)
    r2_scores.append(r2)
    
# Calculate average scores across folds
average_mae = sum(mae_scores) / len(mae_scores)
average_mse = sum(mse_scores) / len(mse_scores)
average_r2 = sum(r2_scores) / len(r2_scores)

# Print or display the average evaluation scores
print(f"Mean Absolute Error: {average_mae}")
print(f"Mean Squared Error: {average_mse}")
print(f"R-squared: {average_r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel('Actual Average Fee Rate')
plt.ylabel('Predicted Average Fee Rate')
plt.title('Actual vs. Predicted Average Fee Rate')
plt.show()

# Calculate and plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, color='red', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Actual Average Fee Rate')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals Plot')
plt.show()