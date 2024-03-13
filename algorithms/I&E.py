# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load historical data (replace 'your_data.csv' with your actual dataset)
data = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)

# Check the first few rows of the dataset
print(data.head())

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Historical Income and Expenses')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

# Define a function to evaluate ARIMA model
def evaluate_arima_model(train_data, test_data, order):
    history = list(train_data)
    predictions = []
    
    # Walk-forward validation
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test_data[t])
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_data, predictions))
    
    return rmse, predictions

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Define the order of the ARIMA model (p, d, q)
order = (5, 1, 0)  # Example values, you may need to fine-tune

# Evaluate the ARIMA model
rmse, predictions = evaluate_arima_model(train, test, order)

# Print RMSE
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(test, label='Actual')
plt.plot(predictions, color='red', label='Predicted')
plt.title('ARIMA Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()
