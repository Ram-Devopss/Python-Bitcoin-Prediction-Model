# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests

# Load the Bitcoin price data
url = 'https://api.coindesk.com/v1/bpi/historical/close.json?start=2013-01-01&end=2024-01-01'
response = requests.get(url)
data = response.json()

# Extracting only the historical price data
prices = data['bpi']

# Creating a DataFrame from the extracted data
data_df = pd.DataFrame(prices.items(), columns=['Date', 'Price'])
data_df['Date'] = pd.to_datetime(data_df['Date'])
data_df.set_index('Date', inplace=True)

# Feature Engineering
data_df['30 Day MA'] = data_df['Price'].rolling(window=30).mean() # 30 Day Moving Average
data_df['Std_dev'] = data_df['Price'].rolling(window=30).std() # 30 Day Standard Deviation
data_df['Upper Band'] = data_df['30 Day MA'] + (data_df['Std_dev'] * 2) # Upper Band
data_df['Lower Band'] = data_df['30 Day MA'] - (data_df['Std_dev'] * 2) # Lower Band

# Creating the feature matrix X and target vector y
X = data_df[['30 Day MA', 'Std_dev', 'Upper Band', 'Lower Band']].dropna()
y = data_df['Price'].loc[X.index]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Making predictions
y_pred = regressor.predict(X_test)

# Visualizing the results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, color='red', label='Actual Bitcoin Price')
plt.plot(y_test.index, y_pred, color='blue', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
