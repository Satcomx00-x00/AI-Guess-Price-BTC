import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from datetime import datetime
import ta
import joblib

# Load the Bitcoin price data
bitcoin_data = pd.read_csv(r'C:\Users\MrBios\Documents\Development\test\csv\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')

# Drop missing values
bitcoin_data.dropna(inplace=True)

# change the name of the columns
bitcoin_data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
# change the name of the column 'Volume_(Currency)' to 'Volume_Currency' to avoid problems with the library ta
bitcoin_data.rename(columns={'Volume_(BTC)': 'Volume'}, inplace=True)
# drop the column 'Weighted_Price' and 'Volume_Currency'
bitcoin_data.drop(columns=['Volume_(Currency)'], inplace=True)

# Keep the necessary columns
bitcoin_data = bitcoin_data[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Weighted_Price']]
print(bitcoin_data.shape)
# Convert Timestamp to datetime format
bitcoin_data['Timestamp'] = pd.to_datetime(bitcoin_data['Timestamp'], unit='s')

# Compute RSI
bitcoin_data['RSI'] = ta.momentum.rsi(close=bitcoin_data["Close"], fillna=True)

# Compute MACD
bitcoin_data['MACD'] = ta.trend.macd(close=bitcoin_data["Close"], fillna=True)
bitcoin_data['MACD_signal'] = ta.trend.macd_signal(close=bitcoin_data["Close"], fillna=True)
bitcoin_data['MACD_diff'] = ta.trend.macd_diff(close=bitcoin_data["Close"], fillna=True)
# Weighted_Price
bitcoin_data['Weighted_Price'] = (bitcoin_data['High'] + bitcoin_data['Low'] + bitcoin_data['Close']) / 3

# Prepare the data for training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin_data.drop(columns=['Timestamp']))

# Define the time step and number of features
time_step = 60
X = []
y = []
for i in range(time_step, len(scaled_data) - 10):
    X.append(scaled_data[i - time_step:i, :])
    y.append(scaled_data[i:i + 10, 3])  # Predicting the 'Close' column
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=y_train.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save the model
version = "V5"
now = datetime.now()
format_date = now.strftime("%Y-%m-%d-%H-%M")
model.save(f'{version}/models/2-{version}-model_{format_date}.h5')

# Save the scaler
scaler_filename = f'{version}/models/{version}-scaler_{format_date}.joblib'
joblib.dump(scaler, scaler_filename)

# save weights
model.save_weights(f'{version}/models/{version}-weights_{format_date}.h5')
