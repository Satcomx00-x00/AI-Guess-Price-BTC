import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pickle

from datetime import datetime

# today minus 730 days
begin_1h = datetime.today() - pd.DateOffset(months=24)
# today minus 30 days
begin_5m = datetime.today() - pd.DateOffset(months=1)


# Define function to retrieve data from Yahoo Finance
def get_data_yfinance(ticker, start, end):
    data = yf.download(ticker, start=begin_1h, interval="60m")
    return data


# Load the trained model
model = load_model('model.h5')

# Load the scalers
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('scaler_close.pkl', 'rb') as f:
    scaler_close = pickle.load(f)

# Set new timeframe
offset = 6
start = '2015-01-01'
end = (datetime.now() - pd.DateOffset(months=offset)).strftime("%Y-%m-%d")

# Retrieve new data
ticker = 'ETH-USD'
data = get_data_yfinance(ticker, start, end)

# Add technical indicators
data['SMA'] = ta.SMA(data['Close'], timeperiod=20)
data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
data['ADX'] = ta.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

# Drop NaN values and select necessary columns
data = data.dropna()
data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'ADX']]

# Normalize the data
data_normalized = scaler.transform(data)

# Create separate scaler for the Close column
data_close = np.array(data['Close']).reshape(-1, 1)
data_close_normalized = scaler_close.transform(data_close)

# Prepare the data
lookback = 60
X_new, y_new = [], []
for i in range(lookback, len(data_normalized)):
    X_new.append(data_normalized[i - lookback:i])
    y_new.append(data_close_normalized[i])

X_new, y_new = np.array(X_new), np.array(y_new)

# Evaluate the model on the new data
loss = model.evaluate(X_new, y_new)

print(f'Loss on new data: {loss}')
