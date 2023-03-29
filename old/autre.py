import ccxt
import numpy as np
import pandas as pd
import talib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Constants
SYMBOL = 'BTC/USDT'
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h',  '12h', '1d', '3d', '1w', '1M']
PREDICTION_STEPS = 10
HISTORY_POINTS = 60

# Initialize the exchange
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
})

# Data retrieval
def get_data(timeframe):
    data = exchange.fetch_ohlcv(SYMBOL, timeframe)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

# Feature engineering
def add_indicators(df):
    df['RSI'] = talib.RSI(df['close'])
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
    df.dropna(inplace=True)
    return df

# Data preprocessing
def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# Model creation
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(PREDICTION_STEPS),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training data preparation
def prepare_training_data(data, history_points):
    X = []
    y = []
    for i in range(history_points, len(data) - PREDICTION_STEPS):
        X.append(data[i - history_points:i])
        y.append(data[i:i + PREDICTION_STEPS, 3])  # 'close' column
    return np.array(X), np.array(y)

# Data preprocessing
# Feature engineering and data preprocessing
def process_data(df):
    df = add_indicators(df)

    # Separate the target column
    target_data = df[['close']]
    input_data = df.drop(columns=['close'])

    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    scaled_input_data = input_scaler.fit_transform(input_data)
    scaled_target_data = target_scaler.fit_transform(target_data)

    scaled_data = np.hstack((scaled_input_data, scaled_target_data))

    return scaled_data, input_scaler, target_scaler

# Constants
SYMBOL = 'BTC/USDT'
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '6h', '12h', '1d', '3d', '1w', '1M']
PREDICTION_STEPS = 10
HISTORY_POINTS = 120

# Train and predict for a specific timeframe
def train_and_predict(timeframe):
    df = get_data(timeframe)
    scaled_data, input_scaler, target_scaler = process_data(df)

    X_train, y_train = prepare_training_data(scaled_data, HISTORY_POINTS)

    # Check if there are enough data points to create the training dataset
    if X_train.size == 0 or y_train.size == 0:
        print(f"Not enough data for the {timeframe} timeframe.")
        return []

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = create_model(input_shape)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    last_sequence = scaled_data[-HISTORY_POINTS:]
    last_sequence = np.expand_dims(last_sequence, axis=0)
    predicted_candles = model.predict(last_sequence)

    # Inverse transform only for the 'close' column using the target scaler
    predicted_candles = target_scaler.inverse_transform(predicted_candles)
    predicted_candles = predicted_candles[0]  # Remove the batch dimension

    return predicted_candles


# Main
for timeframe in TIMEFRAMES:
    print(f"Predicted next 10 candles for the {timeframe} timeframe:")
    predicted_candles = train_and_predict(timeframe)
    for i, close_price in enumerate(predicted_candles):
        print(f"Step {i + 1}: {close_price:.2f}")
    print()
