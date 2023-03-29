import ccxt
import numpy as np
import pandas as pd

import tensorflow as tf

# import add_all_ta_features
from ta import add_all_ta_features

# import tensorflow as tf
import matplotlib.pyplot as plt

# fetch btc ohlcv
# data = ccxt.binance().fetch_ohlcv('BTC/USDT', '1h')

# load data from csv
data = pd.read_csv('btc.csv')

# convert timestamp to datetime
data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

# train test split
train_size = int(len(data) * 0.80)
test_size = len(data) - train_size
# TypeError: list indices must be integers or slices, not tuple
train, test = data[0:train_size], data[train_size:len(data)]

def create_features(df):
    # add all indicators with add_all_ta_features
    df = add_all_ta_features( df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    return df


ohlcv = create_features(data)

# use sklearn to scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(ohlcv)
scaled_data = scaler.transform(ohlcv)

