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

# save to csv
# pd.DataFrame(data)#.to_csv('btc.csv', index=False, header=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# convert timestamp to datetime
data['date'] = pd.to_datetime(data['timestamp'], unit='ms')

# train test split
train_size = int(len(data) * 0.80)
test_size = len(data) - train_size
# TypeError: list indices must be integers or slices, not tuple
train, test = data[0:train_size], data[train_size:len(data)]
# np.asarray(y_train).astype(np.float32)
train = np.asarray(train).astype(np.float32)
test = np.asarray(test).astype(np.float32)


def create_features(df):
    # add all indicators with add_all_ta_features
    df = add_all_ta_features( df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    return df



ohlcv = create_features(data)


# create a cnn model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(ohlcv.shape[1], 1)))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model = create_model()
model.summary()

history = model.fit(ohlcv, data['close'], epochs=10, batch_size=32, verbose=1, shuffle=False)
# Failed to convert a NumPy array to a Tensor (Unsupported object type int).
# plot loss and accuracy
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
# accuracy
plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()

