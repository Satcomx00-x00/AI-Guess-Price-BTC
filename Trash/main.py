from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
import talib as ta

import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential