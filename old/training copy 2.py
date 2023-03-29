import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime

# today minus 730 days
begin_1h = datetime.today() - pd.DateOffset(months=24)
# today minus 30 days
begin_5m = datetime.today() - pd.DateOffset(months=1)


def recuperer_donnees_yfinance(ticker, start, end):
    data = yf.download(ticker, start=begin_1h, interval="60m")
    return data


Offset = 1
# Récupérer les données
ticker = 'BTC-USD'
start = '2015-01-01'
end = (datetime.now() - pd.DateOffset(months=Offset)).strftime("%Y-%m-%d")
donnees = recuperer_donnees_yfinance(ticker, start, end)
print(f"len(donnees) = {len(donnees)}")
print(donnees.tail(5))
# Créer des indicateurs techniques avec TA-Lib
donnees['SMA'] = ta.SMA(donnees['Close'], timeperiod=20)
donnees['RSI'] = ta.RSI(donnees['Close'], timeperiod=14)
donnees['ADX'] = ta.ADX(donnees['High'],
                        donnees['Low'],
                        donnees['Close'],
                        timeperiod=14)
donnees['MACD'], donnees['MACDsignal'], donnees['MACDhist'] = ta.MACD(
    donnees['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
donnees['ADXR'] = ta.ADXR(donnees['High'],
                          donnees['Low'],
                          donnees['Close'],
                          timeperiod=14)
donnees['STOCHF_fastk'], donnees['STOCHF_fastd'] = ta.STOCHF(donnees['High'],
                                                             donnees['Low'],
                                                             donnees['Close'],
                                                             fastk_period=5,
                                                             fastd_period=3,
                                                             fastd_matype=0)
donnees['STOCHRSI_fastk'], donnees['STOCHRSI_fastd'] = ta.STOCHRSI(
    donnees['Close'],
    timeperiod=14,
    fastk_period=5,
    fastd_period=3,
    fastd_matype=0)
donnees['APO'] = ta.APO(donnees['Close'],
                        fastperiod=12,
                        slowperiod=26,
                        matype=0)
donnees['PPO'] = ta.PPO(donnees['Close'],
                        fastperiod=12,
                        slowperiod=26,
                        matype=0)
donnees['MOM'] = ta.MOM(donnees['Close'], timeperiod=10)
donnees['ROC'] = ta.ROC(donnees['Close'], timeperiod=10)
donnees['ROCR'] = ta.ROCR(donnees['Close'], timeperiod=10)
donnees['WILLR'] = ta.WILLR(donnees['High'],
                            donnees['Low'],
                            donnees['Close'],
                            timeperiod=14)
donnees['CCI'] = ta.CCI(donnees['High'],
                        donnees['Low'],
                        donnees['Close'],
                        timeperiod=14)
donnees['ATR'] = ta.ATR(donnees['High'],
                        donnees['Low'],
                        donnees['Close'],
                        timeperiod=14)
donnees['NATR'] = ta.NATR(donnees['High'],
                          donnees['Low'],
                          donnees['Close'],
                          timeperiod=14)
donnees['TRANGE'] = ta.TRANGE(donnees['High'], donnees['Low'],
                              donnees['Close'])

# Supprimer les lignes avec des valeurs NaN
donnees = donnees.dropna()

# Sélectionner les colonnes nécessaires
donnees = donnees[[
    'Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'ADX', 'MACD',
    'MACDsignal', 'MACDhist', 'ADXR', 'STOCHF_fastk', 'STOCHF_fastd',
    'STOCHRSI_fastk', 'STOCHRSI_fastd', 'APO', 'PPO', 'MOM', 'ROC', 'ROCR',
    'WILLR', 'CCI', 'ATR', 'NATR', 'TRANGE'
]]

# Normaliser les données
scaler = MinMaxScaler()
donnees_normalisees = scaler.fit_transform(donnees)

# Créer un scaler séparé pour la colonne "Close"
scaler_close = MinMaxScaler()
donnees_close = np.array(donnees['Close']).reshape(-1, 1)
donnees_close_normalisees = scaler_close.fit_transform(donnees_close)


def preparer_donnees(donnees, donnees_close, pas):
    X, y = [], []
    for i in range(pas, len(donnees)):
        X.append(donnees[i - pas:i])
        y.append(donnees_close[i])
    return np.array(X), np.array(y)


pas = 60
X, y = preparer_donnees(donnees_normalisees, donnees_close_normalisees, pas)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Créer le modèle
modele = Sequential()
modele.add(
    LSTM(50,
         return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])))
modele.add(Dropout(0.2))
modele.add(LSTM(50))
modele.add(Dropout(0.2))
modele.add(Dense(1))

# Compiler le modèle
modele.compile(optimizer='adam', loss='mean_squared_error')

# Entraîner le modèle
modele.fit(X_train,
           y_train,
           epochs=100,
           batch_size=32,
           validation_data=(X_test, y_test))

# Sauvegarder le modèle
modele.save('model.h5')

# Sauvegarder les scalers
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('scaler_close.pkl', 'wb') as f:
    pickle.dump(scaler_close, f)
