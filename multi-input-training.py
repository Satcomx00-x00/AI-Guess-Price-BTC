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
from ta import add_all_ta_features
# today minus 730 days
begin_1h = datetime.today() - pd.DateOffset(months=24)
# today minus 30 days
begin_5m = datetime.today() - pd.DateOffset(months=1)
chunksize = 10000

# def recuperer_donnees_yfinance(ticker, start, end):
#     # PANDAS READ csv
#     data = pd.read_csv(
#         r'C:\Users\MrBios\Documents\Development\test\csv\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
#         dtype={
#             'Timestamp': 'int32',
#             'Open': 'float32',
#             'High': 'float32',
#             'Low': 'float32',
#             'Close': 'float32',
#             'Volume_(BTC)': 'float32',
#             'Volume_(Currency)': 'float32',
#             'Weighted_Price': 'float32'
#         },
#         parse_dates=['Timestamp'])

#     # remove all rows with NaN
#     data = data.dropna()
#     # remove two last columns
#     data = data.iloc[:, :-2]
#     # print columns
#     print(data.columns)
#     # rename columns
#     data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

#     # data = yf.download(ticker, start=begin_1h, interval="60m")
#     return data


def train(datasets, pas=60, epochs=100, batch_size=32, learning_rate=0.001):
    models = []
    scalers = []

    for dataset in datasets:
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_scaled = scaler.fit_transform(dataset)

        # Prepare the data for training
        X, y = [], []
        for i in range(pas, len(dataset_scaled)):
            X.append(dataset_scaled[i - pas:i])
            y.append(dataset_scaled[i])
        X, y = np.array(X), np.array(y)

        # Split the data into training and validation sets
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

        # Add the trained model and scaler to the list
        models.append(model)
        scalers.append(scaler)

    return models, scalers


Offset = 1
# Récupérer les données
ticker = 'BTC-USD'
start = '2015-01-01'
end = (datetime.now() - pd.DateOffset(months=Offset)).strftime("%Y-%m-%d")
# donnees = recuperer_donnees_yfinance(ticker, start, end)
# print(f"len(donnees) = {len(donnees)}")
# print(donnees.tail(5))
# # Créer des indicateurs techniques avec TA-Lib
# donnees = add_all_ta_features(donnees,
#                               open="Open",
#                               high="High",
#                               low="Low",
#                               close="Close",
#                               volume="Volume",
#                               fillna=True)

# # Supprimer les lignes avec des valeurs NaN
# donnees = donnees.dropna()

# # Sélectionner les colonnes nécessaires
# donnees = donnees[[
#     'Open', 'High', 'Low', 'Close', 'Volume', 'volume_adi', 'volume_obv',
#     'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt',
#     'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
#     'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
#     'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch',
#     'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
#     'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
#     'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
#     'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
#     'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
#     'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
#     'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
#     'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
#     'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc',
#     'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
#     'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
#     'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up', 'trend_psar_down',
#     'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi',
#     'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
#     'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
#     'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo',
#     'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo',
#     'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama', 'others_dr',
#     'others_dlr', 'others_cr'
# ]]
# # save data to csv file with donnees as header
# donnees.to_csv(+

#     r'C:\Users\MrBios\Documents\Development\test\csv\Featured_bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
#     index=False)

# input_path = r'C:\Users\MrBios\Documents\Development\test\csv\Featured_bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'
# for chunk in pd.read_csv(input_path, chunksize=chunksize):
#     parsed_data = parsed_data.append(chunk)

# import all datasets from C:\Users\MrBios\Documents\Development\test\csv\chunked
# and append them to parsed_data

# for file in os.listdir(r'C:\Users\MrBios\Documents\Development\test\csv\chunked'):
#     if file.endswith('.csv'):
#         for chunk in pd.read_csv(r'C:\Users\MrBios\Documents\Development\test\csv\chunked' + file, chunksize=chunksize):
#             parsed_data = parsed_data.append(chunk)
# concat all datasets from C:\Users\MrBios\Documents\Development\test\csv\chunked
# and append them to parsed_data
headers = [
    'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'volume_adi',
    'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em',
    'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
    'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
    'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch',
    'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc',
    'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up', 'trend_psar_down',
    'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi',
    'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
    'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
    'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo',
    'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo',
    'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama', 'others_dr',
    'others_dlr', 'others_cr'
]

import os

donnees = pd.DataFrame()

for file in os.listdir(
        r'C:\Users\MrBios\Documents\Development\test\csv\chunked\\'):
    if file.endswith('.csv'):
        # use pandas concat to append all datasets
        donnees = pd.concat([
            donnees,
            pd.read_csv(
                r'C:\Users\MrBios\Documents\Development\test\csv\chunked\\' +
                file,
                names=headers,
                header=0)
        ])

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
modele.save('models/model.h5')

# Sauvegarder les scalers
with open('scalers/scaler_big.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('scalers/scaler_close_big.pkl', 'wb') as f:
    pickle.dump(scaler_close, f)
