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
donnees = add_all_ta_features(donnees,
                              open="Open",
                              high="High",
                              low="Low",
                              close="Close",
                              volume="Volume",
                              fillna=True)

# Supprimer les lignes avec des valeurs NaN
donnees = donnees.dropna()

# Sélectionner les colonnes nécessaires
donnees = donnees[[
    'Open', 'High', 'Low', 'Close', 'Volume', 'volume_adi', 'volume_obv',
    'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt',
    'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
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
modele.save('models/model.h5')

# Sauvegarder les scalers
with open('scalers/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('scalers/scaler_close.pkl', 'wb') as f:
    pickle.dump(scaler_close, f)
