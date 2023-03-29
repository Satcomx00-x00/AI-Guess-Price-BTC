import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from ta import add_all_ta_features
from ta.utils import dropna
from time import sleep as sl
import pandas as pd
import numpy as np
import talib as ta
import yfinance as yf

from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
import ccxt

binance = ccxt.binance()
all_last_price = [0,0,0,0]

# print with time
def printwt(text):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {text}")


def recuperer_dernieres_donnees_yfinance(ticker, start, end):
    # data = yf.download(ticker, start=start, end=end)
    data = binance.fetch_ohlcv(ticker, "15m", limit=9999)
    data = pd.DataFrame(
        data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    last_price = data['Close'].iloc[-1]

    return data


def predire_prochain_prix(donnees, modele, pas, scaler, scaler_close):
    dernier_bloc = donnees[-pas:]
    dernier_bloc = scaler.transform(dernier_bloc)
    dernier_bloc = np.reshape(dernier_bloc, (1, pas, dernier_bloc.shape[1]))
    prediction = modele.predict(dernier_bloc)
    prediction = scaler_close.inverse_transform(prediction)
    return prediction[0, 0]


# Charger le modèle
modele = load_model('model.h5')

while True:
    if len(all_last_price) > 4:
        all_last_price.pop(0)
    # Paramètres
    ticker = 'BTC/USD'
    # start is today minus 4 months
    start = (datetime.now() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    # end is today hour
    end = datetime.now().strftime("%Y-%m-%d")

    # Récupérer les dernières données
    donnees_recentes = recuperer_dernieres_donnees_yfinance(ticker, start, end)
    donnees_recentes=add_all_ta_features(donnees_recentes,open="Open", high="High", low="Low", close="Close", volume="Volume")
    print(donnees_recentes.columns)

    # Supprimer les lignes avec des valeurs NaN
    donnees_recentes = donnees_recentes.dropna()

    # Sélectionner les colonnes nécessaires
    donnees_recentes = donnees_recentes[[
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'ADX', 'MACD',
        'MACDsignal', 'MACDhist', 'ADXR', 'STOCHF_fastk', 'STOCHF_fastd',
        'STOCHRSI_fastk', 'STOCHRSI_fastd', 'APO', 'PPO', 'MOM', 'ROC', 'ROCR',
        'WILLR', 'CCI', 'ATR', 'NATR', 'TRANGE'
    ]]

    # Charger les scalers utilisés pour l'entraînement du modèle
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('scaler_close.pkl', 'rb') as f:
        scaler_close = pickle.load(f)

    pas = 60
    # Effectuer la prédiction
    prochain_prix = predire_prochain_prix(donnees_recentes, modele, pas,
                                          scaler, scaler_close)

    try:
        last_price = donnees_recentes['Close'].iloc[-1]
        percent = (prochain_prix - last_price) / last_price
        printwt(
            f"Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD."
        )
    except:
        last_price = donnees_recentes['Close'][-1]
        percent = (prochain_prix - last_price) / last_price
        printwt(
            f"Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD."
        )

    all_last_price.append(last_price)
    print(f'{all_last_price=}')

    # if price than 4 last price, buy
    if prochain_prix > all_last_price[-1] and prochain_prix > all_last_price[
            -2] and prochain_prix > all_last_price[
                -3] and prochain_prix > all_last_price[-4]:
        printwt("BUY")
    elif prochain_prix < all_last_price[-1] and prochain_prix < all_last_price[
            -2] and prochain_prix < all_last_price[
                -3] and prochain_prix < all_last_price[-4]:
        printwt("SELL")
    else:
        printwt("HOLD")
    sl(60)
