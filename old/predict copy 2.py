import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    # Paramètres
    ticker = 'BTC/USD'
    # start is today minus 4 months
    start = (datetime.now() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    # end is today hour
    end = datetime.now().strftime("%Y-%m-%d")

    # Récupérer les dernières données
    donnees_recentes = recuperer_dernieres_donnees_yfinance(ticker, start, end)

    # Créer des indicateurs techniques avec TA-Lib
    donnees_recentes['SMA'] = ta.SMA(donnees_recentes['Close'], timeperiod=20)
    donnees_recentes['RSI'] = ta.RSI(donnees_recentes['Close'], timeperiod=14)
    donnees_recentes['ADX'] = ta.ADX(donnees_recentes['High'],
                                     donnees_recentes['Low'],
                                     donnees_recentes['Close'],
                                     timeperiod=14)

    # Supprimer les lignes avec des valeurs NaN
    donnees_recentes = donnees_recentes.dropna()

    # Sélectionner les colonnes nécessaires
    donnees_recentes = donnees_recentes[[
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'ADX'
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
        printwt(
            f"Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD."
        )
    except:
        last_price = donnees_recentes['Close'][-1]
        printwt(
            f"Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD."
        )
    # append to file
    with open('prediction.txt', 'a', encoding="UTF8") as f:
        f.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD | actuel: {last_price}.\n"
        )
    # store last price and prediction in csv
    with open('prediction.csv', 'a', encoding="UTF8") as f:
        st = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ticker},{prochain_prix:.2f},{last_price}\n"
        f.write(st)
        
    sl(60)
