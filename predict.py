import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings

warnings.filterwarnings('ignore', 'The behavior of `series.*`', FutureWarning)
warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
import pickle
from datetime import datetime
from time import sleep as sl

import ccxt
import numpy as np
import pandas as pd
import talib as ta
import yfinance as yf
from ta import add_all_ta_features
from tensorflow.keras.models import load_model

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
modele = load_model('models/model.h5')

while True:
    # Paramètres
    ticker = 'BTC/USD'
    # start is today minus 4 months
    start = (datetime.now() - pd.DateOffset(months=4)).strftime("%Y-%m-%d")
    # end is today hour
    end = datetime.now().strftime("%Y-%m-%d")

    # Récupérer les dernières données
    donnees_recentes = recuperer_dernieres_donnees_yfinance(ticker, start, end)
    print(f"len(donnees_recentes) = {len(donnees_recentes)}")
    donnees_recentes = add_all_ta_features(donnees_recentes,
                                           open="Open",
                                           high="High",
                                           low="Low",
                                           close="Close",
                                           volume="Volume",
                                           fillna=True)
    # Supprimer les lignes avec des valeurs NaN
    donnees_recentes = donnees_recentes.dropna()

    # Sélectionner les colonnes nécessaires
    donnees_recentes = donnees_recentes[[
        'Open', 'High', 'Low', 'Close', 'Volume', 'volume_adi', 'volume_obv',
        'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt',
        'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
        'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
        'volatility_bbhi', 'volatility_bbli', 'volatility_kcc',
        'volatility_kch', 'volatility_kcl', 'volatility_kcw', 'volatility_kcp',
        'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
        'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp',
        'volatility_atr', 'volatility_ui', 'trend_macd', 'trend_macd_signal',
        'trend_macd_diff', 'trend_sma_fast', 'trend_sma_slow',
        'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos',
        'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix',
        'trend_mass_index', 'trend_dpo', 'trend_kst', 'trend_kst_sig',
        'trend_kst_diff', 'trend_ichimoku_conv', 'trend_ichimoku_base',
        'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx',
        'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
        'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
        'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
        'trend_psar_down', 'trend_psar_up_indicator',
        'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
        'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
        'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
        'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo',
        'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo',
        'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama',
        'others_dr', 'others_dlr', 'others_cr'
    ]]

    # Charger les scalers utilisés pour l'entraînement du modèle
    with open('scalers/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('scalers/scaler_close.pkl', 'rb') as f:
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
    # append to file
    with open('csv/prediction.txt', 'a', encoding="UTF8") as f:
        f.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Le prix prévu du {ticker} pour la prochaine période est de {prochain_prix:.2f} USD | actuel: {last_price} | diff: {percent:.2f}%\n"
        )
    # store last price and prediction in csv
    with open('csv/prediction.csv', 'a', encoding="UTF8") as f:
        st = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ticker},{prochain_prix:.2f},{last_price},{percent:.2f}\n"
        f.write(st)

    sl(60)
