import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
import json

warnings.filterwarnings('ignore', 'The behavior of `series.*`', FutureWarning)
warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)

import pickle
from datetime import datetime
from time import sleep as sl
from rich.console import Console
from rich.table import Table
import ccxt
import numpy as np
import pandas as pd
# import talib as ta
from ta import add_all_ta_features
from tensorflow.keras.models import load_model
from SatcomDiscord import PredictionMessage
from threading import Thread
import asyncio
# docker run -it --rm -e DISCORD_TOKEN=MTA5MDM4MjU4Mzg5Mzg1NjMwNg.GZajTp.s8xWLSqe16EztPZ47zq4xCkbDU36LUXfIwMA_E -e DISCORD_CHANNEL=1090381983244365904 satcomx00/ai-guess-price-btc:1.0

# import argparse
# parser = argparse.ArgumentParser(prog='AI-Guess-Bot',
#                                  description='Cool Prog',
#                                  epilog='Text at the bottom of help')
# parser.add_argument('-t',
#                     '--token',
#                     type=str,
#                     required=True,
#                     help='The Discord bot token')
# parser.add_argument('-c',
#                     '--channel',
#                     type=int,
#                     required=True,
#                     help='The channel ID to send messages to')
# args = parser.parse_args()
from dotenv import load_dotenv

load_dotenv()
TOKEN = str(os.getenv('DISCORD_TOKEN'))
CHANNEL = int(os.getenv('DISCORD_CHANNEL_ID'))

# Charger les scalers utilisés pour l'entraînement du modèle
with open('scalers/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('scalers/scaler_close.pkl', 'rb') as f:
    scaler_close = pickle.load(f)


def message():
    try:
        prediction_message = PredictionMessage(TOKEN, CHANNEL)
        asyncio.run(prediction_message.run())
    except Exception as e:
        print("Error")
        raise e
        message()

    # end try


thread = Thread(target=message)
thread.start()

binance = ccxt.binance()

# Create a console object to handle rich output
console = Console()

# Define the table columns and headers
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Metric")
table.add_column("Value")

# Set the initial values for metrics
current_price = 0.0
predicted_price = 0.0
delta = 0.0
diff = 0.0
uncertainty = 0.0
eta = 0.0


# Define a function to update the table with new metric values
def update_table():
    global table
    # Define the table columns and headers
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Current Price", f"{current_price:.2f}")
    table.add_row("Predicted Price", f"{predicted_price:.2f}")
    table.add_row("Delta", f"{delta:.2f}")
    table.add_row("Diff", f"{diff:.2f}%")
    table.add_row("Uncertainty", f"{uncertainty:.5f}%")
    table.add_row("ETA", f"{eta:.2f} Hours")


# Define a function to update the console output
def update_console():
    console.print("\n[bold magenta]Metrics:[/bold magenta]\n")
    console.print(table)


# print with time
def printwt(text):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {text}")


def recuperer_dernieres_donnees_yfinance(ticker, start, end):
    # data = yf.download(ticker, start=start, end=end)
    data = binance.fetch_ohlcv(ticker, "1m", limit=9999)
    data = pd.DataFrame(
        data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    last_price = data['Close'].iloc[-1]

    return data


# def predire_prochain_prix(donnees, modele, pas, scaler, scaler_close):
#     dernier_bloc = donnees[-pas:]
#     dernier_bloc = scaler.transform(dernier_bloc)
#     dernier_bloc = np.reshape(dernier_bloc, (1, pas, dernier_bloc.shape[1]))
#     start_time = datetime.now()
#     with tf.device('/CPU:0'):  # Run on CPU to avoid GPU output
#         predictions = modele.predict(dernier_bloc, verbose=0)
#     end_time = datetime.now()
#     mean = scaler_close.inverse_transform(predictions)[0, 0]
#     std = np.std(scaler_close.inverse_transform(predictions))
#     elapsed_time = (end_time - start_time).total_seconds()
#     eta = elapsed_time * (len(donnees) / pas
#                           )  # Estimated time to predict next price
#     return mean, std, eta
def predire_prochain_prix(donnees, modele, pas, scaler, scaler_close):
    dernier_bloc = donnees[-pas:]
    dernier_bloc = scaler.transform(dernier_bloc)
    dernier_bloc = np.reshape(dernier_bloc, (1, pas, dernier_bloc.shape[1]))
    start_time = datetime.now()
    with tf.device('/CPU:0'):  # Run on CPU to avoid GPU output
        predictions = modele.predict(dernier_bloc, verbose=0)
    end_time = datetime.now()
    mean = scaler_close.inverse_transform(predictions)[0, 0]
    std = np.std(scaler_close.inverse_transform(predictions))
    elapsed_time = (end_time - start_time).total_seconds()
    # Calculate the estimated time to predict next price based on the elapsed time and the number of iterations left
    remaining_iterations = len(donnees) // pas
    eta = remaining_iterations * elapsed_time
    return mean, std, eta


# Charger le modèle
modele = load_model('models/model.h5')

# Set the percentage threshold for buying and selling
BUY_THRESHOLD = 0.01
SELL_THRESHOLD = -0.01

# Set the initial position to be empty
position = None

while True:
    print('-' * 50)

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

    pas = 60
    # Effectuer la prédiction
    prochain_prix, uncertainty, eta = predire_prochain_prix(
        donnees_recentes, modele, pas, scaler, scaler_close)
    printwt(
        f"Predicted price: {prochain_prix}, uncertainty: {uncertainty:.5f}%")
    printwt(f"ETA: {eta:.2f} Hours")

    # Get the current price
    current_price = donnees_recentes['Close'].iloc[-1]

    predicted_price = prochain_prix
    delta = predicted_price - current_price
    diff = (predicted_price - current_price) / current_price * 100
    uncertainty = uncertainty

    # Update the table and console output
    update_table()
    update_console()

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

    # store them in json file
    with open('json/prediction.json', 'w') as f:
        json.dump(
            {
                'ticker': str(ticker),
                'prochain_prix': float(prochain_prix),
                'last_price': float(last_price),
                'percent': float(percent),
                'uncertainty': float(uncertainty),
                'eta': str(eta)
            }, f)

    sl(60)
