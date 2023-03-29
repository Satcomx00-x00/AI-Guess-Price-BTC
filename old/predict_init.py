import ccxt
import pandas as pd
import time
import numpy as np
import talib as ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def recuperer_dernieres_donnees_binance(paire, intervalle, limite):
    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(paire, intervalle, limit=limite)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

def predire_prochain_prix(donnees, modele, pas, scaler, scaler_close):
    dernier_bloc = donnees[-pas:]
    dernier_bloc = scaler.transform(dernier_bloc)
    dernier_bloc = np.reshape(dernier_bloc, (1, pas, dernier_bloc.shape[1]))
    prediction = modele.predict(dernier_bloc)
    prediction = scaler_close.inverse_transform(prediction)
    return prediction[0, 0]

def preparer_donnees(donnees, donnees_close, pas):
    X, y = [], []
    for i in range(pas, len(donnees)):
        X.append(donnees[i - pas:i])
        y.append(donnees_close[i])
    return np.array(X), np.array(y)



# Charger le modèle
modele = load_model('modele_4h.h5')

# Paramètres
paire = 'BTC/USDT'
intervalle = '30m'
limite = 200  # Définissez la limite en fonction du pas utilisé pour l'entraînement du modèle

# Récupérer les dernières données
donnees_recentes = recuperer_dernieres_donnees_binance(paire, intervalle, limite)

# Créer des indicateurs techniques avec TA-Lib
donnees_recentes['SMA'] = ta.SMA(donnees_recentes['Close'], timeperiod=20)
donnees_recentes['RSI'] = ta.RSI(donnees_recentes['Close'], timeperiod=14)
donnees_recentes['ADX'] = ta.ADX(donnees_recentes['High'], donnees_recentes['Low'], donnees_recentes['Close'], timeperiod=14)

# Supprimer les lignes avec des valeurs NaN
donnees_recentes = donnees_recentes.dropna()

# Sélectionner les colonnes nécessaires
donnees_recentes = donnees_recentes[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'ADX']]

# Charger les scalers utilisés pour l'entraînement du modèle
# Vous pouvez les sauvegarder et les charger en utilisant la bibliothèque 'pickle'
# Assurez-vous de les charger ici avant de les utiliser
# Normaliser les données
scaler = MinMaxScaler()
scaler = scaler.fit_transform(donnees_recentes)

# Créer un scaler séparé pour la colonne "Close"
scaler_close = MinMaxScaler()
donnees_close = np.array(donnees_recentes['Close']).reshape(-1, 1)
scaler_close = scaler_close.fit_transform(donnees_close)

pas = 60
X, y = preparer_donnees(donnees_recentes, scaler_close, pas)


# Effectuer la prédiction
prochain_prix = predire_prochain_prix(donnees_recentes, modele, pas, scaler, scaler_close)
print(f"Le prix prévu du Bitcoin pour la prochaine période est : {prochain_prix}")
