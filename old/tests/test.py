import ccxt
import pandas as pd
import time

def recuperer_donnees_binance(debut, fin, paire, intervalle):
    binance = ccxt.binance()
    limit = 99999
    donnees_brutes = []

    while debut < fin:
        try:
            ohlcv = binance.fetch_ohlcv(paire, intervalle, since=debut, limit=limit)
            if not ohlcv:
                break
            debut = ohlcv[-1][0] + 1
            donnees_brutes += ohlcv
            time.sleep(binance.rateLimit / 1000)
        except Exception as e:
            print(f"Erreur : {e}")
            time.sleep(binance.rateLimit / 1000)

    df = pd.DataFrame(donnees_brutes, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

# Paramètres
paire = 'BTC/USDT'
intervalle = '1d'
debut_timestamp = 0
fin_timestamp = int(time.time() * 1000)

# Récupérer les données et les stocker dans un DataFrame
donnees_btc = recuperer_donnees_binance(debut_timestamp, fin_timestamp, paire, intervalle)

# Sauvegarder les données dans un fichier CSV
donnees_btc.to_csv('BTC_OHLCV_binance.csv', index=False)
