import ccxt
import pandas as pd
import talib

# Définir le marché et la paire de trading
exchange = ccxt.binance()
symbol = 'BTC/USDT'

# Récupérer les données OHLCV pour toutes les périodes de temps disponibles
timeframes = exchange.timeframes.keys()
ohlcv = {}
for timeframe in timeframes:
    ohlcv[timeframe] = exchange.fetch_ohlcv(symbol, timeframe)

# Convertir les données OHLCV en DataFrame Pandas
df = {}
for timeframe in timeframes:
    df[timeframe] = pd.DataFrame(ohlcv[timeframe], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df[timeframe]['timestamp'] = pd.to_datetime(df[timeframe]['timestamp'], unit='ms')
    df[timeframe].set_index('timestamp', inplace=True)

# Calculer les indicateurs TA-Lib pour chaque période de temps
indicators = {}
for timeframe in timeframes:
    if 'close' not in df[timeframe].columns:
        continue
    indicators[timeframe] = {}
    indicators[timeframe]['EMA20'] = talib.EMA(df[timeframe]['close'], timeperiod=20)
    indicators[timeframe]['EMA50'] = talib.EMA(df[timeframe]['close'], timeperiod=50)
    indicators[timeframe]['SMA200'] = talib.SMA(df[timeframe]['close'], timeperiod=200)
    indicators[timeframe]['MACD'], indicators[timeframe]['MACD_SIGNAL'], indicators[timeframe]['MACD_HIST'] = talib.MACD(df[timeframe]['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators[timeframe]['close'] = df[timeframe]['close']
    indicators[timeframe]['ADX'] = talib.ADX(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['ADXR'] = talib.ADXR(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['APO'] = talib.APO(df[timeframe]['close'])
    indicators[timeframe]['AROONOSC'] = talib.AROONOSC(df[timeframe]['high'], df[timeframe]['low'])
    indicators[timeframe]['BOP'] = talib.BOP(df[timeframe]['open'], df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['CCI'] = talib.CCI(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['CMO'] = talib.CMO(df[timeframe]['close'])
    indicators[timeframe]['DX'] = talib.DX(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['MFI'] = talib.MFI(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'], df[timeframe]['volume'])
    indicators[timeframe]['MINUS_DI'] = talib.MINUS_DI(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['MINUS_DM'] = talib.MINUS_DM(df[timeframe]['high'], df[timeframe]['low'])
    indicators[timeframe]['MOM'] = talib.MOM(df[timeframe]['close'])
    indicators[timeframe]['PLUS_DI'] = talib.PLUS_DI(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['PLUS_DM'] = talib.PLUS_DM(df[timeframe]['high'], df[timeframe]['low'])
    indicators[timeframe]['PPO'] = talib.PPO(df[timeframe]['close'])
    indicators[timeframe]['ROC'] = talib.ROC(df[timeframe]['close'])
    indicators[timeframe]['ROCP'] = talib.ROCP(df[timeframe]['close'])
    indicators[timeframe]['ROCR'] = talib.ROCR(df[timeframe]['close'])
    indicators[timeframe]['ROCR100'] = talib.ROCR100(df[timeframe]['close'])
    indicators[timeframe]['RSI'] = talib.RSI(df[timeframe]['close'])
    indicators[timeframe]['STOCH'] = talib.STOCH(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['STOCHF'] = talib.STOCHF(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['STOCHRSI'] = talib.STOCHRSI(df[timeframe]['close'])
    indicators[timeframe]['TRIX'] = talib.TRIX(df[timeframe]['close'])
    indicators[timeframe]['ULTOSC'] = talib.ULTOSC(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['WILLR'] = talib.WILLR(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'])
    indicators[timeframe]['AD'] = talib.AD(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'], df[timeframe]['volume'])
    indicators[timeframe]['ADOSC'] = talib.ADOSC(df[timeframe]['high'], df[timeframe]['low'], df[timeframe]['close'], df[timeframe]['volume'])
    indicators[timeframe]['OBV'] = talib.OBV(df[timeframe]['close'], df[timeframe]['volume'])
    indicators[timeframe]['HT_TRENDLINE'] = talib.HT_TRENDLINE(df[timeframe]['close'])
    indicators[timeframe]['HT_SINE'] = talib.HT_SINE(df[timeframe]['close'])
    indicators[timeframe]['HT_TRENDMODE'] = talib.HT_TRENDMODE(df[timeframe]['close'])
    indicators[timeframe]['HT_DCPERIOD'] = talib.HT_DCPERIOD(df[timeframe]['close'])
    indicators[timeframe]['HT_DCPHASE'] = talib.HT_DCPHASE(df[timeframe]['close'])
    indicators[timeframe]['HT_PHASOR'] = talib.HT_PHASOR(df[timeframe]['close'])
    
    

# Déterminer si le marché est en hausse ou en baisse pour chaque période de temps
trend = {}
for timeframe in timeframes:
    trend[timeframe] = None
    if timeframe not in indicators:
        continue
    if indicators[timeframe]['EMA20'][-1] > indicators[timeframe]['EMA50'][-1] and indicators[timeframe]['SMA200'][-1] < indicators[timeframe]['close'][-1] and indicators[timeframe]['RSI'][-1] > 50 and indicators[timeframe]['MACD'][-1] > indicators[timeframe]['MACD_SIGNAL'][-1]:
        trend[timeframe] = '✔️'
    elif indicators[timeframe]['EMA20'][-1] < indicators[timeframe]['EMA50'][-1] and indicators[timeframe]['SMA200'][-1] > indicators[timeframe]['close'][-1] and indicators[timeframe]['RSI'][-1] < 50 and indicators[timeframe]['MACD'][-1] < indicators[timeframe]['MACD_SIGNAL'][-1]:
        trend[timeframe] = '❌'
    else:
        trend[timeframe] = '➖'

# Afficher les résultats
for timeframe in timeframes:
    print(f"Tendance {timeframe}: {trend[timeframe]}")

