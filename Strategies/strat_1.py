import pandas as pd
from ta import add_all_ta_features
import ccxt

# Create a Binance exchange object
exchange = ccxt.binance()

# Define the trading pair
pair = 'BTC/USDT'

# Set the time interval for the OHLCV data
timeframe = '1d'

# Retrieve the OHLCV data for the trading pair and timeframe
ohlcv = exchange.fetch_ohlcv(pair, timeframe)

# Convert the OHLCV data into a Pandas dataframe
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Convert the Unix timestamp to a readable datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Add technical indicators to the dataframe
df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume')

# Define the buy and sell thresholds for each indicator
buy_thresholds = {
    'trend_macd': 0,
    'trend_ema_fast': 'cross_above',
    'trend_ema_slow': 'cross_above',
    'momentum_rsi': 30,
    'volatility_bbli': -1,
    'trend_vortex_ind_diff': 'cross_above'
}

sell_thresholds = {
    'trend_macd': 0,
    'trend_ema_fast': 'cross_below',
    'trend_ema_slow': 'cross_below',
    'momentum_rsi': 70,
    'volatility_bbli': 1,
    'trend_vortex_ind_diff': 'cross_below'
}

# Initialize the position to None
position = None

# Loop through each row in the dataframe
for index, row in df.iterrows():
    # Check if the position is already open
    if position is not None:
        # Check if the sell threshold has been reached
        if all((row[indicator]) < float(sell_thresholds[indicator]) for indicator in sell_thresholds):
            # Close the position and print the result
            sell_price = exchange.fetch_ticker(pair)['bid']
            profit = sell_price - position
            print(f"Sold {pair} at {sell_price}. Profit: {profit}")
            position = None
    # Check if the buy threshold has been reached
    elif all((row[indicator]) > float(buy_thresholds[indicator]) for indicator in buy_thresholds):

        # Open the position and print the result
        buy_price = exchange.fetch_ticker(pair)['ask']
        print(f"Bought {pair} at {buy_price}")
        position = buy_price
