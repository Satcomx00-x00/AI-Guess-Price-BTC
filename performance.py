import time
import pandas as pd
from datetime import datetime, timedelta
import ccxt

ccxt.binance()
# Set the interval for fetching new data
INTERVAL = 10 * 60  # 10 minutes in seconds

# Load the input CSV file
df = pd.read_csv(
    r'C:\Users\MrBios\Documents\Development\test\production\Docker\csv\prediction.csv',
    delimiter=',',
    encoding='utf-8')


# Define a function to fetch new data
def fetch_new_data(symbol, start_time, end_time):
    binance = ccxt.binance()
    # Fetch the data from Binance
    print(f"Fetching data for {symbol} from {start_time} to {end_time}")
    start_time = int(datetime.fromisoformat(start_time).timestamp() * 1000)
    end_time = int(datetime.fromisoformat(end_time).timestamp() * 1000)
    data = binance.fetch_ohlcv(symbol, '1m',
                               limit=10,
                               params={
                                   'startTime': start_time,
                                   'endTime': end_time
                               })
    # Convert the data to a DataFrame
    df = pd.DataFrame(
        data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


# Define a function to calculate market performance
def calculate_market_performance(predicted_price, actual_price):
    return (actual_price - predicted_price) / predicted_price

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
    eta = elapsed_time * (len(donnees) / pas
                          )  # Estimated time to predict next price
    return mean, std, eta
# Charger le modèle
from tensorflow.keras.models import load_model
modele = load_model('models/model.h5')
pas = 60
# Iterate over the input rows and fetch new data
for i, row in df.iterrows():
    timedelta(minutes=10)
    # end time is start time + 10 minutes
    end_time = str(datetime.fromisoformat(row['timestamp']) + timedelta(minutes=10))

    # Fetch new data
    data = fetch_new_data(row['symbol'], row['timestamp'], end_time)
    print(data)

    # Drop the last row, which may be incomplete
    data.drop(data.tail(1).index, inplace=True)


    # Get the current price
    current_price = data['close'].iloc[-1]
    # Calculate the market performance
    market_performance = calculate_market_performance(row['predicted_price'],
                                                      current_price)
    # Append the new row to the output DataFrame
    new_row = pd.DataFrame(
        {
            'timestamp': datetime.now(),
            'symbol': row['symbol'],
            'predicted_price': row['predicted_price'],
            'actual_price': current_price,
            'accuracy': row['accuracy'],
            'uncertainty': row['uncertainty'],
            'eta': row['eta'],
            'market_performance': market_performance
        },
        index=[0])
    df = new_row
    # Save the output DataFrame to a CSV file
    df.to_csv('output_file.csv', index=False)
    # Wait for the next iteration
    time.sleep(INTERVAL)
