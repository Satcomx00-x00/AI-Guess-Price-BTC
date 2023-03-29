import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

while True:
    # Load the dataset
    df = pd.read_csv('csv/prediction.csv')

    # Normalize the data
    scaler = MinMaxScaler()
    df['last_price'] = scaler.fit_transform(df['last_price'].values.reshape(-1, 1))
    df['predicted_price'] = scaler.fit_transform(df['predicted_price'].values.reshape(-1, 1))

    # Prepare the data for training the model
    X = df['predicted_price'].values.reshape(-1, 1)
    y = df['last_price'].values.reshape(-1, 1)

    # Train the model
    reg = LinearRegression()
    reg.fit(X, y)

    # Predict the next 20 prices
    last_predicted_price = np.array([df.iloc[-1]['predicted_price']])
    predicted_prices = []
    for i in range(20):
        next_price = reg.predict(last_predicted_price.reshape(-1, 1))
        predicted_prices.append(scaler.inverse_transform(next_price)[0][0])
        last_predicted_price = next_price

    # Plot the results
    real_prices = scaler.inverse_transform(y)
    real_prices_last50 = real_prices[-50:]
    predicted_prices = np.array(predicted_prices)
    predicted_prices = predicted_prices.reshape(-1, 1)
    
    # Mettre à jour la longueur de l'axe x
    x_length = len(real_prices_last50) + len(predicted_prices)
    
    # Concaténer les tableaux pour créer l'axe y
    all_prices = np.concatenate((real_prices_last50, predicted_prices))

    # Plot the graph
    plt.plot(np.arange(x_length - len(predicted_prices)), real_prices_last50, label='Real Prices')
    plt.plot(np.arange(x_length - len(predicted_prices), x_length), predicted_prices, label='Predicted Prices')
    plt.plot(np.arange(x_length), all_prices, linestyle='dashed', label='All Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Ajouter le minuteur
    plt.title('Prix en fonction du temps\nTemps écoulé : 0')
    for i in range(1, 31):
        plt.title(f'Prix en fonction du temps\nTemps écoulé : {i}')
        time.sleep(1)

    plt.show()
