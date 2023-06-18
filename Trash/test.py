import SatcomDiscord as SD
from time import sleep
from threading import Thread
import asyncio

discord_bot = SD.PredictionMessage()

timestamp = 9
symbol = 9
predicted_price = 9
actual_price = 9
accuracy = 9
uncertainty = 9
eta = 9

x = discord_bot.run()

while not discord_bot.bot or not discord_bot.bot.is_ready():
    print('Waiting for bot to be ready')
    
x.update_prediction(timestamp, symbol, predicted_price, actual_price, accuracy,
                    uncertainty, eta)
