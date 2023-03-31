    # if eta < 1:
    #     printwt(f"ETA: {eta:.2f} Hours")
    #     # store last price and prediction in csv
    #     with open('csv/prediction.csv', 'a', encoding="UTF8") as f:
    #         st = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{ticker},{prochain_prix:.2f},{last_price},{percent:.2f},{uncertainty:.2f},{eta:.2f}\n"
    #         f.write(st)
    # # Check if we should buy
    # if prochain_prix > current_price * (1 + BUY_THRESHOLD):
    #     # Buy
    #     printwt(f"Buying {ticker} at {current_price}...")
    #     amount = 100  # set the amount to buy here
    #     # binance.create_market_buy_order(ticker, amount)
    #     position = 'long'
    # # Check if we should sell
    # elif prochain_prix < current_price * (1 + SELL_THRESHOLD):
    #     # Sell
    #     if position == 'long':
    #         printwt(f"Selling {ticker} at {current_price}...")
    #         amount = 100  # set the amount to sell here
    #         # binance.create_market_sell_order(ticker, amount)
    #         position = None
    # printwt(
    #     f"Current price: {current_price}, Predicted price: {prochain_prix}, delta: {prochain_prix - current_price}, diff: {(prochain_prix - current_price) / current_price * 100}%"
    # )