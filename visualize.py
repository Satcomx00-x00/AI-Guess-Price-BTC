import matplotlib.pyplot as plt
import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv(r'C:\Users\MrBios\Documents\Development\test\production\Docker\csv\prediction.csv')

# Convert the timestamp column to a datetime object
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the actual price and predicted price as lines
ax.plot(data['timestamp'], data['actual_price'], label='Actual Price')
ax.plot(data['timestamp'], data['predicted_price'], label='Predicted Price')

# Add a legend and axis labels
ax.legend()
ax.set_xlabel('Timestamp')
ax.set_ylabel('Price (USD)')

# Show the plot
plt.show()
