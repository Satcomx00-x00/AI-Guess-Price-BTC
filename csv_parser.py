
# Replace `path/to/input.csv` with the actual path to your CSV input file
input_path = r'C:\Users\MrBios\Documents\Development\test\csv\Featured_bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'

# Replace `path/to/output.csv` with the desired output path and file name
# output_path = r'C:\Users\MrBios\Documents\Development\test\csv\bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv'

# # Set the number of rows to read in each iteration
# chunksize = 10000

# # Create an empty DataFrame to store the parsed data
# parsed_data = pd.DataFrame()

# # Use a for loop to iterate through the input file in chunks
# for chunk in pd.read_csv(input_path, chunksize=chunksize):
    
#     # Do any necessary parsing or cleaning of the data here
#     # For example:
#     #     chunk = chunk.dropna()  # Drop rows with missing values
    
#     # Append the parsed data to the DataFrame
#     parsed_data = parsed_data.append(chunk)
    
# # Save the parsed data to a new CSV file
# parsed_data.to_csv(output_path, index=False)





import pandas as pd
headers = [
    'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'volume_adi',
    'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em',
    'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi', 'volatility_bbm',
    'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
    'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch',
    'volatility_kcl', 'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
    'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
    'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
    'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
    'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
    'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
    'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc',
    'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
    'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
    'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up', 'trend_psar_down',
    'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_rsi',
    'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
    'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal',
    'momentum_wr', 'momentum_ao', 'momentum_roc', 'momentum_ppo',
    'momentum_ppo_signal', 'momentum_ppo_hist', 'momentum_pvo',
    'momentum_pvo_signal', 'momentum_pvo_hist', 'momentum_kama', 'others_dr',
    'others_dlr', 'others_cr'
]
# Set the chunk size for splitting the file
chunk_size = 200000

# Open the large input CSV file and read it in chunks
with pd.read_csv(r'C:\Users\MrBios\Documents\Development\test\csv\Featured_bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv', chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        # Generate a filename for the current chunk
        filename = fr'C:\Users\MrBios\Documents\Development\test\csv\chunked\chunk_{i}.csv'
        # Save the current chunk to a new CSV file
        
        chunk.to_csv(filename, index=False, header=headers)
