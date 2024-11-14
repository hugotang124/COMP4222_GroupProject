import pandas as pd
import glob
import os

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    return (abs(actual - predicted) / actual).mean() * 100

# Load Crypto data
predicted_data = pd.read_csv('test_predict.csv')
predicted_data['time'] = pd.to_datetime(predicted_data['time'])  # Ensure date is in datetime format

# List all currency files
currency_files = glob.glob('data/*.csv')  # Adjust the path as needed

# Dictionary to store MAPE results
mape_results = {}

for file in currency_files:
    filename = os.path.basename(file)
    currency_name = filename.split('1h')[0]
    currency_data = pd.read_csv(file)
    currency_data['time'] = pd.to_datetime(currency_data['time'])  # Ensure date is in datetime format
    currency_name = currency_name.lower()
    if currency_name not in predicted_data.columns:
        continue  

    # Merge currency data with Bitcoin data on the date column
    merged_data = pd.merge(currency_data, predicted_data, on='time', suffixes=('_actual', '_predicted'))

    # Assuming the close price column for the currency is named 'close_price_currency'
    # and for Bitcoin it is 'close_price_bitcoin'
    if 'close' in merged_data.columns:
        # Calculate MAPE using the actual close price and the predicted price from the predicted_data
        mape = calculate_mape(merged_data['close'], merged_data[currency_name])
        mape_results[currency_name] = mape  # Store MAPE result with currency name
    

# Print MAPE results
for currency_file, mape in mape_results.items():
    print(f"MAPE for {currency_file.upper()}: {mape:.2f}%")