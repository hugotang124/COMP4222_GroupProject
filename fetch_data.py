
import requests                    
import json                        
import pandas as pd       
import os         
from datetime import datetime, timedelta           
import argparse


BINANCE_DATA_COLUMNS = ["time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",\
                        "trades", "buy_base_vol", "buy_quote_vol"]

DATA_DIRECTORY = os.path.join(os.path.split(__file__)[0], "data")

def get_binance_bars(symbol, interval, startTime, endTime):
    '''
    Function to get binance bars of a symbol
    Response from Binance
    [
        [
            1499040000000,      // Kline open time
            "0.01634790",       // Open price
            "0.80000000",       // High price
            "0.01575800",       // Low price
            "0.01577100",       // Close price
            "148976.11427815",  // Volume
            1499644799999,      // Kline Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "0"                 // Unused field, ignore.
        ]
    ] 
    '''

    url = "https://api.binance.com/api/v3/klines"

    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'

    req_params = {"symbol": symbol, 
                  'interval': interval, 
                  'startTime': startTime, 
                  'endTime': endTime, 
                  'limit': limit}

    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text))

    if (len(df.index) == 0):
        return None

    df = df.iloc[:, :-1]
    df.columns = BINANCE_DATA_COLUMNS

    df.drop(columns = "close_time", inplace = True)
    df.set_index("time", inplace = True)
    df.index = pd.to_datetime(df.index, unit = "ms")
    df = df.astype("float")


    return df


def get_single_symbol_data(symbol, interval, start_date, end_date):
    '''
    Function to get the data of single symbol

    symbol: Symbol (e.g., BTCUSDT, ETHUSDT)
    interval: Interval (e.g., 1m, 1h, 1d)
    startTime: Start date (in YYYY-MM-DD)
    endTime: End date (in YYYY-MM-DD)

    '''

    df_list = []
    last_datetime = start_date
    print(f"Fetching data for {symbol}")

    while True:
        print(last_datetime)
        new_df = get_binance_bars(symbol, interval, last_datetime, end_date)
        if new_df is None:
            break
        df_list.append(new_df)
        last_datetime = max(new_df.index) + timedelta(0, 1)
        
    df = pd.concat(df_list)
    df.to_csv(f'{DATA_DIRECTORY}/{symbol}{interval}.csv', index = True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", action = "store", default = "", help = "Symbols")
    parser.add_argument("--start-date", action = "store", default = "", help = "Start Date")
    parser.add_argument("--end-date", action = "store", default = datetime.today().isoformat(), help = "End Date")
    parser.add_argument("--interval", action = "store", default = "1h", help = "Data Interval (e.g., 1m, 1h, 1d)")

    args = parser.parse_args()

    if not args.symbols:
        print("Please insert symbols that you want")
        exit()

    if not args.start_date:
        print("Please insert start date")
        exit()

    symbols = list(args.symbols.split(","))
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    interval = args.interval


    os.makedirs(DATA_DIRECTORY, exist_ok = True)

    for symbol in symbols:
        get_single_symbol_data(symbol, interval, start_date, end_date)



    
