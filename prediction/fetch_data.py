import requests
import pandas as pd
import os
import time
from datetime import datetime

def fetch_binance_data(symbol, interval='1d', limit=365):
    """
    Fetch historical klines from Binance public API.
    No API key required.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Binance response format: [Open time, Open, High, Low, Close, Volume, ...]
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to float
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Convert time
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        
        return df[['date', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def download_data():
    # Map friendly names to Binance symbols
    coins = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'SOL': 'SOLUSDT',
        'PEPE': 'PEPEUSDT',
        'BNB': 'BNBUSDT'
    }
    
    output_dir = "prediction/data"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, symbol in coins.items():
        print(f"Downloading {name} ({symbol}) from Binance...")
        df = fetch_binance_data(symbol)
        
        if df is not None and not df.empty:
            file_path = os.path.join(output_dir, f"{name.lower()}.csv")
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} rows to {file_path}")
        else:
            print(f"Failed to download data for {name}")
            
        # Be nice to the API
        time.sleep(0.5)

if __name__ == "__main__":
    download_data()
