import time
import pandas as pd
import numpy as np
import torch
import os
import sys
import csv
from datetime import datetime
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import CryptoTradingEnv
from agent import RememAgent
from memory import RetrieverMemory

# Constants
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'PEPEUSDT']
WINDOW_SIZE = 10
K_NEIGHBORS = 3
CSV_FILENAME = "verification_remem_only.csv"
DURATION_MINUTES = 60

def get_latest_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        resp = requests.get(url, params={'symbol': symbol})
        data = resp.json()
        return float(data['price'])
    except Exception as e:
        return None

def get_market_history(symbol, limit=20):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': '1m', 'limit': limit}
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        closes = [float(x[4]) for x in data]
        volumes = [float(x[5]) for x in data]
        return closes, volumes
    except Exception as e:
        return [], []

def construct_state(closes, volumes, position=0, balance=10000):
    if len(closes) < WINDOW_SIZE: return None
    base_price = closes[-WINDOW_SIZE]
    base_vol = np.mean(volumes[-WINDOW_SIZE:]) + 1e-8
    norm_closes = [c / base_price for c in closes[-WINDOW_SIZE:]]
    norm_vols = [v / base_vol for v in volumes[-WINDOW_SIZE:]]
    obs_data = []
    for i in range(WINDOW_SIZE):
        obs_data.append(norm_closes[i])
        obs_data.append(norm_vols[i])
    state = np.array(obs_data + [position, balance/10000.0], dtype=np.float32)
    return state

def init_csv():
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Symbol', 'Current_Price', 
                'Action_Pred', 'Method', 'Expected_Move',
                'Next_Price_Real', 'Result_Verified', 'Profit_Loss_Pct'
            ])

def log_result(row):
    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    print(f"[ReMem-Only] Starting Verification for {SYMBOLS}...")
    init_csv()
    
    dummy_state_dim = WINDOW_SIZE * 2 + 2
    action_dim = 3
    # Memory dim for pure RememAgent (Action, Reward, Risk) * K
    memory_dim = 3 * K_NEIGHBORS 
    
    agent = RememAgent(dummy_state_dim, action_dim, memory_dim)
    
    # Load the trained multi-coin model
    model_path = "model_remem_multicoin.pth"
    if os.path.exists(model_path):
        try:
            agent.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.eval()
            print(f"Loaded {model_path}.")
        except:
            print("Error loading model, using random weights.")
    else:
        print(f"{model_path} not found, using random weights.")

    memory = RetrieverMemory(k=K_NEIGHBORS)
    
    # Pre-fill
    for s in SYMBOLS:
        closes, volumes = get_market_history(s, limit=20)
        if len(closes) > WINDOW_SIZE:
             st = construct_state(closes, volumes)
             # Add dummy risk (0) for ReMem format
             memory.add_episode([st], [0], [0], [0]) 
    memory.build()

    for i in range(DURATION_MINUTES):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[ReMem-Only] Minute {i+1}/{DURATION_MINUTES}")
        
        predictions = {}
        
        # 1. Predict
        for symbol in SYMBOLS:
            closes, volumes = get_market_history(symbol, limit=WINDOW_SIZE+5)
            if not closes: continue
            
            current_price = closes[-1]
            state = construct_state(closes, volumes)
            context = memory.retrieve(state)
            
            with torch.no_grad():
                 action, log_prob, value = agent.get_action(state, context)
            
            action_str = ["HOLD", "BUY", "SELL"][action]
            predictions[symbol] = {
                'price': current_price,
                'action': action,
                'action_str': action_str,
                'state': state
            }
            print(f"{symbol}: {current_price} -> {action_str}")
            
        # 2. Wait
        time.sleep(60)
        
        # 3. Verify
        for symbol, pred in predictions.items():
            next_price = get_latest_price(symbol)
            if next_price is None: next_price = pred['price']
            
            start_price = pred['price']
            action = pred['action']
            pct_change = ((next_price - start_price) / start_price) * 100
            
            verified = "NEUTRAL"
            if action == 1: verified = "CORRECT" if next_price > start_price else "WRONG"
            elif action == 2: verified = "CORRECT" if next_price < start_price else "WRONG"
            
            expected = "FLAT"
            if action == 1: expected = "UP"
            if action == 2: expected = "DOWN"
            
            log_result([
                timestamp, symbol, start_price, 
                pred['action_str'], "ReMem-Only", expected,
                next_price, verified, f"{pct_change:.4f}%"
            ])
            
            # Online Learning
            reward = pct_change if action == 1 else (-pct_change if action == 2 else 0)
            # Add with dummy risk metric
            memory.add_episode([pred['state']], [action], [reward], [0])
            
        memory.build()

if __name__ == "__main__":
    main()

