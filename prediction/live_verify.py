import time
import pandas as pd
import numpy as np
import torch
import os
import sys
import csv
from datetime import datetime
import requests

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import CryptoTradingEnv
from agent_hybrid import HybridRememAgent
from memory import RetrieverMemory
from llm_reasoning import LLMReasoningModule

# Constants
# Gemini Embedding size is 768
LLM_EMB_DIM = 768 

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'PEPEUSDT']
WINDOW_SIZE = 10
K_NEIGHBORS = 3
CSV_FILENAME = "verification_20251214.csv"
DURATION_MINUTES = 60
API_KEY = os.getenv("GOOGLE_API_KEY")

def get_latest_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        resp = requests.get(url, params={'symbol': symbol})
        data = resp.json()
        return float(data['price'])
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
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
        print(f"Error fetching history for {symbol}: {e}")
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
                'Action_Pred', 'LLM_Reason', 'Expected_Move',
                'Next_Price_Real', 'Result_Verified', 'Profit_Loss_Pct'
            ])

def log_result(row):
    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def main():
    print(f"Starting Gemini-Powered Live Verification for {SYMBOLS}...")
    init_csv()
    
    dummy_state_dim = WINDOW_SIZE * 2 + 2
    action_dim = 3
    memory_dim = 3 * K_NEIGHBORS
    
    # Initialize Agent with Gemini's Embedding Dim (768)
    # IMPORTANT: The model trained on 1536 dim (OpenAI) will NOT load correctly here.
    # We must re-init the agent. Since we don't have a Gemini-trained model yet, 
    # we will use Random Weights (Untrained) but with correct dimensions.
    # OR we try to adapt/mock it. But typically we need to retrain.
    # For this verification run, we will run the UNTRAINED agent but with REAL reasoning embeddings.
    agent = HybridRememAgent(dummy_state_dim, action_dim, memory_dim, llm_emb_dim=LLM_EMB_DIM)
    
    print("Using Untrained Agent with Gemini Reasoner (Testing Pipeline).")
    
    memory = RetrieverMemory(k=K_NEIGHBORS)
    llm_reasoner = LLMReasoningModule(api_key=API_KEY)
    
    # Pre-fill
    for s in SYMBOLS:
        closes, volumes = get_market_history(s, limit=20)
        if len(closes) > WINDOW_SIZE:
             st = construct_state(closes, volumes)
             memory.add_episode([st], [0], [0], [0])
    memory.build()

    for i in range(DURATION_MINUTES):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- Minute {i+1}/{DURATION_MINUTES} [{timestamp}] ---")
        
        predictions = {}
        
        for symbol in SYMBOLS:
            closes, volumes = get_market_history(symbol, limit=WINDOW_SIZE+5)
            if not closes: continue
            
            current_price = closes[-1]
            state = construct_state(closes, volumes)
            context = memory.retrieve(state)
            
            # Gemini Reasoning
            try:
                llm_emb, reason_text = llm_reasoner.get_strategy_embedding(state, context)
            except Exception as e:
                llm_emb = np.zeros(LLM_EMB_DIM, dtype=np.float32)
                reason_text = f"Error: {e}"

            with torch.no_grad():
                 state_t = torch.FloatTensor(state).unsqueeze(0)
                 context_t = torch.FloatTensor(context).unsqueeze(0)
                 llm_t = torch.FloatTensor(llm_emb).unsqueeze(0)
                 probs, _ = agent.forward(state_t, context_t, llm_t)
                 action = torch.argmax(probs).item()
            
            action_str = ["HOLD", "BUY", "SELL"][action]
            predictions[symbol] = {
                'price': current_price,
                'action': action,
                'action_str': action_str,
                'state': state,
                'reason': reason_text
            }
            print(f"{symbol}: {current_price} -> {action_str} ({reason_text[:50]}...)")
            
        print("Waiting 60s...")
        time.sleep(60)
        
        # Verify
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
            
            print(f"  {symbol}: {verified} ({pct_change:.4f}%)")
            
            log_result([
                timestamp, symbol, start_price, 
                pred['action_str'], pred['reason'], expected,
                next_price, verified, f"{pct_change:.4f}%"
            ])
            
            reward = 0
            if action == 1: reward = pct_change
            if action == 2: reward = -pct_change
            memory.add_episode([pred['state']], [action], [reward], [0])
            
        memory.build()

if __name__ == "__main__":
    main()
