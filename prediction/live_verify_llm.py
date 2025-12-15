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
from agent_llm import LLMTradingAgent # Ensure this handles Gemini or generic API
from memory import RetrieverMemory
from llm_reasoning import LLMReasoningModule 
# Note: agent_llm was originally built for OpenAI. We might need to patch it or create a new wrapper 
# that uses the logic from llm_reasoning but returns an ACTION directly instead of an embedding.

# Let's define a dedicated Gemini Action Agent here for simplicity
import google.generativeai as genai

class GeminiActionAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-3-pro-preview")
        
    def get_action(self, state, retrieved_context):
        prompt = f"""
        You are a crypto trading bot.
        Market State (Last 5 prices normalized): {state[-5:]}
        Similar Past Outcomes (Action/Reward): {retrieved_context}
        
        Based on this, decide the next action:
        0 = HOLD
        1 = BUY
        2 = SELL
        
        Reply ONLY with the single digit (0, 1, or 2).
        """
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if '0' in text: return 0, text
            if '1' in text: return 1, text
            if '2' in text: return 2, text
            return 0, text # Default
        except Exception as e:
            return 0, f"Error: {e}"

# Constants
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'PEPEUSDT']
WINDOW_SIZE = 10
K_NEIGHBORS = 3
CSV_FILENAME = "verification_llm_only.csv"
DURATION_MINUTES = 60
API_KEY = os.getenv("GOOGLE_API_KEY")

def get_latest_price(symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        resp = requests.get(url, params={'symbol': symbol})
        data = resp.json()
        return float(data['price'])
    except: return None

def get_market_history(symbol, limit=20):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': '1m', 'limit': limit}
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        closes = [float(x[4]) for x in data]
        volumes = [float(x[5]) for x in data]
        return closes, volumes
    except: return [], []

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
    print(f"[LLM-Only] Starting Verification for {SYMBOLS}...")
    init_csv()
    
    agent = GeminiActionAgent(api_key=API_KEY)
    memory = RetrieverMemory(k=K_NEIGHBORS)
    
    # Pre-fill
    for s in SYMBOLS:
        closes, volumes = get_market_history(s, limit=20)
        if len(closes) > WINDOW_SIZE:
             st = construct_state(closes, volumes)
             memory.add_episode([st], [0], [0], [0]) # Dummy risk
    memory.build()

    for i in range(DURATION_MINUTES):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[LLM-Only] Minute {i+1}/{DURATION_MINUTES}")
        
        predictions = {}
        
        # 1. Predict
        for symbol in SYMBOLS:
            closes, volumes = get_market_history(symbol, limit=WINDOW_SIZE+5)
            if not closes: continue
            
            current_price = closes[-1]
            state = construct_state(closes, volumes)
            context = memory.retrieve(state)
            
            # Direct LLM Action
            action, reason = agent.get_action(state, context)
            
            action_str = ["HOLD", "BUY", "SELL"][action]
            predictions[symbol] = {
                'price': current_price,
                'action': action,
                'action_str': action_str,
                'state': state,
                'reason': reason
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
                pred['action_str'], pred['reason'][:50], expected,
                next_price, verified, f"{pct_change:.4f}%"
            ])
            
            reward = pct_change if action == 1 else (-pct_change if action == 2 else 0)
            memory.add_episode([pred['state']], [action], [reward], [0])
            
        memory.build()

if __name__ == "__main__":
    main()

