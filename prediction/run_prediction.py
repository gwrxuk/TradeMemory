import numpy as np
import pandas as pd
import torch
import os
import sys

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import CryptoTradingEnv
from agent import RememAgent
from memory import RetrieverMemory

def evaluate(model_path=None):
    # Data to test on
    coins = ['btc', 'eth', 'sol', 'pepe']
    data_dir = "prediction/data"
    
    # Hyperparameters (Must match training)
    WINDOW_SIZE = 10
    K_NEIGHBORS = 5
    
    # Load Agent (Initializing with dummy dims first)
    # In a real scenario, you'd save/load these dims or the model would know.
    # We assume standard dims from our env for now.
    dummy_env = CryptoTradingEnv(window_size=WINDOW_SIZE)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    memory_dim = 2 * K_NEIGHBORS
    
    agent = RememAgent(state_dim, action_dim, memory_dim)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        agent.load_state_dict(torch.load(model_path))
        agent.eval()
    else:
        print("No model path provided or file not found. Using untrained agent for demonstration.")
    
    # Evaluate on each coin
    results = {}
    
    for coin in coins:
        file_path = os.path.join(data_dir, f"{coin}.csv")
        if not os.path.exists(file_path):
            print(f"Data for {coin} not found at {file_path}. Skipping.")
            continue
            
        print(f"\nEvaluating on {coin.upper()}...")
        df = pd.read_csv(file_path)
        
        # Initialize Env with specific data
        env = CryptoTradingEnv(df=df, window_size=WINDOW_SIZE, initial_balance=10000)
        
        # Initialize Memory (Empty for evaluation, or pre-filled if we had a training set)
        # For pure evaluation "zero-shot" or "online adaptation", we start empty or load a memory bank.
        memory = RetrieverMemory(k=K_NEIGHBORS)
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        portfolio_values = []
        
        while not done:
            # 1. Retrieve
            context = memory.retrieve(state)
            
            # 2. Act (Deterministic for eval usually, or sample)
            # We'll stick to sampling or take argmax for strict eval
            full_input = np.concatenate([state, context])
            full_input_tensor = torch.FloatTensor(full_input)
            probs, _ = agent(full_input_tensor)
            action = torch.argmax(probs).item()
            
            # 3. Step
            next_state, reward, done, _, info = env.step(action)
            
            # Record
            portfolio_values.append(info['net_worth'])
            total_reward += reward
            
            # Optional: Online Learning / Memory Update during evaluation?
            # The paper might suggest updating memory during test time (meta-learning/online).
            # We will add to memory to allow "in-episode" learning/retrieval.
            memory.add_episode([state], [action], [reward])
            if len(memory.keys) >= memory.k:
                 memory.build()
                 
            state = next_state
            
        final_value = portfolio_values[-1]
        roi = ((final_value - 10000) / 10000) * 100
        print(f"  Final Balance: ${final_value:.2f}")
        print(f"  ROI: {roi:.2f}%")
        print(f"  Total Reward: {total_reward:.2f}")
        
        results[coin] = roi
        
    print("\nSummary of Performance (ROI):")
    for coin, roi in results.items():
        print(f"{coin.upper()}: {roi:.2f}%")

if __name__ == "__main__":
    evaluate()

