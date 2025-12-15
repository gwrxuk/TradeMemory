import numpy as np
import pandas as pd
import torch
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import CryptoTradingEnv
from agent_hybrid import HybridRememAgent
from memory import RetrieverMemory
from llm_reasoning import LLMReasoningModule

load_dotenv()

def evaluate_hybrid(model_path="model_hybrid.pth"):
    # Data to test on
    coins = ['btc', 'eth', 'sol', 'pepe', 'bnb']
    data_dir = "prediction/data"
    
    # Hyperparameters (Must match training)
    WINDOW_SIZE = 10
    K_NEIGHBORS = 3
    LLM_EMB_DIM = 1536
    
    # Initialize Environment Dims (dummy)
    dummy_env = CryptoTradingEnv(window_size=WINDOW_SIZE)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    memory_dim = 3 * K_NEIGHBORS
    
    # Initialize Hybrid Agent
    agent = HybridRememAgent(state_dim, action_dim, memory_dim, llm_emb_dim=LLM_EMB_DIM)
    
    if model_path and os.path.exists(model_path):
        print(f"Loading hybrid model from {model_path}...")
        try:
            agent.load_state_dict(torch.load(model_path))
            agent.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Model file not found. Please train first.")
        return
    
    # Initialize Components
    memory = RetrieverMemory(k=K_NEIGHBORS)
    llm_reasoner = LLMReasoningModule(model_name="gpt-3.5-turbo", embedding_model="text-embedding-3-small")
    
    results = {}
    
    for coin in coins:
        file_path = os.path.join(data_dir, f"{coin}.csv")
        if not os.path.exists(file_path):
            print(f"Data for {coin} not found. Skipping.")
            continue
            
        print(f"\nEvaluating Hybrid Agent on {coin.upper()}...")
        df = pd.read_csv(file_path)
        env = CryptoTradingEnv(df=df, window_size=WINDOW_SIZE, initial_balance=10000)
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        portfolio_values = []
        
        # Caching Logic
        llm_emb = None
        LLM_UPDATE_FREQ = 50
        step_count = 0
        
        while not done:
            # 1. Retrieve
            context = memory.retrieve(state)
            
            # 2. LLM Reasoning (Cached)
            if llm_emb is None or step_count % LLM_UPDATE_FREQ == 0:
                try:
                    # In eval, we might want to be deterministic or print the thought
                    llm_emb = llm_reasoner.get_strategy_embedding(state, context)
                except Exception as e:
                    if llm_emb is None: llm_emb = np.zeros(LLM_EMB_DIM, dtype=np.float32)
            
            # 3. Act
            # We use no_grad for inference
            with torch.no_grad():
                full_input = np.concatenate([state, context]) # (Just for reference if we weren't using hybrid class)
                # But our Hybrid Agent expects specific args:
                state_t = torch.FloatTensor(state).unsqueeze(0)
                context_t = torch.FloatTensor(context).unsqueeze(0)
                llm_t = torch.FloatTensor(llm_emb).unsqueeze(0)
                
                probs, _ = agent.forward(state_t, context_t, llm_t)
                action = torch.argmax(probs).item()
            
            # 4. Step
            next_state, reward, done, _, info = env.step(action)
            
            # Record
            portfolio_values.append(info['net_worth'])
            total_reward += reward
            
            # Online Learning: Add to memory during evaluation
            memory.add_episode([state], [action], [reward])
            if len(memory.keys) >= memory.k:
                 memory.build()
                 
            state = next_state
            step_count += 1
            
        final_value = portfolio_values[-1]
        roi = ((final_value - 10000) / 10000) * 100
        print(f"  Final Balance: ${final_value:.2f}")
        print(f"  ROI: {roi:.2f}%")
        
        results[coin] = roi
        
    print("\nSummary of Hybrid Performance (ROI):")
    for coin, roi in results.items():
        print(f"{coin.upper()}: {roi:.2f}%")

if __name__ == "__main__":
    evaluate_hybrid()

