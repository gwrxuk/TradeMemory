import numpy as np
import torch
import os
import glob
import pandas as pd
from env import CryptoTradingEnv
from agent import RememAgent
from memory import RetrieverMemory

def calculate_drawdown(prices):
    """Simple drawdown calculation for a series of prices."""
    if len(prices) == 0: return 0.0
    peak = prices[0]
    max_drawdown = 0.0
    for p in prices:
        if p > peak: 
            peak = p
        dd = (peak - p) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return max_drawdown

def main():
    # Hyperparameters
    EPISODES_PER_COIN = 10 # 10 episodes per coin * 5 coins = 50 total
    WINDOW_SIZE = 10
    K_NEIGHBORS = 5
    MODEL_SAVE_PATH = "model_remem_multicoin.pth"
    
    # 1. Load Data for All Coins
    data_dir = "prediction/data"
    # Look for all CSVs (btc, eth, sol, pepe, bnb)
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("No data files found in prediction/data/. Run fetch_data.py first.")
        return

    print(f"Found {len(csv_files)} datasets: {[os.path.basename(f) for f in csv_files]}")

    # Initialize Environment with first file just to get dims
    dummy_df = pd.read_csv(csv_files[0])
    dummy_env = CryptoTradingEnv(df=dummy_df, window_size=WINDOW_SIZE)
    
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    memory_dim = 3 * K_NEIGHBORS # Action, Reward, Risk
    
    # Initialize ReMem Agent
    agent = RememAgent(state_dim, action_dim, memory_dim)
    
    # Initialize Memory
    memory = RetrieverMemory(k=K_NEIGHBORS)
    
    print("Starting Multi-Coin ReMem Training...")
    
    total_episode_count = 0
    
    # Iterate through each coin dataset
    for file_path in csv_files:
        coin_name = os.path.basename(file_path).replace('.csv', '').upper()
        print(f"\n--- Training on {coin_name} ---")
        
        df = pd.read_csv(file_path)
        env = CryptoTradingEnv(df=df, window_size=WINDOW_SIZE)
        
        for episode in range(EPISODES_PER_COIN):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_contexts = []
            episode_log_probs = []
            episode_prices = [] 
            
            step_count = 0
            
            while not done:
                # 1. Retrieve (Cross-Asset Retrieval!)
                # The memory contains experiences from ALL previous coins trained so far
                context = memory.retrieve(state)
                
                # 2. Act
                action, log_prob, value = agent.get_action(state, context)
                
                # 3. Step
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_states.append(state)
                episode_contexts.append(context)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                episode_prices.append(info.get('net_worth', 10000))
                
                state = next_state
                total_reward += reward
                step_count += 1
                
            max_dd = calculate_drawdown(episode_prices)
            risk_metrics = [max_dd] * len(episode_states)
            
            total_episode_count += 1
            print(f"Episode {total_episode_count}: {coin_name} Reward: {total_reward:.2f}, MaxDD: {max_dd:.2%}")
            
            # 4. Update Agent
            rollouts = list(zip(
                episode_states, 
                episode_contexts, 
                episode_actions, 
                episode_rewards, 
                episode_states[1:] + [next_state], 
                episode_log_probs,
                [False]*(len(episode_rewards)-1) + [True]
            ))
            loss = agent.update(rollouts)
            
            # 5. Update Memory
            # We add this coin's experience to the shared memory bank
            memory.add_episode(episode_states, episode_actions, episode_rewards, risk_metrics)
            memory.build()
        
    print("\nTraining Completed.")
    torch.save(agent.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
