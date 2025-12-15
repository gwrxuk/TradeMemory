import numpy as np
import torch
import os
import pandas as pd
from dotenv import load_dotenv

from env import CryptoTradingEnv
from agent_llm import LLMTradingAgent
from memory import RetrieverMemory

# Load .env file
load_dotenv()

def main():
    # Hyperparameters
    EPISODES = 5 # Small number for API testing
    WINDOW_SIZE = 10
    K_NEIGHBORS = 3
    
    # 1. Load Data
    data_path = "prediction/data/btc.csv"
    if os.path.exists(data_path):
        print(f"Loading training data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("Data file not found. Generating synthetic data.")
        df = None

    # Initialize Environment
    env = CryptoTradingEnv(df=df, window_size=WINDOW_SIZE)
    
    # Initialize LLM Agent
    # It will automatically pick up OPENAI_API_KEY from .env via os.getenv
    agent = LLMTradingAgent(model_name="gpt-3.5-turbo")
    
    # Initialize Memory
    memory = RetrieverMemory(k=K_NEIGHBORS)
    
    print("Starting LLM-based Trading Session (with Real API calls)...")
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        step_count = 0
        
        while not done:
            # 1. Retrieve Context
            context = memory.retrieve(state)
            
            # 2. Act (LLM Decides)
            action, _, _ = agent.get_action(state, context)
            
            # 3. Step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Print occasionally to show it's working
            if step_count % 50 == 0:
                print(f"  Step {step_count}: Action {action}, Reward {reward:.2f}")
            
        print(f"Episode {episode+1}/{EPISODES}: Total Reward: {total_reward:.2f}, Steps: {step_count}")
        
        # 4. Update Memory
        memory.add_episode(episode_states, episode_actions, episode_rewards)
        memory.build()
        
    print("Session Completed.")

if __name__ == "__main__":
    main()
