import numpy as np
import torch
import os
import glob
import pandas as pd
from dotenv import load_dotenv

from env import CryptoTradingEnv
from agent_hybrid import HybridRememAgent
from memory import RetrieverMemory
from llm_reasoning import LLMReasoningModule

load_dotenv()

def main():
    # Hyperparameters
    EPISODES_PER_COIN = 2 # Very few for hybrid demo (API latency)
    WINDOW_SIZE = 10
    K_NEIGHBORS = 3
    
    # OpenAI Embeddings are 1536 dimensions
    LLM_EMB_DIM = 1536 
    
    # 1. Load Data
    data_dir = "prediction/data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No data. Run fetch_data.py")
        return

    # Initialize Environment Dims
    dummy_df = pd.read_csv(csv_files[0])
    dummy_env = CryptoTradingEnv(df=dummy_df, window_size=WINDOW_SIZE)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    memory_dim = 3 * K_NEIGHBORS
    
    # Initialize Hybrid Agent
    # Agent takes State + Memory + LLM_Embedding
    agent = HybridRememAgent(state_dim, action_dim, memory_dim, llm_emb_dim=LLM_EMB_DIM)
    
    # Initialize Components
    memory = RetrieverMemory(k=K_NEIGHBORS)
    llm_reasoner = LLMReasoningModule(model_name="gpt-3.5-turbo", embedding_model="text-embedding-3-small")
    
    print("Starting Hybrid ReMem (LLM-Guided) Training...")
    
    for file_path in csv_files:
        coin_name = os.path.basename(file_path).replace('.csv', '').upper()
        print(f"\n--- Training on {coin_name} ---")
        df = pd.read_csv(file_path)
        env = CryptoTradingEnv(df=df, window_size=WINDOW_SIZE)
        
        for episode in range(EPISODES_PER_COIN):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            episode_data = {
                'states': [], 'contexts': [], 'llm_embs': [], 
                'actions': [], 'rewards': [], 'log_probs': []
            }
            
            step_count = 0
            
            llm_emb = None
            LLM_UPDATE_FREQ = 50 # Update strategy every 50 steps
            
            while not done:
                # 1. Retrieve Memory
                context = memory.retrieve(state)
                
                # 2. LLM Reasoning (Think) - OPTIMIZED
                # Only call if it's the first step OR every N steps
                if llm_emb is None or step_count % LLM_UPDATE_FREQ == 0:
                    try:
                        llm_emb = llm_reasoner.get_strategy_embedding(state, context)
                    except Exception as e:
                        print(f"LLM Error, reusing old embedding: {e}")
                        if llm_emb is None: llm_emb = np.zeros(1536, dtype=np.float32)
                
                # 3. Act (Neural Net Execution)
                action, log_prob, value = agent.get_action(state, context, llm_emb)
                
                # 4. Step
                next_state, reward, done, _, info = env.step(action)
                
                # Store
                episode_data['states'].append(state)
                episode_data['contexts'].append(context)
                episode_data['llm_embs'].append(llm_emb)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['log_probs'].append(log_prob)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward {reward:.2f}")

            print(f"Episode {episode+1}: {coin_name} Total Reward: {total_reward:.2f}")
            
            # 5. Update Agent
            # Construct rollouts including LLM embeddings
            rollouts = list(zip(
                episode_data['states'],
                episode_data['contexts'],
                episode_data['llm_embs'],
                episode_data['actions'],
                episode_data['rewards'],
                episode_data['states'][1:] + [next_state], # next_states
                episode_data['log_probs'],
                [False]*(len(episode_data['rewards'])-1) + [True]
            ))
            
            loss = agent.update(rollouts)
            
            # 6. Update Memory
            memory.add_episode(episode_data['states'], episode_data['actions'], episode_data['rewards'])
            memory.build()

    print("Hybrid Training Completed.")
    torch.save(agent.state_dict(), "model_hybrid.pth")

if __name__ == "__main__":
    main()

