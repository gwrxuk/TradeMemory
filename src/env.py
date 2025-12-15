import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CryptoTradingEnv(gym.Env):
    def __init__(self, df=None, window_size=10, initial_balance=10000):
        super(CryptoTradingEnv, self).__init__()
        
        # If no data provided, generate synthetic sine wave data
        if df is None:
            t = np.linspace(0, 100, 1000)
            price = 100 + 10 * np.sin(t) + np.random.normal(0, 1, 1000)
            self.df = pd.DataFrame({'close': price, 'volume': np.random.rand(1000)*100})
        else:
            self.df = df
            
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: window_size x 2 (Close, Volume) + Current Position + Balance
        # Flattened
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size * 2 + 2,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0 # Amount of crypto held
        self.current_step = self.window_size
        self.done = False
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get window of data
        window = self.df.iloc[self.current_step-self.window_size : self.current_step]
        obs_data = window[['close', 'volume']].values.flatten()
        
        # Normalize simple (optional but good for RL)
        # For simplicity, we just pass raw values, but usually we'd normalize by initial price or similar.
        
        state = np.concatenate([obs_data, [self.position, self.balance]])
        return state.astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1: # Buy
            if self.balance >= current_price:
                # Buy 1 unit (simplified)
                self.position += 1
                self.balance -= current_price
        elif action == 2: # Sell
            if self.position >= 1:
                # Sell 1 unit
                self.position -= 1
                self.balance += current_price
        
        # Move to next step
        self.current_step += 1
        
        # Calculate Reward: Change in Net Worth
        net_worth = self.balance + self.position * current_price
        # Simple reward: net worth at end - net worth at start? 
        # Or step-wise change. Step-wise is usually better for learning.
        prev_price = self.df.iloc[self.current_step-1]['close']
        prev_net_worth = self.balance + self.position * prev_price # (Wait, balance changed if we traded)
        # Actually, let's track previous net worth explicitly
        if not hasattr(self, 'prev_net_worth'):
            self.prev_net_worth = self.initial_balance
            
        reward = net_worth - self.prev_net_worth
        self.prev_net_worth = net_worth
        
        # Check done
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        return self._get_observation(), reward, self.done, False, {'net_worth': net_worth}

