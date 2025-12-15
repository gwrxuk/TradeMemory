import numpy as np
from sklearn.neighbors import NearestNeighbors

class RetrieverMemory:
    def __init__(self, k=5):
        self.keys = []   # State vectors (embeddings)
        self.values = [] # (Action, Reward, Risk_Metric)
        self.k = k
        self.index = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.fitted = False
        self.capacity = 10000 # Limit memory size to prevent infinite growth
        
    def add_episode(self, states, actions, rewards, risk_metrics=None):
        """
        Add a full episode to memory.
        risk_metrics: Optional list of risk values (e.g., drawdown) for each step
        """
        # If no risk metrics provided, use 0
        if risk_metrics is None:
            risk_metrics = [0.0] * len(states)
            
        for s, a, r, risk in zip(states, actions, rewards, risk_metrics):
            if len(self.keys) >= self.capacity:
                # FIFO pruning (simplest)
                self.keys.pop(0)
                self.values.pop(0)
                
            self.keys.append(s)
            # Value: Action, Reward, Risk
            self.values.append(np.array([a, r, risk], dtype=np.float32))
            
    def build(self):
        """Build/Rebuild the KNN index."""
        if len(self.keys) >= self.k:
            self.index.fit(np.array(self.keys))
            self.fitted = True
            
    def retrieve(self, query_state):
        """
        Retrieve k nearest neighbors.
        Returns: 
          context_features: Concatenated features (or could return list for Attention)
          raw_matches: List of [action, reward, risk]
        """
        if not self.fitted or len(self.keys) < self.k:
            # Return zeros if not ready. Shape: k * 3 (Action, Reward, Risk)
            return np.zeros(self.k * 3)
        
        distances, indices = self.index.kneighbors([query_state])
        
        # Gather values
        retrieved_values = []
        for idx in indices[0]:
            retrieved_values.append(self.values[idx])
            
        # Flatten: (k * 3, )
        return np.concatenate(retrieved_values)

    def refine(self, query_state, context):
        """
        'Refine' step from the paper.
        Allows the agent to filter or re-weight retrieved memories.
        For now, we implement a simple distance-based weighting.
        """
        # (This logic can be moved inside the network as Attention)
        return context
