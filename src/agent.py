import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RememAgent(nn.Module):
    def __init__(self, state_dim, action_dim, memory_dim, lr=0.0003):
        super(RememAgent, self).__init__()
        
        # Architecture components based on reference:
        # 1. State Encoder
        # 2. Context Encoder (Process retrieved memory)
        # 3. Fusion Layer (Attention or Concat)
        # 4. Policy/Value Heads
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Memory dim is (k * features_per_memory)
        # We process the memory vector directly
        self.context_encoder = nn.Sequential(
            nn.Linear(memory_dim, 128),
            nn.ReLU()
        )
        
        # Fusion: Attention Mechanism
        # Query: Encoded State
        # Key/Value: Encoded Context
        # For simplicity in this implementation, we use a Gated Fusion
        # z = sigmoid(gate_net(state, context))
        # fused = z * state + (1-z) * context
        self.gate_net = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Linear(256, 128)
        
        # Policy Network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, context):
        # 1. Encode
        s_emb = self.state_encoder(state)     # [batch, 128]
        c_emb = self.context_encoder(context) # [batch, 128]
        
        # 2. Fuse (Concatenate + Linear)
        # The reference mentions "Context Constructor" - this serves that role
        combined = torch.cat([s_emb, c_emb], dim=1) # [batch, 256]
        
        # Optional: Gating (ReMem "Think" - decide how much to rely on memory)
        gate = self.gate_net(combined)
        # Weighted combination? Or just process the concatenation.
        # Let's process the concatenation to let the net learn interactions.
        
        hidden = torch.relu(self.fusion_layer(combined))
        
        # 3. Output
        probs = self.actor(hidden)
        value = self.critic(hidden)
        
        return probs, value
        
    def get_action(self, state, memory_context):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        context_t = torch.FloatTensor(memory_context).unsqueeze(0)
        
        probs, value = self.forward(state_t, context_t)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value

    def update(self, rollouts, gamma=0.99):
        states, contexts, actions, rewards, next_states, log_probs, dones = zip(*rollouts)
        
        states_t = torch.FloatTensor(np.array(states))
        contexts_t = torch.FloatTensor(np.array(contexts))
        actions_t = torch.LongTensor(np.array(actions))
        rewards_t = torch.FloatTensor(np.array(rewards))
        
        # Calculate Returns (Discounted Reward)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass
        probs, values = self.forward(states_t, contexts_t)
        values = values.squeeze()
        
        advantage = returns - values.detach()
        
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions_t)
        
        actor_loss = -(new_log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(values, returns)
        
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
