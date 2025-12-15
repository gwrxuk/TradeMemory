import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HybridRememAgent(nn.Module):
    def __init__(self, state_dim, action_dim, memory_dim, llm_emb_dim=128, lr=0.0003):
        super(HybridRememAgent, self).__init__()
        
        # 1. State Encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # 2. Memory Context Encoder (from Retriever)
        self.memory_encoder = nn.Sequential(
            nn.Linear(memory_dim, 128),
            nn.ReLU()
        )
        
        # 3. LLM Strategy Encoder
        # Takes the embedding vector from the LLM (e.g. OpenAI text-embedding-3-small)
        self.llm_encoder = nn.Sequential(
            nn.Linear(llm_emb_dim, 128),
            nn.ReLU()
        )
        
        # 4. Fusion Layer (Concatenate State + Memory + LLM)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 5. Policy Head
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 6. Value Head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state, memory_context, llm_embedding):
        # Encode inputs
        s_emb = self.state_encoder(state)
        m_emb = self.memory_encoder(memory_context)
        l_emb = self.llm_encoder(llm_embedding)
        
        # Fuse
        combined = torch.cat([s_emb, m_emb, l_emb], dim=1)
        hidden = self.fusion_layer(combined)
        
        # Outputs
        probs = self.actor(hidden)
        value = self.critic(hidden)
        
        return probs, value
        
    def get_action(self, state, memory_context, llm_embedding):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        context_t = torch.FloatTensor(memory_context).unsqueeze(0)
        llm_t = torch.FloatTensor(llm_embedding).unsqueeze(0)
        
        probs, value = self.forward(state_t, context_t, llm_t)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), value

    def update(self, rollouts, gamma=0.99):
        # Unpack rollout including llm_embeddings
        states, contexts, llm_embs, actions, rewards, next_states, log_probs, dones = zip(*rollouts)
        
        states_t = torch.FloatTensor(np.array(states))
        contexts_t = torch.FloatTensor(np.array(contexts))
        llm_t = torch.FloatTensor(np.array(llm_embs))
        actions_t = torch.LongTensor(np.array(actions))
        rewards_t = torch.FloatTensor(np.array(rewards))
        
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        probs, values = self.forward(states_t, contexts_t, llm_t)
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

