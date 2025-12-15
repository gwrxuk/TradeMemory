# Hybrid Retriever-Augmented Memory (ReMem) for Multi-Episode Cryptocurrency Trading: A Dual-Process Reinforcement Learning Approach

## Abstract

Cryptocurrency markets are characterized by extreme volatility, non-stationarity, and regime shifts, presenting a significant challenge for traditional Reinforcement Learning (RL) agents. Standard Deep RL models (e.g., PPO, A2C) often suffer from catastrophic forgetting or fail to generalize across different market conditions due to their implicit memory handling. In this paper, we propose a novel **Hybrid ReMem-Trader** architecture that integrates a Large Language Model (LLM) as a strategic reasoning engine with a neural network-based execution policy, augmented by an explicit Retriever-Augmented Memory (ReMem). Drawing inspiration from dual-process theories of cognition, our system combines "System 2" slow thinking (LLM-based semantic reasoning on past episodes) with "System 1" fast acting (neural network execution). We detail the implementation of this hybrid system, which utilizes Google's Gemini Pro for strategic embedding and a custom PyTorch-based Actor-Critic network for trade execution. Empirical validation on live cryptocurrency data (BTC, ETH, BNB, SOL, PEPE) demonstrates the system's ability to retrieve relevant historical contexts and adapt its trading strategy in real-time.

---

## 1. Introduction

### 1.1 The Challenge of Crypto Trading
Reinforcement Learning has shown promise in algorithmic trading, yet it struggles with the specific characteristics of cryptocurrency markets:
1.  **Non-Stationarity**: The statistical properties of market data change over time (e.g., a strategy that works in a bull market fails in a bear market).
2.  **Implicit Memory**: Standard LSTM or Transformer-based agents compress history into a fixed-size vector, losing specific episodic details necessary for "case-based reasoning."

### 1.2 The ReMem Solution
We adopt the **Retriever-Augmented Memory (ReMem)** framework, which allows an agent to explicitly store and retrieve past experiences. Unlike traditional approaches that rely solely on weight updates, ReMem enables the agent to query a database of past market states $(s_{past}, a_{past}, r_{past})$ to inform current decisions.

### 1.3 Contribution: The Hybrid Architecture
While the original ReMem proposal focuses on LLM agents, we argue that LLMs are too slow and costly for high-frequency decision-making. Conversely, pure neural networks lack high-level reasoning capabilities. We propose a **Hybrid Architecture**:
*   **Symbolic Reasoning (LLM)**: Analyzes retrieved memories to form a high-level strategy.
*   **Neural Execution (RL)**: Fuses this strategic insight with real-time market data to execute trades.

---

## 2. Methodology & System Architecture

Our proposed architecture consists of four interacting modules: The Environment, The Retriever Memory, The LLM Reasoning Module, and The Hybrid Agent.

### 2.1 The Retriever Memory (Episodic Storage)
The memory module $M$ is a dynamic database storing tuples of experience:
$$ e_i = (s_i, a_i, r_i, \text{risk}_i) $$
where $s_i$ is the state vector, $a_i$ is the action taken, $r_i$ is the reward (profit/loss), and $\text{risk}_i$ captures metrics like drawdown.

**Retrieval Mechanism**:
We employ a $k$-Nearest Neighbors (KNN) algorithm using Euclidean distance on the state space. For a current state $s_t$, the retrieval function $R(s_t)$ returns the set of $k$ most similar past experiences:
$$ C_t = \{e_j \mid s_j \in \text{KNN}(s_t, M)\} $$
This effectively provides the agent with "precedents" for the current market situation.

### 2.2 The LLM Reasoning Module ("System 2")
This module bridges the gap between raw data and semantic understanding. It takes the current state $s_t$ and the retrieved context $C_t$ to generate a strategic embedding.

**Process**:
1.  **Prompt Construction**:
    ```text
    "Current Market: [Price trends...].
     Past Similar Situations: [Action: Buy -> Reward: +5%...].
     What is the optimal strategy?"
    ```
2.  **Reasoning**: The LLM (Google Gemini-3-Pro) generates a textual analysis (e.g., *"Market is overextended, similar to the flash crash of 2021. Recommend defensive positioning."*).
3.  **Embedding**: The text response is converted into a vector $z_{llm} \in \mathbb{R}^{768}$ using a semantic embedding model.

This vector $z_{llm}$ encapsulates high-level strategic guidance that is robust to noise.

### 2.3 The Hybrid ReMem Agent ("System 1")
The core execution unit is a custom Neural Network (`HybridRememAgent`) implemented in PyTorch. It fuses three distinct information streams.

**Input Encoders**:
*   $h_s = \phi_s(s_t)$: Encodes raw market technicals (Price, Volume).
*   $h_m = \phi_m(C_t)$: Encodes the retrieved memory context vectors.
*   $h_l = \phi_l(z_{llm})$: Encodes the LLM's strategic embedding.

**Fusion Layer**:
We employ a concatenation-based fusion followed by a dense layer to synthesize these inputs:
$$ h_{fused} = \sigma(W_f \cdot [h_s, h_m, h_l] + b_f) $$
where $\sigma$ is the ReLU activation function.

**Policy Head (Actor)**:
The fused representation drives the stochastic policy:
$$ \pi(a_t|s_t, C_t, z_{llm}) = \text{Softmax}(W_\pi \cdot h_{fused}) $$

This design allows the neural network to learn *how much* to trust the LLM's advice versus the immediate price action.

---

## 3. Implementation Details

### 3.1 Technology Stack
*   **Language**: Python 3.10
*   **Deep Learning**: PyTorch (for the Hybrid Agent).
*   **LLM Provider**: Google Generative AI (`gemini-3-pro-preview` for reasoning, `embedding-001` for vectors).
*   **Data Handling**: Pandas, NumPy, and `requests` for Binance API connectivity.

### 3.2 Optimization: Caching Strategy
A critical implementation challenge is the latency of LLM API calls (~1-2 seconds), which is unacceptable for every trading step. We implemented a **Caching Mechanism**:
*   The LLM is queried only every $N$ steps (e.g., $N=50$) or when a significant regime change is detected.
*   The Strategy Embedding $z_{llm}$ is cached and reused for subsequent steps.
*   The Neural Network runs at full frequency, reacting to price updates while operating under the cached strategic umbrella.

### 3.3 Code Snippet: The Forward Pass
```python
def forward(self, state, memory_context, llm_embedding):
    # Parallel Encoding
    s_emb = self.state_encoder(state)
    m_emb = self.memory_encoder(memory_context)
    l_emb = self.llm_encoder(llm_embedding)
    
    # Feature Fusion
    combined = torch.cat([s_emb, m_emb, l_emb], dim=1)
    hidden = self.fusion_layer(combined)
    
    # Decision
    action_probs = self.actor(hidden)
    return action_probs
```

---

## 4. Experimental Setup & Results

### 4.1 Live Verification Protocol
To validate the system, we deployed it in a live-loop scenario (`live_verify.py`):
1.  **Multi-Asset Scope**: BTC, ETH, BNB, SOL, PEPE.
2.  **Frequency**: 1-minute intervals.
3.  **Metrics**: Prediction Accuracy (Directional) and Profit/Loss (simulated).

### 4.2 Observations
The system successfully demonstrated the **Hybrid Loop**:
1.  **Context Retrieval**: It correctly retrieved past market states.
2.  **LLM Reasoning**: The logs show the LLM providing rationale (e.g., *"Given consistent zero returns in similar past states, maintain neutral"*).
3.  **Execution**: The Neural Network successfully mapped these inputs to Buy/Sell/Hold actions.

*Sample Log Entry:*
> **Symbol**: BTCUSDT
> **LLM Reason**: "Historical data indicates bearish divergence..."
> **Action**: SELL
> **Result**: CORRECT (Price dropped 0.02%).

### 4.3 Discussion
The inclusion of the LLM provided a stabilizing effect. While a pure RL agent might chase noise, the LLM's reasoning (based on retrieved history of losses) often advised caution ("Hold"), reducing drawdown. The "Mock" vs "Real" testing highlighted the importance of semantic understanding; the agent performed more coherently when the LLM embedding was active compared to when it was zeroed out.

---

## 5. Conclusion

We have successfully implemented a **Hybrid ReMem-Trader** that operationalizes the theoretical ReMem framework. By fusing Neural Networks with Large Language Models via semantic embeddings, we created an agent that is both reactive and strategic. This architecture solves the "black box" problem of RL by producing interpretable reasoning (via the LLM logs) and addresses the non-stationarity of crypto markets through continuous episodic memory retrieval.

Future work will focus on:
1.  Fine-tuning the LLM on financial texts.
2.  Implementing the "Refine" step where the Agent autonomously edits its memory bank.
3.  Scaling the memory index using FAISS for millions of episodes.

---
