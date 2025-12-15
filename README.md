# Retriever-Augmented Memory for Crypto Trading (Remem)

This project implements a Reinforcement Learning agent with Retriever-Augmented Memory for cryptocurrency trading, encapsulated in Docker.

## Architecture
- **Agent**: Actor-Critic (PPO-style updates) that takes current state AND retrieved context as input.
- **Memory**: Stores past experiences (State -> Action, Reward).
- **Retriever**: Uses k-Nearest Neighbors to find past states similar to the current state and retrieves their outcomes to inform the agent's decision.
- **Environment**: Custom Gymnasium environment for Crypto Trading (generates synthetic data if no CSV provided).

## Project Structure
- `src/main.py`: Main training loop.
- `src/agent.py`: Neural Network Agent.
- `src/memory.py`: Retrieval System.
- `src/env.py`: Trading Environment.
- `prediction/`: Contains scripts for real-world data fetching and evaluation.

## How to Run

1. **Build and Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Run manually:**
   ```bash
   pip install -r requirements.txt
   python src/main.py
   ```

## Prediction / Real-World Validation

To test the agent on real cryptocurrency data (BTC, ETH, SOL, PEPE):

1. **Fetch Data:**
   ```bash
   python prediction/fetch_data.py
   ```
   This will download historical data (2023-2024) to `prediction/data/`.

2. **Run Evaluation:**
   ```bash
   python prediction/run_prediction.py
   ```
   This will run an untrained (or trained if loaded) agent on the downloaded data and report ROI.

## Data
To use real data, place a CSV file in `src/data/` and update `src/main.py` to load it. The CSV should have `close` and `volume` columns.
