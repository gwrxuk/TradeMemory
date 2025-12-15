# Full Econometric & Financial Analysis Results

## III. Investment & Portfolio Analysis
### Factor Model (Crypto-CAPM + Momentum)
| Strategy | Alpha | Beta (MKT) | Beta (MOM) | R-Squared |
|---|---|---|---|---|
| Hybrid | 0.0000 | 1.2496 | -0.3837 | 0.4017 |
| ReMem | -0.0005 | 0.4487 | -0.0233 | 0.1439 |
| LLM | 0.0005 | 1.3017 | 0.4069 | 0.5121 |

## IV. Risk & Volatility Analysis
### GARCH(1,1) Volatility Persistence
| Strategy | Persistence (alpha+beta) |
|---|---|
| Hybrid | 0.8629 |
| ReMem | 1.0000 |
| LLM | 0.8211 |
> Persistence close to 1.0 indicates 'long memory' in volatility (clusters of risk).

## V. Market & Asset Pricing Analysis
### Event Study: Performance during High Volatility Shocks
| Strategy | Avg Abnormal Return during Shocks |
|---|---|
| Hybrid | -0.0242% |
| ReMem | -0.0254% |
| LLM | -0.0424% |
> Positive values indicate the strategy acts as a hedge or profits from chaos.

## VII. Quantitative & Computational Methods
### Microstructure: Effective Spread / Noise Estimate (Roll Model)
| Strategy | Effective Cost/Noise |
|---|---|
| Hybrid | 0.0086 |
| ReMem | 0.0000 |
| LLM | 0.0164 |
> Higher values indicate the strategy returns are noisy or mean-reverting due to execution costs (simulated).
