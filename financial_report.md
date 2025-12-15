# Comprehensive Financial Analysis of Trading Agents

## III. Investment & Portfolio Analysis (MPT/CAPM)
| Strategy | Alpha (Excess) | Beta (Sys Risk) | R-Squared | Info Ratio |
|---|---|---|---|---|
| Hybrid | 0.00087 | 1.1049 | 0.3642 | 0.0256 |
| ReMem | -0.00050 | 0.4399 | 0.1435 | -0.0177 |
| LLM | -0.00037 | 1.4552 | 0.4800 | -0.0101 |
> **Insight**: High Alpha indicates skill (edge) independent of market movements. Beta < 1 indicates defensive characteristics.

## IV. Risk & Volatility Analysis (GARCH)
| Strategy | GARCH Persistence (alpha+beta) | Volatility Half-Life (Steps) |
|---|---|---|
| Hybrid | 0.8629 | 4.7 |
| ReMem | 1.0000 | 94028.4 |
| LLM | 0.8211 | 3.5 |
> **Insight**: High persistence implies that volatility clusters—high risk periods tend to last longer. A lower half-life means the strategy stabilizes quickly after a shock.

## VII. Quantitative Finance: Statistical Arbitrage Potential
Based on the Ornstein-Uhlenbeck process fit (Mean Reversion Speed):
- **ReMem-Only**: Shows characteristics of a momentum strategy (Positive Autocorrelation).
- **Hybrid**: Shows characteristics of a mean-reversion strategy (Negative Autocorrelation), suitable for Statistical Arbitrage.

