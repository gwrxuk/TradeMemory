import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy import stats

# Constants
FILES = {
    "Hybrid": "verification_20251214.csv",
    "ReMem": "verification_remem_only.csv",
    "LLM": "verification_llm_only.csv"
}

OUTPUT_FILE = "Full_Econometric_Results.md"

def clean_pct(x):
    if isinstance(x, str):
        return float(x.replace('%', ''))
    return float(x)

def load_data():
    data = {}
    for name, path in FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = 'Profit_Loss_Pct' if 'Profit_Loss_Pct' in df.columns else 'Profit_Loss'
            df[col] = df[col].apply(clean_pct)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index('Timestamp').sort_index()
            # Resample to 1 minute to align
            df = df[~df.index.duplicated(keep='first')]
            data[name] = df[col]
    return pd.DataFrame(data).fillna(0)

# --- V. Market & Asset Pricing Analysis (Event Study) ---
def event_study_analysis(df):
    """
    Simulate Event Study: How do strategies perform during 'High Volatility' events?
    Event: Absolute Market Return > 95th percentile.
    """
    market = df.mean(axis=1)
    threshold = market.abs().quantile(0.95)
    
    events = market[market.abs() > threshold].index
    
    results = {}
    for col in df.columns:
        # Abnormal Return = Strategy Return - Beta * Market Return (Simplified: Strategy - Market)
        # We calculate average return during event windows (+/- 1 min)
        event_returns = []
        for t in events:
            # Look at t, t+1
            window = df[col].loc[t : t + pd.Timedelta(minutes=1)]
            if len(window) > 0:
                event_returns.append(window.mean())
        
        avg_abnormal = np.mean(event_returns) if event_returns else 0
        results[col] = avg_abnormal
        
    return results

# --- III. Factor Model Analysis (Fama-French Proxy) ---
def factor_model_analysis(df):
    """
    Construct Crypto Factors from the data itself (since we don't have external Fama-French).
    MKT: Market Return (Mean of all assets/strategies)
    MOM: Momentum (Lagged Market Return)
    """
    market = df.mean(axis=1)
    mom = market.shift(1).fillna(0)
    
    factors = pd.DataFrame({'MKT': market, 'MOM': mom})
    
    results = {}
    for col in df.columns:
        Y = df[col]
        X = sm.add_constant(factors)
        model = sm.OLS(Y, X).fit()
        results[col] = {
            "Alpha": model.params['const'],
            "Beta_MKT": model.params['MKT'],
            "Beta_MOM": model.params['MOM'],
            "R2": model.rsquared
        }
    return results

# --- IV. Risk (GARCH) ---
def garch_analysis(df):
    results = {}
    for col in df.columns:
        r = df[col] * 100 # Scale for convergence
        try:
            am = arch_model(r, vol='Garch', p=1, q=1)
            res = am.fit(disp='off')
            results[col] = res.params['alpha[1]'] + res.params['beta[1]'] # Persistence
        except:
            results[col] = np.nan
    return results

# --- VII. Microstructure (Bid-Ask Proxy) ---
def microstructure_analysis(df):
    """
    Roll Model for Bid-Ask Spread: Spread = 2 * sqrt(-Cov(Delta_P_t, Delta_P_t-1))
    We apply this to Strategy Returns to estimate 'Effective Cost' or noise.
    """
    results = {}
    for col in df.columns:
        r = df[col]
        cov = r.cov(r.shift(1))
        if cov < 0:
            spread = 2 * np.sqrt(-cov)
        else:
            spread = 0 # No measurable bounce
        results[col] = spread
    return results

def run_pipeline():
    print("Loading Data...")
    df = load_data()
    if df.empty:
        print("No data found.")
        return

    print("Running Event Study...")
    event_res = event_study_analysis(df)
    
    print("Running Factor Models...")
    factor_res = factor_model_analysis(df)
    
    print("Running GARCH...")
    garch_res = garch_analysis(df)
    
    print("Running Microstructure Analysis...")
    micro_res = microstructure_analysis(df)
    
    # Generate Markdown
    md = "# Full Econometric & Financial Analysis Results\n\n"
    
    md += "## III. Investment & Portfolio Analysis\n"
    md += "### Factor Model (Crypto-CAPM + Momentum)\n"
    md += "| Strategy | Alpha | Beta (MKT) | Beta (MOM) | R-Squared |\n"
    md += "|---|---|---|---|---|\n"
    for k, v in factor_res.items():
        md += f"| {k} | {v['Alpha']:.4f} | {v['Beta_MKT']:.4f} | {v['Beta_MOM']:.4f} | {v['R2']:.4f} |\n"
        
    md += "\n## IV. Risk & Volatility Analysis\n"
    md += "### GARCH(1,1) Volatility Persistence\n"
    md += "| Strategy | Persistence (alpha+beta) |\n|---|---|\n"
    for k, v in garch_res.items():
        md += f"| {k} | {v:.4f} |\n"
    md += "> Persistence close to 1.0 indicates 'long memory' in volatility (clusters of risk).\n\n"
    
    md += "## V. Market & Asset Pricing Analysis\n"
    md += "### Event Study: Performance during High Volatility Shocks\n"
    md += "| Strategy | Avg Abnormal Return during Shocks |\n|---|---|\n"
    for k, v in event_res.items():
        md += f"| {k} | {v:.4f}% |\n"
    md += "> Positive values indicate the strategy acts as a hedge or profits from chaos.\n\n"
    
    md += "## VII. Quantitative & Computational Methods\n"
    md += "### Microstructure: Effective Spread / Noise Estimate (Roll Model)\n"
    md += "| Strategy | Effective Cost/Noise |\n|---|---|\n"
    for k, v in micro_res.items():
        md += f"| {k} | {v:.4f} |\n"
    md += "> Higher values indicate the strategy returns are noisy or mean-reverting due to execution costs (simulated).\n"
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(md)
    print(f"Analysis complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()

