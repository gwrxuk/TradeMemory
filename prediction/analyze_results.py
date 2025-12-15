import pandas as pd
import numpy as np
import os
from scipy import stats

FILES = {
    "Hybrid ReMem (Gemini + Neural)": "verification_20251214.csv",
    "ReMem-Only (Neural)": "verification_remem_only.csv",
    "LLM-Only (Gemini)": "verification_llm_only.csv"
}

def clean_pct(x):
    if isinstance(x, str):
        return float(x.replace('%', ''))
    return float(x)

def calculate_advanced_metrics(df):
    # Filter active trades
    trades = df[df['Action_Pred'] != 'HOLD']['Profit_Loss_Pct']
    
    if len(trades) < 2:
        return {
            "sharpe": 0.0, "volatility": 0.0, "max_dd": 0.0, 
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }
        
    # Economic Metrics
    avg_return = trades.mean()
    volatility = trades.std()
    
    # Sharpe (Assuming risk-free rate ~ 0 for high-freq minute trading)
    # Annualized Sharpe (Minute -> Year = * sqrt(525600)) is extreme, 
    # so we just use per-trade Sharpe.
    sharpe = avg_return / volatility if volatility != 0 else 0
    
    # Max Drawdown (on cumulative curve)
    cumulative = trades.cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    
    # Expectancy
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_rate = len(wins) / len(trades)
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    return {
        "sharpe": sharpe,
        "volatility": volatility,
        "max_dd": max_dd,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }

def analyze_file(name, filepath):
    if not os.path.exists(filepath): return None
    
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0: return None
        
        col_name = 'Profit_Loss_Pct'
        if 'Profit_Loss_Pct' not in df.columns and 'Profit_Loss' in df.columns:
            col_name = 'Profit_Loss'
        df[col_name] = df[col_name].apply(clean_pct)
        df['Profit_Loss_Pct'] = df[col_name]
        
        # Basic Metrics
        active_df = df[df['Action_Pred'] != 'HOLD']
        total_trades = len(active_df)
        
        accuracy = 0.0
        if total_trades > 0:
            correct = len(active_df[active_df['Result_Verified'] == 'CORRECT'])
            accuracy = (correct / total_trades) * 100
            
        cum_return = df['Profit_Loss_Pct'].sum()
        
        # Advanced Metrics
        adv = calculate_advanced_metrics(df)
        
        return {
            "name": name,
            "df": df, # Keep for statistical test
            "total_steps": len(df),
            "trades": total_trades,
            "accuracy": accuracy,
            "cum_return": cum_return,
            **adv
        }
    except Exception as e:
        print(f"Error {name}: {e}")
        return None

def perform_ttest(res1, res2):
    # Independent t-test on trade returns
    returns1 = res1['df'][res1['df']['Action_Pred'] != 'HOLD']['Profit_Loss_Pct']
    returns2 = res2['df'][res2['df']['Action_Pred'] != 'HOLD']['Profit_Loss_Pct']
    
    if len(returns1) < 2 or len(returns2) < 2:
        return None
        
    t_stat, p_val = stats.ttest_ind(returns1, returns2, equal_var=False)
    return t_stat, p_val

def generate_swot(stats):
    name = stats['name']
    swot = {"S": [], "W": [], "O": [], "T": []}
    
    # Generic SWOT based on performance
    if stats['sharpe'] > 0.1: swot["S"].append("Positive Sharpe Ratio (Risk-Adjusted Return)")
    if stats['max_dd'] < -5: swot["W"].append("High Drawdown Risk")
    if stats['accuracy'] > 55: swot["S"].append("High Directional Accuracy")
    
    if "Hybrid" in name:
        swot["S"].extend(["Semantic Context Integration", "Adaptive Strategy"])
        swot["W"].extend(["API Latency/Cost", "Complexity"])
    elif "ReMem" in name:
        swot["S"].extend(["Execution Speed", "Stability"])
        swot["W"].extend(["Lack of Reasoning", "Overfitting Technicals"])
    
    return swot

def write_report(results):
    sorted_res = sorted(results, key=lambda x: x['cum_return'], reverse=True)
    
    md = "# Comprehensive Economic & Statistical Analysis of ReMem Agents\n\n"
    
    md += "## 1. Economic Performance Metrics\n\n"
    md += "| Strategy | Return | Sharpe Ratio | Max Drawdown | Expectancy | Volatility |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    
    for r in sorted_res:
        md += f"| {r['name']} | **{r['cum_return']:.2f}%** | {r['sharpe']:.3f} | {r['max_dd']:.2f}% | {r['expectancy']:.3f}% | {r['volatility']:.3f} |\n"
        
    md += "\n> **Sharpe Ratio**: Measures excess return per unit of risk. Higher is better.\n"
    md += "> **Expectancy**: Average return per trade. Positive means the strategy has a statistical edge.\n\n"
    
    md += "## 2. Statistical Significance (t-test)\n\n"
    
    # Pairwise t-test
    hybrid = next((r for r in results if "Hybrid" in r['name']), None)
    remem = next((r for r in results if "ReMem-Only" in r['name']), None)
    
    if hybrid and remem:
        t_res = perform_ttest(hybrid, remem)
        if t_res:
            t, p = t_res
            sig = "Significant" if p < 0.05 else "Not Significant"
            md += f"### Hybrid vs. ReMem-Only\n"
            md += f"- **t-statistic**: {t:.4f}\n"
            md += f"- **p-value**: {p:.4f}\n"
            md += f"- **Conclusion**: The difference in performance is **{sig}** at 95% confidence.\n"
            if p >= 0.05:
                md += "  *(Note: The LLM's contribution may not yet be statistically distinguishable from noise in this short sample size.)*\n\n"
        else:
            md += "Insufficient data for t-test.\n\n"
            
    md += "## 3. SWOT Analysis\n\n"
    for r in results:
        swot = generate_swot(r)
        md += f"### {r['name']}\n"
        md += f"**Strengths:**\n" + "\n".join([f"- {s}" for s in swot["S"]]) + "\n"
        md += f"**Weaknesses:**\n" + "\n".join([f"- {s}" for s in swot["W"]]) + "\n\n"

    md += "## 4. Final Ranking & Conclusion\n\n"
    for i, r in enumerate(sorted_res):
        md += f"**Rank {i+1}: {r['name']}**\n"
        md += f"- Return: {r['cum_return']:.2f}%\n"
        md += f"- Edge: {r['expectancy']:.3f}% per trade\n\n"

    with open("report.md", "w") as f:
        f.write(md)
    print("Updated report.md with economic analysis.")

if __name__ == "__main__":
    results = []
    for name, path in FILES.items():
        res = analyze_file(name, path)
        if res: results.append(res)
    
    if results:
        write_report(results)
    else:
        print("No results.")
