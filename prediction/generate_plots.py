import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for academic publication
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 300

FILES = {
    "Hybrid ReMem": "verification_20251214.csv",
    "ReMem-Only": "verification_remem_only.csv",
    "LLM-Only": "verification_llm_only.csv"
}

OUTPUT_DIR = "plots"

def clean_pct(x):
    if isinstance(x, str):
        return float(x.replace('%', ''))
    return float(x)

def load_data():
    data_frames = {}
    combined_data = []
    
    for name, path in FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = 'Profit_Loss_Pct' if 'Profit_Loss_Pct' in df.columns else 'Profit_Loss'
            df[col] = df[col].apply(clean_pct)
            df['Strategy'] = name
            
            # Ensure datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Calculate Cumulative Return per coin then average? 
            # Or just cumulative return of the strategy portfolio (assuming equal weight or sequential)
            # For simplicity in this visualization, we assume sequential trades in the log form a single equity curve
            # But better to group by timestamp to handle multi-coin steps
            
            # Group by Timestamp to get portfolio-level return per minute
            portfolio_df = df.groupby('Timestamp')[col].mean().reset_index()
            portfolio_df['Cumulative_Return'] = portfolio_df[col].cumsum()
            portfolio_df['Strategy'] = name
            
            data_frames[name] = portfolio_df
            combined_data.append(portfolio_df)
            
    if not combined_data:
        return None
    return pd.concat(combined_data, ignore_index=True)

def plot_cumulative_returns(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Timestamp', y='Cumulative_Return', hue='Strategy', linewidth=2)
    plt.title('Cumulative Performance Comparison')
    plt.ylabel('Cumulative Return (%)')
    plt.xlabel('Time')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_returns.png")
    print("Saved cumulative_returns.png")

def plot_drawdown(df):
    plt.figure(figsize=(12, 6))
    
    for strategy in df['Strategy'].unique():
        strat_df = df[df['Strategy'] == strategy].sort_values('Timestamp')
        cum = strat_df['Cumulative_Return']
        running_max = cum.cummax()
        drawdown = cum - running_max
        plt.fill_between(strat_df['Timestamp'], drawdown, alpha=0.3, label=strategy)
        
    plt.title('Drawdown Analysis')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Time')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/drawdown.png")
    print("Saved drawdown.png")

def plot_return_distribution(df):
    plt.figure(figsize=(10, 6))
    
    # We need the per-step returns, not cumulative
    # Extract returns from the processed df
    # Note: 'Profit_Loss_Pct' is the mean return at that timestamp (proxy for portfolio return)
    # The column name in 'combined_df' from load_data was based on the groupby, 
    # let's look at the code: grouped by Timestamp, col is 'Profit_Loss_Pct' (mean)
    
    # Renaming for clarity in the plot function if needed, but 'Profit_Loss_Pct' is usually the name after groupby if not renamed?
    # Actually groupby(...)[col].mean() returns a Series with name 'Profit_Loss_Pct'.
    # reset_index() keeps it.
    
    # Check column name. In load_data: portfolio_df[col] is grouped. col is 'Profit_Loss_Pct'.
    
    sns.kdeplot(data=df, x='Profit_Loss_Pct', hue='Strategy', fill=True, common_norm=False, alpha=0.4)
    plt.title('Distribution of Returns (Kernel Density Estimation)')
    plt.xlabel('Return per Minute (%)')
    plt.ylabel('Density')
    plt.xlim(-0.5, 0.5) # Zoom in on center if tails are long
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/return_distribution.png")
    print("Saved return_distribution.png")

def plot_correlation_heatmap(df):
    # Pivot to get columns as strategies
    pivot_df = df.pivot(index='Timestamp', columns='Strategy', values='Profit_Loss_Pct').fillna(0)
    
    plt.figure(figsize=(8, 6))
    corr = pivot_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Strategy Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    print("Saved correlation_matrix.png")

def plot_box_returns(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Strategy', y='Profit_Loss_Pct', palette="Set3")
    plt.title('Return Statistics by Strategy')
    plt.ylabel('Return (%)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/returns_boxplot.png")
    print("Saved returns_boxplot.png")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data...")
    df = load_data()
    
    if df is None or len(df) == 0:
        print("No data found to plot.")
        return

    # Column name adjustment if needed
    if 'Profit_Loss_Pct' not in df.columns and 'Profit_Loss' in df.columns:
        df['Profit_Loss_Pct'] = df['Profit_Loss']

    print("Generating plots...")
    plot_cumulative_returns(df)
    plot_drawdown(df)
    plot_return_distribution(df)
    plot_correlation_heatmap(df)
    plot_box_returns(df)
    print(f"All plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

