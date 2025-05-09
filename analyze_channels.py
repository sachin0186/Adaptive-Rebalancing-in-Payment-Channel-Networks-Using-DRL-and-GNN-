import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This will set up seaborn's default styling

def load_data(file_path):
    """Load and preprocess the payment channel data."""
    df = pd.read_csv(file_path)
    
    # Calculate additional metrics
    df['utilization'] = (df['node1_balance'] + df['node2_balance']) / df['capacity']
    df['balance_imbalance'] = abs(df['node1_balance'] - df['node2_balance']) / df['capacity']
    df['total_tx_rate'] = df['node1_tx_rate'] + df['node2_tx_rate']
    
    return df

def analyze_channels(df):
    """Perform analysis on the payment channel data."""
    analysis = {
        'total_channels': len(df),
        'total_capacity': df['capacity'].sum(),
        'avg_capacity': df['capacity'].mean(),
        'median_capacity': df['capacity'].median(),
        'avg_utilization': df['utilization'].mean(),
        'avg_balance_imbalance': df['balance_imbalance'].mean(),
        'avg_fee_rate': df['fee_rate_ppm'].mean(),
        'channels_by_age': df.groupby(pd.qcut(df['age_days'], q=5))['channel_id'].count().to_dict(),
        'capacity_by_age': df.groupby(pd.qcut(df['age_days'], q=5))['capacity'].sum().to_dict()
    }
    return analysis

def create_visualizations(df, output_dir):
    """Create and save visualizations of the payment channel data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create a single figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Payment Channel Network Analysis', fontsize=16)
    
    # 1. Capacity Distribution
    sns.histplot(data=df, x='capacity', bins=50, ax=axes[0,0])
    axes[0,0].set_title('Distribution of Channel Capacities')
    axes[0,0].set_xlabel('Capacity (sats)')
    axes[0,0].set_ylabel('Number of Channels')
    
    # 2. Utilization vs Balance Imbalance
    sns.scatterplot(data=df, x='utilization', y='balance_imbalance', alpha=0.5, ax=axes[0,1])
    axes[0,1].set_title('Channel Utilization vs Balance Imbalance')
    axes[0,1].set_xlabel('Utilization')
    axes[0,1].set_ylabel('Balance Imbalance')
    
    # 3. Fee Rate Distribution
    sns.histplot(data=df, x='fee_rate_ppm', bins=50, ax=axes[1,0])
    axes[1,0].set_title('Distribution of Fee Rates')
    axes[1,0].set_xlabel('Fee Rate (ppm)')
    axes[1,0].set_ylabel('Number of Channels')
    
    # 4. Age vs Capacity
    sns.scatterplot(data=df, x='age_days', y='capacity', alpha=0.5, ax=axes[1,1])
    axes[1,1].set_title('Channel Age vs Capacity')
    axes[1,1].set_xlabel('Age (days)')
    axes[1,1].set_ylabel('Capacity (sats)')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'all_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_analysis(analysis, output_file):
    """Save the analysis results to a markdown file."""
    with open(output_file, 'w') as f:
        f.write('# Payment Channel Network Analysis\n\n')
        
        f.write('## Summary Statistics\n\n')
        f.write(f'- Total number of channels: {analysis["total_channels"]:,}\n')
        f.write(f'- Total network capacity: {analysis["total_capacity"]:,.0f} sats\n')
        f.write(f'- Average channel capacity: {analysis["avg_capacity"]:,.0f} sats\n')
        f.write(f'- Median channel capacity: {analysis["median_capacity"]:,.0f} sats\n')
        f.write(f'- Average channel utilization: {analysis["avg_utilization"]:.2%}\n')
        f.write(f'- Average balance imbalance: {analysis["avg_balance_imbalance"]:.2%}\n')
        f.write(f'- Average fee rate: {analysis["avg_fee_rate"]:.2f} ppm\n\n')
        
        f.write('## Channel Distribution by Age\n\n')
        for age_range, count in analysis['channels_by_age'].items():
            f.write(f'- {age_range}: {count} channels\n')
        
        f.write('\n## Capacity Distribution by Age\n\n')
        for age_range, capacity in analysis['capacity_by_age'].items():
            f.write(f'- {age_range}: {capacity:,.0f} sats\n')

def main():
    # Load data
    df = load_data('DATA/pcn_network_data.csv')
    
    # Perform analysis
    analysis = analyze_channels(df)
    
    # Create visualizations
    create_visualizations(df, 'analysis_output')
    
    # Save analysis results
    save_analysis(analysis, 'analysis_output/analysis_results.md')
    
    print("Analysis complete! Results saved in 'analysis_output' directory.")

if __name__ == '__main__':
    main() 