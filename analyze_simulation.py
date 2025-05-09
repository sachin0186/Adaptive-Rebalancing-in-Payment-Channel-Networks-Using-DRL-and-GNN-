import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_simulation_results(file_path):
    """Load simulation results from HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        # Get the trajectory data
        traj = f['relay_node_channel_rebalancing']
        run_name = list(traj['results']['runs'].keys())[0]  # Get the first run
        run_data = traj['results']['runs'][run_name]
        
        # Extract results
        results = {}
        for key in run_data.keys():
            if isinstance(run_data[key], h5py.Group):
                # The actual data is nested one level deeper with the same name
                results[key] = run_data[key][key][()]
            
        # Extract parameters
        params = {}
        for key in traj['parameters'].keys():
            if isinstance(traj['parameters'][key], h5py.Group):
                if 'data' in traj['parameters'][key]:
                    params[key] = traj['parameters'][key]['data'][()]
            
    return results, params

def plot_comparison_results(all_results, output_dir):
    """Create comparison plots for different rebalancing policies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # 1. Profit Comparison
    plt.figure(figsize=(12, 6))
    profits = [results['final_fortune'] - results['initial_fortune'] for results in all_results.values()]
    policies = list(all_results.keys())
    plt.bar(policies, profits)
    plt.title('Profit by Rebalancing Policy')
    plt.xlabel('Rebalancing Policy')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Success Rate Comparison
    plt.figure(figsize=(12, 6))
    success_rates = [results['success_rate_node_total'] for results in all_results.values()]
    plt.bar(policies, success_rates)
    plt.title('Success Rate by Rebalancing Policy')
    plt.xlabel('Rebalancing Policy')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Number of Rebalancing Operations
    plt.figure(figsize=(12, 6))
    rebalancing_ops = [np.sum(np.diff(results['balance_history_values'][0]) != 0) for results in all_results.values()]
    plt.bar(policies, rebalancing_ops)
    plt.title('Number of Rebalancing Operations by Policy')
    plt.xlabel('Rebalancing Policy')
    plt.ylabel('Number of Operations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'rebalancing_ops_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Transaction Success vs Failure
    plt.figure(figsize=(15, 6))
    x = np.arange(len(policies))
    width = 0.35
    
    success_counts = [results['success_count_node_total'] for results in all_results.values()]
    failure_counts = [results['failure_count_node_total'] for results in all_results.values()]
    
    plt.bar(x - width/2, success_counts, width, label='Success')
    plt.bar(x + width/2, failure_counts, width, label='Failure')
    plt.title('Transaction Success vs Failure by Policy')
    plt.xlabel('Rebalancing Policy')
    plt.ylabel('Number of Transactions')
    plt.xticks(x, policies, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'transaction_stats_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Channel Balance History Comparison
    plt.figure(figsize=(15, 8))
    for policy, results in all_results.items():
        times = results['balance_history_times']
        balance_values = results['balance_history_values']
        plt.plot(times, balance_values[0], label=f'{policy} - Channel L')
        plt.plot(times, balance_values[1], label=f'{policy} - Channel R', linestyle='--')
    plt.title('Channel Balance History by Policy')
    plt.xlabel('Time')
    plt.ylabel('Balance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'balance_history_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Channel Balance Ratio Over Time
    plt.figure(figsize=(15, 8))
    for policy, results in all_results.items():
        times = results['balance_history_times']
        balance_values = results['balance_history_values']
        balance_ratio = balance_values[0] / (balance_values[0] + balance_values[1])
        plt.plot(times, balance_ratio, label=policy)
    plt.title('Channel Balance Ratio Over Time')
    plt.xlabel('Time')
    plt.ylabel('Balance Ratio (Channel L / Total)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'balance_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Scalability Analysis
    plt.figure(figsize=(15, 8))
    for policy, results in all_results.items():
        times = results['balance_history_times']
        throughput = results['normalized_throughput_node_total']
        # Flatten throughput if needed
        if throughput.shape != times.shape:
            throughput = np.ravel(throughput)
            if throughput.shape[0] > times.shape[0]:
                throughput = throughput[:times.shape[0]]
            elif throughput.shape[0] < times.shape[0]:
                times = times[:throughput.shape[0]]
        plt.plot(times, throughput, label=policy)
    plt.title('Scalability Analysis - Throughput Over Time')
    plt.xlabel('Time')
    plt.ylabel('Normalized Throughput')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'scalability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Find all simulation result files
    result_files = glob.glob('../outputs/results/results_*.hdf5')
    
    # Load results for each policy
    all_results = {}
    for file_path in result_files:
        results, params = load_simulation_results(file_path)
        # Extract policy name from the file path
        policy = file_path.split('_')[-1].split('.')[0]  # Get the policy name from filename
        all_results[policy] = results
    
    # Create comparison plots
    plot_comparison_results(all_results, 'simulation_output')
    
    # Print summary for each policy
    print("\nSimulation Summary by Policy:")
    for policy, results in all_results.items():
        print(f"\n{policy}:")
        print(f"Total Transactions: {results['arrived_count_node_total']}")
        print(f"Success Rate: {results['success_rate_node_total']:.2%}")
        print(f"Initial Fortune: {results['initial_fortune']}")
        print(f"Final Fortune: {results['final_fortune']}")
        print(f"Profit: {results['final_fortune'] - results['initial_fortune']}")
        print(f"Number of Rebalancing Operations: {np.sum(np.diff(results['balance_history_values'][0]) != 0)}")

if __name__ == '__main__':
    main() 