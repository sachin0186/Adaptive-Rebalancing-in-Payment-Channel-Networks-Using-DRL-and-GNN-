import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_hdf5_results(file_paths):
    """Read results from multiple HDF5 files for different policies"""
    all_results = {}
    for policy, file_path in file_paths.items():
        with h5py.File(file_path, 'r') as f:
            base_path = 'relay_node_channel_rebalancing/results/runs/run_00000000/'
            results = {}
            
            # Read balance history
            results['balance_history_L'] = {
                'times': np.array(f[base_path + 'balance_history_times_L/balance_history_times_L']),
                'values': np.array(f[base_path + 'balance_history_values_L/balance_history_values_L']),
                'remote_values': np.array(f[base_path + 'remote_balance_history_values_L/remote_balance_history_values_L'])
            }
            
            results['balance_history_R'] = {
                'times': np.array(f[base_path + 'balance_history_times_R/balance_history_times_R']),
                'values': np.array(f[base_path + 'balance_history_values_R/balance_history_values_R']),
                'remote_values': np.array(f[base_path + 'remote_balance_history_values_R/remote_balance_history_values_R'])
            }

            # Read key metrics
            metrics = {
                'success_rate_L_to_R': f[base_path + 'success_rate_L_to_R/success_rate_L_to_R'][()],
                'success_rate_R_to_L': f[base_path + 'success_rate_R_to_L/success_rate_R_to_L'][()],
                'success_rate_node_total': f[base_path + 'success_rate_node_total/success_rate_node_total'][()],
                'success_count_L_to_R': f[base_path + 'success_count_L_to_R/success_count_L_to_R'][()],
                'success_count_R_to_L': f[base_path + 'success_count_R_to_L/success_count_R_to_L'][()],
                'failure_count_L_to_R': f[base_path + 'failure_count_L_to_R/failure_count_L_to_R'][()],
                'failure_count_R_to_L': f[base_path + 'failure_count_R_to_L/failure_count_R_to_L'][()],
                'success_amount_L_to_R': f[base_path + 'success_amount_L_to_R/success_amount_L_to_R'][()],
                'success_amount_R_to_L': f[base_path + 'success_amount_R_to_L/success_amount_R_to_L'][()],
                'failure_amount_L_to_R': f[base_path + 'failure_amount_L_to_R/failure_amount_L_to_R'][()],
                'failure_amount_R_to_L': f[base_path + 'failure_amount_R_to_L/failure_amount_R_to_L'][()]
            }
            results.update(metrics)
            all_results[policy] = results
    return all_results

def calculate_debal_imbalance(local_balance, remote_balance):
    """Calculate imbalance ratio according to DEBAL methodology"""
    total_balance = local_balance + remote_balance
    return np.abs(local_balance - remote_balance) / total_balance

def detect_rebalancing_events(times, values, threshold=20):
    """Detect significant balance changes that indicate rebalancing"""
    events = []
    changes = []
    for i in range(1, len(values)):
        change = values[i] - values[i-1]
        if abs(change) > threshold:
            events.append(times[i])
            changes.append(change)
    return events, changes

def plot_comparison_metrics(all_results):
    """Create comparative plots for different rebalancing policies"""
    plt.rcParams['figure.figsize'] = [20, 15]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    
    policies = list(all_results.keys())
    
    # Create figure with 2x2 subplots
    fig = plt.figure()
    
    # 1. Profit by Rebalancing Policy
    ax1 = plt.subplot(2, 2, 1)
    profits = []
    for policy in policies:
        results = all_results[policy]
        profit = (results['success_amount_L_to_R'] + results['success_amount_R_to_L'] -
                 results['failure_amount_L_to_R'] - results['failure_amount_R_to_L'])
        profits.append(profit)
    
    ax1.bar(policies, profits)
    ax1.set_title('Profit by Rebalancing Policy')
    ax1.set_ylabel('Profit')
    
    # 2. Success Rate by Rebalancing Policy
    ax2 = plt.subplot(2, 2, 2)
    success_rates = []
    for policy in policies:
        results = all_results[policy]
        success_rate = results['success_rate_node_total'] * 100  # Convert to percentage
        success_rates.append(success_rate)
    
    ax2.bar(policies, success_rates)
    ax2.set_title('Success Rate by Rebalancing Policy')
    ax2.set_ylabel('Success Rate (%)')
    
    # 3. Number of Rebalancing Operations by Policy
    ax3 = plt.subplot(2, 2, 3)
    rebalancing_counts = []
    for policy in policies:
        results = all_results[policy]
        events_L, _ = detect_rebalancing_events(results['balance_history_L']['times'],
                                              results['balance_history_L']['values'])
        events_R, _ = detect_rebalancing_events(results['balance_history_R']['times'],
                                              results['balance_history_R']['values'])
        rebalancing_counts.append(len(events_L) + len(events_R))
    
    ax3.bar(policies, rebalancing_counts)
    ax3.set_title('Number of Rebalancing Operations by Policy')
    ax3.set_ylabel('Count')
    
    # 4. Transaction Success vs Failure by Policy
    ax4 = plt.subplot(2, 2, 4)
    success_counts = []
    failure_counts = []
    for policy in policies:
        results = all_results[policy]
        success = (results['success_count_L_to_R'] + results['success_count_R_to_L'])
        failure = (results['failure_count_L_to_R'] + results['failure_count_R_to_L'])
        success_counts.append(success)
        failure_counts.append(failure)
    
    x = np.arange(len(policies))
    width = 0.35
    ax4.bar(x - width/2, success_counts, width, label='Success')
    ax4.bar(x + width/2, failure_counts, width, label='Failed')
    ax4.set_xticks(x)
    ax4.set_xticklabels(policies)
    ax4.set_title('Transaction Success vs Failure by Policy')
    ax4.set_ylabel('Count')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/results/policy_comparison.png', bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nPolicy Comparison Summary:")
    for policy in policies:
        results = all_results[policy]
        print(f"\n{policy}:")
        print(f"Success Rate: {results['success_rate_node_total']*100:.2f}%")
        print(f"Total Transactions: {results['success_count_L_to_R'] + results['success_count_R_to_L'] + results['failure_count_L_to_R'] + results['failure_count_R_to_L']:.0f}")
        print(f"Total Amount: {results['success_amount_L_to_R'] + results['success_amount_R_to_L']:.0f}")

if __name__ == '__main__':
    # Define paths for different policies
    file_paths = {
        'Autoloop': '../outputs/results/results_autoloop.hdf5',
        'Loopmax': '../outputs/results/results_loopmax.hdf5',
        'None': '../outputs/results/results_none.hdf5',
        'DEBAL': '../outputs/results/results_debal.hdf5'  # Using DEBAL instead of RebEL
    }
    
    all_results = read_hdf5_results(file_paths)
    plot_comparison_metrics(all_results) 