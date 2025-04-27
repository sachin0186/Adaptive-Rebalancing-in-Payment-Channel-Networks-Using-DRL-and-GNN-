import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Open the HDF5 file - using a more reliable path approach
file_path = 'outputs/results/results_test.hdf5'
if not os.path.exists(file_path):
    file_path = '../outputs/results/results_test.hdf5'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find the results file in expected locations")

print(f"Reading results from: {os.path.abspath(file_path)}")
f = h5py.File(file_path, 'r')

# Print the overall structure 
print("\nTop level structure:")
for key in f.keys():
    print(f"  {key}")

# Access the runs from the correct location
runs_path = 'relay_node_channel_rebalancing/results/runs'
if runs_path in f:
    runs = list(f[runs_path].keys())
    print(f"\nFound {len(runs)} simulation runs: {runs}")

    # Collect key metrics for each rebalancing policy
    results_summary = []
    
    # Read simulation parameters
    params_path = 'relay_node_channel_rebalancing/parameters'
    
    # Collect parameters
    params = {}
    for param_name in ['initial_balance_L', 'initial_balance_R', 'capacity_L', 'capacity_R',
                       'on_chain_budget', 'proportional_fee', 'server_swap_fee', 'miner_fee',
                       'total_transactions_L_to_R', 'total_transactions_R_to_L',
                       'exp_mean_L_to_R', 'exp_mean_R_to_L']:
        param_dataset = f.get(f'{params_path}/{param_name}/data')
        if param_dataset is not None:
            params[param_name] = param_dataset[0][0]
    
    print("\nSimulation Parameters:")
    print(f"  Initial balance L: {params.get('initial_balance_L', 'N/A')}")
    print(f"  Initial balance R: {params.get('initial_balance_R', 'N/A')}")
    print(f"  Capacity L: {params.get('capacity_L', 'N/A')}")
    print(f"  Capacity R: {params.get('capacity_R', 'N/A')}")
    print(f"  On-chain budget: {params.get('on_chain_budget', 'N/A')}")
    print(f"  Proportional fee: {params.get('proportional_fee', 'N/A')}")
    print(f"  Server swap fee: {params.get('server_swap_fee', 'N/A')}")
    print(f"  Miner fee: {params.get('miner_fee', 'N/A')}")
    print(f"  Total transactions L→R: {params.get('total_transactions_L_to_R', 'N/A')}")
    print(f"  Total transactions R→L: {params.get('total_transactions_R_to_L', 'N/A')}")
    print(f"  Transaction rate L→R: {params.get('exp_mean_L_to_R', 'N/A')} per minute")
    print(f"  Transaction rate R→L: {params.get('exp_mean_R_to_L', 'N/A')} per minute")
    
    for run in runs:
        run_path = f'{runs_path}/{run}'
        
        # Extract run index
        run_idx = int(run.split('_')[-1])
        
        # Get rebalancing policy - look in the parameter exploration data
        rebalancing_policy = None
        policy_dataset = f.get('relay_node_channel_rebalancing/parameters/rebalancing_policy/explored_data')
        if policy_dataset is not None:
            if run_idx < len(policy_dataset):
                rebalancing_policy = policy_dataset[run_idx][0]
                if isinstance(rebalancing_policy, bytes):
                    rebalancing_policy = rebalancing_policy.decode('utf-8')
        
        if rebalancing_policy is None:
            rebalancing_policy = f"Policy {run_idx}"
        
        # Key metrics
        success_rate = f[f'{run_path}/success_rate_node_total/success_rate_node_total'][()]
        initial_fortune = f[f'{run_path}/initial_fortune/initial_fortune'][()]
        final_fortune = f[f'{run_path}/final_fortune_with_pending_swaps/final_fortune_with_pending_swaps'][()]
        profit = final_fortune - initial_fortune
        
        # Direction-specific success rates
        success_rate_L_to_R = f[f'{run_path}/success_rate_L_to_R/success_rate_L_to_R'][()]
        success_rate_R_to_L = f[f'{run_path}/success_rate_R_to_L/success_rate_R_to_L'][()]
        
        # Count rebalancing operations
        rebalancing_count = 0
        if f'{run_path}/rebalancing_history_types' in f:
            rebalancing_types_dataset = f[f'{run_path}/rebalancing_history_types/rebalancing_history_types']
            rebalancing_count = rebalancing_types_dataset.shape[0] if rebalancing_types_dataset.shape else 0
        
        # Transactions stats
        success_count = f[f'{run_path}/success_count_node_total/success_count_node_total'][()]
        failure_count = f[f'{run_path}/failure_count_node_total/failure_count_node_total'][()]
        total_count = success_count + failure_count
        
        # Direction-specific counts
        success_count_L_to_R = f[f'{run_path}/success_count_L_to_R/success_count_L_to_R'][()]
        success_count_R_to_L = f[f'{run_path}/success_count_R_to_L/success_count_R_to_L'][()]
        failure_count_L_to_R = f[f'{run_path}/failure_count_L_to_R/failure_count_L_to_R'][()]
        failure_count_R_to_L = f[f'{run_path}/failure_count_R_to_L/failure_count_R_to_L'][()]
        
        # Fee-related metrics
        cumulative_fee_losses = f[f'{run_path}/cumulative_fee_losses/cumulative_fee_losses'][()]
        cumulative_rebalancing_fees = f[f'{run_path}/cumulative_rebalancing_fees/cumulative_rebalancing_fees'][()]
        
        # Collect metrics
        results_summary.append({
            'Run': run,
            'Rebalancing Policy': rebalancing_policy,
            'Initial Fortune': initial_fortune,
            'Final Fortune': final_fortune,
            'Profit': profit,
            'Profit %': (profit / initial_fortune) * 100,
            'Success Rate %': success_rate * 100,
            'Success Rate L→R %': success_rate_L_to_R * 100,
            'Success Rate R→L %': success_rate_R_to_L * 100,
            'Rebalancing Operations': rebalancing_count,
            'Successful Transactions': success_count,
            'Failed Transactions': failure_count,
            'Total Transactions': total_count,
            'Success Count L→R': success_count_L_to_R,
            'Success Count R→L': success_count_R_to_L,
            'Failure Count L→R': failure_count_L_to_R,
            'Failure Count R→L': failure_count_R_to_L,
            'Fee Losses': cumulative_fee_losses,
            'Rebalancing Fees': cumulative_rebalancing_fees,
        })

    # Convert to DataFrame for better viewing
    df = pd.DataFrame(results_summary)
    df = df.sort_values(by='Rebalancing Policy')
    
    # Print a more readable table with the most important metrics
    print("\n=== RESULTS SUMMARY ===")
    summary_df = df[['Rebalancing Policy', 'Initial Fortune', 'Final Fortune', 'Profit', 'Profit %', 
                   'Success Rate %', 'Rebalancing Operations', 'Successful Transactions', 'Failed Transactions']]
    
    # Format the numeric columns
    formatted_df = summary_df.copy()
    formatted_df['Profit'] = formatted_df['Profit'].map(lambda x: f"{x:.2f}")
    formatted_df['Profit %'] = formatted_df['Profit %'].map(lambda x: f"{x:.2f}%")
    formatted_df['Success Rate %'] = formatted_df['Success Rate %'].map(lambda x: f"{x:.2f}%")
    
    print(formatted_df.to_string(index=False))
    
    # Print detailed metrics for each policy
    print("\n=== DETAILED METRICS ===")
    for i, row in df.iterrows():
        policy = row['Rebalancing Policy']
        print(f"\n{policy} Policy:")
        print(f"  Profit: {row['Profit']:.2f} ({row['Profit %']:.2f}% increase from initial)")
        print(f"  Transactions:")
        print(f"    Success rate: {row['Success Rate %']:.2f}%")
        print(f"    Direction-specific success rates:")
        print(f"      L→R: {row['Success Rate L→R %']:.2f}%")
        print(f"      R→L: {row['Success Rate R→L %']:.2f}%")
        print(f"    Total transactions: {row['Total Transactions']}")
        print(f"    Successful transactions: {row['Successful Transactions']} ({row['Successful Transactions']/row['Total Transactions']*100:.2f}%)")
        print(f"    Failed transactions: {row['Failed Transactions']} ({row['Failed Transactions']/row['Total Transactions']*100:.2f}%)")
        print(f"    Direction-specific counts:")
        print(f"      L→R Success: {row['Success Count L→R']} | Failure: {row['Failure Count L→R']}")
        print(f"      R→L Success: {row['Success Count R→L']} | Failure: {row['Failure Count R→L']}")
        print(f"  Rebalancing:")
        print(f"    Total operations: {row['Rebalancing Operations']}")
        print(f"    Rebalancing fees: {row['Rebalancing Fees']:.2f}")
        print(f"  Fees:")
        print(f"    Fee losses from failed transactions: {row['Fee Losses']:.2f}")
    
    # Baseline comparison
    baseline_row = df[df['Rebalancing Policy'] == 'None'].iloc[0] if 'None' in df['Rebalancing Policy'].values else None
    
    if baseline_row is not None:
        baseline_profit = baseline_row['Profit']
        print("\n=== COMPARISON TO BASELINE (No Rebalancing) ===")
        for i, row in df.iterrows():
            if row['Rebalancing Policy'] != 'None':
                profit_improvement = row['Profit'] - baseline_profit
                profit_improvement_pct = (profit_improvement / baseline_profit) * 100
                print(f"{row['Rebalancing Policy']}: +{profit_improvement:.2f} ({profit_improvement_pct:.2f}% improvement)")
    
    # Calculate efficiency metrics
    print("\n=== EFFICIENCY METRICS ===")
    for i, row in df.iterrows():
        if row['Rebalancing Operations'] > 0:
            profit_per_op = row['Profit'] / row['Rebalancing Operations']
            success_per_op = row['Successful Transactions'] / row['Rebalancing Operations']
            print(f"{row['Rebalancing Policy']}:")
            print(f"  Profit per rebalancing operation: {profit_per_op:.2f}")
            print(f"  Successful transactions per rebalancing operation: {success_per_op:.2f}")
    
    print("\nResults Summary:")
    print(df)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot 1: Profit by policy
    plt.subplot(2, 2, 1)
    plt.bar(df['Rebalancing Policy'], df['Profit'])
    plt.title('Profit by Rebalancing Policy')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)

    # Plot 2: Success rate by policy
    plt.subplot(2, 2, 2)
    plt.bar(df['Rebalancing Policy'], df['Success Rate %'])
    plt.title('Success Rate by Rebalancing Policy')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)

    # Plot 3: Rebalancing operations by policy
    plt.subplot(2, 2, 3)
    plt.bar(df['Rebalancing Policy'], df['Rebalancing Operations'])
    plt.title('Number of Rebalancing Operations by Policy')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Plot 4: Transaction success vs failure by policy
    plt.subplot(2, 2, 4)
    width = 0.35
    x = np.arange(len(df['Rebalancing Policy']))
    plt.bar(x - width/2, df['Successful Transactions'], width, label='Success')
    plt.bar(x + width/2, df['Failed Transactions'], width, label='Failed')
    plt.title('Transaction Success vs Failure by Policy')
    plt.xlabel('Rebalancing Policy')
    plt.ylabel('Count')
    plt.xticks(x, df['Rebalancing Policy'], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig('outputs/results/simulation_results_comparison.png')
    print("\nPlot saved to 'outputs/results/simulation_results_comparison.png'")

    # For each policy, plot the balance history
    for run in runs:
        run_path = f'{runs_path}/{run}'
        
        # Extract run index
        run_idx = int(run.split('_')[-1])
        
        # Get rebalancing policy - look in the parameter exploration data
        rebalancing_policy = None
        policy_dataset = f.get('relay_node_channel_rebalancing/parameters/rebalancing_policy/explored_data')
        if policy_dataset is not None:
            if run_idx < len(policy_dataset):
                rebalancing_policy = policy_dataset[run_idx][0]
                if isinstance(rebalancing_policy, bytes):
                    rebalancing_policy = rebalancing_policy.decode('utf-8')
        
        if rebalancing_policy is None:
            rebalancing_policy = f"Policy {run_idx}"
            
        # Get the balance history data
        balance_history_times = f[f'{run_path}/balance_history_times/balance_history_times'][:]
        balance_history_L = f[f'{run_path}/balance_history_values_L/balance_history_values_L'][:]
        balance_history_R = f[f'{run_path}/balance_history_values_R/balance_history_values_R'][:]
        
        # Check if we have rebalancing history
        has_rebalancing = f'{run_path}/rebalancing_history_start_times' in f
        
        # Plot the balance history for this policy
        plt.figure(figsize=(12, 6))
        plt.plot(balance_history_times, balance_history_L, label='Channel N-L Balance')
        plt.plot(balance_history_times, balance_history_R, label='Channel N-R Balance')
        
        # Mark rebalancing operations if available
        if has_rebalancing:
            rebalancing_start_times = f[f'{run_path}/rebalancing_history_start_times/rebalancing_history_start_times'][:]
            rebalancing_types = f[f'{run_path}/rebalancing_history_types/rebalancing_history_types'][:]
            
            if len(rebalancing_start_times) > 0:
                for i, t in enumerate(rebalancing_start_times):
                    try:
                        rb_type = rebalancing_types[i]
                        if isinstance(rb_type, bytes):
                            rb_type = rb_type.decode('utf-8')
                        
                        if isinstance(rb_type, str):
                            if 'swap_IN' in rb_type and 'L' in rb_type:
                                plt.axvline(x=t, color='g', linestyle='--', alpha=0.5)
                            elif 'swap_OUT' in rb_type and 'L' in rb_type:
                                plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
                            elif 'swap_IN' in rb_type and 'R' in rb_type:
                                plt.axvline(x=t, color='g', linestyle=':', alpha=0.5)
                            elif 'swap_OUT' in rb_type and 'R' in rb_type:
                                plt.axvline(x=t, color='r', linestyle=':', alpha=0.5)
                    except:
                        # Skip if there's an issue with a particular rebalancing operation
                        pass
        
        plt.title(f'Channel Balance History - {rebalancing_policy} Policy')
        plt.xlabel('Simulation Time')
        plt.ylabel('Balance')
        plt.legend()
        plt.savefig(f'outputs/results/balance_history_{rebalancing_policy}.png')
        print(f"\nBalance history plot saved for {rebalancing_policy} policy")

    # Calculate and plot the channel imbalance for each policy
    plt.figure(figsize=(12, 6))
    
    for i, run in enumerate(runs):
        run_path = f'{runs_path}/{run}'
        
        # Extract run index
        run_idx = int(run.split('_')[-1])
        
        # Get rebalancing policy
        rebalancing_policy = None
        policy_dataset = f.get('relay_node_channel_rebalancing/parameters/rebalancing_policy/explored_data')
        if policy_dataset is not None:
            if run_idx < len(policy_dataset):
                rebalancing_policy = policy_dataset[run_idx][0]
                if isinstance(rebalancing_policy, bytes):
                    rebalancing_policy = rebalancing_policy.decode('utf-8')
        
        if rebalancing_policy is None:
            rebalancing_policy = f"Policy {run_idx}"
        
        # Get balance history
        balance_history_times = f[f'{run_path}/balance_history_times/balance_history_times'][:]
        balance_history_L = f[f'{run_path}/balance_history_values_L/balance_history_values_L'][:]
        balance_history_R = f[f'{run_path}/balance_history_values_R/balance_history_values_R'][:]
        
        # Calculate channel balance ratio (L/(L+R))
        total_balance = balance_history_L + balance_history_R
        balance_ratio = balance_history_L / total_balance
        
        # Plot channel balance ratio over time
        plt.plot(balance_history_times, balance_ratio, label=rebalancing_policy)
    
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Perfect Balance (50/50)')
    plt.title('Channel Balance Ratio (L/(L+R)) Over Time')
    plt.xlabel('Simulation Time')
    plt.ylabel('Balance Ratio')
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig('outputs/results/channel_balance_ratio.png')
    print("\nChannel balance ratio plot saved")

else:
    print(f"Error: Could not find runs at path '{runs_path}' in the HDF5 file.")

# Close the file
f.close()

print("\nAnalysis complete!") 