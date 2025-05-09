import pypet
from simulate_relay_node import *
import argparse

'''

def load_existing_pcn_data():
    """Load existing PCN data from CSV file."""
    data_file = Path('DATA') / 'pcn_network_data.csv'
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"PCN data file not found at {data_file}"
        )
    
    # Load the CSV file
    pcn_df = pd.read_csv(data_file)
    
    # Print basic statistics about the network
    print("\nLoaded PCN Network Statistics:")
    print(f"Total number of channels: {len(pcn_df)}")
    print(f"Average channel capacity: {pcn_df['capacity'].mean():.0f} satoshis")
    print(f"Total network capacity: {pcn_df['capacity'].sum()/100_000_000:.2f} BTC")
    
    return pcn_df




'''



def parse_args():
    parser = argparse.ArgumentParser(description='Run payment channel rebalancing simulation')
    parser.add_argument('--network_size', type=int, default=100, help='Size of the network')
    parser.add_argument('--simulation_duration', type=int, default=1000, help='Duration of the simulation')
    parser.add_argument('--rebalancing_policy', type=str, default='debal', choices=['none', 'autoloop', 'loopmax', 'debal'], help='Rebalancing policy to use')
    return parser.parse_args()

def pypet_wrapper(traj):
    node_parameters = {
        'initial_balance_L': traj.initial_balance_L,
        'initial_balance_R': traj.initial_balance_R,
        'capacity_L': traj.capacity_L,
        'capacity_R': traj.capacity_R,
        'proportional_fee': traj.proportional_fee,
        'base_fee': traj.base_fee,
        'server_swap_fee': traj.server_swap_fee,
        'miner_fee': traj.miner_fee,
        'on_chain_budget': traj.on_chain_budget
    }

    experiment_parameters = {
        'simulation_duration': traj.simulation_duration,
        'total_transactions_L_to_R': traj.total_transactions_L_to_R,
        'total_transactions_R_to_L': traj.total_transactions_R_to_L,
        'exp_mean_L_to_R': traj.exp_mean_L_to_R,
        'exp_mean_R_to_L': traj.exp_mean_R_to_L,
        'amount_distribution_L_to_R': traj.amount_distribution_L_to_R,
        'amount_distribution_R_to_L': traj.amount_distribution_R_to_L,
        'amount_distribution_parameters_L_to_R': traj.amount_distribution_parameters_L_to_R,
        'amount_distribution_parameters_R_to_L': traj.amount_distribution_parameters_R_to_L,
        'verbose': traj.verbose,
        'verbose_also_print_transactions': traj.verbose_also_print_transactions
    }

    rebalancing_parameters = {
        'rebalancing_policy': traj.rebalancing_policy,
        'check_interval': 5.0,  # Check every 5 minutes
        'theta': 0.35,  # Minimum balance ratio threshold
        'tau': 1.5,     # TTD threshold in hours
        'epsilon': 0.001  # Small constant to prevent division by zero
    }

    results = simulate_relay_node(node_parameters, experiment_parameters, rebalancing_parameters)

    # Add results to trajectory
    for key, value in results.items():
        traj.f_add_result(key, value, comment=key.replace('_', ' ').capitalize())


def main():
    args = parse_args()
    '''
    # Load existing PCN data
    pcn_df = load_existing_pcn_data()
    
    # Use the loaded data to set up simulation parameters
    avg_capacity = pcn_df['capacity'].mean()
    '''
    # SIMULATION PARAMETERS
    filename = 'results_test'

    # verbose = True
    verbose = False
    verbose_also_print_transactions = False
    # verbose_also_print_transactions = True
    num_of_experiments = 1

    base_fee = 0    # 4e-4
    # proportional_fee = 3e-5
    proportional_fee = 0.01
    on_chain_budget = 1000

    # Channel N-L
    initial_balance_L = 500
    capacity_L = 1000
    total_transactions_L_to_R = args.network_size * 600  # Scale with network size
    exp_mean_L_to_R = 10 / 1     # transactions per minute
    amount_distribution_L_to_R = "gaussian"
    amount_distribution_parameters_L_to_R = [100, 25, 20]

    # Channel N-R
    initial_balance_R = 500
    capacity_R = 1000
    total_transactions_R_to_L = args.network_size * 150  # Scale with network size
    exp_mean_R_to_L = 2.5 / 1     # transactions per minute
    amount_distribution_R_to_L = "gaussian"
    amount_distribution_parameters_R_to_L = [100, 25, 20]

    # REBALANCING
    server_swap_fee = 0.005      # fraction of swap amount

    # Node parameters
    rebalancing_policy = args.rebalancing_policy.capitalize()
    autoloop_lower_threshold = 0.3
    autoloop_upper_threshold = 0.7
    check_interval = 10     # minutes

    T_conf = 9.99     # minutes
    miner_fee = 2

    safety_margin_in_minutes_L = T_conf/5
    safety_margin_in_minutes_R = T_conf/5

    # Create the environment
    env = pypet.Environment(trajectory='relay_node_channel_rebalancing',
                            filename='../outputs/results/' + filename + '.hdf5',
                            log_folder='../outputs/logs/',
                            log_stdout=True,
                            overwrite_file=True)
    traj = env.traj

    # Encode parameters for pypet

    traj.f_add_parameter('initial_balance_L', initial_balance_L, comment='Initial balance of node N in channel with node L')
    traj.f_add_parameter('capacity_L', capacity_L, comment='Capacity of channel N-L')
    traj.f_add_parameter('total_transactions_L_to_R', total_transactions_L_to_R, comment='Total transactions from L to R')
    traj.f_add_parameter('exp_mean_L_to_R', exp_mean_L_to_R, comment='Rate of exponentially distributed arrivals from L to R')
    traj.f_add_parameter('amount_distribution_L_to_R', amount_distribution_L_to_R, comment='The distribution of the transaction amounts from L to R')
    traj.f_add_parameter('amount_distribution_parameters_L_to_R', amount_distribution_parameters_L_to_R, comment='Parameters of the distribution of the transaction amounts from L to R')

    traj.f_add_parameter('initial_balance_R', initial_balance_R, comment='Initial balance of node N in channel with node R')
    traj.f_add_parameter('capacity_R', capacity_R, comment='Capacity of channel N-R')
    traj.f_add_parameter('total_transactions_R_to_L', total_transactions_R_to_L, comment='Total transactions from R to L')
    traj.f_add_parameter('exp_mean_R_to_L', exp_mean_R_to_L, comment='Rate of exponentially distributed arrivals from R to L')
    traj.f_add_parameter('amount_distribution_R_to_L', amount_distribution_R_to_L, comment='The distribution of the transaction amounts from R to L')
    traj.f_add_parameter('amount_distribution_parameters_R_to_L', amount_distribution_parameters_R_to_L, comment='Parameters of the distribution of the transaction amounts from R to L')

    traj.f_add_parameter('base_fee', base_fee, comment='Base forwarding fee charged by node N')
    traj.f_add_parameter('proportional_fee', proportional_fee, comment='Proportional forwarding fee charged by node N')
    traj.f_add_parameter('on_chain_budget', on_chain_budget, comment='On-chain budget of node N')

    # traj.f_add_parameter('server_min_swap_amount', server_min_swap_amount, comment='Minimum amount the LSP allows for a swap')
    traj.f_add_parameter('server_swap_fee', server_swap_fee, comment='Percentage of swap amount the LSP charges as fees')
    traj.f_add_parameter('rebalancing_policy', rebalancing_policy, comment='Rebalancing policy')
    traj.f_add_parameter('autoloop_lower_threshold', autoloop_lower_threshold, comment='Balance percentage threshold below which the channel needs a swap-in according to the Autoloop policy')
    traj.f_add_parameter('autoloop_upper_threshold', autoloop_upper_threshold, comment='Balance percentage threshold above which the channel needs a swap-out according to the Autoloop policy')
    # traj.f_add_parameter('default_swap_amount', default_swap_amount, comment='Default swap amount node N requests')
    traj.f_add_parameter('check_interval', check_interval, comment='Time in seconds every which a check for rebalancing is performed')
    traj.f_add_parameter('T_conf', T_conf, comment='Confirmation time (seconds) for an on-chain transaction')
    traj.f_add_parameter('miner_fee', miner_fee, comment='Miner fee for an on-chain transaction')
    traj.f_add_parameter('safety_margin_in_minutes_L', safety_margin_in_minutes_L, comment='Safety margin in minutes for swaps under the Loopmax policy for node L')
    traj.f_add_parameter('safety_margin_in_minutes_R', safety_margin_in_minutes_R, comment='Safety margin in minutes for swaps under the Loopmax policy for node R')

    # traj.f_add_parameter('verbose', verbose, comment='Verbose output')
    traj.f_add_parameter('verbose', verbose, comment='Verbose output at rebalancing check times')
    traj.f_add_parameter('verbose_also_print_transactions', verbose_also_print_transactions, comment='Verbose output for all transactions apart from rebalancing check times')
    traj.f_add_parameter('filename', filename, comment='Filename of the results')
    traj.f_add_parameter('num_of_experiments', num_of_experiments, comment='Repetitions of every experiment')
    traj.f_add_parameter('seed', 0, comment='Randomness seed')
    traj.f_add_parameter('simulation_duration', args.simulation_duration, comment='Duration of the simulation')

    seeds = [63621, 87563, 24240, 14020, 84331, 60917, 48692, 73114, 90695, 62302, 52578, 43760, 84941, 30804, 40434, 63664, 25704, 38368, 45271, 34425]

    traj.f_explore(pypet.cartesian_product({
        'rebalancing_policy': [rebalancing_policy],
        'seed': seeds[0:traj.num_of_experiments]
    }))

    # Run wrapping function instead of simulator directly
    env.run(pypet_wrapper)
    env.disable_logging()


if __name__ == '__main__':
    main()
