from numpy import random, round
import simpy
import sys
import pandas as pd
import numpy as np

from entities.node import Node
from entities.transaction import Transaction


def transaction_generator(env, topology, source, destination, total_transactions, exp_mean, amount_distribution, amount_distribution_parameters, all_transactions_list, verbose, verbose_also_print_transactions):
    time_to_next_arrival = random.exponential(1.0 / exp_mean)
    yield env.timeout(time_to_next_arrival)

    for _ in range(total_transactions):
        if amount_distribution == "constant":
            amount = amount_distribution_parameters[0]
        elif amount_distribution == "uniform":
            max_transaction_amount = amount_distribution_parameters[0]
            amount = random.randint(1, max_transaction_amount)
            # amount = random.uniform(0.0, float(max_transaction_amount))
        elif amount_distribution == "gaussian":
            max_transaction_amount = amount_distribution_parameters[0]
            gaussian_mean = amount_distribution_parameters[1]
            gaussian_variance = amount_distribution_parameters[2]
            amount = round(max(1, min(max_transaction_amount, random.normal(gaussian_mean, gaussian_variance))))
            # amount = max(0.00001, min(float(max_transaction_amount), random.normal(gaussian_mean, gaussian_variance)))
        # elif amount_distribution == "pareto":
        #     lower = amount_distribution_parameters[0]  # the lower end of the support
        #     shape = amount_distribution_parameters[1]  # the distribution shape parameter, also known as `a` or `alpha`
        #     size = amount_distribution_parameters[2]  # the size of your sample (number of random values)
        #     amount = random.pareto(shape, size) + lower
        # elif amount_distribution == "powerlaw":
        #     powerlaw.Power_Law(xmin=1, xmax=2, discrete=True, parameters=[1.16]).generate_random(n=10)
        # elif amount_distribution == "empirical_from_csv_file":
        #     dataset = amount_distribution_parameters[0]
        #     data_size = amount_distribution_parameters[1]
        #     amount = dataset[random.randint(0, data_size)]
        else:
            print("Input error: {} is not a supported amount distribution or the parameters {} given are invalid.".format(amount_distribution, amount_distribution_parameters))
            sys.exit(1)

        t = Transaction(env, topology, env.now, source, destination, amount, verbose, verbose_also_print_transactions)
        all_transactions_list.append(t)
        env.process(t.run())

        time_to_next_arrival = random.exponential(1.0 / exp_mean)
        yield env.timeout(time_to_next_arrival)


def simulate_relay_node(node_parameters, experiment_parameters, rebalancing_parameters):
    # Create the node
    N = Node("N")  # Initialize with node ID
    
    # Set up channels
    N.add_channel("L", node_parameters["initial_balance_L"], node_parameters["capacity_L"] - node_parameters["initial_balance_L"], node_parameters["capacity_L"])
    N.add_channel("R", node_parameters["initial_balance_R"], node_parameters["capacity_R"] - node_parameters["initial_balance_R"], node_parameters["capacity_R"])
    
    # Set fee rates
    N.fee_rates["L"] = node_parameters["proportional_fee"]
    N.fee_rates["R"] = node_parameters["proportional_fee"]

    # Create environment and start simulation
    env = simpy.Environment()
    N.env = env  # Add environment reference to node for time tracking
    topology = {"N": N}

    all_transactions_list = []
    
    # Add rebalancing check process for DEBAL
    if rebalancing_parameters["rebalancing_policy"].lower() == "debal":
        def rebalancing_check():
            while env.now < experiment_parameters["simulation_duration"]:
                yield env.timeout(rebalancing_parameters["check_interval"])
                if N.decide_rebalancing():
                    # Perform rebalancing based on DEBAL's decision
                    if experiment_parameters["verbose"]:
                        print(f"Time {env.now}: DEBAL decided to rebalance")
                        print(f"Current state: L={N.local_balances['L']}/{N.remote_balances['L']}, R={N.local_balances['R']}/{N.remote_balances['R']}")
        
        env.process(rebalancing_check())

    # Add transaction generators
    env.process(transaction_generator(
        env, topology, "L", "R",
        experiment_parameters["total_transactions_L_to_R"],
        experiment_parameters["exp_mean_L_to_R"],
        experiment_parameters["amount_distribution_L_to_R"],
        experiment_parameters["amount_distribution_parameters_L_to_R"],
        all_transactions_list,
        experiment_parameters["verbose"],
        experiment_parameters["verbose_also_print_transactions"]
    ))
    
    env.process(transaction_generator(
        env, topology, "R", "L",
        experiment_parameters["total_transactions_R_to_L"],
        experiment_parameters["exp_mean_R_to_L"],
        experiment_parameters["amount_distribution_R_to_L"],
        experiment_parameters["amount_distribution_parameters_R_to_L"],
        all_transactions_list,
        experiment_parameters["verbose"],
        experiment_parameters["verbose_also_print_transactions"]
    ))

    # Run simulation
    env.run(until=experiment_parameters["simulation_duration"])

    # Calculate results
    measurement_interval = [0, env.now]

    success_count_L_to_R = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "SUCCEEDED")))
    success_count_R_to_L = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "SUCCEEDED")))
    success_count_node_total = success_count_L_to_R + success_count_R_to_L

    failure_count_L_to_R = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "FAILED")))
    failure_count_R_to_L = sum(1 for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "FAILED")))
    failure_count_node_total = failure_count_L_to_R + failure_count_R_to_L

    arrived_count_L_to_R = success_count_L_to_R + failure_count_L_to_R
    arrived_count_R_to_L = success_count_R_to_L + failure_count_R_to_L
    arrived_count_node_total = arrived_count_L_to_R + arrived_count_R_to_L

    success_amount_L_to_R = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "SUCCEEDED")))
    success_amount_R_to_L = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "SUCCEEDED")))
    success_amount_node_total = success_amount_L_to_R + success_amount_R_to_L

    failure_amount_L_to_R = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "L") and (t.destination == "R") and (t.status == "FAILED")))
    failure_amount_R_to_L = sum(t.amount for t in all_transactions_list if ((t.time_of_arrival >= measurement_interval[0]) and (t.time_of_arrival < measurement_interval[1]) and (t.source == "R") and (t.destination == "L") and (t.status == "FAILED")))
    failure_amount_node_total = failure_amount_L_to_R + failure_amount_R_to_L

    arrived_amount_L_to_R = success_amount_L_to_R + failure_amount_L_to_R
    arrived_amount_R_to_L = success_amount_R_to_L + failure_amount_R_to_L
    arrived_amount_node_total = arrived_amount_L_to_R + arrived_amount_R_to_L

    success_rate_L_to_R = success_count_L_to_R / arrived_count_L_to_R if arrived_count_L_to_R > 0 else 0
    success_rate_R_to_L = success_count_R_to_L / arrived_count_R_to_L if arrived_count_R_to_L > 0 else 0
    success_rate_node_total = success_count_node_total / arrived_count_node_total if arrived_count_node_total > 0 else 0

    normalized_throughput_L_to_R = success_amount_L_to_R / arrived_amount_L_to_R if arrived_amount_L_to_R > 0 else 0
    normalized_throughput_R_to_L = success_amount_R_to_L / arrived_amount_R_to_L if arrived_amount_R_to_L > 0 else 0
    normalized_throughput_node_total = success_amount_node_total / arrived_amount_node_total if arrived_amount_node_total > 0 else 0

    initial_fortune = node_parameters["initial_balance_L"] + node_parameters["initial_balance_R"]
    final_fortune = N.get_total_outgoing_liquidity()

    # Extract balance history
    balance_history = N.get_balance_history()
    # Flatten times and values for plotting compatibility
    balance_history_times_L = np.array([t for t, _, _ in balance_history["L"]])
    balance_history_times_R = np.array([t for t, _, _ in balance_history["R"]])
    balance_history_values_L = np.array([local for _, local, _ in balance_history["L"]])
    balance_history_values_R = np.array([local for _, local, _ in balance_history["R"]])
    remote_balance_history_values_L = np.array([remote for _, _, remote in balance_history["L"]])
    remote_balance_history_values_R = np.array([remote for _, _, remote in balance_history["R"]])

    # Synchronize time points for both channels
    if len(balance_history_times_L) == len(balance_history_times_R) and np.allclose(balance_history_times_L, balance_history_times_R):
        total_fortune_times = balance_history_times_L
        total_fortune_values = balance_history_values_L + balance_history_values_R
    else:
        # Interpolate the shorter to the longer one's time base
        if len(balance_history_times_L) > len(balance_history_times_R):
            interp_R = np.interp(balance_history_times_L, balance_history_times_R, balance_history_values_R)
            total_fortune_times = balance_history_times_L
            total_fortune_values = balance_history_values_L + interp_R
        else:
            interp_L = np.interp(balance_history_times_R, balance_history_times_L, balance_history_values_L)
            total_fortune_times = balance_history_times_R
            total_fortune_values = interp_L + balance_history_values_R

    # Synchronize balance and remote balance values to the same time base as total_fortune_times
    if len(balance_history_times_L) == len(balance_history_times_R) and np.allclose(balance_history_times_L, balance_history_times_R):
        sync_balance_values_L = balance_history_values_L
        sync_balance_values_R = balance_history_values_R
        sync_remote_values_L = remote_balance_history_values_L
        sync_remote_values_R = remote_balance_history_values_R
    elif len(balance_history_times_L) > len(balance_history_times_R):
        sync_balance_values_L = balance_history_values_L
        sync_balance_values_R = np.interp(balance_history_times_L, balance_history_times_R, balance_history_values_R)
        sync_remote_values_L = remote_balance_history_values_L
        sync_remote_values_R = np.interp(balance_history_times_L, balance_history_times_R, remote_balance_history_values_R)
    else:
        sync_balance_values_L = np.interp(balance_history_times_R, balance_history_times_L, balance_history_values_L)
        sync_balance_values_R = balance_history_values_R
        sync_remote_values_L = np.interp(balance_history_times_R, balance_history_times_L, remote_balance_history_values_L)
        sync_remote_values_R = remote_balance_history_values_R

    results = {
        'measurement_interval_length': measurement_interval[1] - measurement_interval[0],
        'success_count_L_to_R': success_count_L_to_R,
        'success_count_R_to_L': success_count_R_to_L,
        'success_count_node_total': success_count_node_total,
        'failure_count_L_to_R': failure_count_L_to_R,
        'failure_count_R_to_L': failure_count_R_to_L,
        'failure_count_node_total': failure_count_node_total,
        'arrived_count_L_to_R': arrived_count_L_to_R,
        'arrived_count_R_to_L': arrived_count_R_to_L,
        'arrived_count_node_total': arrived_count_node_total,
        'success_amount_L_to_R': success_amount_L_to_R,
        'success_amount_R_to_L': success_amount_R_to_L,
        'success_amount_node_total': success_amount_node_total,
        'failure_amount_L_to_R': failure_amount_L_to_R,
        'failure_amount_R_to_L': failure_amount_R_to_L,
        'failure_amount_node_total': failure_amount_node_total,
        'arrived_amount_L_to_R': arrived_amount_L_to_R,
        'arrived_amount_R_to_L': arrived_amount_R_to_L,
        'arrived_amount_node_total': arrived_amount_node_total,
        'success_rate_L_to_R': success_rate_L_to_R,
        'success_rate_R_to_L': success_rate_R_to_L,
        'success_rate_node_total': success_rate_node_total,
        'normalized_throughput_L_to_R': normalized_throughput_L_to_R,
        'normalized_throughput_R_to_L': normalized_throughput_R_to_L,
        'normalized_throughput_node_total': normalized_throughput_node_total,
        'initial_fortune': initial_fortune,
        'final_fortune': final_fortune,
        'balance_history_times': total_fortune_times,  # Use synchronized array
        'balance_history_values': np.stack([sync_balance_values_L, sync_balance_values_R]),
        'remote_balance_history_values': np.stack([sync_remote_values_L, sync_remote_values_R]),
        'total_fortune_including_pending_swaps_times': total_fortune_times,
        'total_fortune_including_pending_swaps_values': total_fortune_values
    }

    return results

# if __name__ == '__main__':
#     simulate_relay_node()
