import subprocess
import os

def run_simulation(policy, duration=100):
    """Run simulation for a specific policy"""
    print(f"\nRunning simulation for {policy} policy...")
    cmd = f"python simulation_driver_new.py --rebalancing_policy {policy} --simulation_duration {duration}"
    
    # Rename the output file to match the policy
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        old_file = "../outputs/results/results_test.hdf5"
        new_file = f"../outputs/results/results_{policy.lower()}.hdf5"
        if os.path.exists(old_file):
            os.rename(old_file, new_file)
            print(f"Successfully saved results to {new_file}")
    else:
        print(f"Error running simulation for {policy}:")
        print(result.stderr)

def main():
    # Create outputs/results directory if it doesn't exist
    os.makedirs("../outputs/results", exist_ok=True)
    
    # Run simulations for all policies
    policies = ['none', 'autoloop', 'loopmax', 'debal']
    for policy in policies:
        run_simulation(policy)

if __name__ == "__main__":
    main() 