"""
Main script to demonstrate the DEBAL system.
"""

import time
import sys
from src.entities.Node import Node
from src.entities.leader_election import LeaderElection
from src.entities.rebalancing_engine import RebalancingEngine
from src.entities.rebalancing_scheduler import RebalancingScheduler

def main():
    print("Starting DEBAL system...")
    
    # Create nodes
    nodes = [Node(i) for i in range(5)]
    print(f"Created {len(nodes)} nodes")
    
    # Create channels between nodes
    # Node 0
    nodes[0].add_channel(1, 1000, 800, 200, 0.001)  # Imbalanced
    nodes[0].add_channel(2, 1000, 500, 500, 0.001)  # Balanced
    
    # Node 1
    nodes[1].add_channel(0, 1000, 200, 800, 0.001)  # Imbalanced
    nodes[1].add_channel(2, 1000, 500, 500, 0.001)  # Balanced
    nodes[1].add_channel(3, 1000, 300, 700, 0.001)  # Imbalanced
    
    # Node 2
    nodes[2].add_channel(0, 1000, 500, 500, 0.001)  # Balanced
    nodes[2].add_channel(1, 1000, 500, 500, 0.001)  # Balanced
    nodes[2].add_channel(4, 1000, 400, 600, 0.001)  # Imbalanced
    
    # Node 3
    nodes[3].add_channel(1, 1000, 700, 300, 0.001)  # Imbalanced
    nodes[3].add_channel(4, 1000, 500, 500, 0.001)  # Balanced
    
    # Node 4
    nodes[4].add_channel(2, 1000, 600, 400, 0.001)  # Imbalanced
    nodes[4].add_channel(3, 1000, 500, 500, 0.001)  # Balanced
    
    print("Created channels between nodes")
    
    # Initialize components with more lenient parameters
    leader_election = LeaderElection(kappa=0.0, theta=0.1)  # Lower thresholds
    rebalancing_engine = RebalancingEngine(sigma=0.9)  # Higher skewness tolerance
    scheduler = RebalancingScheduler(
        nodes=nodes,
        leader_election=leader_election,
        rebalancing_engine=rebalancing_engine,
        interval=5.0  # Check every 5 seconds
    )
    
    print("Initialized components")
    
    # Start the scheduler
    scheduler.start()
    print("Started scheduler")
    
    try:
        # Let nodes request rebalancing
        for node in nodes:
            if node.decide_rebalancing():
                node.rebalancing_requested = True
                node.needs_rebalancing = True
                print(f"Node {node.id} requested rebalancing")
        
        # Run for 30 seconds
        print("Running DEBAL system for 30 seconds...")
        for i in range(6):  # Check every 5 seconds
            print(f"\nTime elapsed: {i*5} seconds")
            time.sleep(5)
            
            # Print current state
            print("\nCurrent state:")
            leader = leader_election.current_leader
            print(f"Current leader: {leader.id if leader else 'None'}")
            
            for node in nodes:
                print(f"\nNode {node.id}:")
                print(f"  Rebalancing requested: {node.rebalancing_requested}")
                print(f"  Needs rebalancing: {node.needs_rebalancing}")
                for neighbor_id in node.local_balances:
                    print(f"  Channel with {neighbor_id}:")
                    print(f"    Local balance: {node.local_balances[neighbor_id]}")
                    print(f"    Remote balance: {node.remote_balances[neighbor_id]}")
                    print(f"    Capacity: {node.capacities[neighbor_id]}")
                    print(f"    Fee rate: {node.fee_rates[neighbor_id]}")
        
    except KeyboardInterrupt:
        print("\nStopping DEBAL system...")
    finally:
        # Stop the scheduler
        scheduler.stop()
        print("Stopped scheduler")
        
        # Print final state
        print("\nFinal state:")
        for node in nodes:
            print(f"\nNode {node.id}:")
            for neighbor_id in node.local_balances:
                print(f"  Channel with {neighbor_id}:")
                print(f"    Local balance: {node.local_balances[neighbor_id]}")
                print(f"    Remote balance: {node.remote_balances[neighbor_id]}")
                print(f"    Capacity: {node.capacities[neighbor_id]}")
                print(f"    Fee rate: {node.fee_rates[neighbor_id]}")

if __name__ == "__main__":
    main() 