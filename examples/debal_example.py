"""
Example script demonstrating DEBAL system usage.
"""

import simpy
import networkx as nx
from src.entities.node import Node
from src.entities.debal_integration import DEBALManager

def create_test_network():
    """Create a simple test network with 4 nodes."""
    # Create nodes
    nodes = [Node(f"node_{i}") for i in range(4)]
    
    # Create network graph
    G = nx.Graph()
    for i in range(4):
        G.add_node(f"node_{i}")
    
    # Add edges with capacities
    G.add_edge("node_0", "node_1", capacity=1000)
    G.add_edge("node_1", "node_2", capacity=1000)
    G.add_edge("node_2", "node_3", capacity=1000)
    G.add_edge("node_0", "node_2", capacity=1000)
    
    # Set up channels with more imbalanced initial state
    # Node 0 -> Node 1
    nodes[0].add_channel("node_1", 800, 200, 1000)  # More imbalanced
    nodes[1].add_channel("node_0", 200, 800, 1000)
    
    # Node 1 -> Node 2
    nodes[1].add_channel("node_2", 500, 500, 1000)
    nodes[2].add_channel("node_1", 500, 500, 1000)
    
    # Node 2 -> Node 3
    nodes[2].add_channel("node_3", 300, 700, 1000)  # More imbalanced
    nodes[3].add_channel("node_2", 700, 300, 1000)
    
    # Node 0 -> Node 2
    nodes[0].add_channel("node_2", 900, 100, 1000)  # More imbalanced
    nodes[2].add_channel("node_0", 100, 900, 1000)
    
    return nodes, G

def main():
    """Run the DEBAL system with a test network."""
    # Create environment
    env = simpy.Environment()
    
    # Create test network
    nodes, network_graph = create_test_network()
    
    # Initialize DEBAL manager
    debal_manager = DEBALManager(
        env=env,
        nodes=nodes,
        network_graph=network_graph,
        kappa=0.5,  # Minimum outgoing balance threshold
        theta=0.2,  # Minimum balance ratio threshold
        tau=2.0,    # TTD threshold in hours
        delta_t=600.0  # Re-election interval
    )
    
    # Start the system
    debal_manager.start()
    
    # Request rebalancing for multiple nodes
    nodes[0].request_rebalancing()
    nodes[2].request_rebalancing()
    
    # Run simulation for 2 hours
    env.run(until=7200)
    
    # Print final network state
    state = debal_manager.get_network_state()
    print("\nFinal Network State:")
    print(f"Leader: {state['leader_id']}")
    print(f"Election Time: {state['election_time']}")
    print(f"Pending Requests: {state['pending_requests']}")
    
    print("\nNode States:")
    for node_id, node_state in state['node_states'].items():
        print(f"\nNode {node_id}:")
        print(f"  Local Balances: {node_state['local_balances']}")
        print(f"  Remote Balances: {node_state['remote_balances']}")
        print(f"  Rebalancing Requested: {node_state['rebalancing_requested']}")

if __name__ == "__main__":
    main() 