"""
Example script to run the DEBAL system with detailed logging.
"""

import simpy
import networkx as nx
from src.entities.node import Node
from src.entities.debal_integration import DEBALManager

def create_test_network():
    """Create a test network with 4 nodes and specified channel balances."""
    # Create nodes
    nodes = [Node(f"node_{i}") for i in range(4)]
    
    # Create network graph
    G = nx.Graph()
    for i in range(4):
        G.add_node(f"node_{i}")
    
    # Add edges with capacities
    G.add_edge("node_0", "node_1", capacity=100)  # A->B
    G.add_edge("node_1", "node_2", capacity=200)  # B->C
    G.add_edge("node_2", "node_3", capacity=200)  # C->D
    G.add_edge("node_3", "node_0", capacity=100)  # D->A
    G.add_edge("node_1", "node_3", capacity=200)  # B->D (50 + 150)
    
    # Set up channels with specified initial balances
    # A->B
    nodes[0].add_channel("node_1", 100, 0, 100)  # (lu→v, ru→v, Cc)
    nodes[1].add_channel("node_0", 0, 100, 100)
    
    # B->C
    nodes[1].add_channel("node_2", 200, 0, 200)
    nodes[2].add_channel("node_1", 0, 200, 200)
    
    # C->D
    nodes[2].add_channel("node_3", 200, 0, 200)
    nodes[3].add_channel("node_2", 0, 200, 200)
    
    # D->A
    nodes[3].add_channel("node_0", 100, 0, 100)
    nodes[0].add_channel("node_3", 0, 100, 100)
    
    # B->D
    nodes[1].add_channel("node_3", 50, 150, 200)
    nodes[3].add_channel("node_1", 150, 50, 200)
    
    # Set node traffic rates (units/hour)
    nodes[0].traffic_in = 50   # A
    nodes[0].traffic_out = 150
    nodes[1].traffic_in = 100  # B
    nodes[1].traffic_out = 50
    nodes[2].traffic_in = 200  # C
    nodes[2].traffic_out = 50
    nodes[3].traffic_in = 100  # D
    nodes[3].traffic_out = 200
    
    return nodes, G

def print_network_state(env, debal_manager, title="Network State"):
    """Print detailed network state."""
    print(f"\n{title}")
    print("=" * 50)
    print(f"Leader: {debal_manager.leader_election.current_leader}")
    print(f"Election Time: {debal_manager.leader_election.leader_timeout}")
    
    # Count nodes that have requested rebalancing
    pending_requests = sum(1 for node in debal_manager.nodes if node.rebalancing_requested)
    print(f"Pending Requests: {pending_requests}")
    
    # Print channel balances
    print("\nChannel Balances:")
    print("Channel (u→v)   Balances (lu→v, ru→v)")
    print("---------------------------------------------")
    # Create a sorted list of unique channel pairs
    channel_pairs = []
    for node in debal_manager.nodes:
        for peer_id in node.local_balances.keys():
            if (node.id, peer_id) not in channel_pairs and (peer_id, node.id) not in channel_pairs:
                if node.id < peer_id:
                    channel_pairs.append((node.id, peer_id))
                else:
                    channel_pairs.append((peer_id, node.id))
    
    # Print each channel's balances
    for node_id, peer_id in sorted(channel_pairs):
        node = next(n for n in debal_manager.nodes if n.id == node_id)
        local = node.local_balances[peer_id]
        remote = node.remote_balances[peer_id]
        print(f"{node_id}→{peer_id}       ({local:>4.1f}, {remote:>4.1f})")
    
    # Print node states
    print("\nNode States:")
    print("Node  Requesting Outgoing Ratio  Incoming Ratio")
    print("--------------------------------------------------")
    for node in debal_manager.nodes:
        requesting = node.rebalancing_requested
        total_local = sum(node.local_balances.values())
        total_remote = sum(node.remote_balances.values())
        total = total_local + total_remote
        out_ratio = total_local / total if total > 0 else 0.0
        in_ratio = total_remote / total if total > 0 else 0.0
        print(f"{node.id} {requesting!s:<9} {out_ratio:.2f}            {in_ratio:.2f}")
    
    # Print requesting nodes
    requesting_nodes = [
        (node.id, node.get_total_outgoing_liquidity() / (node.get_total_outgoing_liquidity() + node.get_total_incoming_liquidity()))
        for node in debal_manager.nodes
        if node.rebalancing_requested
    ]
    if requesting_nodes:
        print("\nRequesting Nodes:")
        for node_id, ratio in requesting_nodes:
            print(f"  Node {node_id} (Outgoing Ratio: {ratio:.2f})")

def print_changes(initial_state, final_state):
    """Print a summary of balance changes for each node."""
    print("\nSummary of Changes:")
    print("=" * 50)
    print()
    
    # Group changes by node
    for node_id in initial_state['node_states']:
        initial = initial_state['node_states'][node_id]
        final = final_state['node_states'][node_id]
        
        print(f"Node {node_id} Changes:")
        for peer in initial['local_balances']:
            initial_local = initial['local_balances'][peer]
            final_local = final['local_balances'][peer]
            change = final_local - initial_local
            print(f"  {peer}: {change:+.1f} (Local Balance Change)")
        print()

def main():
    """Run the DEBAL system with a test network and detailed logging."""
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
        delta_t=10.0  # Re-election interval (reduced to 10 seconds)
    )
    
    # Print initial state
    initial_state = debal_manager.get_network_state()
    print_network_state(env, debal_manager, "Initial Network State")
    
    # Start the system
    debal_manager.start()
    
    print("\nRequesting rebalancing for nodes 0 and 2...")
    # Request rebalancing for multiple nodes
    nodes[0].request_rebalancing()
    nodes[2].request_rebalancing()
    
    # Run simulation for 5 minutes with state updates every 30 seconds
    for i in range(10):
        env.run(until=(i + 1) * 30)  # Run for 30 seconds
        print_network_state(env, debal_manager, f"Network State at {(i + 1) * 30} seconds")
    
    # Print final state
    final_state = debal_manager.get_network_state()
    print_network_state(env, debal_manager, "Final Network State")
    
    # Print summary of changes
    print_changes(initial_state, final_state)

if __name__ == "__main__":
    main() 