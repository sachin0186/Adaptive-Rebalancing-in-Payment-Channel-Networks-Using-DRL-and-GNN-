"""
DEBAL (Decentralized Balance-Aware) Integration Module

This module integrates all components of the DEBAL methodology for payment channel
network rebalancing:
1. Node-Level DRL for liquidity assessment
2. Decentralized Leader Election
3. BalanceAware GNN for path calculation
4. Multi-Cycle Rebalancing
5. Periodic Re-Election

The module provides a high-level interface for managing the rebalancing process.
"""

from typing import List, Dict, Optional
import simpy
import networkx as nx
from src.entities.node import Node
from src.entities.debal_components import DEBALNodeState
from src.entities.leader_election import LeaderElection
from src.entities.rebalancing_engine import RebalancingEngine
from src.entities.scheduler import RebalancingScheduler
from src.learning.balance_aware_gnn import BalanceAwareGNN

class DEBALManager:
    def __init__(self, env: simpy.Environment, nodes: List[Node],
                 network_graph: nx.Graph,
                 kappa: float = 0.5,  # Minimum outgoing balance threshold
                 theta: float = 0.2,  # Minimum balance ratio threshold
                 tau: float = 2.0,    # TTD threshold in hours
                 delta_t: float = 600.0):  # Re-election interval
        """
        Initialize the DEBAL manager.
        
        Args:
            env: SimPy environment
            nodes: List of nodes in the network
            network_graph: NetworkX graph representing the PCN
            kappa: Minimum outgoing balance threshold for leader eligibility
            theta: Minimum balance ratio threshold
            tau: Time To Depletion (TTD) threshold in hours
            delta_t: Time interval between re-elections
        """
        self.env = env
        self.nodes = nodes
        self.network_graph = network_graph
        
        # Initialize components
        self.node_states = {node: DEBALNodeState(theta, tau) for node in nodes}
        self.leader_election = LeaderElection(kappa=kappa, theta=theta, delta_t=delta_t)
        self.rebalancing_engine = RebalancingEngine(sigma=0.2)  # Allow more skewness
        self.rebalancing_engine.nodes = {node.id: node for node in nodes}  # Initialize nodes
        self.scheduler = RebalancingScheduler(
            env=env,
            nodes=nodes,
            leader_election=self.leader_election,
            rebalancing_engine=self.rebalancing_engine,
            delta_t=delta_t
        )
        self.gnn = BalanceAwareGNN()
        
        # Initialize node states
        for node in nodes:
            node.debal_state = self.node_states[node]
            node.leader_id = None
            node.election_timestamp = None
            node.rebalancing_requested = False
            
    def start(self):
        """
        Start the DEBAL rebalancing process.
        """
        # Start the scheduler process
        self.env.process(self.scheduler.run())
        
        # Initialize GNN with current network state
        self.env.process(self._update_gnn_state())
        
    def _update_gnn_state(self):
        """Update GNN state periodically."""
        while True:
            # Prepare node features
            node_features = []
            for node in self.nodes:
                if not node.local_balances:
                    node_features.append([0.0, 0.0])  # Default features for isolated nodes
                    continue
                    
                # Calculate average balance ratios
                local_ratios = [bal / node.capacities[n] for n, bal in node.local_balances.items()]
                remote_ratios = [bal / node.capacities[n] for n, bal in node.remote_balances.items()]
                avg_local = sum(local_ratios) / len(local_ratios)
                avg_remote = sum(remote_ratios) / len(remote_ratios)
                node_features.append([avg_local, avg_remote])
            
            # Prepare edge features (channel capacities, current balances)
            edge_features = []
            edge_index = []
            for i, node in enumerate(self.nodes):
                for neighbor_id, capacity in node.capacities.items():
                    j = int(neighbor_id.split('_')[1])  # Get node index from ID
                    edge_index.append([i, j])
                    edge_features.append([
                        capacity,
                        node.local_balances[neighbor_id],
                        node.remote_balances[neighbor_id]
                    ])
            
            # Update GNN state
            self.gnn.update_state(node_features, edge_index, edge_features)
            
            # Wait before next update
            yield self.env.timeout(60.0)  # Update every minute
        
    def handle_rebalancing_request(self, node: Node):
        """
        Handle a rebalancing request from a node.
        
        Args:
            node: Node requesting rebalancing
        """
        # Mark node as requesting rebalancing
        node.rebalancing_requested = True
        
        # If no leader exists, trigger election
        if not self.scheduler.current_leader:
            self.scheduler.perform_election()
            
    def execute_rebalancing(self, source: Node, target: Node, amount: float) -> bool:
        """
        Execute rebalancing between two nodes.
        
        Args:
            source: Source node
            target: Target node
            amount: Rebalancing amount
            
        Returns:
            bool: True if rebalancing was successful, False otherwise
        """
        # Update GNN state
        self._update_gnn_state()
        
        # Get ranked paths
        paths = self.gnn.rank_paths(
            self.network_graph,
            source.id,
            target.id,
            self.gnn.node_embeddings,
            self.gnn.constraint_scores,
            self.gnn.edge_index
        )
        
        # Try each path until successful
        for score, path in paths:
            if self.rebalancing_engine.validate_path(path, amount):
                if self.rebalancing_engine.execute_transfer(path, amount):
                    return True
                    
        return False
        
    def get_network_state(self) -> Dict:
        """
        Get current network state.
        
        Returns:
            Dict containing:
                - leader_id: Current leader node ID
                - election_time: Last election timestamp
                - node_states: Dictionary of node states
                - pending_requests: Number of pending rebalancing requests
        """
        return {
            "leader_id": self.scheduler.current_leader.id if self.scheduler.current_leader else None,
            "election_time": self.scheduler.last_election_time,
            "node_states": {
                node.id: {
                    "local_balances": node.local_balances,
                    "remote_balances": node.remote_balances,
                    "rebalancing_requested": node.rebalancing_requested
                }
                for node in self.nodes
            },
            "pending_requests": sum(1 for node in self.nodes if node.rebalancing_requested)
        } 