"""Tests for the DEBAL integration module."""

import unittest
import simpy
import networkx as nx
from src.entities.node import Node
from src.entities.debal_integration import DEBALManager

class TestDEBALIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment and nodes."""
        self.env = simpy.Environment()
        
        # Create test nodes
        self.nodes = []
        for i in range(4):
            node = Node(f"node_{i}")
            self.nodes.append(node)
            
        # Set up channels with specific balance configurations
        # Node 0 -> Node 1: Imbalanced but within skewness limit
        self.nodes[0].local_balances = {"node_1": 600}
        self.nodes[0].remote_balances = {"node_1": 400}
        self.nodes[0].capacities = {"node_1": 1000}
        
        # Node 1 -> Node 2: Balanced
        self.nodes[1].local_balances = {"node_0": 400, "node_2": 500}
        self.nodes[1].remote_balances = {"node_0": 600, "node_2": 500}
        self.nodes[1].capacities = {"node_0": 1000, "node_2": 1000}
        
        # Node 2 -> Node 3: Slightly imbalanced
        self.nodes[2].local_balances = {"node_1": 500, "node_3": 550}
        self.nodes[2].remote_balances = {"node_1": 500, "node_3": 450}
        self.nodes[2].capacities = {"node_1": 1000, "node_3": 1000}
        
        # Node 3: End node
        self.nodes[3].local_balances = {"node_2": 450}
        self.nodes[3].remote_balances = {"node_2": 550}
        self.nodes[3].capacities = {"node_2": 1000}
        
        # Create network graph
        self.network_graph = nx.Graph()
        for node in self.nodes:
            self.network_graph.add_node(node.id)
            for neighbor in node.local_balances:
                self.network_graph.add_edge(node.id, neighbor)
                
        # Initialize DEBAL manager
        self.debal = DEBALManager(
            self.env,
            self.nodes,
            self.network_graph,
            kappa=0.5,
            theta=0.2,
            tau=2.0,
            delta_t=600.0
        )
        
    def test_initialization(self):
        """Test DEBAL manager initialization."""
        self.assertIsNotNone(self.debal.node_states)
        self.assertIsNotNone(self.debal.leader_election)
        self.assertIsNotNone(self.debal.rebalancing_engine)
        self.assertIsNotNone(self.debal.scheduler)
        self.assertIsNotNone(self.debal.gnn)
        
        # Check node state initialization
        for node in self.nodes:
            self.assertIn(node, self.debal.node_states)
            self.assertIsNotNone(node.debal_state)
            
    def test_rebalancing_request(self):
        """Test handling of rebalancing requests."""
        # Initially no leader
        self.assertIsNone(self.debal.scheduler.current_leader)
        
        # Request rebalancing from node 0
        self.nodes[0].request_rebalancing()
        self.debal.handle_rebalancing_request(self.nodes[0])
        
        # Should trigger election
        self.assertIsNotNone(self.debal.scheduler.current_leader)
        
    def test_leader_election(self):
        """Test leader election process."""
        # Request rebalancing from eligible node
        self.nodes[0].request_rebalancing()
        self.debal.handle_rebalancing_request(self.nodes[0])
        
        # Verify leader election
        leader = self.debal.scheduler.current_leader
        self.assertIsNotNone(leader)
        self.assertTrue(leader.rebalancing_requested)
        
        # Check network state
        state = self.debal.get_network_state()
        self.assertEqual(state["leader_id"], leader.id)
        self.assertIsNotNone(state["election_time"])
        
    def test_rebalancing_execution(self):
        """Test rebalancing execution."""
        # Set up rebalancing request
        self.nodes[0].request_rebalancing()
        self.debal.handle_rebalancing_request(self.nodes[0])
        
        # Execute rebalancing
        success = self.debal.execute_rebalancing(
            self.nodes[0],
            self.nodes[2],
            100
        )
        
        # Verify rebalancing result
        self.assertTrue(success)
        
        # Check balance updates
        self.assertEqual(self.nodes[0].local_balances["node_1"], 500)
        self.assertEqual(self.nodes[0].remote_balances["node_1"], 500)
        
    def test_network_state(self):
        """Test network state monitoring."""
        # Get initial state
        initial_state = self.debal.get_network_state()
        
        # Request rebalancing
        self.nodes[0].request_rebalancing()
        self.debal.handle_rebalancing_request(self.nodes[0])
        
        # Get updated state
        updated_state = self.debal.get_network_state()
        
        # Verify state changes
        self.assertNotEqual(initial_state["leader_id"], updated_state["leader_id"])
        self.assertNotEqual(initial_state["election_time"], updated_state["election_time"])
        self.assertEqual(updated_state["pending_requests"], 1)
        
    def test_gnn_state_update(self):
        """Test GNN state updates."""
        # Initial GNN state
        initial_embeddings = self.debal.gnn.node_embeddings
        
        # Update balances
        self.nodes[0].update_balances("node_1", 100, -100)
        
        # Update GNN state
        self.debal._update_gnn_state()
        
        # Verify GNN state update
        self.assertNotEqual(initial_embeddings, self.debal.gnn.node_embeddings)
        
    def test_scheduler_operation(self):
        """Test scheduler operation."""
        # Start scheduler
        self.debal.start()
        
        # Run simulation for a short time
        self.env.run(until=100)
        
        # Verify scheduler state
        self.assertIsNotNone(self.debal.scheduler.current_leader)
        self.assertIsNotNone(self.debal.scheduler.last_election_time)
        
if __name__ == '__main__':
    unittest.main() 