"""Tests for the rebalancing scheduler module."""

import unittest
import simpy
from src.entities.node import Node
from src.entities.scheduler import RebalancingScheduler

class TestRebalancingScheduler(unittest.TestCase):
    def setUp(self):
        """Set up test environment and nodes."""
        self.env = simpy.Environment()
        
        # Create test nodes
        self.nodes = []
        for i in range(4):
            node = Node(f"node_{i}")
            self.nodes.append(node)
            
        # Set up channels with specific balance configurations
        # Node 0: Eligible leader (high outgoing balance)
        self.nodes[0].local_balances = {"node_1": 800}
        self.nodes[0].remote_balances = {"node_1": 200}
        self.nodes[0].capacities = {"node_1": 1000}
        self.nodes[0].rebalancing_requested = True
        
        # Node 1: Ineligible (low outgoing balance)
        self.nodes[1].local_balances = {"node_0": 200, "node_2": 200}
        self.nodes[1].remote_balances = {"node_0": 800, "node_2": 800}
        self.nodes[1].capacities = {"node_0": 1000, "node_2": 1000}
        
        # Node 2: Eligible but not requesting
        self.nodes[2].local_balances = {"node_1": 700, "node_3": 600}
        self.nodes[2].remote_balances = {"node_1": 300, "node_3": 400}
        self.nodes[2].capacities = {"node_1": 1000, "node_3": 1000}
        
        # Node 3: End node
        self.nodes[3].local_balances = {"node_2": 400}
        self.nodes[3].remote_balances = {"node_2": 600}
        self.nodes[3].capacities = {"node_2": 1000}
        
        # Initialize scheduler
        self.scheduler = RebalancingScheduler(
            self.env,
            self.nodes,
            delta_t=600.0  # 10 minutes
        )
        
    def test_initialization(self):
        """Test scheduler initialization."""
        self.assertIsNotNone(self.scheduler.leader_election)
        self.assertIsNone(self.scheduler.current_leader)
        self.assertEqual(self.scheduler.last_election_time, 0)
        self.assertEqual(self.scheduler.delta_t, 600.0)
        
    def test_election_trigger(self):
        """Test election trigger conditions."""
        # Initially no leader, should trigger election
        self.assertTrue(self.scheduler.should_trigger_election())
        
        # Set current leader and time
        self.scheduler.current_leader = self.nodes[0]
        self.scheduler.last_election_time = self.env.now
        
        # Within delta_t, should not trigger
        self.assertFalse(self.scheduler.should_trigger_election())
        
        # Force time past delta_t
        self.env.run(until=700)
        self.assertTrue(self.scheduler.should_trigger_election())
        
        # Make current leader ineligible
        self.nodes[0].rebalancing_requested = False
        self.assertTrue(self.scheduler.should_trigger_election())
        
    def test_election_process(self):
        """Test election process."""
        # Perform initial election
        self.scheduler.perform_election()
        
        # Verify leader election
        self.assertIsNotNone(self.scheduler.current_leader)
        self.assertEqual(self.scheduler.current_leader.id, "node_0")
        self.assertEqual(self.scheduler.last_election_time, self.env.now)
        
        # Verify announcement
        announcement = self.scheduler.leader_election.announce_leader(
            self.scheduler.current_leader,
            self.env.now
        )
        self.assertTrue(self.scheduler.leader_election.verify_announcement(
            announcement,
            self.nodes
        ))
        
    def test_leader_transition(self):
        """Test leader transition handling."""
        # Set initial leader
        self.scheduler.current_leader = self.nodes[0]
        self.scheduler.last_election_time = self.env.now
        
        # Add some state to old leader
        self.nodes[0].rebalancing_state = {"some_state": "value"}
        
        # Perform new election
        self.nodes[2].rebalancing_requested = True
        self.scheduler.perform_election()
        
        # Verify transition
        self.assertEqual(self.scheduler.current_leader.id, "node_2")
        self.assertEqual(self.nodes[2].rebalancing_state, {"some_state": "value"})
        
    def test_scheduler_run(self):
        """Test scheduler run process."""
        # Start scheduler
        self.env.process(self.scheduler.run())
        
        # Run simulation for a short time
        self.env.run(until=100)
        
        # Verify initial state
        self.assertIsNotNone(self.scheduler.current_leader)
        
        # Run past delta_t
        self.env.run(until=700)
        
        # Verify re-election
        self.assertIsNotNone(self.scheduler.current_leader)
        self.assertEqual(self.scheduler.last_election_time, 700)
        
    def test_rebalancing_request_handling(self):
        """Test handling of rebalancing requests."""
        # Initially no requests
        self.assertFalse(any(node.rebalancing_requested for node in self.nodes))
        
        # Request rebalancing from node 2
        self.nodes[2].rebalancing_requested = True
        self.scheduler.perform_election()
        
        # Verify request handling
        self.assertFalse(self.nodes[2].rebalancing_requested)
        self.assertEqual(self.scheduler.current_leader.id, "node_2")
        
    def test_node_state_updates(self):
        """Test node state updates during election."""
        # Perform election
        self.scheduler.perform_election()
        
        # Verify node states
        for node in self.nodes:
            self.assertEqual(node.leader_id, self.scheduler.current_leader.id)
            self.assertEqual(node.election_timestamp, self.env.now)
            
if __name__ == '__main__':
    unittest.main() 