"""Tests for the leader election module."""

import unittest
import time
from src.entities.leader_election import LeaderElection
from src.entities.node import Node

class TestLeaderElection(unittest.TestCase):
    def setUp(self):
        """Set up test nodes and leader election instance."""
        self.election = LeaderElection(kappa=0.5, theta=0.2, delta_t=3600.0)
        
        # Create test nodes with different balance configurations
        self.nodes = []
        
        # Node 1: Eligible leader (high outgoing balance, good ratios)
        node1 = Node("node1")
        node1.local_balances = {"ch1": 800, "ch2": 700}
        node1.remote_balances = {"ch1": 200, "ch2": 300}
        node1.capacities = {"ch1": 1000, "ch2": 1000}
        node1.rebalancing_requested = True
        self.nodes.append(node1)
        
        # Node 2: Ineligible (low outgoing balance)
        node2 = Node("node2")
        node2.local_balances = {"ch1": 100, "ch2": 100}
        node2.remote_balances = {"ch1": 900, "ch2": 900}
        node2.capacities = {"ch1": 1000, "ch2": 1000}
        node2.rebalancing_requested = True
        self.nodes.append(node2)
        
        # Node 3: Ineligible (poor balance ratios)
        node3 = Node("node3")
        node3.local_balances = {"ch1": 950, "ch2": 950}
        node3.remote_balances = {"ch1": 50, "ch2": 50}
        node3.capacities = {"ch1": 1000, "ch2": 1000}
        node3.rebalancing_requested = True
        self.nodes.append(node3)
        
        # Node 4: Eligible but not requesting rebalancing
        node4 = Node("node4")
        node4.local_balances = {"ch1": 600, "ch2": 600}
        node4.remote_balances = {"ch1": 400, "ch2": 400}
        node4.capacities = {"ch1": 1000, "ch2": 1000}
        node4.rebalancing_requested = False
        self.nodes.append(node4)
        
        # Node 5: Another eligible leader
        node5 = Node("node5")
        node5.local_balances = {"ch1": 700, "ch2": 600}
        node5.remote_balances = {"ch1": 300, "ch2": 400}
        node5.capacities = {"ch1": 1000, "ch2": 1000}
        node5.rebalancing_requested = True
        self.nodes.append(node5)
        
    def test_hash_computation(self):
        """Test that hash computation is consistent and unique."""
        # Same input should produce same hash
        hash1 = self.election.compute_hash("node1", 1000.0)
        hash2 = self.election.compute_hash("node1", 1000.0)
        self.assertEqual(hash1, hash2)
        
        # Different timestamps should produce different hashes
        hash3 = self.election.compute_hash("node1", 2000.0)
        self.assertNotEqual(hash1, hash3)
        
        # Different node IDs should produce different hashes
        hash4 = self.election.compute_hash("node2", 1000.0)
        self.assertNotEqual(hash1, hash4)
        
    def test_leader_eligibility(self):
        """Test leader eligibility criteria."""
        # Node 1 should be eligible
        self.assertTrue(self.election.is_eligible_leader(self.nodes[0]))
        
        # Node 2 should be ineligible (low outgoing balance)
        self.assertFalse(self.election.is_eligible_leader(self.nodes[1]))
        
        # Node 3 should be ineligible (poor ratios)
        self.assertFalse(self.election.is_eligible_leader(self.nodes[2]))
        
        # Node 4 should be ineligible (not requesting)
        self.assertFalse(self.election.is_eligible_leader(self.nodes[3]))
        
        # Node 5 should be eligible
        self.assertTrue(self.election.is_eligible_leader(self.nodes[4]))
        
    def test_leader_election(self):
        """Test the leader election process."""
        # Elect initial leader
        leader = self.election.elect_leader(self.nodes)
        self.assertIsNotNone(leader)
        self.assertTrue(self.election.is_eligible_leader(leader))
        
        # Leader should remain the same within timeout
        time.sleep(0.1)  # Small delay
        new_leader = self.election.elect_leader(self.nodes)
        self.assertEqual(leader, new_leader)
        
        # Force timeout and re-election
        self.election.leader_timeout = time.time() - 4000  # Past delta_t
        new_leader = self.election.elect_leader(self.nodes)
        self.assertIsNotNone(new_leader)
        self.assertTrue(self.election.is_eligible_leader(new_leader))
        
    def test_announcement_verification(self):
        """Test leader announcement and verification."""
        # Get a valid leader
        leader = self.election.elect_leader(self.nodes)
        timestamp = time.time()
        
        # Create and verify valid announcement
        announcement = self.election.announce_leader(leader, timestamp)
        self.assertTrue(self.election.verify_announcement(announcement, self.nodes))
        
        # Test with tampered announcement
        tampered = announcement.copy()
        tampered["leader_id"] = "fake_node"
        self.assertFalse(self.election.verify_announcement(tampered, self.nodes))
        
        # Test with invalid timestamp
        tampered = announcement.copy()
        tampered["timestamp"] = timestamp + 1000
        self.assertFalse(self.election.verify_announcement(tampered, self.nodes))
        
if __name__ == '__main__':
    unittest.main() 