"""Tests for the Node class."""

import unittest
from src.entities.node import Node

class TestNode(unittest.TestCase):
    def setUp(self):
        """Set up test node with initial state."""
        self.node = Node("test_node")
        
        # Initialize channel balances and capacities
        self.node.local_balances = {
            "node1": 500,
            "node2": 300,
            "node3": 200
        }
        self.node.remote_balances = {
            "node1": 500,
            "node2": 700,
            "node3": 800
        }
        self.node.capacities = {
            "node1": 1000,
            "node2": 1000,
            "node3": 1000
        }
        
    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.id, "test_node")
        self.assertFalse(self.node.rebalancing_requested)
        self.assertIsNone(self.node.leader_id)
        self.assertIsNone(self.node.election_timestamp)
        
    def test_balance_ratios(self):
        """Test balance ratio calculations."""
        # Test ratio calculation for a balanced channel
        ratio = self.node.get_balance_ratio("node1")
        self.assertEqual(ratio, 0.5)  # 500/1000
        
        # Test ratio calculation for an imbalanced channel
        ratio = self.node.get_balance_ratio("node2")
        self.assertEqual(ratio, 0.3)  # 300/1000
        
    def test_channel_skewness(self):
        """Test channel skewness calculations."""
        # Test skewness for a balanced channel
        skewness = self.node.get_channel_skewness("node1")
        self.assertEqual(skewness, 0.0)  # (500-500)/1000
        
        # Test skewness for an imbalanced channel
        skewness = self.node.get_channel_skewness("node2")
        self.assertEqual(skewness, 0.4)  # (700-300)/1000
        
    def test_total_liquidity(self):
        """Test total liquidity calculations."""
        # Test total outgoing liquidity
        outgoing = self.node.get_total_outgoing_liquidity()
        self.assertEqual(outgoing, 1000)  # 500 + 300 + 200
        
        # Test total incoming liquidity
        incoming = self.node.get_total_incoming_liquidity()
        self.assertEqual(incoming, 2000)  # 500 + 700 + 800
        
    def test_rebalancing_request(self):
        """Test rebalancing request handling."""
        # Initially not requesting
        self.assertFalse(self.node.rebalancing_requested)
        
        # Request rebalancing
        self.node.request_rebalancing()
        self.assertTrue(self.node.rebalancing_requested)
        
        # Clear rebalancing request
        self.node.clear_rebalancing_request()
        self.assertFalse(self.node.rebalancing_requested)
        
    def test_balance_updates(self):
        """Test balance update operations."""
        # Test successful balance update
        success = self.node.update_balances("node1", 100, -100)
        self.assertTrue(success)
        self.assertEqual(self.node.local_balances["node1"], 600)
        self.assertEqual(self.node.remote_balances["node1"], 400)
        
        # Test invalid update (exceeds capacity)
        success = self.node.update_balances("node1", 500, -500)
        self.assertFalse(success)
        
        # Test invalid update (negative balances)
        success = self.node.update_balances("node1", -700, 700)
        self.assertFalse(success)
        
    def test_channel_management(self):
        """Test channel management operations."""
        # Add new channel
        self.node.add_channel("node4", 400, 600, 1000)
        self.assertIn("node4", self.node.local_balances)
        self.assertIn("node4", self.node.remote_balances)
        self.assertIn("node4", self.node.capacities)
        
        # Remove channel
        self.node.remove_channel("node4")
        self.assertNotIn("node4", self.node.local_balances)
        self.assertNotIn("node4", self.node.remote_balances)
        self.assertNotIn("node4", self.node.capacities)
        
    def test_leader_state(self):
        """Test leader state management."""
        # Set leader
        self.node.set_leader("leader_node", 1234567890)
        self.assertEqual(self.node.leader_id, "leader_node")
        self.assertEqual(self.node.election_timestamp, 1234567890)
        
        # Clear leader
        self.node.clear_leader()
        self.assertIsNone(self.node.leader_id)
        self.assertIsNone(self.node.election_timestamp)
        
    def test_channel_validation(self):
        """Test channel validation."""
        # Test valid channel
        self.assertTrue(self.node.is_valid_channel("node1"))
        
        # Test non-existent channel
        self.assertFalse(self.node.is_valid_channel("nonexistent"))
        
        # Test channel with invalid balances
        self.node.local_balances["node1"] = -100
        self.assertFalse(self.node.is_valid_channel("node1"))
        
if __name__ == '__main__':
    unittest.main() 