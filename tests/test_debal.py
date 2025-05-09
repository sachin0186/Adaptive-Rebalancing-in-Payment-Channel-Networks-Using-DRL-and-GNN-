"""
Test script for DEBAL system.
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.entities import Node
from src.learning.pytorch_soft_actor_critic import SAC, ReplayMemory

class TestDEBAL(unittest.TestCase):
    def setUp(self):
        self.node = Node(0)
        
    def test_node_initialization(self):
        self.assertEqual(self.node.id, 0)
        self.assertEqual(len(self.node.local_balances), 0)
        self.assertEqual(len(self.node.remote_balances), 0)
        self.assertEqual(len(self.node.capacities), 0)
        self.assertEqual(len(self.node.fee_rates), 0)
        
    def test_add_channel(self):
        self.node.add_channel(1, 1000, 500, 500, 0.001)
        self.assertEqual(self.node.local_balances[1], 500)
        self.assertEqual(self.node.remote_balances[1], 500)
        self.assertEqual(self.node.capacities[1], 1000)
        self.assertEqual(self.node.fee_rates[1], 0.001)
        
    def test_get_state(self):
        self.node.add_channel(1, 1000, 500, 500, 0.001)
        state = self.node.get_state()
        self.assertEqual(len(state), 4)
        self.assertEqual(state[0], 500)  # avg_local
        self.assertEqual(state[1], 500)  # avg_remote
        self.assertEqual(state[2], 1000)  # avg_capacity
        self.assertEqual(state[3], 0.001)  # avg_fee
        
    def test_decide_rebalancing(self):
        self.node.add_channel(1, 1000, 500, 500, 0.001)
        decision = self.node.decide_rebalancing()
        self.assertIsInstance(decision, bool)
        
    def test_calculate_reward(self):
        self.node.add_channel(1, 1000, 500, 500, 0.001)
        state = self.node.get_state()
        action = 0.5
        next_state = np.array([600, 400, 1000, 0.001])
        reward = self.node._calculate_reward(state, action, next_state)
        self.assertIsInstance(reward, float)
        
if __name__ == '__main__':
    unittest.main() 