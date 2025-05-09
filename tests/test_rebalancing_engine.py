import unittest
from src.entities.rebalancing_engine import RebalancingEngine
from src.entities.node import Node

class TestRebalancingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RebalancingEngine(
            sigma=0.8,  # Maximum allowed skewness
            m_cycles=3,  # Number of rebalancing cycles
            reduction_factor=0.5,  # Partial transfer factor
            epsilon=0.001  # Early convergence threshold
        )
        
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
        
    def test_path_validation(self):
        """Test path validation with skewness constraint"""
        path = [self.nodes[0], self.nodes[1]]
        
        # Test valid amount (should not exceed skewness)
        self.assertTrue(self.engine.validate_path(path, 100))
        
        # Test amount that would cause excessive skewness
        self.assertFalse(self.engine.validate_path(path, 500))
        
        # Test amount exceeding available balance
        self.assertFalse(self.engine.validate_path(path, 800))
        
    def test_transfer_execution(self):
        """Test transfer execution with rollback"""
        path = [self.nodes[0], self.nodes[1]]
        
        # Store initial balances
        initial_balances = [
            (node.local_balances.copy(), node.remote_balances.copy())
            for node in path
        ]
        
        # Execute small transfer that should succeed
        success, improvement = self.engine.execute_transfer(path, 50)
        self.assertTrue(success)
        self.assertGreater(improvement, 0)
        
        # Reset balances
        for i, node in enumerate(path):
            node.local_balances = initial_balances[i][0].copy()
            node.remote_balances = initial_balances[i][1].copy()
        
        # Execute large transfer that should fail and rollback
        success, improvement = self.engine.execute_transfer(path, 500)
        self.assertFalse(success)
        self.assertEqual(improvement, 0)
        
        # Verify rollback
        for i, node in enumerate(path):
            self.assertEqual(node.local_balances, initial_balances[i][0])
            self.assertEqual(node.remote_balances, initial_balances[i][1])
            
    def test_multi_cycle_rebalancing(self):
        """Test iterative multi-cycle rebalancing"""
        path = [self.nodes[0], self.nodes[1]]
        
        # Store initial balances
        initial_balances = [
            (node.local_balances.copy(), node.remote_balances.copy())
            for node in path
        ]
        
        # Start with a reasonable amount
        success, improvement = self.engine.rebalance_path(path, 100)
        self.assertTrue(success)
        self.assertGreater(improvement, 0)
        
        # Verify balances changed
        for i, node in enumerate(path):
            self.assertNotEqual(node.local_balances, initial_balances[i][0])
            self.assertNotEqual(node.remote_balances, initial_balances[i][1])
            
    def test_improvement_calculation(self):
        """Test liquidity improvement calculation"""
        path = [self.nodes[0], self.nodes[1]]
        
        # Calculate improvement for a small transfer
        improvement = self.engine.calculate_improvement(path, 50)
        self.assertGreater(improvement, 0)
        
        # Calculate improvement for an invalid transfer
        improvement = self.engine.calculate_improvement(path, 800)
        self.assertEqual(improvement, 0)
        
if __name__ == '__main__':
    unittest.main() 