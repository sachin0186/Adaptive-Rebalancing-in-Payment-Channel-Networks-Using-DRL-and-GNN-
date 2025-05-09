import unittest
import torch
import networkx as nx
from src.learning.balance_aware_gnn import BalanceAwareGNN
from torch_geometric.data import Data

class TestBalanceAwareGNN(unittest.TestCase):
    def setUp(self):
        self.gnn = BalanceAwareGNN(
            input_dim=2,  # Node features dimension
            hidden_dim1=16,  # First GCN layer dimension
            hidden_dim2=8,  # Second GCN layer dimension
            dropout_rate=0.2
        )
        
        # Create a simple graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(4))
        self.graph.add_edges_from([
            (0, 1), (1, 2), (2, 3), (0, 3)
        ])
        
        # Create PyTorch Geometric data
        num_nodes = 4
        num_edges = 4
        
        # Node features: [local_balance_ratio, remote_balance_ratio]
        x = torch.tensor([
            [0.7, 0.3],  # Node 0
            [0.5, 0.5],  # Node 1
            [0.4, 0.6],  # Node 2
            [0.5, 0.5]   # Node 3
        ], dtype=torch.float)
        
        # Edge index
        edge_index = torch.tensor([
            [0, 1, 2, 0],
            [1, 2, 3, 3]
        ], dtype=torch.long)
        
        # Edge features: [capacity, fee_rate, skewness]
        edge_attr = torch.tensor([
            [1000, 0.001, 0.4],  # Edge 0-1
            [1000, 0.001, 0.2],  # Edge 1-2
            [1000, 0.001, 0.2],  # Edge 2-3
            [1000, 0.001, 0.0]   # Edge 0-3
        ], dtype=torch.float)
        
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    def test_forward_pass_dimensions(self):
        """Test GNN forward pass dimensions"""
        # Forward pass
        node_embeddings, constraint_scores = self.gnn(self.data)
        
        # Check dimensions
        self.assertEqual(node_embeddings.shape, (4, 8))  # num_nodes x hidden_dim2
        self.assertEqual(constraint_scores.shape, (4, 1))  # num_edges x 1
        
    def test_constraint_layer(self):
        """Test constraint layer with sigmoid activation"""
        # Forward pass
        _, constraint_scores = self.gnn(self.data)
        
        # Check constraint scores are between 0 and 1
        self.assertTrue(torch.all(constraint_scores >= 0))
        self.assertTrue(torch.all(constraint_scores <= 1))
        
    def test_path_scoring(self):
        """Test path scoring MLP"""
        # Forward pass to get embeddings
        node_embeddings, constraint_scores = self.gnn(self.data)
        
        # Test valid path
        path = [0, 1, 2]
        score = self.gnn.score_path(path, node_embeddings, constraint_scores, self.data.edge_index)
        self.assertGreater(score, -float('inf'))
        
        # Test invalid path
        invalid_path = [0, 3, 1]  # No direct edge 3-1
        score = self.gnn.score_path(invalid_path, node_embeddings, constraint_scores, self.data.edge_index)
        self.assertEqual(score, -float('inf'))
        
    def test_candidate_paths(self):
        """Test candidate path finding"""
        # Find paths from node 0 to node 2
        paths = self.gnn.find_candidate_paths(self.graph, 0, 2, max_length=3)
        
        # Should find at least one path
        self.assertGreater(len(paths), 0)
        
        # All paths should be valid
        for path in paths:
            self.assertTrue(nx.is_path(self.graph, path))
            
    def test_path_ranking(self):
        """Test path ranking"""
        # Forward pass to get embeddings
        node_embeddings, constraint_scores = self.gnn(self.data)
        
        # Rank paths
        ranked_paths = self.gnn.rank_paths(
            self.graph, 0, 2, node_embeddings, 
            constraint_scores, self.data.edge_index
        )
        
        # Should have ranked paths
        self.assertGreater(len(ranked_paths), 0)
        
        # Check scores are in descending order
        scores = [score for score, _ in ranked_paths]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
if __name__ == '__main__':
    unittest.main() 