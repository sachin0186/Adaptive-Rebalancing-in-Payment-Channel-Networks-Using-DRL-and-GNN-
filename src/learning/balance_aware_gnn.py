"""
Balance-Aware Graph Neural Network for path ranking in DEBAL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Dict, Tuple
import networkx as nx

class BalanceAwareGNN(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 32):
        """
        Initialize the Balance-Aware GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Node embedding layers
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Edge embedding layers
        self.edge_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3 features: capacity, local_balance, remote_balance
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Path scoring layer
        self.path_scoring = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize embeddings
        self.node_embeddings = None
        self.edge_index = None
        self.constraint_scores = None
        
    def update_state(self, node_features: List[List[float]], 
                    edge_index: List[List[int]], 
                    edge_features: List[List[float]]):
        """
        Update the GNN state with new network features.
        
        Args:
            node_features: List of node feature vectors
            edge_index: List of edge indices
            edge_features: List of edge feature vectors
        """
        # Convert to tensors
        node_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_tensor = torch.tensor(edge_features, dtype=torch.float)
        
        # Compute node embeddings
        self.node_embeddings = self.node_embedding(node_tensor)
        
        # Compute edge embeddings
        edge_embeddings = self.edge_embedding(edge_tensor)
        
        # Store edge index
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Compute constraint scores (balance ratios)
        self.constraint_scores = torch.min(
            edge_tensor[:, 1] / edge_tensor[:, 0],  # local_balance / capacity
            edge_tensor[:, 2] / edge_tensor[:, 0]   # remote_balance / capacity
        )
        
    def rank_paths(self, graph: nx.Graph, source: str, target: str,
                  node_embeddings: torch.Tensor, constraint_scores: torch.Tensor,
                  edge_index: torch.Tensor) -> List[Tuple[float, List[str]]]:
        """
        Rank possible paths between source and target nodes.
        
        Args:
            graph: NetworkX graph
            source: Source node ID
            target: Target node ID
            node_embeddings: Node embeddings tensor
            constraint_scores: Constraint scores tensor
            edge_index: Edge index tensor
            
        Returns:
            List of (score, path) tuples, sorted by score
        """
        # Get all simple paths
        paths = list(nx.all_simple_paths(graph, source, target, cutoff=4))
        
        # Score each path
        scored_paths = []
        for path in paths:
            # Get node indices
            node_indices = [int(node.split('_')[1]) for node in path]
            
            # Get embeddings for nodes in path
            path_embeddings = node_embeddings[node_indices]
            
            # Compute path score
            source_embedding = path_embeddings[0]
            target_embedding = path_embeddings[-1]
            path_score = self.path_scoring(
                torch.cat([source_embedding, target_embedding])
            ).item()
            
            # Adjust score based on constraint scores
            for i in range(len(path) - 1):
                u = int(path[i].split('_')[1])
                v = int(path[i + 1].split('_')[1])
                edge_idx = (edge_index[0] == u) & (edge_index[1] == v)
                if edge_idx.any():
                    path_score *= constraint_scores[edge_idx].item()
                    
            scored_paths.append((path_score, path))
            
        # Sort paths by score
        return sorted(scored_paths, key=lambda x: x[0], reverse=True)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        return self.node_embedding(x)
        
    def score_path(self, path: List[int], node_embeddings: torch.Tensor, 
                  constraint_scores: torch.Tensor, edge_index: torch.Tensor) -> float:
        """
        Score a candidate path using the path scoring MLP.
        
        Args:
            path: List of node indices representing the path
            node_embeddings: Node embeddings from GNN
            constraint_scores: Edge constraint scores
            edge_index: Graph connectivity
            
        Returns:
            float: Path score
        """
        if len(path) < 2:
            return -float('inf')
            
        # Get embeddings for path nodes
        path_embeddings = node_embeddings[path]
        
        # Calculate path features
        path_features = []
        for i in range(len(path) - 1):
            # Get edge constraint score
            edge_mask = (edge_index[0] == path[i]) & (edge_index[1] == path[i + 1])
            if not edge_mask.any():
                return -float('inf')
                
            edge_score = constraint_scores[edge_mask].item()
            
            # Concatenate node embeddings
            node_pair = torch.cat([path_embeddings[i], path_embeddings[i + 1]])
            path_features.append(node_pair * edge_score)
            
        # Average path features
        path_features = torch.stack(path_features).mean(dim=0)
        
        # Score path
        score = self.path_scoring(path_features)
        return score.item()
        
    def find_candidate_paths(self, graph: nx.Graph, source: int, target: int, 
                           max_length: int = 3) -> List[List[int]]:
        """
        Find candidate paths between source and target nodes.
        
        Args:
            graph: NetworkX graph
            source: Source node index
            target: Target node index
            max_length: Maximum path length
            
        Returns:
            List of candidate paths
        """
        paths = []
        for path in nx.all_simple_paths(graph, source=source, target=target, 
                                      cutoff=max_length):
            if len(path) >= 2:  # Ensure path has at least one edge
                paths.append(path)
        return paths
        
    def rank_paths(self, graph: nx.Graph, source: int, target: int, 
                  node_embeddings: torch.Tensor, constraint_scores: torch.Tensor,
                  edge_index: torch.Tensor, max_length: int = 3) -> List[Tuple[float, List[int]]]:
        """
        Rank candidate paths between source and target nodes.
        
        Args:
            graph: NetworkX graph
            source: Source node index
            target: Target node index
            node_embeddings: Node embeddings from GNN
            constraint_scores: Edge constraint scores
            edge_index: Graph connectivity
            max_length: Maximum path length
            
        Returns:
            List of (score, path) tuples, sorted by score in descending order
        """
        paths = self.find_candidate_paths(graph, source, target, max_length)
        scored_paths = []
        
        for path in paths:
            score = self.score_path(path, node_embeddings, constraint_scores, edge_index)
            if score > -float('inf'):  # Only include valid paths
                scored_paths.append((score, path))
                
        return sorted(scored_paths, key=lambda x: x[0], reverse=True) 