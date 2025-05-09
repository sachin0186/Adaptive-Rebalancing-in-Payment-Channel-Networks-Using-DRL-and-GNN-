"""
Rebalancing engine module for DEBAL.
"""

from typing import List, Dict, Optional, Tuple
import networkx as nx
from .node import Node

class RebalancingEngine:
    def __init__(self, sigma: float = 0.1):
        """
        Initialize the rebalancing engine.
        
        Args:
            sigma: Maximum allowed skewness
        """
        self.sigma = sigma
        self.nodes = {}  # Will be initialized by scheduler
        
    def validate_path(self, path: List[str], amount: float) -> bool:
        """
        Check if a rebalancing path is feasible.
        
        Args:
            path: List of node IDs in the path
            amount: Amount to transfer
            
        Returns:
            bool: True if path is feasible, False otherwise
        """
        if len(path) < 2:
            return False
            
        # Check each hop in the path
        for i in range(len(path) - 1):
            node_id = path[i]
            next_node_id = path[i + 1]
            
            # Get nodes
            if node_id not in self.nodes or next_node_id not in self.nodes:
                return False
                
            node = self.nodes[node_id]
            next_node = self.nodes[next_node_id]
            
            # Check if channel exists
            if next_node_id not in node.local_balances:
                return False
                
            # Get current balances and capacity
            local_balance = node.local_balances[next_node_id]
            remote_balance = node.remote_balances[next_node_id]
            capacity = node.capacities[next_node_id]
            
            # Check if sufficient balance for transfer
            if local_balance < amount:
                return False
                
            # Calculate new balances after transfer
            new_local = local_balance - amount
            new_remote = remote_balance + amount
            
            # Check minimum balance requirements (20% of capacity)
            min_balance = 0.2 * capacity
            if new_local < min_balance or new_remote < min_balance:
                return False
                
            # Calculate skewness before and after transfer
            current_skew = abs(local_balance - remote_balance) / capacity
            new_skew = abs(new_local - new_remote) / capacity
            
            # Check if new skewness exceeds threshold or makes imbalance worse
            if new_skew > self.sigma and new_skew >= current_skew:
                return False
                
        return True
        
    def execute_transfer(self, path: List[str], amount: float) -> bool:
        """
        Execute a transfer along a valid path.
        
        Args:
            path: List of node IDs in the path
            amount: Amount to transfer
            
        Returns:
            bool: True if transfer was successful, False otherwise
        """
        if not self.validate_path(path, amount):
            return False
            
        # Update balances along the path
        for i in range(len(path) - 1):
            node_id = path[i]
            next_node_id = path[i + 1]
            
            # Get nodes
            node = self.nodes[node_id]
            next_node = self.nodes[next_node_id]
            
            # Update local balances
            node.local_balances[next_node_id] -= amount
            next_node.local_balances[node_id] += amount
            
            # Update remote balances
            node.remote_balances[next_node_id] += amount
            next_node.remote_balances[node_id] -= amount
            
        return True
        
    def calculate_improvement(self, path: List[str], amount: float) -> float:
        """
        Calculate the liquidity improvement from a rebalancing operation.
        
        Args:
            path: List of node IDs in the path
            amount: Amount to transfer
            
        Returns:
            float: Improvement in balance ratios
        """
        if amount > 0 and not self.validate_path(path, amount):
            return 0.0
            
        # Calculate initial balance ratios
        initial_ratios = []
        for i in range(len(path) - 1):
            node_id = path[i]
            next_node_id = path[i + 1]
            
            node = self.nodes[node_id]
            next_node = self.nodes[next_node_id]
            
            local = node.local_balances[next_node_id]
            remote = node.remote_balances[next_node_id]
            capacity = node.capacities[next_node_id]
            
            ratio = abs(local - remote) / capacity
            initial_ratios.append(ratio)
            
        # Calculate final balance ratios after transfer
        final_ratios = []
        for i in range(len(path) - 1):
            node_id = path[i]
            next_node_id = path[i + 1]
            
            node = self.nodes[node_id]
            next_node = self.nodes[next_node_id]
            
            local = node.local_balances[next_node_id] - amount
            remote = node.remote_balances[next_node_id] + amount
            capacity = node.capacities[next_node_id]
            
            ratio = abs(local - remote) / capacity
            final_ratios.append(ratio)
            
        # Return improvement in balance ratios
        return sum(initial_ratios) - sum(final_ratios) 