"""
Rebalancing scheduler module for DEBAL.
"""

from typing import List, Dict, Optional
import time
import threading
from src.entities.Node import Node
from src.entities.leader_election import LeaderElection
from src.entities.rebalancing_engine import RebalancingEngine

class RebalancingScheduler:
    def __init__(self, 
                 nodes: List[Node],
                 leader_election: LeaderElection,
                 rebalancing_engine: RebalancingEngine,
                 interval: float = 60.0):
        """
        Initialize rebalancing scheduler.
        
        Args:
            nodes: List of nodes in the network
            leader_election: Leader election component
            rebalancing_engine: Rebalancing engine component
            interval: Time between rebalancing attempts in seconds
        """
        self.nodes = nodes
        self.leader_election = leader_election
        self.rebalancing_engine = rebalancing_engine
        self.interval = interval
        self.running = False
        self.scheduler_thread = None
        
    def start(self):
        """Start the rebalancing scheduler."""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
    def stop(self):
        """Stop the rebalancing scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
            
    def _scheduler_loop(self):
        """Main scheduler loop that periodically triggers rebalancing."""
        while self.running:
            # Elect leader if needed
            if not self.leader_election.current_leader:
                leader = self.leader_election.elect_leader(self.nodes)
                if leader:
                    print(f"Elected node {leader.id} as leader")
                else:
                    print("Failed to elect leader")
                    time.sleep(self.interval)
                    continue
                    
            # Get nodes requesting rebalancing
            rebalancing_nodes = [node for node in self.nodes if node.needs_rebalancing]
            
            if not rebalancing_nodes:
                time.sleep(self.interval)
                continue
                
            # Process rebalancing requests
            for node in rebalancing_nodes:
                self._process_rebalancing_request(node)
                
            time.sleep(self.interval)
            
    def _process_rebalancing_request(self, node: Node):
        """
        Process a rebalancing request from a node.
        
        Args:
            node: Node requesting rebalancing
        """
        # Find candidate paths
        paths = self._find_rebalancing_paths(node)
        
        if not paths:
            print(f"No valid paths found for node {node.id}")
            return
            
        print(f"Found {len(paths)} potential paths for node {node.id}")
            
        # Try rebalancing with different amounts
        amounts = [1000, 750, 500, 250, 100]  # More granular amounts
        
        for path in paths:
            path_str = " -> ".join(str(n.id) for n in path)
            print(f"Trying path: {path_str}")
            
            for amount in amounts:
                improvement = self.rebalancing_engine.calculate_improvement(path, amount)
                print(f"  Amount {amount}: improvement {improvement}")
                
                if improvement > 0:
                    if self.rebalancing_engine.execute_transfer(path, amount):
                        print(f"Successfully rebalanced {amount} along path {path_str}")
                        node.needs_rebalancing = False
                        return
                        
        print(f"Failed to find a valid rebalancing for node {node.id}")
                        
    def _find_rebalancing_paths(self, node: Node, max_paths: int = 10) -> List[List[Node]]:
        """
        Find potential rebalancing paths for a node.
        
        Args:
            node: Node to find paths for
            max_paths: Maximum number of paths to return
            
        Returns:
            List[List[Node]]: List of potential rebalancing paths
        """
        paths = []
        
        # Find nodes with complementary imbalances
        for other_node in self.nodes:
            if other_node == node:
                continue
                
            # Check if nodes have complementary imbalances
            if self._has_complementary_imbalance(node, other_node):
                # Try to find multiple paths between these nodes
                for _ in range(3):  # Try to find up to 3 different paths
                    path = self._find_path(node, other_node, paths)
                    if path and len(path) <= 5:  # Limit path length
                        paths.append(path)
                        
            if len(paths) >= max_paths:
                break
                
        return paths
        
    def _has_complementary_imbalance(self, node1: Node, node2: Node) -> bool:
        """
        Check if two nodes have complementary imbalances.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            bool: True if nodes have complementary imbalances
        """
        # Calculate total balances
        node1_total_local = sum(node1.local_balances.values())
        node1_total_remote = sum(node1.remote_balances.values())
        node2_total_local = sum(node2.local_balances.values())
        node2_total_remote = sum(node2.remote_balances.values())
        
        # Calculate imbalance ratios
        node1_ratio = node1_total_local / (node1_total_local + node1_total_remote)
        node2_ratio = node2_total_local / (node2_total_local + node2_total_remote)
        
        # Check if ratios are significantly different
        return abs(node1_ratio - node2_ratio) > 0.2
                
    def _find_path(self, source: Node, target: Node, 
                  existing_paths: List[List[Node]] = None) -> Optional[List[Node]]:
        """
        Find a path between two nodes, avoiding existing paths if possible.
        
        Args:
            source: Source node
            target: Target node
            existing_paths: List of paths to avoid
            
        Returns:
            Optional[List[Node]]: Path between nodes if found
        """
        visited = {source}
        queue = [(source, [source])]
        
        while queue:
            node, path = queue.pop(0)
            
            # Get neighbors in random order to find different paths
            neighbors = list(node.local_balances.keys())
            import random
            random.shuffle(neighbors)
            
            for neighbor_id in neighbors:
                neighbor = next((n for n in self.nodes if n.id == neighbor_id), None)
                
                if not neighbor or neighbor in visited:
                    continue
                    
                new_path = path + [neighbor]
                
                # Check if this path is significantly different from existing paths
                if existing_paths and not self._is_unique_path(new_path, existing_paths):
                    continue
                    
                if neighbor == target:
                    return new_path
                    
                visited.add(neighbor)
                queue.append((neighbor, new_path))
                
        return None
        
    def _is_unique_path(self, new_path: List[Node], 
                       existing_paths: List[List[Node]]) -> bool:
        """
        Check if a path is significantly different from existing paths.
        
        Args:
            new_path: Path to check
            existing_paths: List of existing paths
            
        Returns:
            bool: True if path is unique enough
        """
        if not existing_paths:
            return True
            
        for existing_path in existing_paths:
            common_nodes = set(n.id for n in new_path) & set(n.id for n in existing_path)
            # If more than 50% of nodes are common, consider it too similar
            if len(common_nodes) > len(new_path) * 0.5:
                return False
                
        return True 