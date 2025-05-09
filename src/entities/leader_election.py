"""
Leader election module for DEBAL.
"""

import time
import hashlib
from typing import List, Dict, Optional, Any
from .node import Node

class LeaderElection:
    def __init__(self, kappa: float = 0.5, theta: float = 0.2, delta_t: float = 3600.0):
        """
        Initialize leader election with DEBAL parameters.
        
        Args:
            kappa: Minimum outgoing balance threshold (as a ratio of total capacity)
            theta: Minimum balance ratio threshold
            delta_t: Re-election interval in seconds
        """
        self.kappa = kappa
        self.theta = theta
        self.delta_t = delta_t
        self.leader_timeout = 0.0
        self.current_leader = None
        
    def compute_hash(self, node_id: str, timestamp: float) -> str:
        """
        Compute the SHA-256 hash of node ID and timestamp.
        
        Args:
            node_id: Node identifier
            timestamp: Current timestamp
            
        Returns:
            str: Hexadecimal hash value
        """
        message = f"{node_id}{timestamp}".encode('utf-8')
        return hashlib.sha256(message).hexdigest()
        
    def is_eligible_leader(self, node: Node) -> bool:
        """
        Check if a node is eligible to be a leader.
        
        A node is eligible if:
        1. It has requested rebalancing
        2. At least one of its channels has an outgoing balance ratio exceeding kappa
           OR at least one channel has a significant imbalance
        
        Args:
            node: Node to check eligibility for
            
        Returns:
            True if the node is eligible, False otherwise
        """
        if not node.rebalancing_requested:
            return False
            
        # Check if any channel meets the criteria
        for channel_id in node.local_balances:
            local = node.local_balances[channel_id]
            remote = node.remote_balances[channel_id]
            capacity = node.capacities[channel_id]
            
            # Check outgoing balance ratio
            if local / capacity >= self.kappa:
                return True
                
            # Check for significant imbalance
            if abs(local - remote) >= capacity * self.theta:
                return True
                
        return False
        
    def elect_leader(self, nodes: List[Node], timestamp: float) -> tuple[Optional[Node], float]:
        """
        Elect a leader from the list of nodes.
        
        The leader is chosen as the eligible node with the highest hash value.
        
        Args:
            nodes: List of nodes participating in election
            timestamp: Current timestamp
            
        Returns:
            Tuple of (elected leader node or None if no eligible leaders, election timestamp)
        """
        # Check if current leader is still valid
        if (self.current_leader is not None and 
            timestamp - self.leader_timeout < self.delta_t and
            self.is_eligible_leader(self.current_leader)):
            return self.current_leader, self.leader_timeout
            
        # Find eligible nodes
        eligible_nodes = [node for node in nodes if self.is_eligible_leader(node)]
        if not eligible_nodes:
            return None, timestamp
            
        # Compute hashes for eligible nodes
        node_hashes = [(node, self.compute_hash(node.id, timestamp)) 
                      for node in eligible_nodes]
        
        # Select node with highest hash
        leader_node, _ = max(node_hashes, key=lambda x: x[1])
        
        # Update leader state
        self.current_leader = leader_node
        self.leader_timeout = timestamp
        
        return leader_node, timestamp
        
    def announce_leader(self, leader: Node, timestamp: float) -> Dict[str, Any]:
        """
        Create a leader announcement message.
        
        Args:
            leader: Elected leader node
            timestamp: Election timestamp
            
        Returns:
            Dictionary containing announcement details
        """
        return {
            "leader_id": leader.id,
            "timestamp": timestamp,
            "hash": self.compute_hash(leader.id, timestamp)
        }
        
    def verify_announcement(self, announcement: Dict[str, Any], nodes: List[Node]) -> bool:
        """
        Verify a leader announcement message.
        
        Args:
            announcement: Leader announcement message
            nodes: List of nodes to verify against
            
        Returns:
            True if announcement is valid, False otherwise
        """
        leader_id = announcement["leader_id"]
        timestamp = announcement["timestamp"]
        hash_value = announcement["hash"]
        
        # Find leader node
        leader_node = next((node for node in nodes if node.id == leader_id), None)
        if leader_node is None:
            return False
            
        # Verify hash
        if hash_value != self.compute_hash(leader_id, timestamp):
            return False
            
        # Verify eligibility
        return self.is_eligible_leader(leader_node) 