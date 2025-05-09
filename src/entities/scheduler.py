import simpy
from typing import List, Optional
from src.entities.node import Node
from src.entities.leader_election import LeaderElection
from src.entities.rebalancing_engine import RebalancingEngine

class RebalancingScheduler:
    def __init__(self, env: simpy.Environment, nodes: List[Node],
                 leader_election: LeaderElection,
                 rebalancing_engine: RebalancingEngine,
                 delta_t: float = 600.0):  # 10 minutes default
        """
        Initialize the rebalancing scheduler.
        
        Args:
            env: SimPy environment
            nodes: List of nodes in the network
            leader_election: Leader election component
            rebalancing_engine: Rebalancing engine component
            delta_t: Time interval between re-elections
        """
        self.env = env
        self.nodes = nodes
        self.delta_t = delta_t
        self.leader_election = leader_election
        self.rebalancing_engine = rebalancing_engine
        self.current_leader = None
        self.last_election_time = 0
        
        # Initialize rebalancing engine with nodes
        self.rebalancing_engine.nodes = {node.id: node for node in nodes}
        
    def run(self):
        """
        Run the scheduler process.
        """
        while True:
            # Check if re-election is needed
            if self.should_trigger_election():
                self.perform_election()
            
            # If we have a leader and pending requests, trigger rebalancing
            if self.current_leader and any(node.rebalancing_requested for node in self.nodes):
                self.trigger_rebalancing()
            
            # Wait for next cycle
            yield self.env.timeout(self.delta_t / 10)  # Check more frequently
                
    def should_trigger_election(self) -> bool:
        """
        Check if re-election should be triggered.
        
        Returns:
            bool: True if re-election should be triggered, False otherwise
        """
        current_time = self.env.now
        
        # Always trigger election if we have no leader and there are pending requests
        if not self.current_leader and any(node.rebalancing_requested for node in self.nodes):
            return True
            
        # Check if current leader is still valid
        if self.current_leader:
            if not self.leader_election.is_eligible_leader(self.current_leader):
                return True
            if current_time - self.last_election_time >= self.delta_t:
                return True
                
        # Check if any node has requested rebalancing and we have no leader
        if not self.current_leader and any(node.rebalancing_requested for node in self.nodes):
            return True
            
        return False
        
    def perform_election(self):
        """
        Perform leader election and handle leader transition.
        """
        # Get timestamp for election
        timestamp = self.env.now
        
        # Perform election
        new_leader, election_time = self.leader_election.elect_leader(self.nodes, timestamp)
        
        if new_leader:
            # Create and broadcast announcement
            announcement = self.leader_election.announce_leader(new_leader, election_time)
            
            # Verify announcement
            if self.leader_election.verify_announcement(announcement, self.nodes):
                # Handle leader transition
                if self.current_leader and self.current_leader != new_leader:
                    self.handle_leader_transition(self.current_leader, new_leader)
                    
                # Update leader state
                self.current_leader = new_leader
                self.last_election_time = election_time
                
    def trigger_rebalancing(self):
        """
        Trigger rebalancing for nodes that have requested it.
        """
        for node in self.nodes:
            if node.rebalancing_requested:
                # Find imbalanced channels
                imbalanced_channels = []
                for peer_id, local_balance in node.local_balances.items():
                    remote_balance = node.remote_balances[peer_id]
                    capacity = node.capacities[peer_id]
                    
                    # Calculate skewness
                    skewness = abs(local_balance - remote_balance) / capacity
                    if skewness > 0.2:  # If channel is significantly imbalanced
                        imbalanced_channels.append({
                            'peer_id': peer_id,
                            'local_balance': local_balance,
                            'remote_balance': remote_balance,
                            'capacity': capacity,
                            'skewness': skewness
                        })
                
                if imbalanced_channels:
                    # Sort channels by skewness
                    imbalanced_channels.sort(key=lambda x: x['skewness'], reverse=True)
                    
                    # Try to rebalance each channel
                    for channel in imbalanced_channels:
                        peer_id = channel['peer_id']
                        local_balance = channel['local_balance']
                        remote_balance = channel['remote_balance']
                        capacity = channel['capacity']
                        
                        # Calculate target balance that maintains 20% minimum
                        if local_balance > remote_balance:
                            # We want to move funds from local to remote
                            # Calculate maximum transfer that keeps both balances above 20%
                            max_transfer = min(
                                local_balance - (0.2 * capacity),  # Keep local above 20%
                                (0.8 * capacity) - remote_balance  # Keep remote below 80%
                            )
                            if max_transfer > 0:
                                path = [node.id, peer_id]
                                if self.rebalancing_engine.execute_transfer(path, max_transfer):
                                    print(f"Rebalancing successful: {path} with amount {max_transfer}")
                                    node.rebalancing_requested = False
                                    break
                                else:
                                    print(f"Rebalancing failed: {path} with amount {max_transfer}")
                        else:
                            # We want to move funds from remote to local
                            # Calculate maximum transfer that keeps both balances above 20%
                            max_transfer = min(
                                remote_balance - (0.2 * capacity),  # Keep remote above 20%
                                (0.8 * capacity) - local_balance   # Keep local below 80%
                            )
                            if max_transfer > 0:
                                path = [peer_id, node.id]
                                if self.rebalancing_engine.execute_transfer(path, max_transfer):
                                    print(f"Rebalancing successful: {path} with amount {max_transfer}")
                                    node.rebalancing_requested = False
                                    break
                                else:
                                    print(f"Rebalancing failed: {path} with amount {max_transfer}")
                    
    def handle_leader_transition(self, old_leader: Node, new_leader: Node):
        """
        Handle transition from old leader to new leader.
        
        Args:
            old_leader: Previous leader node
            new_leader: New leader node
        """
        # Update node states
        for node in self.nodes:
            node.leader_id = new_leader.id
            node.election_timestamp = self.env.now 