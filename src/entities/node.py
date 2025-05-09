"""
Node class for the DEBAL system.
"""

import numpy as np
from src.learning.pytorch_soft_actor_critic.replay_memory import ReplayMemory
from src.learning.pytorch_soft_actor_critic.sac import SAC
from src.learning.pytorch_soft_actor_critic.utils import get_action

class Node:
    def __init__(self, node_id: str):
        """
        Initialize a node in the payment channel network.
        
        Args:
            node_id: Unique identifier for the node
        """
        self.id = node_id
        self.local_balances = {}  # {neighbor_id: balance}
        self.remote_balances = {}  # {neighbor_id: balance}
        self.capacities = {}      # {neighbor_id: capacity}
        self.fee_rates = {}       # {neighbor_id: fee_rate}
        self.balance_history = {} # {neighbor_id: [(time, local_balance, remote_balance)]}
        
        # Initialize DRL agent
        self.replay_memory = ReplayMemory(1000000)
        self.agent = SAC(
            state_dim=4,  # [local_balance, remote_balance, capacity, fee_rate]
            action_dim=1,  # [rebalancing_amount]
            action_range=(-1, 1)
        )
        
        # Node state
        self.debal_state = None
        self.leader_id = None
        self.election_timestamp = None
        self.rebalancing_requested = False
        self.needs_rebalancing = False
        
    def get_balance_ratio(self, channel_id: str) -> float:
        """
        Calculate the balance ratio for a channel.
        
        Args:
            channel_id: Channel identifier
            
        Returns:
            float: Balance ratio (local_balance/capacity)
        """
        if channel_id not in self.capacities:
            return 0.0
        return self.local_balances[channel_id] / self.capacities[channel_id]
        
    def get_channel_skewness(self, channel_id: str) -> float:
        """
        Calculate the skewness of a channel.
        
        Args:
            channel_id: Channel identifier
            
        Returns:
            float: Channel skewness (|local_balance - remote_balance|/capacity)
        """
        if channel_id not in self.capacities:
            return 0.0
        local = self.local_balances[channel_id]
        remote = self.remote_balances[channel_id]
        capacity = self.capacities[channel_id]
        return abs(local - remote) / capacity
        
    def get_total_outgoing_liquidity(self) -> float:
        """
        Calculate total outgoing liquidity.
        
        Returns:
            float: Sum of local balances
        """
        return sum(self.local_balances.values())
        
    def get_total_incoming_liquidity(self) -> float:
        """
        Calculate total incoming liquidity.
        
        Returns:
            float: Sum of remote balances
        """
        return sum(self.remote_balances.values())
        
    def request_rebalancing(self):
        """Request rebalancing for this node."""
        self.rebalancing_requested = True
        
    def clear_rebalancing_request(self):
        """Clear rebalancing request."""
        self.rebalancing_requested = False
        
    def update_balances(self, channel_id: str, local_delta: float, remote_delta: float) -> bool:
        """
        Update channel balances.
        
        Args:
            channel_id: Channel identifier
            local_delta: Change in local balance
            remote_delta: Change in remote balance
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if channel_id not in self.capacities:
            return False
            
        new_local = self.local_balances[channel_id] + local_delta
        new_remote = self.remote_balances[channel_id] + remote_delta
        
        # Check capacity constraints
        if new_local < 0 or new_remote < 0:
            return False
        if new_local + new_remote > self.capacities[channel_id]:
            return False
            
        self.local_balances[channel_id] = new_local
        self.remote_balances[channel_id] = new_remote
        
        # Record balance change in history
        if hasattr(self, 'env'):  # Check if we have access to simulation environment
            current_time = self.env.now
        else:
            current_time = max(t for t, _, _ in self.balance_history[channel_id]) + 1 if self.balance_history[channel_id] else 0
        self.balance_history[channel_id].append((current_time, new_local, new_remote))
        
        return True
        
    def add_channel(self, channel_id: str, local_balance: float, remote_balance: float, capacity: float):
        """
        Add a new payment channel.
        
        Args:
            channel_id: Channel identifier
            local_balance: Initial local balance
            remote_balance: Initial remote balance
            capacity: Channel capacity
        """
        self.local_balances[channel_id] = local_balance
        self.remote_balances[channel_id] = remote_balance
        self.capacities[channel_id] = capacity
        # Initialize balance history with initial balances at time 0
        self.balance_history[channel_id] = [(0, local_balance, remote_balance)]
        
    def remove_channel(self, channel_id: str):
        """
        Remove a payment channel.
        
        Args:
            channel_id: Channel identifier
        """
        if channel_id in self.local_balances:
            del self.local_balances[channel_id]
        if channel_id in self.remote_balances:
            del self.remote_balances[channel_id]
        if channel_id in self.capacities:
            del self.capacities[channel_id]
        if channel_id in self.fee_rates:
            del self.fee_rates[channel_id]
        if channel_id in self.balance_history:
            del self.balance_history[channel_id]
            
    def set_leader(self, leader_id: str, timestamp: float):
        """
        Set leader information.
        
        Args:
            leader_id: Leader node identifier
            timestamp: Election timestamp
        """
        self.leader_id = leader_id
        self.election_timestamp = timestamp
        
    def clear_leader(self):
        """Clear leader information."""
        self.leader_id = None
        self.election_timestamp = None
        
    def is_valid_channel(self, channel_id: str) -> bool:
        """
        Check if a channel is valid.
        
        Args:
            channel_id: Channel identifier
            
        Returns:
            bool: True if channel is valid, False otherwise
        """
        if channel_id not in self.capacities:
            return False
            
        local = self.local_balances[channel_id]
        remote = self.remote_balances[channel_id]
        capacity = self.capacities[channel_id]
        
        # Check basic constraints
        if local < 0 or remote < 0:
            return False
        if local + remote > capacity:
            return False
            
        return True
        
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the node for DEBAL.
        
        Returns:
            np.ndarray: State vector [local_balance_L, local_balance_R, remote_balance_L, remote_balance_R]
        """
        # Get channel states
        local_L = self.local_balances.get("L", 0)
        local_R = self.local_balances.get("R", 0)
        remote_L = self.remote_balances.get("L", 0)
        remote_R = self.remote_balances.get("R", 0)
        
        # Normalize by capacities
        capacity_L = self.capacities.get("L", 1)
        capacity_R = self.capacities.get("R", 1)
        
        return np.array([
            local_L / capacity_L,
            local_R / capacity_R,
            remote_L / capacity_L,
            remote_R / capacity_R
        ])
        
    def decide_rebalancing(self) -> bool:
        """
        Decide whether to request rebalancing using DEBAL.
        
        Returns:
            bool: True if rebalancing is needed, False otherwise
        """
        state = self.get_state()
        
        # Get current channel states
        local_L_ratio = state[0]
        local_R_ratio = state[1]
        remote_L_ratio = state[2]
        remote_R_ratio = state[3]
        
        # Calculate imbalance metrics
        L_imbalance = abs(local_L_ratio - remote_L_ratio)
        R_imbalance = abs(local_R_ratio - remote_R_ratio)
        
        # Print current state for debugging
        if hasattr(self, 'env') and hasattr(self.env, 'now'):
            print(f"\nTime {self.env.now}: Checking rebalancing")
            print(f"Channel L: Local={self.local_balances['L']}, Remote={self.remote_balances['L']}, Imbalance={L_imbalance:.2f}")
            print(f"Channel R: Local={self.local_balances['R']}, Remote={self.remote_balances['R']}, Imbalance={R_imbalance:.2f}")
        
        # DEBAL decision criteria
        # 1. Check if any channel is significantly imbalanced (>20% difference)
        if L_imbalance > 0.2 or R_imbalance > 0.2:
            if hasattr(self, 'env') and hasattr(self.env, 'now'):
                print(f"Time {self.env.now}: Channel imbalance detected")
            # Calculate rebalancing amount
            if L_imbalance > R_imbalance:
                # Rebalance L channel
                target_balance = (self.capacities["L"] * 0.5)  # Target 50% balance
                rebalance_amount = target_balance - self.local_balances["L"]
                self.perform_rebalancing("L", rebalance_amount)
            else:
                # Rebalance R channel
                target_balance = (self.capacities["R"] * 0.5)  # Target 50% balance
                rebalance_amount = target_balance - self.local_balances["R"]
                self.perform_rebalancing("R", rebalance_amount)
            return True
            
        # 2. Check if local balances are too low (<30% of capacity)
        if local_L_ratio < 0.3 or local_R_ratio < 0.3:
            if hasattr(self, 'env') and hasattr(self.env, 'now'):
                print(f"Time {self.env.now}: Low local balance detected")
            # Rebalance the channel with lower local balance
            if local_L_ratio < local_R_ratio:
                target_balance = (self.capacities["L"] * 0.4)  # Target 40% balance
                rebalance_amount = target_balance - self.local_balances["L"]
                self.perform_rebalancing("L", rebalance_amount)
            else:
                target_balance = (self.capacities["R"] * 0.4)  # Target 40% balance
                rebalance_amount = target_balance - self.local_balances["R"]
                self.perform_rebalancing("R", rebalance_amount)
            return True
            
        # 3. Check if remote balances are too low (<30% of capacity)
        if remote_L_ratio < 0.3 or remote_R_ratio < 0.3:
            if hasattr(self, 'env') and hasattr(self.env, 'now'):
                print(f"Time {self.env.now}: Low remote balance detected")
            # Rebalance the channel with lower remote balance
            if remote_L_ratio < remote_R_ratio:
                target_balance = (self.capacities["L"] * 0.6)  # Target 60% balance
                rebalance_amount = target_balance - self.local_balances["L"]
                self.perform_rebalancing("L", rebalance_amount)
            else:
                target_balance = (self.capacities["R"] * 0.6)  # Target 60% balance
                rebalance_amount = target_balance - self.local_balances["R"]
                self.perform_rebalancing("R", rebalance_amount)
            return True
            
        return False
        
    def perform_rebalancing(self, channel_id: str, amount: float):
        """
        Perform rebalancing on a channel.
        
        Args:
            channel_id: Channel identifier
            amount: Amount to rebalance (positive for increase, negative for decrease)
        """
        if channel_id not in self.capacities:
            return
            
        # Calculate new balances
        new_local = self.local_balances[channel_id] + amount
        new_remote = self.remote_balances[channel_id] - amount
        
        # Ensure balances stay within capacity
        if new_local < 0:
            new_local = 0
            new_remote = self.capacities[channel_id]
        elif new_local > self.capacities[channel_id]:
            new_local = self.capacities[channel_id]
            new_remote = 0
            
        # Update balances
        self.local_balances[channel_id] = new_local
        self.remote_balances[channel_id] = new_remote
        
        # Record in history
        if hasattr(self, 'env'):
            current_time = self.env.now
        else:
            current_time = max(t for t, _, _ in self.balance_history[channel_id]) + 1 if self.balance_history[channel_id] else 0
        self.balance_history[channel_id].append((current_time, new_local, new_remote))

    def _calculate_reward(self, state: np.ndarray, action: float, 
                         next_state: np.ndarray) -> float:
        """
        Calculate reward for DEBAL.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        # Calculate current and next imbalance
        current_imbalance = abs(state[0] - state[1]) + abs(state[2] - state[3])
        next_imbalance = abs(next_state[0] - next_state[1]) + abs(next_state[2] - next_state[3])
        
        # Reward based on imbalance reduction
        imbalance_improvement = current_imbalance - next_imbalance
        
        # Penalty for rebalancing cost
        rebalancing_cost = abs(action) * np.mean(list(self.fee_rates.values()))
        
        return imbalance_improvement - rebalancing_cost

    def get_balance_history(self, channel_id: str = None):
        """
        Get balance history for a specific channel or all channels.
        
        Args:
            channel_id: Channel identifier (optional)
            
        Returns:
            dict: Balance history for specified channel or all channels
        """
        if channel_id is not None:
            return self.balance_history.get(channel_id, [])
        return self.balance_history
