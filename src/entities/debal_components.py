import numpy as np
from typing import Dict, List, Tuple

class DEBALNodeState:
    def __init__(self, theta: float = 0.35, tau: float = 1.5, epsilon: float = 0.001):
        self.theta = theta  # Minimum balance ratio threshold
        self.tau = tau      # TTD threshold in hours
        self.epsilon = epsilon  # Small constant to prevent division by zero
        
    def calculate_ttd(self, local_balances: Dict[str, float], 
                     outgoing_rate: float, incoming_rate: float) -> float:
        """
        Calculate Time To Depletion (TTD) based on current balances and transaction rates.
        
        Args:
            local_balances: Dictionary of local balances for each channel
            outgoing_rate: Rate of outgoing transactions
            incoming_rate: Rate of incoming transactions
            
        Returns:
            float: Estimated time until liquidity depletion in hours
        """
        total_local_balance = sum(local_balances.values())
        net_flow = max(outgoing_rate - incoming_rate, self.epsilon)
        return total_local_balance / net_flow
        
    def check_balance_ratios(self, local_balances: Dict[str, float], 
                           remote_balances: Dict[str, float], 
                           capacities: Dict[str, float]) -> bool:
        """
        Check if any channel violates the minimum balance ratio constraint.
        
        Args:
            local_balances: Dictionary of local balances
            remote_balances: Dictionary of remote balances
            capacities: Dictionary of channel capacities
            
        Returns:
            bool: True if any channel violates the constraint, False otherwise
        """
        for channel in local_balances.keys():
            local_ratio = local_balances[channel] / capacities[channel]
            remote_ratio = remote_balances[channel] / capacities[channel]
            min_ratio = min(local_ratio, remote_ratio)
            if min_ratio < self.theta:
                return True
        return False
        
    def calculate_skewness(self, local_balance: float, remote_balance: float, 
                          capacity: float) -> float:
        """
        Calculate channel skewness.
        
        Args:
            local_balance: Local balance of the channel
            remote_balance: Remote balance of the channel
            capacity: Channel capacity
            
        Returns:
            float: Skewness value between 0 and 1
        """
        return abs(local_balance - remote_balance) / capacity
        
    def should_request_rebalancing(self, local_balances: Dict[str, float],
                                 remote_balances: Dict[str, float],
                                 capacities: Dict[str, float],
                                 outgoing_rate: float,
                                 incoming_rate: float) -> bool:
        """
        Determine if rebalancing should be requested based on TTD and balance ratios.
        
        Args:
            local_balances: Dictionary of local balances
            remote_balances: Dictionary of remote balances
            capacities: Dictionary of channel capacities
            outgoing_rate: Rate of outgoing transactions
            incoming_rate: Rate of incoming transactions
            
        Returns:
            bool: True if rebalancing should be requested, False otherwise
        """
        # Check balance ratios
        if self.check_balance_ratios(local_balances, remote_balances, capacities):
            return True
            
        # Check TTD
        ttd = self.calculate_ttd(local_balances, outgoing_rate, incoming_rate)
        if ttd < self.tau:
            return True
            
        return False 