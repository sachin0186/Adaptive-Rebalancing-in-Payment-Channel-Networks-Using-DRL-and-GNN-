# Payment Channel Rebalancing Simulation Results

## Overview

This document summarizes the results of running simulations for different rebalancing policies in a payment channel network. The simulation models a relay node with two payment channels that forwards traffic and can perform rebalancing operations.

## Rebalancing Policies Tested

1. **None**: No rebalancing operations performed
2. **Autoloop**: A heuristic policy based on low and high thresholds
3. **Loopmax**: A policy based on expected demand that tries to rebalance infrequently with maximum amounts
4. **RebEL**: "Rebalancing Enabled by Learning" - A Deep Reinforcement Learning-based policy

## Simulation Parameters

- **Initial configuration**: 
  - Two channels with initial balance of 500 each
  - Channel capacity of 1000 each
  - On-chain budget of 1000
  - Total initial fortune: 2000

- **Traffic pattern**:
  - L→R: 60,000 transactions with rate of 10 per minute
  - R→L: 15,000 transactions with rate of 2.5 per minute
  - Payment amounts follow a Gaussian distribution

- **Fee structure**:
  - Proportional fee: 1%
  - Swap server fee: 0.5%
  - Miner fee: 2

## Key Findings

### 1. Profit Comparison

All rebalancing policies significantly outperformed the baseline (No Rebalancing):

| Policy   | Initial Fortune | Final Fortune | Profit | Profit Increase |
|----------|----------------|---------------|--------| --------------- |
| None     | 2000           | 2998          | 998    | Baseline        |
| Autoloop | 2000           | 8708          | 6708   | +572% over baseline |
| Loopmax  | 2000           | 7749          | 5749   | +476% over baseline |
| RebEL    | 2000           | 9263          | 7263   | +628% over baseline |

The reinforcement learning approach (RebEL) demonstrated the highest profitability, generating 7.26x more profit than without rebalancing.

### 2. Transaction Success Rate

| Policy   | Success Rate | Failed Transactions | Successful Transactions |
|----------|--------------|---------------------|-------------------------|
| None     | 8.2%         | 68,630              | 6,095                   |
| Autoloop | 59.1%        | 30,592              | 44,133                  |
| Loopmax  | 60.6%        | 29,458              | 45,267                  |
| RebEL    | 51.6%        | 36,186              | 38,539                  |

Loopmax achieved the highest success rate at 60.6%, followed closely by Autoloop at 59.1%. RebEL prioritized profit over success rate.

### 3. Rebalancing Operations

| Policy   | Number of Rebalancing Operations |
|----------|----------------------------------|
| None     | 0                                |
| Autoloop | 601                              |
| Loopmax  | 1198                             |
| RebEL    | 498                              |

RebEL performed the fewest rebalancing operations while achieving the highest profit, showing its efficiency. Loopmax performed the most rebalancing operations.

### 4. Channel Balance History

The balance history plots show how each policy maintained channel balance over time. Without rebalancing, the channels quickly became imbalanced, with the L channel depleting and the R channel accumulating funds.

The rebalancing policies actively maintained balance in the channels, with different strategies:

- **Autoloop**: Used fixed thresholds, resulting in regular rebalancing operations
- **Loopmax**: Performed more frequent rebalancing with larger amounts
- **RebEL**: Strategically timed rebalancing operations to maximize profit

## Conclusion

The simulation results clearly demonstrate that intelligent rebalancing policies can significantly improve the profitability of relay nodes in payment channel networks. The machine learning-based approach (RebEL) showed the best overall performance, generating the highest profit while requiring fewer rebalancing operations.

Key takeaways:
1. Rebalancing is essential for maintaining channel liquidity and maximizing profit
2. The RebEL policy's ability to learn and adapt to traffic patterns resulted in the most efficient use of rebalancing operations
3. While Loopmax achieved the highest success rate, RebEL optimized for overall profit
4. The baseline (No Rebalancing) quickly reached a state where most transactions failed 