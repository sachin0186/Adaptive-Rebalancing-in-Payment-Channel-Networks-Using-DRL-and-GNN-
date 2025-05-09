"""
Replay memory for Soft Actor-Critic.
"""

import os
import pickle
import random
import numpy as np
import torch
from collections import deque


class ReplayMemory:
    def __init__(self, capacity):
        """
        Initialize replay memory.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from memory.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return (torch.FloatTensor(state),
                torch.FloatTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))

    def __len__(self):
        """
        Get current size of memory.
        
        Returns:
            int: Number of transitions in memory
        """
        return len(self.memory)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.memory = pickle.load(f)
