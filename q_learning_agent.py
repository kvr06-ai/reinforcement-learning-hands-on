"""
This module implements a Q-learning agent for reinforcement learning tasks.
The agent learns by updating a Q-table based on experiences from interacting with
an environment.
"""

import numpy as np
import random


class QLearningAgent:
    """
    A Q-learning agent that learns to make decisions in an environment to maximize
    cumulative rewards.
    """
    
    def __init__(
        self, 
        state_space_size, 
        action_space_size, 
        learning_rate=0.1, 
        discount_factor=0.95, 
        exploration_rate=1.0, 
        min_exploration_rate=0.01, 
        exploration_decay_rate=0.001
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            state_space_size (int): Number of possible states in the environment.
            action_space_size (int): Number of possible actions in the environment.
            learning_rate (float): Alpha parameter - how quickly the agent updates Q-values.
            discount_factor (float): Gamma parameter - how much the agent values future rewards.
            exploration_rate (float): Epsilon parameter - probability of choosing a random action.
            min_exploration_rate (float): Minimum value for exploration rate.
            exploration_decay_rate (float): Rate at which exploration probability decreases.
        """
        # Initialize the Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        
        # Tracking variables
        self.training_episode = 0
    
    def choose_action(self, state, training=True):
        """
        Choose an action using an epsilon-greedy policy.
        
        Args:
            state (int): Current state of the environment.
            training (bool): If True, use exploration, otherwise always exploit.
            
        Returns:
            int: The chosen action.
        """
        if training and random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            return random.randint(0, self.q_table.shape[1] - 1)
        else:
            # Exploit: choose the action with highest Q-value
            # If multiple actions have the same highest value, randomly choose one of them
            actions_with_max_value = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
            return np.random.choice(actions_with_max_value)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for a state-action pair using the Q-learning formula.
        
        Args:
            state (int): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (int): Next state after taking action.
        """
        # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )
    
    def update_exploration_rate(self):
        """
        Decrease the exploration rate over time to shift from exploration to exploitation.
        """
        self.exploration_rate = self.min_exploration_rate + (
            self.max_exploration_rate - self.min_exploration_rate
        ) * np.exp(-self.exploration_decay_rate * self.training_episode)
        
        self.training_episode += 1
    
    def get_best_action(self, state):
        """
        Get the best action for a state based on current Q-values.
        
        Args:
            state (int): Current state.
            
        Returns:
            int: Action with the highest Q-value.
        """
        # If multiple actions have the same highest value, randomly choose one of them
        actions_with_max_value = np.where(self.q_table[state] == np.max(self.q_table[state]))[0]
        return np.random.choice(actions_with_max_value)
    
    def save_q_table(self, filename="q_table.npy"):
        """
        Save the Q-table to a file.
        
        Args:
            filename (str): Name of the file to save the Q-table.
        """
        np.save(filename, self.q_table)
        print(f"Q-table saved to {filename}")
    
    def load_q_table(self, filename="q_table.npy"):
        """
        Load the Q-table from a file.
        
        Args:
            filename (str): Name of the file to load the Q-table from.
        """
        try:
            self.q_table = np.load(filename)
            print(f"Q-table loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"File {filename} not found. Q-table not loaded.")
            return False


if __name__ == "__main__":
    # Simple test for the Q-learning agent
    agent = QLearningAgent(state_space_size=16, action_space_size=4)
    print("Initial Q-table:")
    print(agent.q_table)
    
    # Test choose_action
    state = 0
    action = agent.choose_action(state)
    print(f"Chosen action for state {state}: {action}")
    
    # Test update_q_value
    reward = 1.0
    next_state = 1
    agent.update_q_value(state, action, reward, next_state)
    print("Q-table after update:")
    print(agent.q_table) 