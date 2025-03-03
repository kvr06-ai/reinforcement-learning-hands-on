"""
This module handles the setup of the FrozenLake environment from OpenAI Gymnasium.
It provides functions to create and configure the environment for the Q-learning agent.
"""

import gymnasium as gym
import numpy as np


def create_environment(is_slippery=True, render_mode=None):
    """
    Creates and returns a FrozenLake environment.
    
    Args:
        is_slippery (bool): If True, the agent will slip on the ice with some probability.
        render_mode (str): The render mode for the environment (None, 'human', 'rgb_array').
    
    Returns:
        gym.Env: The initialized FrozenLake environment.
        int: Number of possible states in the environment.
        int: Number of possible actions in the environment.
    """
    # Create the FrozenLake environment
    env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode=render_mode)
    
    # Get the size of the state and action spaces
    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    
    print(f"Environment: FrozenLake-v1")
    print(f"States: {state_space_size}, Actions: {action_space_size}")
    print(f"Is slippery: {is_slippery}")
    
    if is_slippery:
        print("Note: In the slippery version, actions may not always result in the intended direction.")
    
    # Print a visual representation of the FrozenLake grid
    print("\nEnvironment layout:")
    print("S = Start, F = Frozen (safe), H = Hole, G = Goal\n")
    
    # Default 4x4 FrozenLake map
    lake_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    
    for row in lake_map:
        print(row)
    
    return env, state_space_size, action_space_size


if __name__ == "__main__":
    # Test the environment setup
    env, states, actions = create_environment(is_slippery=True, render_mode="human")
    
    # Reset the environment to get an initial state
    initial_state, _ = env.reset()
    print(f"\nInitial state: {initial_state}")
    
    # Take a random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Action taken: {action}")
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Episode terminated: {terminated}")
    
    # Close the environment
    env.close() 