"""
This module handles the training process for the Q-learning agent in the FrozenLake environment.
It includes functions to train the agent over multiple episodes and track its performance.
"""

import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from environment_setup import create_environment
from q_learning_agent import QLearningAgent


def train_agent(
    agent, 
    env, 
    num_episodes=10000, 
    max_steps_per_episode=100, 
    progress_interval=1000,
    render_training=False,
    render_interval=None
):
    """
    Train the Q-learning agent in the given environment.
    
    Args:
        agent (QLearningAgent): The agent to train.
        env (gym.Env): The environment to train in.
        num_episodes (int): Number of episodes to train for.
        max_steps_per_episode (int): Maximum steps per episode before termination.
        progress_interval (int): How often to print progress (in episodes).
        render_training (bool): Whether to render the environment during training.
        render_interval (int): If rendering, how often to render (in episodes).
        
    Returns:
        list: Rewards for each episode.
        list: Success rate over time (1 if reached goal, 0 otherwise).
    """
    # Lists to track rewards and success rates
    rewards_all_episodes = []
    success_all_episodes = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        # Reset the environment for a new episode
        state, _ = env.reset()
        done = False
        current_reward = 0
        
        # Render first frame if requested
        if render_training and (render_interval is None or episode % render_interval == 0):
            env.render()
            time.sleep(0.1)
        
        # Loop through each step in the episode
        for step in range(max_steps_per_episode):
            # Choose action using epsilon-greedy policy
            action = agent.choose_action(state, training=True)
            
            # Take action and observe new state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-table using the Q-learning formula
            agent.update_q_value(state, action, reward, new_state)
            
            # Render frame if requested
            if render_training and (render_interval is None or episode % render_interval == 0):
                env.render()
                time.sleep(0.1)
            
            # Update state and accumulate reward
            state = new_state
            current_reward += reward
            
            # Exit loop if episode is finished
            if done:
                break
        
        # Update exploration rate
        agent.update_exploration_rate()
        
        # Track rewards and success
        rewards_all_episodes.append(current_reward)
        success_all_episodes.append(1 if current_reward > 0 else 0)
        
        # Print progress update
        if episode % progress_interval == 0 and episode > 0:
            # Calculate success rate in recent episodes
            recent_success_rate = np.mean(success_all_episodes[-progress_interval:]) * 100
            recent_reward = np.mean(rewards_all_episodes[-progress_interval:])
            
            clear_output(wait=True)
            print(f"Episode: {episode}/{num_episodes}")
            print(f"Recent success rate: {recent_success_rate:.2f}%")
            print(f"Recent average reward: {recent_reward:.4f}")
            print(f"Exploration rate: {agent.exploration_rate:.4f}")
    
    return rewards_all_episodes, success_all_episodes


def calculate_moving_average(data, window_size=100):
    """
    Calculate the moving average of a list of values.
    
    Args:
        data (list): List of values to calculate moving average for.
        window_size (int): Size of the moving window.
        
    Returns:
        list: Moving averages of the data.
    """
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window_average = np.mean(data[i:i+window_size])
        moving_averages.append(window_average)
    return moving_averages


def plot_training_results(rewards, successes, window_size=100):
    """
    Plot the results of training.
    
    Args:
        rewards (list): Rewards for each episode.
        successes (list): Success indicator for each episode (1 for success, 0 for failure).
        window_size (int): Size of the moving window for averaging.
    """
    # Calculate moving averages
    if len(rewards) > window_size:
        moving_avg_rewards = calculate_moving_average(rewards, window_size)
        moving_avg_success = calculate_moving_average(successes, window_size)
        x_values = range(window_size - 1, len(rewards))
    else:
        moving_avg_rewards = [np.mean(rewards)]
        moving_avg_success = [np.mean(successes)]
        x_values = [len(rewards) // 2]
    
    plt.figure(figsize=(12, 5))
    
    # Plot success rate
    plt.subplot(1, 2, 1)
    plt.plot(x_values, moving_avg_success)
    plt.title(f"Success Rate (Moving Avg, Window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(x_values, moving_avg_rewards)
    plt.title(f"Average Reward (Moving Avg, Window={window_size})")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()


if __name__ == "__main__":
    # Create environment
    env, state_space_size, action_space_size = create_environment(is_slippery=True)
    
    # Create agent with custom hyperparameters
    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,         # How quickly the agent updates its Q-values
        discount_factor=0.99,      # How much the agent values future rewards
        exploration_rate=1.0,      # Initial exploration rate (1.0 = 100% random actions)
        min_exploration_rate=0.01, # Minimum exploration rate
        exploration_decay_rate=0.001  # How quickly to decrease exploration
    )
    
    # Train the agent
    print("Starting training...")
    rewards, successes = train_agent(
        agent=agent,
        env=env,
        num_episodes=5000,
        max_steps_per_episode=100,
        progress_interval=500,
        render_training=False
    )
    
    # Plot results
    plot_training_results(rewards, successes, window_size=100)
    
    # Save the trained agent's Q-table
    agent.save_q_table("trained_q_table.npy")
    
    # Close the environment
    env.close()
    
    print("Training complete. Results saved to 'training_results.png'.") 