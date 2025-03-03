"""
This module provides functions to test a trained Q-learning agent in the FrozenLake environment.
It includes visualization of the agent's behavior using the learned policy.
"""

import numpy as np
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

from environment_setup import create_environment
from q_learning_agent import QLearningAgent


def test_agent(agent, env, num_episodes=10, max_steps=100, render=True, delay=0.3):
    """
    Test a trained agent in the environment and record performance metrics.
    
    Args:
        agent (QLearningAgent): The trained agent to test.
        env (gym.Env): The environment to test in.
        num_episodes (int): Number of episodes to test.
        max_steps (int): Maximum steps per episode.
        render (bool): Whether to render the environment during testing.
        delay (float): Delay between steps when rendering (seconds).
        
    Returns:
        float: Success rate as a percentage.
        float: Average number of steps to goal (for successful episodes).
        list: Steps taken in each episode.
    """
    successes = 0
    steps_list = []
    rewards_list = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        
        # Clear the display and show episode info
        if render:
            clear_output(wait=True)
            print(f"Test Episode: {episode+1}/{num_episodes}")
        
        while not done and steps < max_steps:
            # Always choose the best action according to the Q-table
            action = agent.get_best_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            episode_reward += reward
            
            # Render the environment 
            if render:
                env.render()
                time.sleep(delay)  # Pause to make movement visible
            
            # Update state
            state = next_state
        
        # Record results
        if episode_reward > 0:  # Successful episode (reached goal)
            successes += 1
            steps_list.append(steps)
        
        rewards_list.append(episode_reward)
        
        if render:
            print(f"Episode {episode+1} finished: {'Success' if episode_reward > 0 else 'Failure'}")
            print(f"Steps taken: {steps}")
            print(f"Total reward: {episode_reward}")
            print("\n")
            time.sleep(1)  # Pause between episodes
    
    # Calculate metrics
    success_rate = (successes / num_episodes) * 100
    avg_steps = np.mean(steps_list) if steps_list else 0
    
    # Print summary
    print(f"\n===== Testing Results =====")
    print(f"Success rate: {success_rate:.1f}%")
    if steps_list:
        print(f"Average steps to goal (successful episodes): {avg_steps:.1f}")
    else:
        print("No successful episodes")
    
    return success_rate, avg_steps, steps_list


def test_with_different_parameters(agent, hyperparameters, num_tests=5, max_steps=100):
    """
    Test the agent with different hyperparameter configurations.
    
    Args:
        agent (QLearningAgent): Base agent with a trained Q-table.
        hyperparameters (dict): Dictionary of hyperparameter settings to test.
        num_tests (int): Number of test episodes for each configuration.
        max_steps (int): Maximum steps per episode.
        
    Returns:
        dict: Dictionary of results for each configuration.
    """
    results = {}
    
    # Create environment (no rendering for batch testing)
    env, _, _ = create_environment(is_slippery=True)
    
    for name, params in hyperparameters.items():
        # Create a copy of the agent with the new parameters but same Q-table
        test_agent = QLearningAgent(
            state_space_size=agent.q_table.shape[0],
            action_space_size=agent.q_table.shape[1],
            **params
        )
        test_agent.q_table = agent.q_table.copy()  # Use the same learned Q-table
        
        print(f"Testing configuration: {name}")
        success_rate, avg_steps, steps_list = test_agent(
            test_agent, env, num_episodes=num_tests, max_steps=max_steps, render=False
        )
        
        results[name] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'steps_list': steps_list
        }
    
    return results


def compare_slippery_vs_non_slippery(agent, num_episodes=10, max_steps=100):
    """
    Compare agent performance in slippery vs. non-slippery environments.
    
    Args:
        agent (QLearningAgent): The trained agent to test.
        num_episodes (int): Number of episodes to test for each environment.
        max_steps (int): Maximum steps per episode.
        
    Returns:
        dict: Dictionary containing performance metrics for both environments.
    """
    results = {}
    
    # Test in slippery environment
    print("Testing in slippery environment...")
    env_slippery, _, _ = create_environment(is_slippery=True, render_mode="human")
    success_rate_slippery, avg_steps_slippery, _ = test_agent(
        agent, env_slippery, num_episodes=num_episodes, max_steps=max_steps
    )
    env_slippery.close()
    
    # Test in non-slippery environment
    print("\nTesting in non-slippery environment...")
    env_non_slippery, _, _ = create_environment(is_slippery=False, render_mode="human")
    success_rate_non_slippery, avg_steps_non_slippery, _ = test_agent(
        agent, env_non_slippery, num_episodes=num_episodes, max_steps=max_steps
    )
    env_non_slippery.close()
    
    # Store results
    results['slippery'] = {
        'success_rate': success_rate_slippery,
        'avg_steps': avg_steps_slippery
    }
    results['non_slippery'] = {
        'success_rate': success_rate_non_slippery,
        'avg_steps': avg_steps_non_slippery
    }
    
    # Compare results
    print("\n===== Environment Comparison =====")
    print(f"Slippery: {success_rate_slippery:.1f}% success, {avg_steps_slippery:.1f} avg steps")
    print(f"Non-slippery: {success_rate_non_slippery:.1f}% success, {avg_steps_non_slippery:.1f} avg steps")
    
    return results


def plot_test_results(results):
    """
    Plot the results of agent testing with different configurations.
    
    Args:
        results (dict): Dictionary of test results for different configurations.
    """
    config_names = list(results.keys())
    success_rates = [results[name]['success_rate'] for name in config_names]
    avg_steps = [results[name]['avg_steps'] for name in config_names]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot success rates
    ax1.bar(config_names, success_rates, color='skyblue')
    ax1.set_title('Success Rate by Configuration')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 105)
    
    # Plot average steps
    ax2.bar(config_names, avg_steps, color='salmon')
    ax2.set_title('Average Steps to Goal by Configuration')
    ax2.set_ylabel('Average Steps')
    
    # Add value labels on the bars
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 3, f"{v:.1f}%", ha='center')
    
    for i, v in enumerate(avg_steps):
        if not np.isnan(v):
            ax2.text(i, v + 0.5, f"{v:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("test_results_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Try to load a trained Q-table
    try:
        q_table = np.load("trained_q_table.npy")
        print("Loaded trained Q-table")
        
        # Create environment and agent
        env, state_space_size, action_space_size = create_environment(
            is_slippery=True, render_mode="human"
        )
        
        agent = QLearningAgent(
            state_space_size=state_space_size,
            action_space_size=action_space_size
        )
        agent.q_table = q_table
        
        # Test the agent
        print("Testing the trained agent...")
        success_rate, avg_steps, _ = test_agent(
            agent, env, num_episodes=5, max_steps=100, render=True, delay=0.3
        )
        
        # Close the environment
        env.close()
        
        # Optionally compare performance in different environments
        # compare_slippery_vs_non_slippery(agent, num_episodes=3)
        
        # Optionally test with different parameter settings
        # hyperparameters = {
        #     'Standard': {'learning_rate': 0.1, 'discount_factor': 0.99},
        #     'High Discount': {'learning_rate': 0.1, 'discount_factor': 0.99},
        #     'Low Discount': {'learning_rate': 0.1, 'discount_factor': 0.8},
        # }
        # results = test_with_different_parameters(agent, hyperparameters)
        # plot_test_results(results)
        
    except FileNotFoundError:
        print("No trained Q-table found. Please train an agent first by running training.py.") 