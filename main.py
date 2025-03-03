"""
Main script for the Q-Learning tutorial that combines all components:
1. Setting up the environment
2. Creating and training a Q-learning agent
3. Visualizing the Q-table and learning progress
4. Testing the trained agent
"""

import argparse
import numpy as np
import os
import time

from environment_setup import create_environment
from q_learning_agent import QLearningAgent
from training import train_agent, plot_training_results
from visualization import (
    visualize_q_table, 
    create_agent_path_animation, 
    visualize_heatmap,
    collect_q_tables_during_training,
    visualize_learning_progress
)
from testing import test_agent, compare_slippery_vs_non_slippery


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Q-Learning in FrozenLake Environment')
    
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'visualize', 'all'],
                        help='Mode to run the script in')
    
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes for training')
    
    parser.add_argument('--slippery', type=bool, default=True,
                        help='Whether the environment is slippery')
    
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (alpha) for the agent')
    
    parser.add_argument('--discount_factor', type=float, default=0.99,
                        help='Discount factor (gamma) for future rewards')
    
    parser.add_argument('--exploration_rate', type=float, default=1.0,
                        help='Initial exploration rate (epsilon)')
    
    parser.add_argument('--min_exploration_rate', type=float, default=0.01,
                        help='Minimum exploration rate')
    
    parser.add_argument('--exploration_decay', type=float, default=0.001,
                        help='Rate at which exploration probability decreases')
    
    parser.add_argument('--visualize_training', action='store_true',
                        help='Whether to render the environment during training')
    
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of episodes for testing')
    
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    
    parser.add_argument('--render_delay', type=float, default=0.3,
                        help='Delay between rendered frames (seconds)')
    
    parser.add_argument('--q_table_path', type=str, default='trained_q_table.npy',
                        help='Path to save/load the Q-table')
    
    return parser.parse_args()


def train_mode(args):
    """Train a Q-learning agent and save results."""
    print("===== TRAINING MODE =====")
    
    # Create environment
    env, state_space_size, action_space_size = create_environment(
        is_slippery=args.slippery,
        render_mode="human" if args.visualize_training else None
    )
    
    # Create agent
    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        exploration_rate=args.exploration_rate,
        min_exploration_rate=args.min_exploration_rate,
        exploration_decay_rate=args.exploration_decay
    )
    
    # Train the agent
    print(f"Starting training for {args.episodes} episodes...")
    rewards, successes = train_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        progress_interval=max(1, args.episodes // 10),
        render_training=args.visualize_training
    )
    
    # Plot training results
    plot_training_results(rewards, successes)
    
    # Save the trained agent's Q-table
    agent.save_q_table(args.q_table_path)
    
    # Close the environment
    env.close()
    
    print(f"Training complete. Q-table saved to '{args.q_table_path}'.")
    return agent


def test_mode(args):
    """Test a trained Q-learning agent."""
    print("===== TESTING MODE =====")
    
    try:
        # Load the Q-table
        q_table = np.load(args.q_table_path)
        print(f"Loaded Q-table from '{args.q_table_path}'")
        
        # Create environment
        env, state_space_size, action_space_size = create_environment(
            is_slippery=args.slippery,
            render_mode="human"  # Always render in test mode
        )
        
        # Create agent with loaded Q-table
        agent = QLearningAgent(
            state_space_size=state_space_size,
            action_space_size=action_space_size
        )
        agent.q_table = q_table
        
        # Test the agent
        print(f"Testing agent for {args.test_episodes} episodes...")
        success_rate, avg_steps, _ = test_agent(
            agent=agent,
            env=env,
            num_episodes=args.test_episodes,
            max_steps=args.max_steps,
            render=True,
            delay=args.render_delay
        )
        
        # Close the environment
        env.close()
        
        print("Testing complete.")
        return agent
    
    except FileNotFoundError:
        print(f"Error: Could not find Q-table at '{args.q_table_path}'.")
        print("Please train an agent first or specify the correct path.")
        return None


def visualize_mode(args):
    """Visualize a trained Q-learning agent."""
    print("===== VISUALIZATION MODE =====")
    
    try:
        # Load the Q-table
        q_table = np.load(args.q_table_path)
        print(f"Loaded Q-table from '{args.q_table_path}'")
        
        # Create environment and agent
        env, state_space_size, action_space_size = create_environment(
            is_slippery=args.slippery
        )
        agent = QLearningAgent(state_space_size, action_space_size)
        agent.q_table = q_table
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Visualize the Q-table
        print("1. Generating Q-table visualization...")
        visualize_q_table(q_table, save_path="q_table_visualization.png")
        
        # Create a heatmap
        print("2. Generating Q-value heatmap...")
        visualize_heatmap(q_table, save_path="q_value_heatmap.png")
        
        # Create an animation of the agent's path
        print("3. Generating agent path animation...")
        env_render, _, _ = create_environment(
            is_slippery=args.slippery,
            render_mode="rgb_array"
        )
        create_agent_path_animation(env_render, agent, filename="agent_path.gif")
        env_render.close()
        
        print("Visualizations complete.")
        return agent
    
    except FileNotFoundError:
        print(f"Error: Could not find Q-table at '{args.q_table_path}'.")
        print("Please train an agent first or specify the correct path.")
        
        # Offer to run training to collect visualizations
        response = input("Would you like to train a new agent and collect visualizations? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("Collecting Q-tables during training for visualization...")
            q_tables = collect_q_tables_during_training(
                num_episodes=args.episodes,
                save_interval=max(1, args.episodes // 10)
            )
            visualize_learning_progress(q_tables, interval=max(1, args.episodes // 10))
            visualize_q_table(q_tables[-1], save_path="final_q_table_visualization.png")
        
        return None


def all_mode(args):
    """Run training, testing, and visualization in sequence."""
    print("===== ALL MODES =====")
    
    # Train the agent
    agent = train_mode(args)
    
    # Add a short pause to let the user see the training results
    print("\nTraining complete. Moving to testing phase in 3 seconds...")
    time.sleep(3)
    
    # Test the trained agent
    test_mode(args)
    
    # Add a short pause
    print("\nTesting complete. Moving to visualization phase in 3 seconds...")
    time.sleep(3)
    
    # Visualize the agent
    visualize_mode(args)
    
    print("\nAll phases complete!")


def main():
    """Main function to run the Q-learning tutorial."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs("q_table_snapshots", exist_ok=True)
    
    # Run the selected mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'visualize':
        visualize_mode(args)
    elif args.mode == 'all':
        all_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")


if __name__ == "__main__":
    main() 