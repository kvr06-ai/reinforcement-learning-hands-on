"""
This module provides functions to visualize the Q-learning process, including:
1. Visualizing the Q-table as a heatmap
2. Creating animations of the agent's behavior during training
3. Plotting performance metrics like rewards and success rates
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.animation as animation
import gymnasium as gym
import imageio
import os
from tqdm import tqdm

from environment_setup import create_environment
from q_learning_agent import QLearningAgent


def visualize_q_table(q_table, save_path=None):
    """
    Create a visual representation of the Q-table showing action preferences at each state.
    
    Args:
        q_table (numpy.ndarray): The Q-table to visualize.
        save_path (str): Optional path to save the visualization.
    """
    # Assuming a 4x4 FrozenLake environment with 4 actions
    n_states = q_table.shape[0]
    grid_size = int(np.sqrt(n_states))  # Assuming square grid
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define the FrozenLake map (for reference)
    lake_map = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ]
    
    # Define colors for different cell types
    color_map = {
        'S': 'lightblue',  # Start
        'F': 'white',      # Frozen (safe)
        'H': 'black',      # Hole
        'G': 'green'       # Goal
    }
    
    # Define action names and directions
    action_names = ["Left", "Down", "Right", "Up"]
    action_arrows = ["←", "↓", "→", "↑"]
    action_coords = [(-0.2, 0), (0, -0.2), (0.2, 0), (0, 0.2)]  # Offsets for arrows
    
    # Plot the grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            cell_type = lake_map[i][j]
            cell_color = color_map[cell_type]
            
            # Create the grid cell
            rect = patches.Rectangle((j, grid_size-i-1), 1, 1, linewidth=1, 
                                    edgecolor='black', facecolor=cell_color, alpha=0.3)
            ax.add_patch(rect)
            
            # Skip adding arrows for terminal states (holes and goal)
            if cell_type == 'H' or cell_type == 'G':
                if cell_type == 'G':
                    ax.text(j+0.5, grid_size-i-0.5, "GOAL", ha='center', va='center', fontsize=12)
                elif cell_type == 'H':
                    ax.text(j+0.5, grid_size-i-0.5, "HOLE", ha='center', va='center', fontsize=12)
                continue
            
            # Get Q-values for this state
            q_values = q_table[state]
            best_action = np.argmax(q_values)
            
            # Add arrows for each action, with size proportional to Q-value
            max_q = np.max(np.abs(q_values))
            if max_q > 0:  # Avoid division by zero
                normalized_q = q_values / max_q
            else:
                normalized_q = q_values
            
            for action, (dx, dy), arrow in zip(range(4), action_coords, action_arrows):
                x = j + 0.5 + dx
                y = (grid_size - i - 1) + 0.5 + dy
                q_val = q_values[action]
                
                # Skip actions with very low Q-values
                if abs(q_val) < 0.01:
                    continue
                
                # Determine color (blue for positive, red for negative)
                arrow_color = 'blue' if q_val >= 0 else 'red'
                alpha = min(0.3 + abs(normalized_q[action]) * 0.7, 1.0)  # Scale opacity with Q-value
                
                # Make best action more prominent
                fontsize = 12 if action == best_action else 8
                weight = 'bold' if action == best_action else 'normal'
                
                ax.text(x, y, arrow, ha='center', va='center', fontsize=fontsize,
                       color=arrow_color, alpha=alpha, weight=weight)
                
                # Add small Q-value text
                small_dx = dx * 1.5
                small_dy = dy * 1.5
                ax.text(j+0.5+small_dx, (grid_size-i-1)+0.5+small_dy, f"{q_val:.2f}", 
                       ha='center', va='center', fontsize=6, alpha=0.7)
    
    # Add state numbers
    for i in range(grid_size):
        for j in range(grid_size):
            state = i * grid_size + j
            ax.text(j+0.1, grid_size-i-0.1, f"s{state}", fontsize=8, alpha=0.7)
    
    # Set axis properties
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size+1, 1))
    ax.set_yticks(np.arange(0, grid_size+1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    plt.title("Q-Table Visualization\nArrow Size: Q-Value Magnitude | Blue: Positive | Red: Negative")
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()


def create_agent_path_animation(env, agent, max_steps=100, filename="agent_path.gif"):
    """
    Create an animation showing the agent's path through the environment using the learned policy.
    
    Args:
        env (gym.Env): The environment to animate.
        agent (QLearningAgent): The trained agent.
        max_steps (int): Maximum steps to allow in the episode.
        filename (str): Path to save the animation file.
    """
    # Set up the environment in RGB rendering mode
    env_rgb, _, _ = create_environment(is_slippery=True, render_mode="rgb_array")
    
    # Reset the environment
    state, _ = env_rgb.reset()
    frames = [env_rgb.render()]
    
    done = False
    steps = 0
    
    # Get the agent's path
    while not done and steps < max_steps:
        # Choose the best action based on learned Q-values
        action = agent.get_best_action(state)
        
        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env_rgb.step(action)
        done = terminated or truncated
        
        # Render the environment
        frames.append(env_rgb.render())
        
        # Update state
        state = next_state
        steps += 1
    
    env_rgb.close()
    
    # Save frames as GIF
    imageio.mimsave(
        filename, 
        frames, 
        fps=3,
        loop=0  # 0 means loop indefinitely
    )
    
    print(f"Animation saved to {filename}")
    return filename


def visualize_learning_progress(q_tables, interval=500, filename="learning_progress.gif"):
    """
    Create an animation showing how the Q-table evolves during training.
    
    Args:
        q_tables (list): List of Q-tables at different stages of training.
        interval (int): The interval at which Q-tables were saved.
        filename (str): Path to save the animation file.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        q_table = q_tables[frame]
        
        # Assuming a 4x4 FrozenLake environment with 4 actions
        n_states = q_table.shape[0]
        grid_size = int(np.sqrt(n_states))  # Assuming square grid
        
        # Define the FrozenLake map (for reference)
        lake_map = [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ]
        
        # Define colors for different cell types
        color_map = {
            'S': 'lightblue',  # Start
            'F': 'white',      # Frozen (safe)
            'H': 'black',      # Hole
            'G': 'green'       # Goal
        }
        
        # Define action names and directions
        action_arrows = ["←", "↓", "→", "↑"]
        action_coords = [(-0.2, 0), (0, -0.2), (0.2, 0), (0, 0.2)]  # Offsets for arrows
        
        # Plot the grid cells
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j
                cell_type = lake_map[i][j]
                cell_color = color_map[cell_type]
                
                # Create the grid cell
                rect = patches.Rectangle((j, grid_size-i-1), 1, 1, linewidth=1, 
                                        edgecolor='black', facecolor=cell_color, alpha=0.3)
                ax.add_patch(rect)
                
                # Skip adding arrows for terminal states (holes and goal)
                if cell_type == 'H' or cell_type == 'G':
                    continue
                
                # Get Q-values for this state
                q_values = q_table[state]
                best_action = np.argmax(q_values)
                
                # Add arrows for each action, with size proportional to Q-value
                max_q = np.max(np.abs(q_values))
                if max_q > 0:  # Avoid division by zero
                    normalized_q = q_values / max_q
                else:
                    normalized_q = q_values
                
                for action, (dx, dy), arrow in zip(range(4), action_coords, action_arrows):
                    x = j + 0.5 + dx
                    y = (grid_size - i - 1) + 0.5 + dy
                    q_val = q_values[action]
                    
                    # Skip actions with very low Q-values
                    if abs(q_val) < 0.01:
                        continue
                    
                    # Determine color (blue for positive, red for negative)
                    arrow_color = 'blue' if q_val >= 0 else 'red'
                    alpha = min(0.3 + abs(normalized_q[action]) * 0.7, 1.0)  # Scale opacity with Q-value
                    
                    # Make best action more prominent
                    fontsize = 12 if action == best_action else 8
                    weight = 'bold' if action == best_action else 'normal'
                    
                    ax.text(x, y, arrow, ha='center', va='center', fontsize=fontsize,
                           color=arrow_color, alpha=alpha, weight=weight)
        
        # Set axis properties
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks(np.arange(0, grid_size+1, 1))
        ax.set_yticks(np.arange(0, grid_size+1, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_title(f"Q-Table Evolution: Episode {frame * interval}")
        
        return ax,
    
    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(q_tables), interval=500)
    
    # Save the animation
    anim.save(filename, writer='pillow', fps=2)
    plt.close(fig)
    
    print(f"Learning progress animation saved to {filename}")
    return filename


def visualize_heatmap(q_table, save_path=None):
    """
    Create a heatmap visualization of the Q-table, showing the maximum Q-value for each state.
    
    Args:
        q_table (numpy.ndarray): The Q-table to visualize.
        save_path (str): Optional path to save the visualization.
    """
    # Assuming a 4x4 FrozenLake environment
    max_q_values = np.max(q_table, axis=1).reshape(4, 4)
    
    # Create a custom colormap - red for negative, white for zero, blue for positive
    cmap = colors.LinearSegmentedColormap.from_list(
        'custom_diverging', ['red', 'white', 'blue'], N=256)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap
    im = plt.imshow(max_q_values, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, label="Max Q-Value")
    
    # Add grid
    plt.grid(which='major', color='black', linestyle='-', linewidth=2, alpha=0.7)
    
    # Add state numbers and best actions
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            best_action = np.argmax(q_table[state])
            action_symbols = ["←", "↓", "→", "↑"]
            
            # Skip holes in the default FrozenLake map
            if (i == 1 and j == 1) or (i == 1 and j == 3) or (i == 2 and j == 3) or (i == 3 and j == 0):
                plt.text(j, i, "H", ha="center", va="center", color="white", fontsize=16)
            # Mark the goal
            elif i == 3 and j == 3:
                plt.text(j, i, "G", ha="center", va="center", color="white", fontsize=16)
            # Mark the start
            elif i == 0 and j == 0:
                plt.text(j, i, "S", ha="center", va="center", color="black", fontsize=16)
            else:
                plt.text(j, i, action_symbols[best_action], ha="center", va="center", 
                        color="black", fontsize=16, fontweight="bold")
    
    # Add labels and title
    plt.title("Maximum Q-Value and Best Action for Each State")
    plt.xlabel("Column")
    plt.ylabel("Row")
    
    # Set tick positions and labels
    plt.xticks(np.arange(4), labels=np.arange(4))
    plt.yticks(np.arange(4), labels=np.arange(4))
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()


def collect_q_tables_during_training(num_episodes=5000, save_interval=500):
    """
    Train an agent and collect Q-tables at regular intervals to show learning progress.
    
    Args:
        num_episodes (int): Number of training episodes.
        save_interval (int): Interval at which to save Q-tables.
        
    Returns:
        list: List of Q-tables at different stages of training.
    """
    # Create environment and agent
    env, state_space_size, action_space_size = create_environment(is_slippery=True)
    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        min_exploration_rate=0.01,
        exploration_decay_rate=0.001
    )
    
    # Create a directory to save snapshots if it doesn't exist
    snapshot_dir = "q_table_snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # List to store Q-tables
    q_tables = []
    
    # Training loop
    print("Training agent and collecting Q-tables...")
    for episode in tqdm(range(num_episodes)):
        # Reset the environment
        state, _ = env.reset()
        done = False
        
        # Episode loop
        while not done:
            # Choose action using epsilon-greedy policy
            action = agent.choose_action(state, training=True)
            
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-value
            agent.update_q_value(state, action, reward, next_state)
            
            # Update state
            state = next_state
        
        # Update exploration rate
        agent.update_exploration_rate()
        
        # Save Q-table at specific intervals
        if episode % save_interval == 0 or episode == num_episodes - 1:
            # Make a copy of the Q-table (important to avoid reference issues)
            q_tables.append(agent.q_table.copy())
            
            # Save to file as well
            np.save(f"{snapshot_dir}/q_table_episode_{episode}.npy", agent.q_table)
    
    env.close()
    print(f"Collected {len(q_tables)} Q-tables during training.")
    return q_tables


if __name__ == "__main__":
    # Load a trained Q-table or create a new one
    try:
        q_table = np.load("trained_q_table.npy")
        print("Loaded existing Q-table")
        
        # Create an agent with the loaded Q-table
        env, state_space_size, action_space_size = create_environment(is_slippery=True)
        agent = QLearningAgent(state_space_size, action_space_size)
        agent.q_table = q_table
        
        # Visualize the Q-table
        visualize_q_table(q_table, save_path="q_table_visualization.png")
        
        # Create a heatmap visualization
        visualize_heatmap(q_table, save_path="q_value_heatmap.png")
        
        # Create an animation of the agent's path
        create_agent_path_animation(env, agent, filename="agent_path.gif")
        
    except FileNotFoundError:
        print("No trained Q-table found. Collecting Q-tables during training...")
        
        # Train agent and collect Q-tables to visualize learning progress
        q_tables = collect_q_tables_during_training(num_episodes=5000, save_interval=500)
        
        # Visualize the learning progress
        visualize_learning_progress(q_tables, interval=500, filename="learning_progress.gif")
        
        # Visualize the final Q-table
        visualize_q_table(q_tables[-1], save_path="final_q_table_visualization.png") 