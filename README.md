# Reinforcement Learning with Python: Q-Learning in FrozenLake

This repository contains a complete implementation of Q-learning for the FrozenLake environment from OpenAI's Gymnasium. The code demonstrates how reinforcement learning agents learn through interaction with their environment, without explicit programming of the solution.

## Project Overview

The FrozenLake environment provides a simple grid world where an agent must navigate from a starting position to a goal while avoiding holes in the ice. The "slippery" version introduces stochasticity, as the agent's movements may not always result in the intended direction, making the learning task more challenging.

This implementation:
- Creates a Q-learning agent that learns optimal behavior through trial and error
- Tracks and visualizes the learning progress
- Provides tools to test the trained agent
- Creates animations showing the agent's behavior before and after training

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/reinforcement-learning-python.git
cd reinforcement-learning-python
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project is organized into modular components that can be run separately or together using the main script.

### Running the Complete Pipeline

To train an agent, test its performance, and generate visualizations:

```bash
python main.py --mode all
```

### Training

To just train the agent:

```bash
python main.py --mode train --episodes 5000
```

You can customize the training parameters:
```bash
python main.py --mode train --episodes 10000 --learning_rate 0.2 --discount_factor 0.95
```

### Testing

To test a previously trained agent:

```bash
python main.py --mode test --test_episodes 10
```

### Visualization

To generate visualizations for a trained agent:

```bash
python main.py --mode visualize
```

## Project Structure

- `environment_setup.py`: Creates and configures the FrozenLake environment
- `q_learning_agent.py`: Implements the Q-learning algorithm
- `training.py`: Contains functions for training the agent and tracking performance
- `visualization.py`: Provides tools to visualize the Q-table and create animations
- `testing.py`: Contains functions for testing the trained agent
- `main.py`: Combines all components for end-to-end execution

## Visualizations

This implementation generates several visualizations:

1. **Training Performance**: Plots showing success rate and average reward over time
2. **Q-Table Visualization**: A graphical representation of the learned Q-values
3. **Agent Path Animation**: A GIF showing how the agent navigates the environment
4. **Learning Progress**: An animation showing how the Q-table evolves during training

## Command Line Arguments

The main script accepts various command line arguments to customize the learning process:

```
--mode              Mode to run: 'train', 'test', 'visualize', or 'all'
--episodes          Number of episodes for training (default: 5000)
--slippery          Whether the environment is slippery (default: True)
--learning_rate     Learning rate (alpha) for the agent (default: 0.1)
--discount_factor   Discount factor (gamma) for future rewards (default: 0.99)
--exploration_rate  Initial exploration rate (epsilon) (default: 1.0)
--visualize_training Render the environment during training (flag)
--test_episodes     Number of episodes for testing (default: 5)
```

For a complete list of options:
```bash
python main.py --help
```

## Example Output

After training, the agent typically achieves a success rate of 70-90% in the slippery environment, with each successful episode taking approximately 15-25 steps to reach the goal. The visualizations show how the agent has learned to navigate around holes and find an optimal path to the goal.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Gymnasium](https://gymnasium.farama.org/) for providing the reinforcement learning environments
- [NumPy](https://numpy.org/) for numerical operations
- [Matplotlib](https://matplotlib.org/) for visualizations
- [imageio](https://imageio.readthedocs.io/) for creating animations 