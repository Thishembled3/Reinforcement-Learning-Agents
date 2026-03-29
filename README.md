# Reinforcement-Learning-Agents

A collection of Reinforcement Learning (RL) agents implemented in Python, showcasing various algorithms like Q-learning, SARSA, and policy gradients, with environments for testing.

## Features
- **Multiple RL Algorithms**: Implementations of Q-learning, SARSA, DQN, A2C, and more.
- **Modular Design**: Easy to extend with new algorithms and environments.
- **OpenAI Gym Integration**: Compatible with popular OpenAI Gym environments.
- **Visualization Tools**: Tools for visualizing agent performance and learning progress.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
from rl_agents.q_learning import QLearningAgent
from rl_agents.environments import SimpleGridWorld

# Create an environment
env = SimpleGridWorld()

# Initialize and train a Q-learning agent
agent = QLearningAgent(env)
agent.train(episodes=1000)

# Evaluate the agent
agent.evaluate(episodes=10)
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
