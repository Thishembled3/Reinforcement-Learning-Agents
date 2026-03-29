import numpy as np
import random

class QLearningAgent:
    """
    A Q-learning agent implementation.
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay_rate=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.q_table = {}  # Q-table to store Q-values
        self._init_q_table()

    def _init_q_table(self):
        """
        Initializes the Q-table with zeros for all state-action pairs.
        """
        for state in self.env.get_all_states():
            self.q_table[state] = {action: 0.0 for action in self.env.get_actions(state)}

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.q_table[state].keys()))  # Explore
        else:
            return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def learn(self, state, action, reward, next_state):
        """
        Updates the Q-value for a given state-action pair using the Q-learning update rule.
        """
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

    def train(self, episodes=1000):
        """
        Trains the Q-learning agent for a specified number of episodes.
        """
        print(f"Starting Q-learning training for {episodes} episodes...")
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            # Decay epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay_rate

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        print("Q-learning training complete!")

    def evaluate(self, episodes=10):
        """
        Evaluates the trained Q-learning agent.
        """
        print(f"Starting Q-learning evaluation for {episodes} episodes...")
        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = max(self.q_table[state], key=self.q_table[state].get) # Greedy action
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}/{episodes}, Reward: {episode_reward}")
        print(f"Average evaluation reward: {np.mean(total_rewards):.2f}")


class SimpleGridWorld:
    """
    A simple grid world environment for testing RL agents.
    """
    def __init__(self, size=4, start=(0, 0), goal=(3, 3), pits=[(1, 1), (1, 3), (2, 0)]):
        self.size = size
        self.start = start
        self.goal = goal
        self.pits = pits
        self.current_state = start
        self.actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }

    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.current_state = self.start
        return self.current_state

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and whether the episode is done.
        """
        dr, dc = self.actions[action]
        next_row, next_col = self.current_state[0] + dr, self.current_state[1] + dc

        # Check boundaries
        if not (0 <= next_row < self.size and 0 <= next_col < self.size):
            return self.current_state, -1, False, {}

        next_state = (next_row, next_col)
        self.current_state = next_state

        reward = -0.1  # Small penalty for each step
        done = False

        if next_state == self.goal:
            reward = 10
            done = True
        elif next_state in self.pits:
            reward = -10
            done = True

        return next_state, reward, done, {}

    def get_all_states(self):
        """
        Returns a list of all possible states in the grid world.
        """
        states = []
        for r in range(self.size):
            for c in range(self.size):
                states.append((r, c))
        return states

    def get_actions(self, state):
        """
        Returns a list of possible actions from a given state.
        """
        return list(self.actions.keys())


if __name__ == "__main__":
    env = SimpleGridWorld()
    agent = QLearningAgent(env)
    agent.train(episodes=2000)
    agent.evaluate(episodes=5)
