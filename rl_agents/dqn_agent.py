import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture.
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    A Deep Q-Learning agent implementation.
    """
    def __init__(self, state_size, action_size, env,
                 buffer_size=10000, batch_size=64, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 learning_rate=5e-4, update_every=4):

        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.t_step = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size).to(self.device)
        self.qnetwork_target = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=buffer_size)

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use it to learn every `update_every` time steps.
        """
        self.memory.append((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self._sample_experiences()
                self._learn(experiences)

    def _sample_experiences(self):
        """
        Randomly sample a batch of experiences from replay memory.
        """
        return random.sample(self.memory, k=self.batch_size)

    def _learn(self, experiences):
        """
        Update Q-network parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters. θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state):
        """
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def train_agent(self, num_episodes=2000):
        """
        Trains the DQN agent over a number of episodes.
        """
        print(f"Starting DQN training for {num_episodes} episodes...")
        scores = []
        for i_episode in range(1, num_episodes + 1):
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
            scores.append(score)

            # Decrease epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

            if i_episode % 100 == 0:
                print(f"Episode {i_episode}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}, Epsilon: {self.epsilon:.4f}")
        print("DQN training complete!")

# Example usage with a dummy environment
if __name__ == "__main__":
    class DummyEnv:
        def __init__(self):
            self.state_space = 4
            self.action_space = 2
            self.current_state = np.zeros(self.state_space)

        def reset(self):
            self.current_state = np.random.rand(self.state_space)
            return self.current_state

        def step(self, action):
            reward = 1 if action == 0 else -1
            next_state = np.random.rand(self.state_space)
            done = random.random() < 0.1 # 10% chance to be done
            return next_state, reward, done, {}

    dummy_env = DummyEnv()
    agent = DQNAgent(state_size=dummy_env.state_space, action_size=dummy_env.action_space, env=dummy_env)
    agent.train_agent(num_episodes=500)
// Update on 2024-01-08 00:00:00 - 899
