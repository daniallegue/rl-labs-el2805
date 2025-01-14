# Student 1: Daniel de Dios Allegue - 20041029-T074
# Student 2: Rodrigo Montero González - 20040421-T054
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import random
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def running_average(x, N):
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the ReplayBuffer.

        Parameters:
            buffer_size (int): Maximum buffer size.
            batch_size (int): Batch size for sampling.
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode ended.
        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Neural Network for Q-value approximation.
    """

    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    """
    DQN Agent for interacting with and learning from the environment.
    """

    def __init__(self, env, gamma=0.99, epsilon=0.1, lr=1e-3, batch_size=64, buffer_size=10000, target_update_freq=20):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.target_update_freq = target_update_freq

        self.model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.target_model = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.update_target_model()

    def update_target_model(self):
        """
        Update the target network to match the current network.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def train(self):
        if self.buffer.size() < self.batch_size:
            return

        batch = self.buffer.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)


def plot_results(episode_rewards, running_avg_window, title="Training Results"):
    plt.figure(figsize=(16, 9))
    plt.plot(episode_rewards, label='Rewards')
    plt.plot(running_average(episode_rewards, running_avg_window), label=f'Running Average (Window={running_avg_window})')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()



def train_agent(env, episodes=700, gamma=0.9, epsilon=0.1, lr=0.0005, batch_size=16, buffer_size=10000,
                target_update_freq=200, reward_threshold=100, critical_reward=-3000, save_model=False):
    agent = DQNAgent(env, gamma, epsilon, lr, batch_size, buffer_size, target_update_freq)

    rewards = []
    steps = []

    for episode in trange(episodes, desc="Training"):
        state = env.reset()[0]
        done = False
        total_reward = 0
        total_steps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            total_steps += 1

            if total_reward < critical_reward:
                print(f"Run stopped early at episode {episode + 1} due to critical reward {total_reward:.2f}")
                break

        rewards.append(total_reward)
        steps.append(total_steps)




        # Update the target model at specified intervals
        if episode % agent.target_update_freq == 0:
            agent.update_target_model()

        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # Check running average reward and stop early if threshold is reached
        if reward_threshold is not None and len(rewards) >= 50:
            running_avg = np.mean(rewards[-50:])
            if running_avg >= reward_threshold:
                print(f"Stopping early at episode {episode + 1} with average reward {running_avg:.2f}")
                break

    # Save the trained model if required
    if save_model:
        torch.save(agent.model, 'neural-network-1.pth')
        print(f"Model saved")

    return rewards, steps


def grid_search(env, episodes, param_grid):
    import itertools

    keys, values = zip(*param_grid.items())
    configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = {}
    for idx, config in enumerate(configurations):
        print(f"Testing configuration {idx + 1}/{len(configurations)}: {config}")
        rewards, _ = train_agent(
            env,
            episodes=episodes,
            gamma=config['gamma'],
            epsilon=config['epsilon'],
            lr=config['lr'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            target_update_freq=config['target_update_freq'],
        )
        avg_reward = np.mean(rewards[-50:])  # Average reward over last 50 episodes
        print(f"Average reward for last 50 episodes was {avg_reward}")
        results[tuple(config.items())] = avg_reward

    return results


def visualize_q_values():
    model = torch.load('neural-network-1.pth')
    y_values = np.linspace(0, 1.5, 100)
    omega_values = np.linspace(-np.pi, np.pi, 100)
    max_q_values = np.zeros((len(y_values), len(omega_values)))
    optimal_actions = np.zeros_like(max_q_values, dtype=int)

    for i, y in enumerate(y_values):
        for j, omega in enumerate(omega_values):
            state = torch.tensor([0, y, 0, 0, omega, 0, 0, 0], dtype=torch.float32)
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
            max_q_values[i, j] = q_values.max().item()
            optimal_actions[i, j] = q_values.argmax().item()

    # Plot max_a Q(s(y, ω), a)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Y, Omega = np.meshgrid(y_values, omega_values)
    ax.plot_surface(Y, Omega, max_q_values.T, cmap='viridis')
    ax.set_xlabel('y')
    ax.set_ylabel('omega')
    ax.set_zlabel('max_a Q')
    plt.title('Max Q-values')
    plt.savefig('max_q_values.png')

    # Plot argmax_a Q(s(y, ω), a) - 2D Discrete Plot
    plt.figure()
    plt.imshow(optimal_actions.T, origin='lower', extent=(y_values.min(), y_values.max(), omega_values.min(), omega_values.max()), cmap='coolwarm', aspect='auto')
    plt.colorbar(ticks=np.unique(optimal_actions))
    plt.xlabel('y')
    plt.ylabel('omega')
    plt.title('Optimal Actions (Discrete)')
    plt.savefig('optimal_actions_discrete.png')

    # 3D Plot for argmax_a Q(s(y, ω), a)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = ListedColormap(['blue', 'orange', 'green', 'red'])
    ax.plot_surface(Y, Omega, optimal_actions.T, cmap=cmap, rstride=1, cstride=1, edgecolor='none')
    ax.set_xlabel('y')
    ax.set_ylabel('omega')
    ax.set_zlabel('Optimal Actions (Discrete)')
    plt.title('3D Plot of Optimal Actions')
    plt.savefig('optimal_actions_3d.png')


def compare_with_random(env):
    random_rewards = []
    trained_rewards = []

    # random
    for _ in range(50):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        random_rewards.append(total_reward)

    # trained
    model = torch.load('neural-network-1.pth')
    for _ in range(50):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        trained_rewards.append(total_reward)

    plt.figure()
    plt.plot(random_rewards, label='Random Agent')
    plt.plot(trained_rewards, label='Trained Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Total Rewards')
    plt.legend()
    plt.grid()
    plt.savefig('random_vs_trained.png')


def main():
    env = gym.make('LunarLander-v3')

    print("Starting task 1")
    episodes = 700
    gamma = 0.9
    rewards, steps = train_agent(env, episodes, gamma, save_model=True)
    plot_results(rewards, running_avg_window=50, title=f'Rewards over {episodes} Episodes (gamma={gamma})')

    print("Starting task 2")
    gammas = [1.0, 0.9, 0.5]
    for gamma in gammas:
        rewards, steps = train_agent(env, episodes, gamma)
        plot_results(rewards, running_avg_window=50,
                     title=f'Rewards with Gamma={gamma}')
        plt.savefig(f'rewards_gamma_{gamma}.png')

    print("Starting task 3")
    episode_variants = [200, 700, 1000]
    memory_sizes = [5000, 10000, 20000]
    for ep in episode_variants:
        for mem_size in memory_sizes:
            rewards, steps = train_agent(env, ep, gamma=0.9, buffer_size=mem_size)
            plot_results(rewards, running_avg_window=50,
                         title=f'Task 3: {ep} Episodes, Buffer Size={mem_size}')
            plt.savefig(f'rewards_eps_{ep}_mem_{mem_size}.png')

    print("Starting task f")
    visualize_q_values()

    print("Starting task g")
    compare_with_random(env)

    print("All tasks completed")



if __name__ == "__main__":
    main()


# TODO these were the models that reached some of the best results
"""
{'gamma': 0.9, 'epsilon': 0.1, 'lr': 0.0005, 'batch_size': 8, 'buffer_size': 10000, 'target_update_freq': 200} Reward: 59.13, 506 episodes
{'gamma': 0.9, 'epsilon': 0.1, 'lr': 0.0005, 'batch_size': 16, 'buffer_size': 10000, 'target_update_freq': 200} Reward: 52.66, 276 episodes
{'gamma': 0.9, 'epsilon': 0.5, 'lr': 0.0001, 'batch_size': 16, 'buffer_size': 10000, 'target_update_freq': 200} Reward: 52.97, 530 episodes
{'gamma': 0.9, 'epsilon': 0.5, 'lr': 0.0005, 'batch_size': 8, 'buffer_size': 10000, 'target_update_freq': 200} Reward 52.53, 493 episodes
{'gamma': 0.9, 'epsilon': 0.5, 'lr': 0.0005, 'batch_size': 16, 'buffer_size': 10000, 'target_update_freq': 200} Reward: 53.28, 518 episodes

"""