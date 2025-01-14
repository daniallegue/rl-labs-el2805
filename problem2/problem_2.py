# Student 1: Daniel de Dios Allegue - 20041029-T074
# Student 2: Rodrigo Montero González - 20040421-T054
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from DDPG_soft_updates import soft_updates


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU()
        )
        self.concat_layer = nn.Sequential(
            nn.Linear(400 + action_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward2(self, state, action):
        state_out = self.state_layer(state)
        x = torch.cat([state_out, action], dim=1)
        return self.concat_layer(x)

    def forward(self, state, action):
        state_out = self.state_layer(state)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 2 and action.shape[0] != state_out.shape[0]:
            action = action.unsqueeze(1) if len(action.shape) == 1 else action
        x = torch.cat([state_out, action], dim=1)
        return self.concat_layer(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        action = self.net(state)
        return action.squeeze(0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# Soft Updates Function
def soft_updates2(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)


def prefill_buffer(env, buffer, buffer_size, action_dim):
    state, _ = env.reset()
    while len(buffer) < buffer_size:
        action = np.random.uniform(-1, 1, size=action_dim)
        next_state, reward, done, truncated, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state


def train_ddpg(env, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, buffer_capacity, params):
    gamma, tau, batch_size, noise_std, max_episodes, max_steps = params
    rewards = []
    steps_per_episode = []

    buffer = ReplayBuffer(buffer_capacity)
    prefill_buffer(env, buffer, buffer_size=buffer_capacity, action_dim=2)

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().flatten()
            action += np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
            next_state, reward, done, truncated, _ = env.step(action)

            buffer.push(state, action.flatten(), reward, next_state, done)

            if len(buffer) > batch_size:
                states, actions, rewards_batch, next_states, dones = buffer.sample(batch_size)
                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.float32)
                rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                target_actions = target_actor(next_states_tensor)
                target_q_values = target_critic(next_states_tensor, target_actions)
                y = rewards_tensor + gamma * target_q_values * (1 - dones_tensor)
                critic_loss = nn.MSELoss()(critic(states_tensor, actions_tensor), y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()

                actor_loss = -critic(states_tensor, actor(states_tensor)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                soft_updates(actor, target_actor, tau)
                soft_updates(critic, target_critic, tau)

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        rewards.append(total_reward)
        steps_per_episode.append(steps)
        print(f"Episode {episode + 1}/{max_episodes}, Reward: {total_reward:.2f}, Steps: {steps}")

    return rewards, steps_per_episode


def plot_training_results(rewards, steps_per_episode):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episodic Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode, label="Steps Per Episode", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps Per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()


def experiment_discount_factors2(env, state_dim, action_dim, params):
    discount_factors = [0.99, 1.0, 0.5]
    results = {}
    for gamma in discount_factors:
        print(f"Training with γ = {gamma}")
        # Reset actor and critic networks
        actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
        critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
        target_actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
        target_critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

        actor_optimizer = optim.Adam(actor.parameters(), lr=5e-5)
        critic_optimizer = optim.Adam(critic.parameters(), lr=5e-4)
        # Reset replay buffer
        buffer = ReplayBuffer(capacity=30000)


        # Update params with the current gamma
        current_params = (gamma, *params[1:])
        rewards, _ = train_ddpg(env, actor, critic, target_actor, target_critic,
                                actor_optimizer, critic_optimizer, buffer_capacity=30000, params=current_params)

        results[gamma] = rewards
        plt.plot(rewards, label=f"γ = {gamma}")
        torch.save(actor, f"neural-network-2-actor-gamma{gamma}.pth")
        torch.save(critic, f"neural-network-2-critic-gamma{gamma}.pth")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Effect of Discount Factor")
    plt.legend()
    plt.show()

    return results


def experiment_replay_buffer_sizes2(env, state_dim, action_dim, params):
    buffer_sizes = [15000, 50000]
    plt.figure()

    for buffer_capacity in buffer_sizes:
        print(f"Training with Buffer Size = {buffer_capacity}")
        # Reset actor and critic networks
        actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
        critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
        target_actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
        target_critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

        actor_optimizer = optim.Adam(actor.parameters(), lr=5e-5)
        critic_optimizer = optim.Adam(critic.parameters(), lr=5e-4)

        buffer = ReplayBuffer(buffer_capacity)

        rewards, _ = train_ddpg(env, actor, critic, target_actor, target_critic,
                                actor_optimizer, critic_optimizer, buffer_capacity, params)

        plt.plot(rewards, label=f"Buffer Size = {buffer_capacity}")
        torch.save(actor, f"neural-network-2-actor-bufcap{buffer_capacity}.pth")
        torch.save(critic, f"neural-network-2-critic-bufcap{buffer_capacity}.pth")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Effect of Replay Buffer Size")
    plt.legend()
    plt.show()


def visualize_q_values(actor, critic):
    y_vals = np.linspace(0, 1.5, 50)
    omega_vals = np.linspace(-np.pi, np.pi, 50)
    Y, Omega = np.meshgrid(y_vals, omega_vals)
    q_values = np.zeros_like(Y)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            state = np.array([0, Y[i, j], 0, 0, Omega[i, j], 0, 0, 0], dtype=np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor(state_tensor).detach().numpy()
            q_values[i, j] = critic(state_tensor, torch.tensor(action, dtype=torch.float32)).item()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, Omega, q_values, cmap='viridis')
    ax.set_xlabel("y (Height)")
    ax.set_ylabel("ω (Angle)")
    ax.set_zlabel("Q-value")
    plt.title("Q-value vs. Height and Angle")
    plt.show()


def visualize_engine_direction(actor):
    y_vals = np.linspace(0, 1.5, 50)
    omega_vals = np.linspace(-np.pi, np.pi, 50)
    Y, Omega = np.meshgrid(y_vals, omega_vals)
    engine_directions = np.zeros_like(Y)

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            state = np.array([0, Y[i, j], 0, 0, Omega[i, j], 0, 0, 0], dtype=np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().squeeze()
            engine_directions[i, j] = action[1]  # Second element of the action vector

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Y, Omega, engine_directions, cmap='viridis')
    ax.set_xlabel("y (Height)")
    ax.set_ylabel("ω (Angle)")
    ax.set_zlabel("Engine Direction")
    plt.title("Engine Direction vs. Height and Angle")
    plt.show()


def random_agent(env, episodes=50):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.random.uniform(-1, 1, size=env.action_space.shape[0])
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return total_rewards


def trained_agent(env, actor, episodes=50):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().flatten()
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    return total_rewards


def compare_agents(env, trained_actor, episodes=50):
    random_rewards = random_agent(env, episodes)
    trained_rewards = trained_agent(env, trained_actor, episodes)

    plt.figure(figsize=(10, 6))
    plt.plot(random_rewards, label="Random Agent", color='blue')
    plt.plot(trained_rewards, label="Trained Agent", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Comparison of Random Agent vs Trained Agent")
    plt.legend()
    plt.show()

    print(f"Average reward for Random Agent: {np.mean(random_rewards):.2f}")
    print(f"Average reward for Trained Agent: {np.mean(trained_rewards):.2f}")


def main():
    env = gym.make('LunarLanderContinuous-v3')

    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    gamma = 0.99
    tau = 1e-3
    actor_lr = 5e-5
    critic_lr = 5e-4
    max_episodes = 300
    max_steps = 300
    buffer_capacity = 30000
    batch_size = 64
    noise_std = 0.2

    actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
    critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
    target_actor = Actor(state_dim, action_dim).to(torch.device("cpu"))
    target_critic = Critic(state_dim, action_dim).to(torch.device("cpu"))
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    # Initialize replay buffer
    #buffer = ReplayBuffer(buffer_capacity)

    params = (gamma, tau, batch_size, noise_std, max_episodes, max_steps)
    rewards, steps_per_episode = train_ddpg(env, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, buffer_capacity, params)
    torch.save(actor, "neural-network-2-actor.pth")
    torch.save(critic, "neural-network-2-critic.pth")

    #plot training results
    plot_training_results(rewards, steps_per_episode)

    #experiment with discount factors
    experiment_discount_factors2(env, state_dim, action_dim, params)

    #experiment with replay buffer sizes
    experiment_replay_buffer_sizes2(env, state_dim, action_dim, params)

    actor = torch.load("neural-network-2-actor.pth")
    critic = torch.load("neural-network-2-critic.pth")
    #visualize Q-values
    visualize_q_values(actor, critic)

    visualize_engine_direction(actor)


    trained_actor = torch.load("neural-network-2-actor.pth")
    compare_agents(env, trained_actor, episodes=50)

    env.close()


if __name__ == "__main__":
    main()
