# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.
# Student 1: Daniel de Dios Allegue - 20041029-T074
# Student 2: Rodrigo Montero González - 20040421-T054

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import itertools
import seaborn as sns

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n
low, high = env.observation_space.low, env.observation_space.high

# Parameters
N_episodes = 100        # Number of episodes to run for training
discount_factor = 1.    # Value of gamma


def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def scale_state_variables(s, low, high):
    """Normalize state to [0, 1]^2."""
    return (s - low) / (high - low)


def fourier_features(s, eta):
    """Compute Fourier basis features."""
    return np.cos(np.pi * np.dot(eta, s))


def select_action(Q, epsilon, strategy='epsilon_greedy'):
    if strategy == 'epsilon_greedy':
        return np.random.randint(0, k) if np.random.rand() < epsilon else np.argmax(Q)
    elif strategy == 'boltzmann':
        tau = 0.5 # Hyperparamater that needs tuning
        exp_Q = np.exp(Q / tau)
        probs = exp_Q / np.sum(exp_Q)
        return np.random.choice(np.arange(k), p=probs)

def sarsa_lambda(env, eta, N_episodes, alpha, gamma, lambda_, epsilon, init_strategy='zero', exploration_strategy='epsilon_greedy'):
    k = env.action_space.n  # Number of actions
    low, high = env.observation_space.low, env.observation_space.high
    m = eta.shape[0]  # Number of features
    W = np.zeros((k, m))  # Weight matrix for Q-values
    if init_strategy == 'zero':
        W = np.zeros((k, m))  # Zero initialization
    elif init_strategy == 'optimistic':
        W = np.full((k, m), 10.0)  # Optimistic initialization
    elif init_strategy == 'random':
        W = np.random.uniform(-0.01, 0.01, size=(k, m))  # Random initialization
    episode_reward_list = []

    v = np.zeros_like(W)
    m = 0.9

    for episode in range(N_episodes):
        state, _ = env.reset()
        state = scale_state_variables(state, low, high)
        features = fourier_features(state, eta)

        # Initialize eligibility traces
        E = np.zeros_like(W)

        # Select action using ε-greedy policy
        Q = np.dot(W, features)
        #action = np.random.randint(0, k) if np.random.rand() < epsilon else np.argmax(Q)

        action = select_action(Q, epsilon, strategy=exploration_strategy)

        total_episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_variables(next_state, low, high)
            next_features = fourier_features(next_state, eta)
            total_episode_reward += reward

            # Select next action using ε-greedy policy
            next_Q = np.dot(W, next_features)
            #next_action = np.random.randint(0, k) if np.random.rand() < epsilon else np.argmax(next_Q)

            next_action = select_action(next_Q, epsilon, strategy=exploration_strategy)

            # Compute TD error
            td_target = reward + gamma * next_Q[next_action] * (not done)
            td_error = td_target - Q[action]

            # Update eligibility traces and clip them to avoid large updates
            E = np.clip(E * gamma * lambda_, -5, 5)
            E[action] += features

            # Apply momentum
            v = m * v + alpha * td_error * E
            W += v

            state = next_state
            features = next_features
            action = next_action
            Q = next_Q

        episode_reward_list.append(total_episode_reward)

        epsilon = max(0.01, epsilon * 0.999)

    return W, episode_reward_list


def plot_rewards(episode_rewards, N_episodes):
    """Plot episode rewards and their running average."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_episodes + 1), episode_rewards, label='Episode reward')
    plt.plot(range(1, N_episodes + 1), running_average(episode_rewards, 10), label='10-episode moving average', color='red', linewidth=1.5)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_value_function(W, eta, low, high, num_points=50):
    """Plot the value function over the state space."""
    x = np.linspace(low[0], high[0], num_points)
    y = np.linspace(low[1], high[1], num_points)
    X, Y = np.meshgrid(x, y)

    V = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([X[i, j], Y[i, j]])
            state_scaled = scale_state_variables(state, low, high)
            features = fourier_features(state_scaled, eta)
            Q_values = np.dot(W, features)
            V[i, j] = np.max(Q_values)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title('Value Function')
    plt.show()


def plot_policy(W, eta, low, high, num_points=50):
    """Plot the optimal policy over the state space."""
    x = np.linspace(low[0], high[0], num_points)
    y = np.linspace(low[1], high[1], num_points)
    X, Y = np.meshgrid(x, y)

    policy = np.zeros_like(X, dtype=int)
    for i in range(num_points):
        for j in range(num_points):
            state = np.array([X[i, j], Y[i, j]])
            state_scaled = scale_state_variables(state, low, high)
            features = fourier_features(state_scaled, eta)
            Q_values = np.dot(W, features)
            policy[i, j] = np.argmax(Q_values)

    action_vectors = {0: [-1, 0], 1: [0, 0], 2: [1, 0]}  # Left, No move, Right
    U = np.vectorize(lambda a: action_vectors[a][0])(policy)
    V = np.vectorize(lambda a: action_vectors[a][1])(policy)

    plt.figure(figsize=(10, 7))
    plt.quiver(X, Y, U, V, scale=20, color='blue')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Optimal Policy (Arrows indicate action direction)')
    plt.grid(alpha=0.3)
    plt.show()


# Main Training Logic
def train_mountain_car(p, N_episodes, alpha, gamma, lambda_, epsilon, init_strategy='zero', exploration_strategy='epsilon_greedy'):
    """Train the Mountain Car agent and return results."""
    env = gym.make('MountainCar-v0')
    env.reset()

    # Fourier Basis Parameters
    eta = np.array([np.array([i, j]) for i in range(p + 1) for j in range(p + 1)])
    #eta = np.array([np.array([i, j]) for i in range(p + 1) for j in range(p + 1) if not (i == 0 and j == 0)])

    low, high = env.observation_space.low, env.observation_space.high

    # Train using Sarsa(λ)
    W, episode_reward_list = sarsa_lambda(env, eta, N_episodes, alpha, gamma, lambda_, epsilon, init_strategy=init_strategy, exploration_strategy=exploration_strategy)

    return W, episode_reward_list, eta, low, high


def analyze_alpha_lambda(env, eta, p, N_episodes, gamma, epsilon):
    # Parameters to analyze
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5]
    lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results_alpha = []
    results_lambda = []
    print("Starting to analyze alpha lambda")
    # Test effect of alpha
    for alpha in alpha_values:
        rewards = []
        for _ in range(3):
            W, episode_rewards, _, _, _ = train_mountain_car(p, N_episodes, alpha, gamma, 0.5, epsilon)
            rewards.append(np.mean(episode_rewards[-50:]))  # Average over last 50 episodes
        results_alpha.append((alpha, np.mean(rewards), np.std(rewards)))

    # Test effect of lambda
    for lambda_ in lambda_values:
        rewards = []
        for _ in range(3):
            W, episode_rewards, _, _, _ = train_mountain_car(p, N_episodes, 0.001, gamma, lambda_, epsilon)
            rewards.append(np.mean(episode_rewards[-50:]))  # Average over last 50 episodes
        results_lambda.append((lambda_, np.mean(rewards), np.std(rewards)))

    # Plot results
    def plot_results(results, param_name, title):
        param_values, means, stds = zip(*results)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=param_values, y=means, label='Average Total Reward')
        plt.fill_between(param_values, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
        plt.xlabel(param_name)
        plt.ylabel('Average Total Reward')
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.show()

    plot_results(results_alpha, 'Alpha (Learning Rate)', 'Effect of Alpha on Average Reward')
    plot_results(results_lambda, 'Lambda (Eligibility Trace)', 'Effect of Lambda on Average Reward')


def grid_search():
    # Initialize environment
    env = gym.make('MountainCar-v0')
    env.reset()

    # Fourier Basis Parameters
    p = 2  # Order of Fourier basis
    eta = np.array([np.array([i, j]) for i in range(p + 1) for j in range(p + 1)])  # Fourier coefficients

    # Define a grid of hyperparameters for grid search
    hyperparameter_grid = {
        'alpha': [0.001],  # Learning rates
        'gamma': [1.0],        # Discount factors
        'lambda_': [0.5],      # Trace decay
        'epsilon': [0.01],      # Exploration rates
        'N_episodes': [700]         # Number of episodes
    }
    # Run 1: (0.01, 1.0, 0.5, 0.1, 1000)
    # Run 2: (0.001, 1.0, 0.9, 0.01, 1000)
    # Run 3: (0.001, 1.0, 0.5, 0.1, 700)
    # Run 4: (0.001, 1.0, 0.5, 0.001, 700) - -134.97
    # Prepare grid search
    param_combinations = list(itertools.product(
        hyperparameter_grid['alpha'],
        hyperparameter_grid['gamma'],
        hyperparameter_grid['lambda_'],
        hyperparameter_grid['epsilon'],
        hyperparameter_grid['N_episodes']
    ))

    # Grid search results
    results = []
    for params in param_combinations:
        alpha, gamma, lambda_, epsilon, N_episodes = params
        print(f"Testing parameters: alpha={alpha}, gamma={gamma}, lambda_={lambda_}, epsilon={epsilon}, N_episodes={N_episodes}")

        W, episode_reward_list = sarsa_lambda(env, eta, N_episodes, alpha, gamma, lambda_, epsilon)

        average_reward = np.mean(episode_reward_list[-100:])
        print(f"Results for these parameters: {average_reward}")
        results.append((params, average_reward))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    best_params, best_performance = sorted_results[0]
    print(f"Best parameters: {best_params} with average reward: {best_performance}")

    best_alpha, best_gamma, best_lambda_, best_epsilon, best_N_episodes = best_params
    W, episode_reward_list = sarsa_lambda(env, eta, best_N_episodes, best_alpha, best_gamma, best_lambda_, best_epsilon)

    plt.plot(range(1, best_N_episodes + 1), episode_reward_list, label='Episode reward')
    plt.plot(range(1, best_N_episodes + 1), running_average(episode_reward_list, 10), label='Average reward (10 episodes)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


    save_path = 'weights.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({'W': W, 'N': eta}, f)

    print("Training complete.")

if __name__ == "__main__":
    # Parameters
    grid_search = False
    p = 2  # Fourier basis order
    N_episodes = 500
    alpha = 0.001
    gamma = 1.0
    lambda_ = 0.5
    epsilon = 0.01
    init_strategy='zero'
    # Init strategies are: zero, optimistic, random
    exploration_strategy='epsilon_greedy'
    # Exploration strategies are: epsilon_greedy, boltzmann

    if grid_search:
        grid_search()
    else:
        # Train the model
        W, episode_rewards, eta, low, high = train_mountain_car(p, N_episodes, alpha, gamma, lambda_, epsilon, init_strategy=init_strategy, exploration_strategy=exploration_strategy)

        # Plot results, uncomment them to check each question part
        plot_rewards(episode_rewards, N_episodes)
        #plot_value_function(W, eta, low, high)
        #plot_policy(W, eta, low, high)

        #analyze_alpha_lambda(env, eta, p, N_episodes, gamma, epsilon)


        save_path = 'weights.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump({'W': W, 'N': eta}, f)
        print(f"Training complete. Weights and Fourier coefficients saved to {save_path}.")

