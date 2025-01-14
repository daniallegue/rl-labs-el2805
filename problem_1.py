# Copyright [2024] [KTH Royal Institute of Technology]
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.
# Student 1: Daniel de Dios Allegue - 20041029-T074
# Student 2: Rodrigo Montero González - 20040421-T054

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import os

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1  # TODO
    GOAL_REWARD = 10  # TODO
    IMPOSSIBLE_REWARD = -5  # TODO
    MINOTAUR_REWARD = -100  # TODO
    POISON_REWARD = -10

    def __init__(self, maze, poisoned=False, poison_probability=0.2,  minotaur_can_stay=False):
        """ Constructor of the environment Maze.
        """
        self.poisoned = poisoned  # Flag to indicate if the agent is poisoned
        self.poison_probability = poison_probability  # The probability of the agent getting poisoned
        self.minotaur_can_stay = minotaur_can_stay # Sets possibility of minotaur staying
        self.maze = maze
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

    def __init__2(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()
        self.minotaur_can_stay = False

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):

        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i, j] != 1:
                            states[s] = ((i, j), (k, l))
                            map[((i, j), (k, l))] = s
                            s += 1

        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1

        states[s] = 'Win'
        map['Win'] = s

        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the player stays in place.

            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """

        if self.states[state] == 'Eaten' or self.states[state] == 'Win':  # In these states, the game is over
            return [self.states[state]]

        else:
            # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0]  # Row of the player's next position
            col_player = self.states[state][0][1] + self.actions[action][1]  # Column of the player's next position

            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = (
                row_player < 0 or row_player >= self.maze.shape[0] or
                col_player < 0 or col_player >= self.maze.shape[1]
            ) or (
                0 <= row_player < self.maze.shape[0] and
                0 <= col_player < self.maze.shape[1] and
                self.maze[row_player, col_player] == 1
            )

            if impossible_action_player:
                row_player, col_player = self.states[state][0]

            # Possible moves for the Minotaur
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]]
            if self.minotaur_can_stay:
                actions_minotaur.append([0, 0])  # Include STAY action if allowed

            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Check if the Minotaur's move is out of bounds
                new_row = self.states[state][1][0] + actions_minotaur[i][0]
                new_col = self.states[state][1][1] + actions_minotaur[i][1]
                impossible_action_minotaur = (
                    new_row < 0 or new_row >= self.maze.shape[0] or
                    new_col < 0 or new_col >= self.maze.shape[1]
                )

                if not impossible_action_minotaur:
                    rows_minotaur.append(new_row)
                    cols_minotaur.append(new_col)

            # Based on the impossibility check, return the next possible states
            states = []
            for i in range(len(rows_minotaur)):
                minotaur_pos = (rows_minotaur[i], cols_minotaur[i])

                if (row_player, col_player) == minotaur_pos:
                    states.append('Eaten')  # The agent is caught by the Minotaur
                elif self.maze[row_player, col_player] == 2:
                    states.append('Win')  # The agent reaches the exit
                else:
                    # The player and the Minotaur move
                    player_pos = (row_player, col_player)
                    states.append((player_pos, minotaur_pos))

            return states


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # TODO: Compute the transition probabilities.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__move(s, a)
                prob_each = 1.0 / len(next_states)  # Since Minotaur moves randomly
                for next_state in next_states:
                    next_state_idx = self.map[next_state]
                    transition_probabilities[s, next_state_idx, a] = prob_each

        return transition_probabilities

    def __rewards(self):

        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):

                if self.states[s] == 'Eaten':  # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD

                elif self.states[s] == 'Win':  # The player has won
                    rewards[s, a] = self.GOAL_REWARD

                else:
                    next_states = self.__move(s, a)
                    next_s = next_states[
                        0]  # The reward does not depend on the next position of the minotaur, we just consider the first one

                    if self.states[s][0] == next_s[0] and a != self.STAY:  # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD

                    else:  # Regular move
                        rewards[s, a] = self.STEP_REWARD

                if self.poisoned:  # Poison mechanic
                    if random.random() < self.poison_probability:  # Poison probability check
                        rewards[s, a] = self.POISON_REWARD  # Apply poison penalty

        return rewards


    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()

        if method == 'DynProg':
            horizon = policy.shape[1]  # Deduce the horizon from the policy shape
            t = 0  # Initialize current time
            s = self.map[start]  # Initialize current state
            path.append(start)  # Add the starting position in the maze to the path

            while t < horizon - 1:
                a = policy[s, t]  # Action at time t
                next_states = self.__move(s, a)
                next_s = next_states[random.randint(0, len(next_states) - 1)]
                path.append(next_s)
                t += 1
                s = self.map[next_s]

                # Check for poison
                if self.poisoned and random.random() < self.poison_probability:
                    path.append('Poisoned')
                    break  # Agent dies due to poison

                if self.states[s] == 'Win' or self.states[s] == 'Eaten':
                    break  # Game over

        if method == 'ValIter':
            s = self.map[start]
            path.append(start)
            t = 0  # Initialize time
            while True:
                a = policy[s]
                next_states = self.__move(s, a)
                next_s = next_states[random.randint(0, len(next_states) - 1)]
                path.append(next_s)
                t += 1
                s = self.map[next_s]

                # Check for poison
                if self.poisoned and random.random() < self.poison_probability:
                    path.append('Poisoned')
                    break  # Agent dies due to poison

                if self.states[s] == 'Win' or self.states[s] == 'Eaten':
                    break  # Game over

                if t > 1000:  # Avoid infinite loops
                    break

        return [path, t]

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


class MazeKeys:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 100
    IMPOSSIBLE_REWARD = -5
    MINOTAUR_REWARD = -100
    POISON_REWARD = -10
    KEY_REWARD = 10

    def __init__(self, maze, poisoned=False, poison_probability=0.02):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.poisoned = poisoned
        self.poison_probability = poison_probability
        self.gamma = 0.9  # Discount factor
        self.reset()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        for key_status in [0, 1]:
                            if self.maze[i, j] != 1 and self.maze[k, l] != 1:
                                states[s] = ((i, j), (k, l), key_status)
                                map[((i, j), (k, l), key_status)] = s
                                s += 1

        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1

        states[s] = 'Win'
        map['Win'] = s

        return states, map

    def reset(self):
        # Initialize agent and minotaur positions
        self.agent_pos = (0, 0)  # Starting position of the agent
        self.minotaur_pos = (6, 5)  # Starting position of the minotaur
        self.key_status = 0  # Agent does not have the keys initially
        self.initial_state = (self.agent_pos, self.minotaur_pos, self.key_status)
        return self.initial_state

    def step(self, state, action):
        """ Executes one time step within the environment """
        if state == 'Eaten' or state == 'Win':
            return state, 0, True

        agent_pos, minotaur_pos, key_status = state

        # Agent movement
        move = self.actions[action]
        next_agent_row = agent_pos[0] + move[0]
        next_agent_col = agent_pos[1] + move[1]

        # Check if agent's move is valid
        if (0 <= next_agent_row < self.maze.shape[0] and
            0 <= next_agent_col < self.maze.shape[1] and
            self.maze[next_agent_row, next_agent_col] != 1):
            next_agent_pos = (next_agent_row, next_agent_col)
        else:
            next_agent_pos = agent_pos  # Agent stays in the same position

        # Update key status
        if next_agent_pos == (4, 3):  # Position C where keys are located
            key_status = 1

        # Minotaur movement
        minotaur_moves = []
        if random.random() < 0.35:
            # Minotaur moves towards the agent
            minotaur_moves = self.__minotaur_moves_towards_agent(minotaur_pos, next_agent_pos)
        else:
            # Minotaur moves randomly
            minotaur_moves = self.__minotaur_random_moves(minotaur_pos)

        # For simplicity, we will choose one possible minotaur move
        next_minotaur_pos = random.choice(minotaur_moves)

        # Check for minotaur encounter
        if next_agent_pos == next_minotaur_pos:
            return 'Eaten', self.MINOTAUR_REWARD, True

        # Check for poison
        if self.poisoned and random.random() < self.poison_probability:
            return 'Eaten', self.POISON_REWARD, True

        # Check for successful exit
        if next_agent_pos == (6, 5) and key_status == 1:
            return 'Win', self.GOAL_REWARD, True

        # Compute reward
        reward = self.STEP_REWARD

        # Construct next state
        next_state = (next_agent_pos, next_minotaur_pos, key_status)

        return next_state, reward, False

    def __minotaur_random_moves(self, minotaur_pos):
        moves = []
        for move in self.actions.values():
            next_row = minotaur_pos[0] + move[0]
            next_col = minotaur_pos[1] + move[1]
            if (0 <= next_row < self.maze.shape[0] and
                0 <= next_col < self.maze.shape[1] and
                self.maze[next_row, next_col] != 1):
                moves.append((next_row, next_col))
        return moves

    def __minotaur_moves_towards_agent(self, minotaur_pos, agent_pos):
        moves = []
        min_row, min_col = minotaur_pos
        agent_row, agent_col = agent_pos
        if min_row < agent_row:
            moves.append((min_row + 1, min_col))
        elif min_row > agent_row:
            moves.append((min_row - 1, min_col))
        if min_col < agent_col:
            moves.append((min_row, min_col + 1))
        elif min_col > agent_col:
            moves.append((min_row, min_col - 1))
        # Filter out invalid moves
        valid_moves = []
        for move in moves:
            row, col = move
            if (0 <= row < self.maze.shape[0] and
                0 <= col < self.maze.shape[1] and
                self.maze[row, col] != 1):
                valid_moves.append(move)
        if not valid_moves:
            valid_moves = [minotaur_pos]  # Minotaur stays in place if no valid moves
        return valid_moves

    def render(self, state):
        # Simple text-based rendering
        if state == 'Eaten':
            print("Agent has been eaten by the Minotaur!")
        elif state == 'Win':
            print("Agent has successfully exited the maze with the keys!")
        else:
            agent_pos, minotaur_pos, key_status = state
            maze_copy = self.maze.copy()
            maze_copy[agent_pos] = -2  # Agent
            maze_copy[minotaur_pos] = -1  # Minotaur
            print(maze_copy)

    def sample_action(self):
        return random.choice(list(self.actions.keys()))

    def get_possible_actions(self, state):
        return list(self.actions.keys())



def dynamic_programming(env, horizon):
    V = np.zeros((env.n_states, horizon + 1))
    policy = np.zeros((env.n_states, horizon), dtype=int)

    for t in range(horizon - 1, -1, -1):
        # Calculate action values
        action_values = np.zeros((env.n_states, env.n_actions))
        for a in range(env.n_actions):
            # Vectorized multiplication over next states
            next_state_values = env.transition_probabilities[:, :, a] @ V[:, t + 1]
            action_values[:, a] = env.rewards[:, a] + next_state_values

        # Optimal value and policy extraction
        V[:, t] = np.max(action_values, axis=1)
        policy[:, t] = np.argmax(action_values, axis=1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : Accuracy threshold for convergence.
        :return numpy.array V     : Optimal values for every state,
                                    dimension S
        :return numpy.array policy: Optimal policy at every state,
                                    dimension S
    """
    V = np.zeros(env.n_states)  # Initialize value function
    policy = np.zeros(env.n_states, dtype=int)  # Initialize policy

    iteration = 0  # To keep track of the number of iterations
    while True:
        V_prev = V.copy()  # Keep a copy of the value function to check for convergence
        Q = np.zeros((env.n_states, env.n_actions))  # Initialize Q-values

        # Vectorized computation of Q-values for all states and actions
        for a in range(env.n_actions):
            # Compute the expected value for action 'a' across all states
            Q[:, a] = env.rewards[:, a] + gamma * env.transition_probabilities[:, :, a].dot(V)

        # Update the value function and policy
        V = np.max(Q, axis=1)
        policy = np.argmax(Q, axis=1)

        # Check for convergence
        delta = np.max(np.abs(V - V_prev))
        iteration += 1
        if delta < epsilon:
            print(f"Value iteration converged after {iteration} iterations.")
            break

    return V, policy



def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: 'white', 1: 'black', 2: 'lightgreen', -1: 'red', -2: 'purple'}

    rows, cols = maze.shape  # Size of the maze

    # Ensure the save folder exists
    if not os.path.exists('save_folder'):
        os.makedirs('save_folder')

    # Loop through the path and animate it
    for i in range(1, len(path)):
        fig, ax = plt.subplots(figsize=(cols, rows))  # Create a fresh figure each time
        ax.set_title('Policy simulation')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

        # Create a table to color
        grid = ax.table(
            cellText=None,
            cellColours=colored_maze,
            cellLoc='center',
            loc='center',
            edges='closed'
        )

        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0 / rows)
            cell.set_width(1.0 / cols)

        # Update the colors for the path positions
        if path[i - 1] != 'Eaten' and path[i - 1] != 'Win':
            grid.get_celld()[(path[i - 1][0][0], path[i - 1][0][1])].set_facecolor(
                col_map[maze[path[i - 1][0][0], path[i - 1][0][1]]])
            grid.get_celld()[(path[i - 1][1][0], path[i - 1][1][1])].set_facecolor(
                col_map[maze[path[i - 1][1][0], path[i - 1][1][1]]])

        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0][0], path[i][0][1])].set_facecolor(col_map[-2])  # Player's position
            grid.get_celld()[(path[i][1][0], path[i][1][1])].set_facecolor(col_map[-1])  # Minotaur's position

        # Display the current figure
        display.display(fig)

        # Save the current frame as an image
        filename = os.path.join('save_folder', f"frame_{i:03d}.png")
        plt.savefig(filename)
        plt.close(fig)  # Close the figure after saving to avoid memory issues

        # Pause for animation effect
        time.sleep(0.1)
        display.clear_output(wait=True)  # Clear previous output to avoid cluttering the display


def format_policy_for_report(policy_visualization):
    # Mapping action symbols to display characters
    action_symbols = {
        'S': '•',  # Stay
        'L': '←',  # Move Left
        'R': '→',  # Move Right
        'U': '↑',  # Move Up
        'D': '↓',  # Move Down
        '': ' '    # Empty cell (wall or undefined)
    }

    formatted_rows = []
    for row in policy_visualization:
        formatted_row = [action_symbols.get(cell, '?') for cell in row]
        formatted_rows.append(' '.join(formatted_row))

    return '\n'.join(formatted_rows)

def extract_policy_for_visualization(env, policy, horizon, minotaur_initial_pos=(6, 5)):
    # Initialize a simplified policy view for the maze (7x8 grid)
    policy_visualization = np.full((env.maze.shape[0], env.maze.shape[1]), '', dtype='<U1')

    # Map action indices to their corresponding directions (strings)
    action_map = {
        env.STAY: 'S',        # Stay
        env.MOVE_LEFT: 'L',   # Move Left
        env.MOVE_RIGHT: 'R',  # Move Right
        env.MOVE_UP: 'U',     # Move Up
        env.MOVE_DOWN: 'D'    # Move Down
    }

    # Iterate over all positions in the maze
    for i in range(env.maze.shape[0]):
        for j in range(env.maze.shape[1]):
            if env.maze[i, j] == 1:
                continue  # Skip walls

            # Create the state corresponding to the player's position (i, j) and the fixed Minotaur position
            state = ((i, j), minotaur_initial_pos)

            # Check if this state exists in the state mapping
            if state in env.map:
                s = env.map[state]
                # Get the action for time t=0
                best_action = policy[s, 0]
                # Map the action to a direction (string)
                policy_visualization[i, j] = action_map[best_action]

    return policy_visualization

def simulate_multiple_episodes(env, policy, method, num_episodes):
    success_count = 0
    for _ in range(num_episodes):
        start = ((0, 0), (6, 5))
        path, _ = env.simulate(start, policy, method)
        if path[-1] == 'Win':
            success_count += 1
    probability_of_success = success_count / num_episodes
    return probability_of_success

def q_learning(env, num_episodes, epsilon, alpha, gamma):
    """
    Implements the episodic Q-learning algorithm.
    """
    # Initialize Q-values arbitrarily for all state-action pairs
    Q = {}
    n_sa = {}  # Counts of state-action pairs

    # For tracking the value function over episodes
    value_per_episode = []

    for episode in range(num_episodes):
        # Reset Q-values between episodes
        Q = {}
        n_sa = {}
        total_reward = 0

        # Initialize the environment
        state = env.reset()
        done = False

        while not done:
            s = state

            # Initialize Q-values for new state-action pairs
            if s not in Q:
                Q[s] = np.zeros(env.n_actions)
                n_sa[s] = np.zeros(env.n_actions)

            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(env.n_actions)
            else:
                action = np.argmax(Q[s])

            # Take action and observe the next state and reward
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Initialize Q-values for the next state if not seen before
            if next_state not in Q:
                Q[next_state] = np.zeros(env.n_actions)
                n_sa[next_state] = np.zeros(env.n_actions)

            # Update counts
            n_sa[s][action] += 1
            alpha_t = 1 / (n_sa[s][action]) ** alpha

            # Q-learning update
            Q[s][action] += alpha_t * (reward + gamma * np.max(Q[next_state]) - Q[s][action])

            # Move to the next state
            state = next_state

        # Store the value of the initial state after each episode
        initial_state = env.initial_state
        if initial_state in Q:
            value_per_episode.append(np.max(Q[initial_state]))
        else:
            value_per_episode.append(0)

    return Q, value_per_episode

def sarsa(env, num_episodes, epsilon, alpha, gamma, epsilon_decay=False, delta=None):
    """
    Implements the SARSA algorithm.
    """
    # Initialize Q-values arbitrarily for all state-action pairs
    Q = {}
    n_sa = {}  # Counts of state-action pairs

    # For tracking the value function over episodes
    value_per_episode = []

    for episode in range(1, num_episodes + 1):
        # Reset the environment
        state = env.reset()
        done = False

        # Initialize Q-values for new state-action pairs
        if state not in Q:
            Q[state] = np.zeros(env.n_actions)
            n_sa[state] = np.zeros(env.n_actions)

        # Update epsilon if decay is enabled
        if epsilon_decay:
            epsilon = min(1.0, 1 / (episode ** delta))

        # Choose action using policy derived from Q (ε-greedy)
        if np.random.rand() < epsilon:
            action = np.random.choice(env.n_actions)
        else:
            action = np.argmax(Q[state])

        while not done:
            s = state
            a = action

            # Take action and observe the next state and reward
            next_state, reward, done = env.step(s, a)

            # Initialize Q-values for new state-action pairs
            if next_state not in Q:
                Q[next_state] = np.zeros(env.n_actions)
                n_sa[next_state] = np.zeros(env.n_actions)

            # Choose next action using policy derived from Q (ε-greedy)
            if epsilon_decay:
                # epsilon already updated
                pass
            if np.random.rand() < epsilon:
                next_action = np.random.choice(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # Update counts
            n_sa[s][a] += 1
            alpha_t = 1 / (n_sa[s][a] ** alpha)

            # SARSA update
            Q[s][a] += alpha_t * (reward + gamma * Q[next_state][next_action] - Q[s][a])

            # Move to the next state and action
            state = next_state
            action = next_action

        # Store the value of the initial state after each episode
        initial_state = env.initial_state
        if initial_state in Q:
            value_per_episode.append(np.max(Q[initial_state]))
        else:
            value_per_episode.append(0)

    return Q, value_per_episode

def extract_policy_from_q_values(Q, env):
    """
    Extracts a policy from the Q-values.
    """
    policy = {}
    for state in Q.keys():
        policy[state] = np.argmax(Q[state])
    return policy

def estimate_success_probability(env, policy, num_episodes):
    """
    Estimates the probability of exiting the maze successfully using the given policy.
    """
    success_count = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if state in policy:
                action = policy[state]
            else:
                action = np.random.choice(env.n_actions)  # Random action if state not in policy
            next_state, reward, done = env.step(state, action)
            if next_state == 'Win':
                success_count += 1
                break
            elif next_state == 'Eaten':
                break
            else:
                state = next_state
    probability_of_success = success_count / num_episodes
    return probability_of_success




def main():
    # Initialize maze and environment
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    env = Maze(maze)

    # Get user input to decide which question to solve
    problem = input("Enter the problem to solve (c, d, e, f, g, h, i, j, k): ")

    if problem == "c":
        # Question (c): Find policy for T=20 that maximizes survival probability
        print("Solving question (c)...")
        horizon = 20
        V, policy = dynamic_programming(env, horizon)
        start = ((0, 0), (6, 5))
        path, _ = env.simulate(start, policy, "DynProg")
        animate_solution(maze, path)  # Illustrate policy
        policy_visualization = extract_policy_for_visualization(env, policy, horizon)
        formatted_policy = format_policy_for_report(policy_visualization)
        print(formatted_policy)

    elif problem == "d":
        print("Solving question (d)...")

        # Initialize lists to store survival probabilities
        probabilities_no_stay = []
        probabilities_with_stay = []

        # Initial state
        start_state = ((0, 0), (6, 5))

        # Number of simulations per time horizon
        num_simulations = 10000

        # Scenario 1: Minotaur cannot stand still
        print("Minotaur cannot stand still...")
        for horizon in range(1, 31):
            print(f"Computing for horizon T={horizon}")
            # Compute the policy using dynamic programming
            V, policy = dynamic_programming(env, horizon)

            # Simulate multiple episodes to estimate survival probability
            success_count = 0
            for _ in range(num_simulations):
                path, _ = env.simulate(start_state, policy, "DynProg")
                if path[-1] == 'Win':
                    success_count += 1
            survival_probability = success_count / num_simulations
            probabilities_no_stay.append(survival_probability)

        # Scenario 2: Minotaur can stand still
        print("Minotaur can stand still...")

        # Create a new environment where the Minotaur can stand still
        env_stay = Maze(maze, minotaur_can_stay=True)
        env_stay.actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]  # Include STAY action
        env_stay.transition_probabilities = env_stay.transition_probabilities
        env_stay.rewards = env_stay.rewards

        for horizon in range(1, 31):
            print(f"Computing for horizon T={horizon}")
            # Compute the policy using dynamic programming
            V_stay, policy_stay = dynamic_programming(env_stay, horizon)

            # Simulate multiple episodes to estimate survival probability
            success_count = 0
            for _ in range(num_simulations):
                path, _ = env_stay.simulate(start_state, policy_stay, "DynProg")
                if path[-1] == 'Win':
                    success_count += 1
            survival_probability = success_count / num_simulations
            probabilities_with_stay.append(survival_probability)

        print(probabilities_no_stay)
        print(probabilities_with_stay)

        # Plot the estimated survival probabilities
        plt.plot(range(1, 31), probabilities_no_stay, label='Minotaur cannot stand still')
        plt.plot(range(1, 31), probabilities_with_stay, label='Minotaur can stand still')
        plt.title("Survival Probability vs. Time Horizon (T=1 to T=30)")
        plt.xlabel("Time Horizon (T)")
        plt.ylabel("Estimated Survival Probability")
        plt.legend()
        plt.grid(True)
        plt.show()


    elif problem == "e":
        # Question (e): Modify for poisoned agent with geometrically distributed life
        print("Solving question (e)...")
        env.poisoned = True
        env.poison_probability = 1.0 / 30.0  # Mean life of 30 steps
        gamma = 0.95  # Discount factor less than 1 to help convergence
        epsilon = 1e-5

        # Run value iteration to compute the optimal policy
        V, policy = value_iteration(env, gamma, epsilon)

        # Simulate multiple episodes to estimate the probability of exiting the maze
        num_episodes = 10000
        probability_of_success = simulate_multiple_episodes(env, policy, "ValIter", num_episodes)
        print(f"Estimated probability of exiting the maze from the start state: {probability_of_success:.4f}")

        # Optionally, simulate and animate a single episode
        start = ((0, 0), (6, 5))
        path, _ = env.simulate(start, policy, "ValIter")
        animate_solution(maze, path)

    elif problem == "f":
        # Question (f): Simulate 10,000 games
        print("Solving question (f)...")
        env.poisoned = True
        env.poison_probability = 1.0 / 30.0
        V, policy = value_iteration(env, 0.95, 1e-5)
        survival_count = 0
        for i in range(10000):
            print("starting ", i)
            start = ((0, 0), (6, 5))
            path, _ = env.simulate(start, policy, "ValIter")
            if path[-1] == 'Win':
                survival_count += 1

        survival_prob = survival_count / 10000

        print(f"Estimated survival probability: {survival_prob}")

    env = MazeKeys(maze)

    if problem == 'i':
        # Parameters
        num_episodes = 5000
        epsilon_values = [0.1, 0.9]
        alpha = 2 / 3
        gamma = 0.9

        # For storing the results
        values_per_epsilon = {}

        for epsilon in epsilon_values:
            Q, value_per_episode = q_learning(env, num_episodes, epsilon, alpha, gamma)
            values_per_epsilon[epsilon] = value_per_episode

        # Plotting the results for different ε values
        plt.figure(figsize=(10, 6))
        for epsilon in epsilon_values:
            plt.plot(values_per_epsilon[epsilon], label=f'ε = {epsilon}')
        plt.title('Value Function over Episodes for Different ε Values')
        plt.xlabel('Episode')
        plt.ylabel('Value of Initial State')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Now vary α while keeping ε fixed
        alpha_values = [0.6, 0.9]
        epsilon = 0.1
        values_per_alpha = {}

        for alpha in alpha_values:
            Q, value_per_episode = q_learning(env, num_episodes, epsilon, alpha, gamma)
            values_per_alpha[alpha] = value_per_episode

        # Plotting the results for different α values
        plt.figure(figsize=(10, 6))
        for alpha in alpha_values:
            plt.plot(values_per_alpha[alpha], label=f'α = {alpha}')
        plt.title('Value Function over Episodes for Different α Values')
        plt.xlabel('Episode')
        plt.ylabel('Value of Initial State')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif problem == 'j':
        # Parameters
        num_episodes = 50000
        epsilon_values = [0.2, 0.1]
        alpha = 2 / 3
        gamma = 0.9

        # For storing the results
        values_per_epsilon = {}

        for epsilon in epsilon_values:
            print(f"Running SARSA with ε = {epsilon}")
            Q, value_per_episode = sarsa(env, num_episodes, epsilon, alpha, gamma)
            values_per_epsilon[epsilon] = value_per_episode

        # Plotting the results for different ε values
        plt.figure(figsize=(12, 6))
        for epsilon in epsilon_values:
            plt.plot(values_per_epsilon[epsilon], label=f'ε = {epsilon}')
        plt.title('Value Function over Episodes for Different ε Values (SARSA)')
        plt.xlabel('Episode')
        plt.ylabel('Value of Initial State')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Now consider decreasing epsilon
        delta = 0.7  # Choose a delta in (0.5, 1]
        epsilon_decay = True

        # For α > δ and α < δ
        alpha_values = [0.8, 0.6]  # α = 0.8 (> δ), α = 0.6 (< δ)
        values_per_alpha = {}

        for alpha in alpha_values:
            print(f"Running SARSA with decreasing ε_k, α = {alpha}, δ = {delta}")
            Q, value_per_episode = sarsa(env, num_episodes, epsilon=None, alpha=alpha, gamma=gamma, epsilon_decay=True,
                                         delta=delta)
            values_per_alpha[alpha] = value_per_episode

        # Plotting the results
        plt.figure(figsize=(12, 6))
        for alpha in alpha_values:
            plt.plot(values_per_alpha[alpha], label=f'α = {alpha}')
        plt.title(f'Value Function over Episodes with Decreasing ε_k (δ={delta})')
        plt.xlabel('Episode')
        plt.ylabel('Value of Initial State')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif problem == 'k':
        if problem == 'k':
            # Parameters
            num_episodes_training = 50000
            num_episodes_testing = 10000
            epsilon = 0.1
            alpha = 2 / 3
            gamma = 0.9

            Q_qlearning, _ = q_learning(env, num_episodes_training, epsilon, alpha, gamma)

            # Extract policy from Q-learning
            policy_qlearning = extract_policy_from_q_values(Q_qlearning, env)

            prob_qlearning = estimate_success_probability(env, policy_qlearning, num_episodes_testing)
            print(f"Estimated probability of exiting the maze using Q-learning policy: {prob_qlearning:.4f}")

            # Get Q-value of initial state
            initial_state = env.initial_state
            q_value_initial_qlearning = np.max(Q_qlearning.get(initial_state, np.zeros(env.n_actions)))
            print(f"Q-value of the initial state (Q-learning): {q_value_initial_qlearning:.4f}")

            # SARSA
            print("Training SARSA policy...")
            Q_sarsa, _ = sarsa(env, num_episodes_training, epsilon, alpha, gamma)

            policy_sarsa = extract_policy_from_q_values(Q_sarsa, env)

            prob_sarsa = estimate_success_probability(env, policy_sarsa, num_episodes_testing)
            print(f"Estimated probability of exiting the maze using SARSA policy: {prob_sarsa:.4f}")

            q_value_initial_sarsa = np.max(Q_sarsa.get(initial_state, np.zeros(env.n_actions)))
            print(f"Q-value of the initial state (SARSA): {q_value_initial_sarsa:.4f}")



if __name__ == '__main__':
    main()