"""Implementing TD(0) on a grid world"""

import numpy as np
import matplotlib.pyplot as plt
import math

grid_row_len = 5
grid_col_len = 5
episode_len = 5000
action_list = ['up', 'down', 'right', 'left']
c = (0.1, 1, 10)
initial_state = (1, 13, 25)


def state_tuple_converter(state_tuple):
    state = grid_row_len * state_tuple[0] + state_tuple[1] +1
    return state


def state_converter(state):
    q = (state -1) // grid_row_len
    r = (state -1) % grid_row_len
    return q, r


def take_action(action, state_tuple):
    if action == 'up':
        return max(state_tuple[0] - 1, 0), state_tuple[1]
    if action == 'down':
        return min(state_tuple[0] + 1, grid_row_len - 1), state_tuple[1]
    if action == 'left':
        return state_tuple[0], max(state_tuple[1] - 1, 0)
    if action == 'right':
        return state_tuple[0], min(state_tuple[1] + 1, grid_col_len - 1)


def phi(state_tuple):
    arr = np.zeros([grid_col_len*grid_row_len, 1])
    arr[state_tuple_converter(state_tuple) -1] = 1
    return arr


def generate_episode(curr_state_tuple, c, i):
    start_state = state_tuple_converter(curr_state_tuple)
    avg_reward = np.zeros(episode_len)
    do_error = np.zeros(episode_len-1)
    weight = np.zeros([grid_row_len*grid_col_len, episode_len])
    for t in range(0, episode_len - 1):
        alpha = 1 / (math.ceil((t + 1) / 10))
        beta = c*alpha
        generate_episode.state_visited.append(curr_state_tuple)
        if curr_state_tuple == (0, 1):
            next_state_tuple = (4, 1)
            reward = 10
        elif curr_state_tuple == (0, 3):
            next_state_tuple = (2, 3)
            reward = 5
        else:
            next_state_tuple = take_action(np.random.choice(action_list, p=[0.25, 0.25, 0.25, 0.25]), curr_state_tuple)
            if next_state_tuple == curr_state_tuple:
                reward = -1
            else:
                reward = 0
        avg_reward[t+1] = avg_reward[t] + beta*(reward - avg_reward[t])
        do_error[t] = reward - avg_reward[t+1] + np.dot(weight[:, t], phi(next_state_tuple)) - np.dot(weight[:, t], phi(curr_state_tuple))
        weight[:, t+1] = weight[:, t] + alpha*do_error[t]*np.transpose(phi(curr_state_tuple))
        curr_state_tuple = next_state_tuple
    print(np.around(np.reshape(weight[:, episode_len - 1], (5, 5)), decimals=3))
    plt.figure(i)
    plt.suptitle("Plot of Value Function Estimates for c="+ str(c))
    plt.subplot(3, 1, generate_episode.counter)
    plt.title("Start State =" + str(start_state), loc='right')
    plt.plot(weight[0, :], label="Value of state 1")
    plt.plot(weight[12, :], label="Value of state 13")
    plt.plot(weight[24, :], label="Value of state 25")
    plt.subplots_adjust(hspace=0.4, top=0.9)
    plt.xlabel("t")
    plt.ylabel("Value of State")
    plt.legend()

    plt.figure(3+i)
    plt.suptitle("Plot of Average Reward Estimates for c="+ str(c))
    plt.subplot(3, 1, generate_episode.counter)
    plt.plot(avg_reward, label="Start state=" + str(start_state))
    plt.subplots_adjust(hspace=0.4, top=0.9)
    plt.xlabel("t")
    plt.ylabel("Ravg")
    plt.legend()

    if start_state == 1:
        plt.figure(6)
        plt.suptitle("Plot of Value Function Estimates for start state=1")
        plt.subplot(3, 1, i+1)
        plt.title("c =" + str(c), loc='right')
        plt.plot(weight[0, :], label="Value of state 1")
        plt.subplots_adjust(hspace=0.4, top=0.9)
        plt.xlabel("t")
        plt.ylabel("Value of State")
        plt.legend()

        plt.figure(7)
        plt.suptitle("Plot of Average Reward Estimates for start state=1")
        plt.subplot(3, 1, i+1)
        plt.title("c =" + str(c), loc='right')
        plt.plot(avg_reward)
        plt.subplots_adjust(hspace=0.4, top=0.9)
        plt.xlabel("t")
        plt.ylabel("Ravg")
    generate_episode.counter+=1


for i in range(len(c)):
    generate_episode.counter = 1
    start_state = 1
    start_state_tuple = state_converter(start_state)
    generate_episode.state_visited = []
    generate_episode(start_state_tuple, c[i], i)

    start_state = 13
    start_state_tuple = state_converter(start_state)
    generate_episode.state_visited = []
    generate_episode(start_state_tuple, c[i], i)

    start_state = 25
    start_state_tuple = state_converter(start_state)
    generate_episode.state_visited = []
    generate_episode(start_state_tuple, c[i], i)

plt.show()
