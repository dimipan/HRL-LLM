import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

def get_file_type(document_path):
    # Split the path and get the extension
    _, file_extension = os.path.splitext(document_path)
    # Return the file extension without the period
    return file_extension[1:] if file_extension else None

### for simple Q and attention
def evaluate_policy(env, num_eval_episodes, max_eval_steps_per_episode, agent):
    total_reward = 0
    steps = 0
    cnt = 0
    for _ in range(num_eval_episodes):
        state = env.reset()
        for _ in range(max_eval_steps_per_episode):
            action = agent.max_Action(state)
            state_, reward, done, _ = env.step(action)
            total_reward += reward
            print(f"In {state} --> {env.get_actiondict()[action]} --> get {reward} reward | TOTAL REWARD {total_reward}")
            state = state_
            if state[0] in env.fires:
                cnt += 1
            steps += 1

            if done:
                print(f"Episode finished after {steps} steps with total reward {total_reward} and {cnt} collisions")
                break
    mean_reward = total_reward / num_eval_episodes
    print(f"Mean reward: {mean_reward:.2f}")


def plot_learning_curve(total_rewards_list, EPISODES, labels, colors, optimal_reward):
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, total_rewards in enumerate(total_rewards_list):
        mean_rewards_1, mean_rewards_50 = np.zeros(EPISODES), np.zeros(EPISODES)
        for t in range(EPISODES):
            mean_rewards_1[t] = np.mean(total_rewards[max(0, t-3):(t+1)])
            mean_rewards_50[t] = np.mean(total_rewards[max(0, t-60):(t+1)])
        ax.plot(mean_rewards_50, label=f'{labels[i]}', alpha=0.9, color=colors[i])
        ax.fill_between(range(EPISODES), mean_rewards_1, mean_rewards_50, color=colors[i], alpha=0.15)
        # Check for 20 consecutive iterations with reward >= optimal reward
        for t in range(EPISODES - 20):
            if all(total_rewards[t:t+20] >= optimal_reward):
                ax.axvline(x=t+20, color=colors[i], linestyle='dotted')
                print(f"Line appears at episode: {t+20} for agent {labels[i]}")
                break
    ax.axhline(y=optimal_reward, color='green', linestyle='--', label='Optimal Reward')
    ax.legend()
    ax.grid(True, alpha=0.4)  # Add this line to enable gridlines
    plt.xlabel("Episodes")
    plt.xticks(np.arange(0, EPISODES, step=250))
    plt.ylabel("Total Rewards")
    plt.title("Mean Total Rewards Comparison")
    plt.show()


def animate_policy(env, agent, iterations):
    s = env.reset()
    Rewards = 0
    for i in range(iterations):
        print(f"episode {i} has started")
        done = False
        Rewards = 0
        while not done:
            env.render()
            time.sleep(0.5)
            action = agent.max_Action(s)
            s_prime, reward, done, _ = env.step(action)
            Rewards += reward
            print(f"state {s[0]} | act: {action} | next: {s_prime[0]} | R: {reward} | status: {s[1]}")
            s = s_prime
            if done:
                s = env.reset()
                break
    print(Rewards)


def plot_steps(total_steps_list, episodes):
    # Assuming steps_per_episode_simple and steps_per_episode_guided are lists or NumPy arrays
    simple_series = pd.Series(total_steps_list[0])
    attention_series_1 = pd.Series(total_steps_list[1])
    window_size = 50  # Adjust the window size as needed
    # Calculate the rolling mean
    simple_smoothed = simple_series.rolling(window=window_size).mean()
    attention_smoothed_1 = attention_series_1.rolling(window=window_size).mean()
    # Plot the smoothed data
    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), simple_smoothed, label='Q', color='blue')
    plt.plot(range(episodes), attention_smoothed_1, label='Q+', color='magenta')
    plt.title('Smoothed Number of Steps Taken per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps (Smoothed)')
    plt.legend()
    plt.grid()
    plt.show()


def visitation_heatmap(visit_counts, GRID):
    # Create a grid of visit counts from the dictionary
    grid_counts = np.zeros((GRID[0], GRID[1]))  # Assuming a 7x7 grid
    for state, count in visit_counts.items():
        grid_counts[state] = count
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(14, 14))
    heatmap = ax.imshow(grid_counts, cmap='viridis', interpolation='nearest')
    # Add visitation counts to the cells
    for i in range(grid_counts.shape[0]):
        for j in range(grid_counts.shape[1]):
            count = grid_counts[i, j]

            ax.text(j, i, str(int(count)), ha='center', va='center', color='w', fontsize=8)
    # Add colorbar
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel('Visit Count', rotation=-90, va="bottom")
    # Set axis limits and labels
    ax.set_xlim([-0.5, grid_counts.shape[1] - 0.5])
    ax.set_ylim([grid_counts.shape[0] - 0.5, -0.5])
    ax.set_title('Visit Count Heatmap')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    # Set axis ticks to display numbers from 0 to 14
    ax.set_xticks(range(grid_counts.shape[1]))
    ax.set_yticks(range(grid_counts.shape[0]))
    # Set tick labels to 0-14
    ax.set_xticklabels(range(grid_counts.shape[1]))
    ax.set_yticklabels(range(grid_counts.shape[0]))
    plt.show()
