import numpy as np
import random
from tqdm.auto import tqdm

class HierarchicalQLearningAgent:
    def __init__(self, environment, actions, alpha, gamma):
        self.actions = actions
        self.env = environment
        self.alpha = alpha
        self.gamma = gamma
        self.high_level_q_table = {}  # Maps state -> option -> Q-value
        self.low_level_q_table = {}  # Maps (state, option, action) -> Q-value
        
    def _get_high_q_value(self, state, option):
        return self.high_level_q_table.get((state, option), 0)

    def _get_low_q_value(self, state, action):
        return self.low_level_q_table.get((state, action), 0)

    def select_option(self, state, exploration_rate):
        if random.random() < exploration_rate:
            return random.choice(list(self.actions.keys()))
        else:
            q_values = {option: self._get_high_q_value(state, option) for option in self.actions.keys()}
            return max(q_values, key=q_values.get)

    def select_action(self, state, option, exploration_rate):
        if random.random() < exploration_rate:
            return random.choice(self.actions[option])
        else:
            q_values = {(state, option, action): self._get_low_q_value((state, option), action) for action in self.actions[option]}
            return max(q_values, key=q_values.get)[2]

    def update(self, state, option, action, reward, next_state, done):
        # Update low-level Q-value
        current_q = self._get_low_q_value((state, option), action)
        next_max_q = max(self._get_low_q_value((next_state, option), a) for a in self.actions[option]) if not done else 0
        self.low_level_q_table[((state, option), action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

        # Update high-level Q-value
        current_option_q = self._get_high_q_value(state, option)
        next_max_option_q = max(self._get_high_q_value(next_state, o) for o in self.actions.keys()) if not done else 0
        self.high_level_q_table[(state, option)] = current_option_q + self.alpha * (reward + self.gamma * next_max_option_q - current_option_q)

    def execute_option(self, initial_state, option, exploration_rate, visit_counts):
        state = initial_state
        total_reward = 0
        done = False
        option_terminated = False 
        self.env.totalTurns = 0
        self.env.totalAsks = 0
        self.env.totalSaves = 0
        
        while not done:
            action = self.select_action(state, option, exploration_rate)
            next_state, reward, done, option_terminated, ditch_event = self.env.step(action, option)
            if ditch_event:
                # Apply the penalty for the ditch event but reset the state for the next iteration
                self.update(state, option, action, reward, state, done)
                #print(f"O: {option} | {state} | a: {action} | s_: {next_state} | r: {reward}")
                state = self.env.reset()  # Reset to start state after applying penalty
            else:
                # Regular update when no ditch event
                self.update(state, option, action, reward, next_state, done)
                #print(f"O: {option} | {state} | a: {action} | s_: {next_state} | r: {reward}")
                state = next_state
            
            total_reward += reward
            if state[0] in visit_counts:
                visit_counts[state[0]] += 1
            else:
                visit_counts[state[0]] = 1
                
            
            if option_terminated or done:
                break

        return state, total_reward, done, visit_counts
    
    
    # Training Loop
    def train_agent(self, episodes):
        exploration_rate = 1.0
        min_exploration_rate = 0.01
        exploration_decay = 2
        rewards = []
        total_reward = 0
        total = np.zeros(episodes)
        max_steps_per_option = 50
        option_steps = 0
        visit_counts = {}

        for episode in tqdm(range(episodes)):
            if episode % 200 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Rate: {exploration_rate}")
            state = self.env.reset()
            total_reward = 0
            Rewards = 0
            done = False
            option_steps = 0

            while not done and option_steps < max_steps_per_option:
              
                      
                option = self.select_option(state, exploration_rate)
                #print(f"\n----- option selected: {option} at episode: {episode} | ep_step: {option_steps}")
                state, reward, done, visit_counts = self.execute_option(state, option, exploration_rate, visit_counts)
                total_reward += reward
                Rewards += reward
                option_steps += 1

            # Decay exploration rate
            #exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
            
            if exploration_rate > 0.1:
                exploration_rate -= exploration_decay/episodes
            else:
                exploration_rate = min_exploration_rate
            

            rewards.append(total_reward)
            total[episode] = Rewards

        return rewards, total, visit_counts