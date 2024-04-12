import numpy as np
import random
from tqdm.auto import tqdm

class HierarchicalQLearningAgent_ATTENTION:
    def __init__(self, environment, actions, alpha, gamma):
        self.actions = actions
        self.env = environment
        self.alpha = alpha
        self.gamma = gamma
        self.high_level_q_table = {}  # Maps state -> option -> Q-value
        self.low_level_q_table = {}  # Maps (state, option, action) -> Q-value
        self.exploit_start_episode = None
        self.exploit_end_episode = None
        self.exploit_episodes = 1000  # 30% of the episodes
        self.exploit_initiated = False
        self.exploit_mode = False
        
        self.attention_space_HIGH = {}
        self.attention_space_LOW = {}
    
    def identify_changes_states(self, readings):
        updated_locations = [updated_loc for updated_loc, sensor_value in readings.items() if readings]
        return updated_locations
            
    ### Finds states connected to a given target state, considering possible actions and their inverse
    def get_connected_states(self, target_state):
        ### {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'} -- Inverse action mapping for movement actions
        inverse_actions = {self.env.actionspace[self.env.optionspace[0]][0]: self.env.actionspace[self.env.optionspace[0]][1],
                        self.env.actionspace[self.env.optionspace[0]][1]: self.env.actionspace[self.env.optionspace[0]][0],
                        self.env.actionspace[self.env.optionspace[0]][2]: self.env.actionspace[self.env.optionspace[0]][3],
                        self.env.actionspace[self.env.optionspace[0]][3]: self.env.actionspace[self.env.optionspace[0]][2]}
        connected_states_pairs = []   # Generate state-action pairs
        for action in self.actions[self.env.optionspace[0]]:   ## we just care about the navigation connections
            inverse_action = inverse_actions[action] # Apply the inverse action to the target state
            possible_prev_state = self.env.next_state_vision(target_state[0], inverse_action)
            #print(f"poss prev: {possible_prev_state}")
            # Check if the resulting state is valid and leads to the target state
            if possible_prev_state != target_state[0] and possible_prev_state not in self.env.ditches:
                connected_states_pairs.append((possible_prev_state, action))
        #print(f"out from get_connected: {connected_states_pairs}")
        return connected_states_pairs

    ### Updates the attention space, which influences the agent's decision-making based on environmental changes. ### more like policy-shaping
    def update_attention_space(self, connection, sensor_readings):
        connected_states = self.get_connected_states(connection)
        print(connected_states)
        # Determine the value to add based on sensor reading
        value_to_add = 5.0 if sensor_readings[connection] > 0 else -10.0
        for connected_state, action in connected_states:    # Update the attention space value for the state-action pair
            ## here we update the attention space low that directly affects the low-level Q table (low level action selection)
            self.attention_space_LOW[(((connected_state, True, False, False, False), self.env.optionspace[0]), action)] = value_to_add  ### (((connected_state), True, False, False), 'EXPLORE'), 'action'): value_to_add
            ## here we update the attention space high that directly affects the high-level Q table (high level action selection -- option selection)
            if value_to_add > 0: # For positive sensor readings, enhance the value for 'EXPLORE' option
                self.attention_space_HIGH[(((connected_state), True, False, False, False), self.env.optionspace[0])] = value_to_add ### (((connected_state), False, False, False), 'option'): value
            else: # For negative sensor readings, decrease the value for options other than 'EXPLORE'# For negative sensor readings, decrease the value for options other than 'EXPLORE'
                self.attention_space_HIGH[(((connected_state), True, False, False, False), self.env.optionspace[1])] = value_to_add
                self.attention_space_HIGH[(((connected_state), True, False, False, False), self.env.optionspace[2])] = value_to_add
            
            ## # Positive value adds to 'EXPLORE', negative to 'ASK' (and could be expanded to others)
            #self.attention_space_HIGH[((connected_state, True, False, False), 'EXPLORE' if value_to_add > 0 else 'ASK')] = value_to_add
        return self.attention_space_HIGH, self.attention_space_LOW


    def update_q_tables_from_attention(self):
        if self.attention_space_HIGH:
            for key, value in self.attention_space_HIGH.items():
                if key in self.high_level_q_table:
                    self.high_level_q_table[key] = self.high_level_q_table[key] + value
                else:
                    self.high_level_q_table[key] = value
                ###here it works
        if self.attention_space_LOW:
            for key, value in self.attention_space_LOW.items():
                if key in self.low_level_q_table:
                    self.low_level_q_table[key] = self.low_level_q_table[key] + value
                else:
                    self.low_level_q_table[key] = value
        
        
    def _get_high_q_value(self, state, option):
        return self.high_level_q_table.get((state, option), 0)

    def _get_low_q_value(self, state, action):
        return self.low_level_q_table.get((state, action), 0)

    def select_option(self, state, exploration_rate):
        if not self.env.visited_information_state:
            if random.random() < exploration_rate:
                return random.choice(list(self.actions.keys()))
            else:
                q_values = {option: self._get_high_q_value(state, option) for option in self.actions.keys()}
                return max(q_values, key=q_values.get)
        else:
            # if random.random() < 0.01:
            #     return random.choice(list(self.actions.keys()))
            # else:
                q_values = {option: self._get_high_q_value(state, option) for option in self.actions.keys()}
                return max(q_values, key=q_values.get)

    def select_action(self, state, option, exploration_rate):
        if not self.env.visited_information_state:
            if random.random() < exploration_rate:
                return random.choice(self.actions[option])
            else:
                q_values = {(state, option, action): self._get_low_q_value((state, option), action) for action in self.actions[option]}
                return max(q_values, key=q_values.get)[2]
        else:
            # if random.random() < 0.001:
            #     return random.choice(self.actions[option])
            # else:
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
        option_terminated = False ##
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
    
    ### Manages the transition between exploration and exploitation modes based on information received
    def exploitation_strategy(self, received, current_episode):
        if received and not self.exploit_mode and not self.exploit_initiated:
            self.exploit_start_episode = current_episode
            self.exploit_end_episode = self.exploit_start_episode + self.exploit_episodes
            self.exploit_mode = True
            self.exploit_initiated = True
            print(f"Exploitation mode started at episode {self.exploit_start_episode}")
    
    
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
        received_input = False
        attention_space_OPTIONS, attention_space_PRIMITIVES = {}, {}

        for episode in tqdm(range(episodes)):
            if episode % 200 == 0:
                print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Rate: {exploration_rate}")
            state = self.env.reset()
            total_reward = 0
            Rewards = 0
            done = False
            option_steps = 0

            while not done and option_steps < max_steps_per_option:
                
                if self.env.visited_information_state and not received_input:
                    identified_states = self.identify_changes_states(self.env.sensor_readings)
                    received_input = True 
                    print(f"got the info needed at ep {episode}")
                    if identified_states:
                        for informed_state in identified_states:
                            attention_space_OPTIONS, attention_space_PRIMITIVES = self.update_attention_space(informed_state, self.env.sensor_readings)
                    self.update_q_tables_from_attention()
                            
                
                option = self.select_option(state, exploration_rate)
                #print(f"\n----- option selected: {option} at episode: {episode} | ep_step: {option_steps}")
                state, reward, done, visit_counts = self.execute_option(state, option, exploration_rate, visit_counts)
                total_reward += reward
                Rewards += reward
                option_steps += 1

            # Decay exploration rate
            #exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
            if not self.env.visited_information_state:
                if exploration_rate > 0.1:
                        exploration_rate -= exploration_decay/episodes
                else:
                    exploration_rate = min_exploration_rate
            else:
                if exploration_rate > 0.1:
                    exploration_rate -= 10*exploration_decay/episodes
                else:
                    exploration_rate = min_exploration_rate
            
            # ### for now verbal input is considered truthful --- see (5b) for explanations
            # self.exploitation_strategy(received_input, episode)
            # if self.exploit_mode and self.exploit_start_episode <= episode < self.exploit_end_episode:
            #     if exploration_rate > 0.1:
            #         exploration_rate -= exploration_decay/episodes
            #     else:
            #         exploration_rate = min_exploration_rate
            #     if episode == self.exploit_end_episode - 1:
            #         self.exploit_mode = False
            #         print(f"Exploitation mode ends at episode {episode}")
            # else:
            #     if exploration_rate > 0.1:
            #         exploration_rate -= 5*exploration_decay/episodes
            #     else:
            #         exploration_rate = min_exploration_rate
            

            rewards.append(total_reward)
            total[episode] = Rewards
        print(f"POIs identified during training {self.env.POIs}")
        print(f"fires identified during training {self.env.fires}")

        return rewards, total, visit_counts, attention_space_OPTIONS, attention_space_PRIMITIVES