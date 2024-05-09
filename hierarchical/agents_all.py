import numpy as np
import random
from tqdm.auto import tqdm

### QLearningAgent --> Flat
class QLearningAgent:
  def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, EPSILON_MIN, DECAY_RATE, EPISODES, name='Flat'):
    self.env = env
    self.ALPHA = ALPHA
    self.GAMMA = GAMMA
    self.EPSILON_MAX = EPSILON_MAX
    self.EPSILON_MIN = EPSILON_MIN
    self.EPISODES = EPISODES
    self.DECAY_RATE = DECAY_RATE
    self.name = name
    self.explored_count = 0
    self.exploited_count = 0
    self.exploration_counts = []
    self.exploitation_counts = []
    self.Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.previous_Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.EPSILON = EPSILON_MAX
    self.real_interaction = False

  def max_Action(self, state):
    #values = np.array([self.Q[state, a] for a in env.get_actionspace()])
    values = np.array([self.Q[(state, a)] for a in self.env.get_actionspace()])
    greedy_action = np.argmax(values)
    return greedy_action
        

  def get_action(self, state):
    rand = np.random.random()
    if rand < self.EPSILON:
      self.explored_count += 1  # Increment exploration count
      a = np.random.choice(self.env.get_actionspace())
    else:
      self.exploited_count += 1  # Increment exploitation count
      a = self.max_Action(state)
    return a

  def decay_epsilon(self):
    if self.EPSILON > 0.1:
      self.EPSILON -= self.DECAY_RATE/self.EPISODES
    else:
      self.EPSILON = self.EPSILON_MIN
    return self.EPSILON

  def update(self, state, action, next_state, next_action, reward):
    target = reward + self.GAMMA*self.Q[next_state, next_action]
    td_error = self.ALPHA*(target - self.Q[state, action])
    self.Q[state, action] = self.Q[state, action] + td_error
    # self.Q[state, action] = self.Q[state, action] + self.ALPHA*(reward + self.GAMMA*self.Q[next_state, next_action] - self.Q[state, action])
    return self.Q

  def train_agent(self):
    total_rewards = np.zeros(self.EPISODES)
    Rewards = 0
    visit_counts = {}
    steps_per_episode = []
    Q_history = []
    epsilon_values = []

    for ep in tqdm(range(self.EPISODES)):
      if ep % 250 == 0:
        print(f"episode: {ep} | reward: {Rewards} | ε: {self.EPSILON}")

      s = self.env.reset()
      done = False
      Rewards = 0
      steps = 0
      while not done:
                
        a = self.get_action(s)
        s_, r, done, info = self.env.step(a)
        #r = self.compute_reward(s, s_, a)
        a_ = self.max_Action(s_)
        # env.render()
        # time.sleep(0.05)
        Rewards += r
        updated_Q = self.update(s, a, s_, a_, r)
        # print(f"State: {s} | AV_ACT: {env.get_actionspace()} | Action: {a} | next_state: {s_} | NEXT_AV_ACT: {env.get_actionspace()} | Next_Action: {a_} | Reward: {r}")
        s = s_
        steps += 1


        if s[0] in visit_counts:
            visit_counts[s[0]] += 1
        else:
            visit_counts[s[0]] = 1

      self.EPSILON = self.decay_epsilon()
      # if ep % 2000 == 0:
      #   Q_history.append(updated_Q.copy())
      total_rewards[ep] = Rewards
      steps_per_episode.append(steps)
      self.exploration_counts.append(self.explored_count)
      self.exploitation_counts.append(self.exploited_count)
      epsilon_values.append(self.EPSILON)
    print(f"Exploration count: {self.explored_count}")
    print(f"Exploitation count: {self.exploited_count}")


    return updated_Q, total_rewards, visit_counts, self.exploration_counts, self.exploitation_counts, steps_per_episode, epsilon_values


### AGENT RECEIVES INFORMATION -- UPDATES THE ENVIRONMENT -- ACTS BASED ON THAT
### QLearningAgentAttention --> Flat_Attention
class QLearningAgentAttention:
  def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, EPSILON_MIN, DECAY_RATE, EPISODES, name='Flat-Attention'):
    self.env = env
    self.ALPHA = ALPHA
    self.GAMMA = GAMMA
    self.EPSILON_MAX = EPSILON_MAX
    self.EPSILON_MIN = EPSILON_MIN
    self.EPISODES = EPISODES
    self.DECAY_RATE = DECAY_RATE
    self.name = name
    self.explored_count = 0
    self.exploited_count = 0
    self.exploration_counts = []
    self.exploitation_counts = []
    self.exploit_start_episode = None
    self.exploit_end_episode = None
    self.exploit_episodes = 1000  # 30% of the episodes
    self.exploit_initiated = False
    self.exploit_mode = False
    self.should_exploit = False
    self.received_input = False  # tracks if verbal input has been received
    self.Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.attention_space = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.previous_Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.EPSILON = EPSILON_MAX
    self.POIs_identified = []
    self.fires_identified = []

  ### Identifies states that have changed based on sensor readings.
  def identify_changed_states(self, readings):
    changed_states = [i for i, value in readings.items() if value != 1]
    return changed_states

  ### Finds states connected to a given target state, considering possible actions and their inverse
  def get_connected_states(self, target_state):
    # Inverse action mapping for movement actions
    inverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}
    # Generate state-action pairs
    connected_states_pairs = []
    for action in self.env.actionspace:
        # Skip 'ASK' and 'SAVE' actions
        if action in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            continue
        # Apply the inverse action to the target state
        inverse_action = inverse_actions[action]
        possible_prev_state = self.env.next_state_vision(target_state, inverse_action)
        # Check if the resulting state is valid and leads to the target state
        if possible_prev_state != target_state and possible_prev_state in self.env.statespace and possible_prev_state[0] not in self.env.ditches:
            connected_states_pairs.append((possible_prev_state, action))
    return connected_states_pairs


  ### Updates the attention space, which influences the agent's decision-making based on environmental changes.
  def update_attention_space(self, connection, sensor_readings):
    connected_states = self.get_connected_states(connection)
    # Determine the value to add based on sensor reading
    value_to_add = 2.0 if sensor_readings[connection] > 0 else -100.0
    for connected_state, action in connected_states:
      # Update the attention space value for the state-action pair
        self.attention_space[(connected_state, action)] = value_to_add
    # # Check if the new information refers to a victim state and update exclusively for SAVE action
    if connection[0] in self.env.victimStates:
        victim_state_action_pair = ((connection[0], True, True, True), 5) # Action 5 corresponds to 'SAVE'
        self.attention_space[victim_state_action_pair] = 100
    return self.attention_space

  def max_Action(self, state):
    #values = np.array([self.Q[state, a] for a in env.get_actionspace()])
    values = np.array([self.Q[(state, a)] for a in self.env.get_actionspace()])
    greedy_action = np.argmax(values)
    return greedy_action

  # ### Chooses an action based on the current policy (exploitation vs. exploration)
  # def get_action(self, state, sensor_readings): # this action is the same to the next state
  #   if self.exploit_mode:
  #     sensor_influenced_states = [state for state, reading in sensor_readings.items() if reading != 1.0]
  #     if sensor_influenced_states:
  #         self.exploited_count += 1
  #         a = self.max_Action(state)
  #     else:
  #         self.explored_count += 1
  #         a = np.random.choice(env.get_actionspace())
  #   else:
  #     rand = np.random.random()
  #     if rand < self.EPSILON:
  #       self.explored_count += 1  # Increment exploration count
  #       a = np.random.choice(env.get_actionspace())
  #     else:
  #       self.exploited_count += 1  # Increment exploitation count
  #       a = self.max_Action(state)
  #   return a

  
  def get_action(self, state, sensor_readings):
    if not self.env.visited_information_state:
      rand = np.random.random()
      if rand < self.EPSILON:
        self.explored_count += 1  # Increment exploration count
        a = np.random.choice(self.env.get_actionspace())
      else:
        self.exploited_count += 1  # Increment exploitation count
        a = self.max_Action(state)
    else:
      # rand = np.random.random()
      # if rand < 0.001:
      #   self.explored_count += 1  # Increment exploration count
      #   a = np.random.choice(env.get_actionspace())
      # else:
        self.exploited_count += 1  # Increment exploitation count
        a = self.max_Action(state)
    return a

  ### Methods for decaying the exploration rate (epsilon) over time
  def decay_epsilon_exploit(self):
    if self.EPSILON > 0.1:
      self.EPSILON -= (8*self.DECAY_RATE)/self.EPISODES
    else:
      self.EPSILON = self.EPSILON_MIN
    return self.EPSILON

  def decay_epsilon(self):
    if self.EPSILON > 0.1:
      self.EPSILON -= self.DECAY_RATE/self.EPISODES
    else:
      self.EPSILON = self.EPSILON_MIN
    return self.EPSILON

  ### Manages the transition between exploration and exploitation modes based on information received
  def exploitation_strategy(self, received, current_episode):
    if received and not self.exploit_mode and not self.exploit_initiated:
        self.exploit_start_episode = current_episode
        self.exploit_end_episode = self.exploit_start_episode + self.exploit_episodes
        self.exploit_mode = True
        self.exploit_initiated = True
        print(f"Exploitation mode started at episode {self.exploit_start_episode}")

  ### Updates the Q-values based on the temporal difference learning formula
  def update(self, state, action, next_state, next_action, reward):
    target = reward + self.GAMMA*self.Q[next_state, next_action]
    td_error = self.ALPHA*(target - self.Q[state, action])
    self.Q[state, action] = self.Q[state, action] + td_error
    # self.Q[state, action] = self.Q[state, action] + self.ALPHA*(reward + self.GAMMA*self.Q[next_state, next_action] - self.Q[state, action])
    return self.Q

  ### the training of the agent over a specified number of episodes
  def train_agent(self):
    total_rewards = np.zeros(self.EPISODES)
    Rewards = 0
    visit_counts = {}
    Q_history = []
    steps_per_episode = []
    identified_states = []
    epsilon_values = []
    sensor_readings = self.env.sensor_readings
    attention_space = {}
    for ep in tqdm(range(self.EPISODES)):
      if ep % 250 == 0:
        print(f"episode: {ep} | reward: {Rewards} | ε: {self.EPSILON}")

      s = self.env.reset()
      done = False
      Rewards = 0
      steps = 0
      while not done:

        if self.env.visited_information_state and not self.received_input:
          identified_states = self.identify_changed_states(sensor_readings)
          self.received_input = True
          self.env.generate_annotated_image()
          print(f"Got the info needed at ep {ep} and location {s[0]}")
          if identified_states:
            for next_state in identified_states:
              attention_space = self.update_attention_space(next_state, sensor_readings)
              ## normally, if we'd like to directly import this update to the Q table, we have to set a metric of
              ## confidence .. that this information is concrete
          for key in attention_space.keys():
              self.Q[key] = self.Q[key] + attention_space[key]


        a = self.get_action(s, sensor_readings)
        s_, r, done, info = self.env.step(a)
        a_ = self.max_Action(s_)
        Rewards += r
        updated_Q = self.update(s, a, s_, a_, r)
        # print(f"State: {s} | AV_ACT: {env.get_actionspace()} | Action: {a} | next_state: {s_} | NEXT_AV_ACT: {env.get_actionspace()} | Next_Action: {a_} | Reward: {r}")
        s = s_
        steps += 1


        if s[0] in visit_counts:
            visit_counts[s[0]] += 1
        else:
            visit_counts[s[0]] = 1

      # ### for now verbal input is considered truthful --- see (5b) for explanations
      # self.exploitation_strategy(self.received_input, ep)
      # if self.exploit_mode and self.exploit_start_episode <= ep < self.exploit_end_episode:
      #   self.EPSILON = self.decay_epsilon_exploit()
      #   if ep == self.exploit_end_episode - 1:
      #       self.exploit_mode = False
      #       print(f"Exploitation mode ends at episode {ep}")
      # else:
      #   self.EPSILON = self.decay_epsilon()

      if not self.env.visited_information_state:
        self.EPSILON = self.decay_epsilon()
      else:
        self.EPSILON = self.decay_epsilon_exploit()
        
      
      self.previous_Q = updated_Q.copy()
      total_rewards[ep] = Rewards
      steps_per_episode.append(steps)
      # save the Q table every 500 episodes
      # if ep % 250 == 0:
      #   Q_history.append(updated_Q.copy())
      self.exploration_counts.append(self.explored_count)
      self.exploitation_counts.append(self.exploited_count)
      epsilon_values.append(self.EPSILON)
      self.POIs_identified.append(self.env.POIs)
      self.fires_identified.append(self.env.fires)
    print(f"Exploration count: {self.explored_count}")
    print(f"Exploitation count: {self.exploited_count}")
    print(f"POIs identified during training {self.env.POIs}")
    print(f"fires identified during training {self.env.fires}")

    return updated_Q, total_rewards, visit_counts, self.exploration_counts, self.exploitation_counts, steps_per_episode, epsilon_values, attention_space



### Hiearchical Q learning --> HRL (multiple workers - action decomposition - hierarchical structure)
class HierarchicalQLearningAgent:
    def __init__(self, env, state_space, action_space, learning_rate=0.1, discount_factor=0.998, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, decay_rate=2, name='HRL'):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.env = env
        self.name = name

    def choose_action(self, state, evaluation=False):
        state_index = self.state_to_index(state)
        if evaluation or np.random.rand() > self.epsilon:
            return np.argmax(self.q_table[state_index])
        else:
            return np.random.randint(0, self.q_table.shape[1])

    def update(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error

    def state_to_index(self, state):
        # Convert state to index for the Q-table
        return np.dot(state, [1, 7, 49, 98, 196, 392]) 

    def decay_epsilon(self, episodes):
        if self.epsilon > 0.1:
            self.epsilon -= self.decay_rate / episodes
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon


    def train(self, manager, workers, num_episodes):
        total_rewards = np.zeros(num_episodes)
        Rewards = 0
        for episode in range(num_episodes):
            if episode % 250 == 0:    
                print(f"Episode: {episode}, Total Reward: {Rewards}, Exploration Rate: {workers[0].epsilon}")

            total_reward, Rewards = 0, 0
            state = self.env.reset()
            done = False
            step_count = 0
            while not done:
                option = self.env.current_option  # Get current option from the environment
                worker = workers[option]  # Choose the correct worker for the option
                action = worker.choose_action(state)  # Worker decides the action
                next_state, reward, done, _ = self.env.step(action)  # Execute the action in the environment
                Rewards += reward
                # Update the worker's Q-table
                worker.update(state, action, reward, next_state)
                manager.update(state, option, reward, next_state)

                # Print the details of the current step
                # print(f"  Step {step_count + 1}: State {state}, Option {option}, Action {action}, Reward {reward}, Next State {next_state}")
                
                # Move to the next state
                state = next_state
                total_reward += reward
                step_count += 1
            total_rewards[episode] = Rewards
            manager.epsilon = manager.decay_epsilon(num_episodes)
            for w in workers.values():
                    w.decay_epsilon(num_episodes)
            # workers[0].epsilon = workers[0].decay_epsilon(num_episodes)
            # workers[1].epsilon = workers[1].decay_epsilon(num_episodes)
            # workers[2].epsilon = workers[2].decay_epsilon(num_episodes)
        
        return total_rewards, workers





### Hiearchical Q learning --> HRL-Attention  (multiple workers - action decomposition - hierarchical structure - attention)
class HierarchicalQLearningAgentAttention:
    def __init__(self, env, state_space, action_space, learning_rate=0.1, discount_factor=0.998, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, decay_rate=2, name='HRL-Attention'):
        self.q_table = np.zeros((state_space, action_space))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.attention_space_low = np.zeros((state_space, action_space))
        self.received_input = False
        self.env = env
        self.name = name
    
    ### Identifies states that have changed based on sensor readings.
    def identify_changed_states(self, readings):
        changed_states = [i for i, value in readings.items() if value != 1]
        return changed_states

    ### Finds states connected to a given target state, considering possible actions and their inverse
    def get_connected_states(self, target_state):
        # Inverse action mapping for movement actions
        inverse_actions = {0: 1, 1: 0, 2: 3, 3: 2}
        # Generate state-action pairs
        connected_states_pairs = []
        for action in range(self.q_table.shape[1]):
            inverse_action = inverse_actions[action]
            possible_prev_state = self.env.next_state_vision(target_state, inverse_action)
            # print(possible_prev_state)
            # Check if the resulting state is valid and leads to the target state
            if possible_prev_state != tuple([target_state[0], target_state[1]]) and possible_prev_state not in self.env.ditches:
                connected_states_pairs.append((possible_prev_state, action))
        return connected_states_pairs
    
    ### Updates the attention space, which influences the agent's decision-making based on environmental changes.
    def update_attention_space(self, connection, readings):
        connected_states = self.get_connected_states(connection)
        # Determine the value to add based on sensor reading
        value_to_add = 2.0 if readings[connection] > 0 else -100.0
        for connected_state, action in connected_states:
            # print(connected_state)
        # Update the attention space value for the state-action pair
            updated_state = [connected_state[0], connected_state[1], 1, 1, 1, 0]
            updated_state_index = self.state_to_index(updated_state)
            self.attention_space_low[updated_state_index][action] = value_to_add
        # # Check if the new information refers to a victim state and update exclusively for SAVE action
        if tuple([connection[0], connection[1]]) == self.env.final_location:
            updated_victim_state = [connection[0], connection[1], 1, 1, 1, 0]
            updated_victim_state_index = self.state_to_index(updated_victim_state)
            self.attention_space_low[updated_victim_state_index][0] = 100
            ##  action 'save' is highly favored in the Q-table when the agent is at the final location, guiding the save_worker to prioritize this action.
        return self.attention_space_low
    
    def apply_attention_to_q_table(self):
        for index, action_values in np.ndenumerate(self.attention_space_low):
            state_index, action = index
            if action_values != 0:
                self.q_table[state_index][action] = action_values
                
    
    def choose_action(self, state, evaluation=False):
        state_index = self.state_to_index(state)
        if not self.env.visited_information_state:
            if evaluation or np.random.rand() > self.epsilon:
                return np.argmax(self.q_table[state_index])
            else:
                return np.random.randint(0, self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state_index])
            

    def update(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
        td_error = td_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.learning_rate * td_error
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def state_to_index(self, state):
        # Convert state to index for the Q-table
        return np.dot(state, [1, 7, 49, 98, 196, 392])

    def decay_epsilon(self, episodes):
        if self.epsilon > 0.1:
            self.epsilon -= self.decay_rate / episodes
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon
    
    def decay_epsilon_exploit(self, episodes):
        if self.epsilon > 0.1:
            self.epsilon -= (8*self.decay_rate) / episodes
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon


    def train(self, manager, workers, num_episodes):
        total_rewards = np.zeros(num_episodes)
        Rewards = 0
        attention_space = {}  
        for episode in range(num_episodes):
            if episode % 250 == 0:    
                print(f"Episode: {episode}, Total Reward: {Rewards}, Exploration Rate: {workers[0].epsilon}")

            total_reward, Rewards = 0, 0
            state = self.env.reset()
            done = False
            step_count = 0
            while not done:
                if self.env.visited_information_state and not manager.received_input:
                    identified_states = manager.identify_changed_states(self.env.sensor_readings)
                    manager.received_input = True
                    print(f"Got the info needed at ep {episode} and location {tuple([state[0], state[1]])}")
                    if identified_states:
                        # print(f"identified states: {identified_states}")
                        for each_state in identified_states:
                            if tuple([each_state[0], each_state[1]]) != self.env.final_location: 
                                attention_space = workers[0].update_attention_space(each_state, self.env.sensor_readings)
                                workers[0].apply_attention_to_q_table()  # Update the Q-table of the explorer worker
                            else:
                                attention_space = workers[2].update_attention_space(each_state, self.env.sensor_readings)
                                workers[2].apply_attention_to_q_table()  # Update the Q-table of the save worker
                    
                option = self.env.current_option  # Get current option from the environment
                worker = workers[option]  # Choose the correct worker for the option
                action = worker.choose_action(state)  # Worker decides the action
                next_state, reward, done, _ = self.env.step(action)  # Execute the action in the environment
                Rewards += reward
                # Update the worker's Q-table
                worker.update(state, action, reward, next_state)
                manager.update(state, option, reward, next_state)

                # Print the details of the current step
                # print(f"  Step {step_count + 1}: State {state}, Option {option}, Action {action}, Reward {reward}, Next State {next_state}")
                
                # Move to the next state
                state = next_state
                total_reward += reward
                step_count += 1
            total_rewards[episode] = Rewards
            if not self.env.visited_information_state:
                manager.epsilon = manager.decay_epsilon(num_episodes)
                for w in workers.values():
                    w.decay_epsilon(num_episodes)
            else:
                manager.epsilon = manager.decay_epsilon_exploit(num_episodes)
                for w in workers.values():
                    w.decay_epsilon_exploit(num_episodes)
        
        return total_rewards, attention_space, workers

"""This setup ensures that the attention mechanism's outputs 
are integrated into the Q-learning process, allowing the agents to adaptively focus 
on areas of interest based on new environmental information. 
Each agent's Q-table is dynamically updated based on these attention adjustments, 
and the training process reflects these changes in real-time to enhance decision-making."""