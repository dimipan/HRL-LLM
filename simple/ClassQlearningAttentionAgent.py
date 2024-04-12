import numpy as np
from tqdm.auto import tqdm

### policy-shaping
### AGENT RECEIVES INFORMATION -- UPDATES THE ENVIRONMENT -- ACTS BASED ON THAT
class QLearningAgentAttention:
  def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, EPSILON_MIN, DECAY_RATE, EPISODES, image_path):
    self.ALPHA = ALPHA
    self.GAMMA = GAMMA
    self.EPSILON_MAX = EPSILON_MAX
    self.EPSILON_MIN = EPSILON_MIN
    self.EPISODES = EPISODES
    self.DECAY_RATE = DECAY_RATE
    self.env = env
    self.image_path = image_path
    self.explored_count = 0
    self.exploited_count = 0
    self.exploration_counts = []
    self.exploitation_counts = []
    self.exploit_start_episode = None
    self.exploit_end_episode = None
    self.exploit_episodes = self.EPISODES * 0.3  # 30% of the episodes
    self.exploit_initiated = False
    self.exploit_mode = False
    self.should_exploit = False
    self.received_input = False  # tracks if verbal input has been received
    self.similarity_threshold = 0.9
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
        if action in [4, 5]:
            continue
        # Apply the inverse action to the target state
        inverse_action = inverse_actions[action]
        possible_prev_state = self.env.next_state_vision(target_state, inverse_action)
        # Check if the resulting state is valid and leads to the target state
        if possible_prev_state != target_state and possible_prev_state in self.env.statespace and possible_prev_state[0] not in self.env.ditches:
            connected_states_pairs.append((possible_prev_state, action))
    return connected_states_pairs


  ### Updates the attention space, which influences the agent's decision-making based on environmental changes. ### more like policy-shaping
  def update_attention_space(self, connection, sensor_readings):
    connected_states = self.get_connected_states(connection)
    # Determine the value to add based on sensor reading
    value_to_add = 10.0 if sensor_readings[connection] > 0 else -10.0
    for connected_state, action in connected_states:
      # Update the attention space value for the state-action pair
        self.attention_space[(connected_state, action)] = value_to_add
    # # Check if the new information refers to a victim state and update exclusively for SAVE action
    if connection[0] in self.env.victimStates:
        victim_state_action_pair = ((connection[0], True), 5) # Action 5 corresponds to 'SAVE'
        self.attention_space[victim_state_action_pair] = 100
    return self.attention_space

  def max_Action(self, state):
    #values = np.array([self.Q[state, a] for a in env.get_actionspace()])
    values = np.array([self.Q[(state, a)] for a in self.env.get_actionspace()])
    greedy_action = np.argmax(values)
    return greedy_action

  ### Chooses an action based on the current policy (exploitation vs. exploration)
  def get_action(self, state, sensor_readings): # this action is the same to the next state
    if self.exploit_mode:
      sensor_influenced_states = [state for state, reading in sensor_readings.items() if reading != 1.0]
      if sensor_influenced_states:
          self.exploited_count += 1
          a = self.max_Action(state)
      else:
          self.explored_count += 1
          a = np.random.choice(self.env.get_actionspace())
    else:
      rand = np.random.random()
      if rand < self.EPSILON:
        self.explored_count += 1  # Increment exploration count
        a = np.random.choice(self.env.get_actionspace())
      else:
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
          self.env.generate_annotated_image(self.image_path)
          print(f"Got the info needed at ep {ep}")
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
        # print(f"State: {s} | AV_ACT: {self.env.get_actionspace()} | Action: {a} | next_state: {s_} | NEXT_AV_ACT: {self.env.get_actionspace()} | Next_Action: {a_} | Reward: {r}")
        s = s_
        steps += 1


        if s[0] in visit_counts:
            visit_counts[s[0]] += 1
        else:
            visit_counts[s[0]] = 1

      ### for now verbal input is considered truthful --- see (5b) for explanations
      self.exploitation_strategy(self.received_input, ep)
      if self.exploit_mode and self.exploit_start_episode <= ep < self.exploit_end_episode:
        self.EPSILON = self.decay_epsilon_exploit()
        if ep == self.exploit_end_episode - 1:
            self.exploit_mode = False
            print(f"Exploitation mode ends at episode {ep}")
      else:
        self.EPSILON = self.decay_epsilon()

      self.previous_Q = updated_Q.copy()
      total_rewards[ep] = Rewards
      steps_per_episode.append(steps)
      # save the Q table every 500 episodes
      if ep % 250 == 0:
        Q_history.append(updated_Q.copy())
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