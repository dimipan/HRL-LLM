
import numpy as np
from tqdm.auto import tqdm

class QLearningAgent:
  def __init__(self, env, ALPHA, GAMMA, EPSILON_MAX, EPSILON_MIN, DECAY_RATE, EPISODES):
    self.env = env
    self.ALPHA = ALPHA
    self.GAMMA = GAMMA
    self.EPSILON_MAX = EPSILON_MAX
    self.EPSILON_MIN = EPSILON_MIN
    self.EPISODES = EPISODES
    self.DECAY_RATE = DECAY_RATE
    self.explored_count = 0
    self.exploited_count = 0
    self.exploration_counts = []
    self.exploitation_counts = []
    self.Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.previous_Q = {(s, a): 0.0 for s in self.env.get_statespace() for a in self.env.get_actionspace()}
    self.EPSILON = EPSILON_MAX
    self.real_interaction = False

  def max_Action(self, state):
    #values = np.array([self.Q[state, a] for a in self.env.get_actionspace()])
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
        # self.env.render()
        # time.sleep(0.05)
        Rewards += r
        updated_Q = self.update(s, a, s_, a_, r)
        # print(f"State: {s} | AV_ACT: {self.env.get_actionspace()} | Action: {a} | next_state: {s_} | NEXT_AV_ACT: {self.env.get_actionspace()} | Next_Action: {a_} | Reward: {r}")
        s = s_
        steps += 1
        
        if s[1] and s[0] in self.env.fires:
          Rewards -= 2


        if s[0] in visit_counts:
            visit_counts[s[0]] += 1
        else:
            visit_counts[s[0]] = 1

      self.EPSILON = self.decay_epsilon()
      if ep % 250 == 0:
        Q_history.append(updated_Q.copy())
      total_rewards[ep] = Rewards
      steps_per_episode.append(steps)
      self.exploration_counts.append(self.explored_count)
      self.exploitation_counts.append(self.exploited_count)
      epsilon_values.append(self.EPSILON)
    print(f"Exploration count: {self.explored_count}")
    print(f"Exploitation count: {self.exploited_count}")


    return updated_Q, total_rewards, visit_counts, self.exploration_counts, self.exploitation_counts, steps_per_episode, epsilon_values