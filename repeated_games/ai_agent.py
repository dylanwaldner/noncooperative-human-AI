import copy
import numpy as np
import random
from scipy.special import softmax

class AIAgent:
    """
    Standard RL agent without PT
    Implements vanilla epsilon greedy q learning, no PT transformations, no beliefs, no reference points
    Should be super straightforward
    """

    def __init__(self, state_size, action_size, opp_action_size, agent_id=0):
        self.state_size = state_size
        self.action_size = action_size
        self.opp_action_size = opp_action_size
        self.agent_id = agent_id

        # Initialize q values as a dictionary
        self.q_values = dict()
       
        # Add an entry for each state 
        # And initialize q values as Q(s, a), no opponent conditioning because no beliefs for AI
        for state in range(self.state_size):
            self.q_values[state] = np.zeros(self.action_size) 

        # state counter
        self.state_visit_counter = dict()

        # Q-learning parameters, set from code i inherited, and all is converging so I see no issue. 
        # self.gamma = 0.95
        self.avg_rew = 0.0
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01

        # tiebreaker variables
        self.tau = 0.1 # threshold
        self.temp = 1.3 # softmax temp, high to encourage randomness
        self.softmax_counter = 0

    def act(self, state):
        # Epsilon greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Get optimal action from q values for this state
        q_values = self.q_values[state]
        optimal_action = np.argmax(q_values).item() 

        # Get the second best value for pathology detection
        # Copy to prevent editing original list
        suboptimal_q_values = q_values.copy()
        # mask out current best
        suboptimal_q_values[optimal_action] = -np.inf
        # get best in list with current best msked
        second_best_action = np.argmax(suboptimal_q_values)

        # find difference between best and second best
        gap = q_values[optimal_action] - q_values[second_best_action]

        if gap < self.tau:
            # Log tie break
            self.softmax_counter += 1

            vals = q_values - q_values.max() # Normalize to prevent explosions
            probs = softmax(vals / self.temp, axis=0)
            action = np.random.choice(len(probs), p=probs) # sample

            return action

        # theres no tie, just return the best
        else:
            return optimal_action
 

    def update(self, state, action, next_state, reward=None, done=False):
        ''' Just vanilla q learning here, no PT and nothing fancy like with the beliefs. 
        '''
        assert reward is not None, "Reward Undefined"
        
        # Update state count
        if state not in self.state_visit_counter.keys():
            self.state_visit_counter[state] = 0

        self.state_visit_counter[state] += 1

        # Get the present state value
        curr_q_val = self.q_values[state][action]

        # Get the max val for the next state
        max_next_q_val = self.q_values[next_state].max()

        if self.state_size == 1:
            max_next_q_val = 0
 
        # TD update
        target = reward - self.avg_rew + max_next_q_val

        q_vals = self.get_q_values()


        # COnvex combination style, just a stylistic difference not a numeric one
        self.q_values[state][action] = (1 - self.alpha) * curr_q_val + self.alpha * target  
       
    # Deprecated Code for the Q-Value convergence metric I was fixated on
    # keeping it in because, im attached
    def get_q_values(self):
        q_values = np.zeros((self.action_size, self.opp_action_size))

        total_visits = sum(self.state_visit_counter.values())

        if total_visits == 0:
            return q_values

        for state, q_val in self.q_values.items():
            num_visits = self.state_visit_counter.get(state, 0)

            if num_visits == 0:
                continue

            weight = num_visits / total_visits

            q_values += weight * q_val

        return q_values
