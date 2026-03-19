from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class LearningHumanPTAgent:
    """
    Learning Human PT Agent Doesn't know game structure, learns via RL, PT preferences at decision time

    Implements epsilon-greedy Q learning with a joint action Q table Q(s, a_i, a_-i) 
    which is split so that we can apply beliefs to weigh the opponent action values and then maximize over a regular
    Q(s, a) table. 

    parameter choices are all pretty standard, values are converging so i haven't spent much time tuning those. 

    There is a tie break logic for when pt values at decision time are very similar (< tau), 
    in which case we use softmax and randomize. 

    Running belief, reference point scalars are tracked and updated after each step of the environment
    """

    def __init__(self, state_size, action_size, opp_action_size, pt_params, agent_id=0, ref_setting='Fixed', lambda_ref=0.95, payoff_matrix = None, B=5):
        self.state_size = state_size * B - 1
        self.action_size = action_size
        self.opp_action_size = opp_action_size
        self.pt = ProspectTheory(**pt_params)
        self.agent_id = agent_id
        self.ref_point = pt_params['r']


        self.max_payoff, self.min_payoff = payoff_matrix[:, :, agent_id].max(), payoff_matrix[:, :, agent_id].min()

        self.B = B

        # Initialize beliefs function and q values as dictionaries
        self.beliefs = dict()
        self.q_values = dict()

        # Initialize belief and reference point lambda parameters. 0.95 is a standard setting, carries across episodes
        self.lam_b = 0.95


        self.init_lam_ref = lambda_ref
        self.lam_r = self.init_lam_ref
        self.ref_k = 0.6

        # Set reference point update mode:
        # options: Fixed, EMA, Max Q value (conditioned on beliefs over opp actions), 
        # EMAOR (Opponent reward not own reward)
        self.ref_update_mode = ref_setting 
       
        # Add an entry for each state populated with uniform probabilities over opponent action set size
        # And initialize q values
        for state in range(self.state_size + 1):
            # Belief function is size: opp action size because they are beliefs over opponent actions
            # we divide by the action size to get equiprobably starting points
            self.beliefs[state] = np.ones(self.opp_action_size) / self.opp_action_size
            
            # Q-values Q(s, a_i, a_-i) represent joint action estimates
            self.q_values[state] = np.zeros((self.action_size, self.opp_action_size)) 

        # track state visits for analysis
        self.state_visit_counter = dict()
        # Track pathology triggers
        self.softmax_counter = 0

        # Q-learning parameters, all from paper. Perhaps gamma could be set to 0.99?
        #self.gamma = 0.95 
        self.avg_rew = 0.0
        self.epsilon = 0.3
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.alpha = 0.1 # standard init alpha value from what I can tell
        self.init_alpha = self.alpha
        self.k = 0.7

        # Pathology Detection parameters
        self.tau = 0.1 # Threshold parameter

        self.temperature = 1.3 # softmax temperature, high to encourage randomness in the tie breaks for exploration

        # Track raw vs PT rewards
        self.raw_rewards = []
        self.pt_rewards = []
       
        self.pt_l2_dists = []
        self.action_changed_flags = []

    def transform_state(self, state):
        # Transform the states from the s(H) to s(H)B format
        # First, normalize for simplicity
        low = min(self.min_payoff, self.ref_point)
        high = max(self.max_payoff, self.ref_point)

        self.min_payoff, self.max_payoff = low, high

        denom = high - low
        if denom == 0:
            norm_ref_point = 0
        else:
            norm_ref_point = (self.ref_point - low) / denom

        ref_bin = None

        ref_bin = min(int(norm_ref_point * self.B), self.B - 1)

        if ref_bin is None:
            raise TypeError("Ref bin never updated")

        pt_state = state * self.B + ref_bin
        return pt_state

    def act(self, state):
        # Epsilon exploration (lines 16-17 in alg 1)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Pathology detection (lines 14, 18, 19 in alg 1)
        ## Generate action values (see method below)
        action_values, EU_action_values = self.calculate_action_values(state)

        # We get the L2 distance between the EU and CPT actions 
        PT_L2_dist = np.linalg.norm(action_values - EU_action_values)

        ## Identify Optimal action for tie break check
        optimal_action = np.argmax(action_values) 
 
        # And then we check if the PT transformation caused a different action to be taken
        EU_optimal_action = np.argmax(EU_action_values)

        # And then, just to guard against argmax tiebreaking in weird ways, 
        # We mark all near tie values to be False to ensure integrity
        tol = 1e-8

        EU_gap = EU_action_values[0] - EU_action_values[1]
        PT_gap = action_values[0] - action_values[1]

        # ignore ties
        EU_tie = abs(EU_gap) < tol
        PT_tie = abs(PT_gap) < tol

        if EU_tie or PT_tie:
            action_changed = 0
        else:
            action_changed = int(np.sign(EU_gap) != np.sign(PT_gap))
 

        self.pt_l2_dists.append(PT_L2_dist)
        self.action_changed_flags.append(action_changed)

        ## Identify second best action for tie break check
        ### copy to prevent mutating original
        non_optimal_actions = action_values.copy()
        ### remove the best action
        non_optimal_actions[optimal_action] = -np.inf
        ### take the best action with the original best action removes
        second_best_action = non_optimal_actions.max()

        ### find the difference
        gap = action_values[optimal_action] - second_best_action
        #print(f"[Debug] gap value LH: {gap}")

        ## Check for pathology:
        if gap < self.tau:
            self.softmax_counter += 1
            ##Softmax
            ## Normalize to help prevent logit explosion, doesnt change answer just stability
            vals = action_values - action_values.max()
            # Recall temp is set to 1.3
            action_probs = softmax(vals/self.temperature, axis=0)
            # Sample action randomly with the softmax odds
            action = np.random.choice(len(action_probs), p=action_probs)
            return action
        
        # Optimal Action (lins 20, 21 in alg 1)
        # If its not a tie, we just return the best option. 
        else:
            return int(optimal_action)

    def calculate_action_values(self, state):
        # Define action space
        action_values = np.zeros(self.action_size)
        # This is for tracking the effect the PT transformation has (EU baseline)
        EU_action_values = np.zeros(self.action_size)
        ## Calculate V for each state, opp action tuple by forming a lottery 
        # with Q vals and beliefs
        # The idea here is that each effective Q value Q(s, a) is conditioned on 
        # the beliefs over what the opp reply will be. 
        # Otherwise we would just be naively estimating values. 
        # Models how humans learn from experience, 
        # ***this holds for one shot and repeated***
        for action in range(self.action_size):
            probabilities = self.beliefs[state] 

            # An old safeguard
            assert np.isclose(probabilities.sum(), 1.0, atol=1e-5), \
            "Beliefs don't sum to 1"

            # select the opp action values (x, y) for the current state action pair
            # length self.opp_action_size
            outcomes = self.q_values[state][action]

            # PT transformation using opp action values and beliefs, 
            # introduces human warping instead of just integrating
            action_val = self.pt.expected_pt_value(outcomes, probabilities) 
            # Get EU value
            EU_action_val = outcomes.dot(probabilities)

            EU_action_values[action] = EU_action_val

            # Update the list
            action_values[action] = action_val

        return action_values, EU_action_values

    def belief_update(self, state, opp_action):
        # Simple EMA for belief updates
        one_hot = np.zeros(self.opp_action_size)
        one_hot[opp_action] = 1
        self.beliefs[state] = self.lam_b * self.beliefs[state] + (1 - self.lam_b) * one_hot

    def alpha_update(self, state, action):
        step = self.state_visit_counter[state][action]
        self.alpha = self.init_alpha / step ** self.k 

    def lambda_ref_update(self):
        step = sum(sum(v) for v in self.state_visit_counter.values())
        self.lam_r = self.init_lam_ref / step ** self.ref_k

    def ref_update(self, payoff, state, opp_payoff):
        # Just slowly moving in the new direction instead of all at once, this seems sufficient for our pruposes
        # alternatively we could do some kind of bayesian update, but that feels like overkill to me
        if self.ref_update_mode == "EMA":
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoff

        # This is where we select the V val for our reference point (very greedy human)
        # We talked about this in meeting a couple of times, please let me know if it holds
        elif self.ref_update_mode == 'V':
            # Set reference point to maximum, normalized q value
            '''
            weighted_q_vals = np.zeros(self.action_size)
            for action in range(self.action_size):
                # Beliefs are over opp actions, so we take the PT transformation for the lottery here 
                # PT transformation here made ref points explode, so usign EU
                weighted_q_val = self.pt.expected_pt_value(self.q_values[state][action], self.beliefs[state])
                print("Weighted Q val for action ", action, " : ", weighted_q_val)
                weighted_q_vals[action] = weighted_q_val
            ''' 

            q_vals = self.get_q_values()
            #print("Q vals: ", q_vals)
            beliefs = self.get_avg_beliefs()
            weighted_q_vals = q_vals.dot(beliefs)
            # Since our policy is just epsilon greedy, we weigh the q vals wrt the policy
            # To get a V(s)
            greedy_action = np.argmax(weighted_q_vals)
            policy = np.ones(self.action_size) * (self.epsilon / self.action_size) 
            policy[greedy_action] += 1.0 - self.epsilon

            V_val = policy.dot(weighted_q_vals)
            #print("V val: ", V_val)
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * V_val

            #print("ref poitn: ", self.ref_point)

        elif self.ref_update_mode == 'EMAOR':
            # EMA, but now using the opponents rewards 
            # (to test to see if knowledge about how other player is doing changes behavior)
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * opp_payoff
            
        # Make sure to update ref point in pt function
        self.pt.r = self.ref_point

    def q_value_update(self, state, next_state, action, opp_action, reward, done=False):
        '''The point here is to align our agent with PT-EB principles. We treat our own decision making
        as certain (so q values are in reward/outcome space), but we allot our uncertainty to opp actions.
        That is why you will see the expectation taken over opponent actions conditioned on our beliefs
        in the boot strap. 
        '''
        if state not in self.state_visit_counter.keys():
            self.state_visit_counter[state] = [0] * self.action_size

        self.state_visit_counter[state][action] += 1

        self.lambda_ref_update()
        self.alpha_update(state, action)

        ## get avg state value for q vals and beliefs
        q_values = self.get_q_values()
        beliefs = self.get_avg_beliefs()
        #print(f"beliefs: {beliefs}")

        # Get maximuj value (not index)
        ## - inf because rewards can be negative
        avg_state_q_value = -np.inf

        eps = 1e-8 # For tie breaks
  
        # Integrate out opp actions (Q(s, a_i, a_-i) -> Q(s, a_i)) in line with PT EB philosophy
        # Then get the max for the bellman update (max Q(s, a))
        for a_prime in range(self.action_size):
            q_val = q_values[a_prime] # this is the average value of this action across states

            # linear expectation of beliefs and values (integrate out opp acts)
            # no pt here because we don't want non linearities in the learning space, just in the decision space
            # preserves rl guarantees and behavior, importantly
            # Future work could include the pt transformation here, Phade did it in one of the papers i found
            weighted_q_val = np.dot(beliefs, q_val)

            if weighted_q_val > avg_state_q_value + eps:
                avg_state_q_value = weighted_q_val

        # Get stored value (state, joint action value) for bootstrap 
        q_value = self.q_values[state][action][opp_action]

        # Calculate current state value
        state_q_val = max(
            self.q_values[state][a].dot(self.beliefs[state])
            for a in range(self.action_size)
        )

        # Calculate delta in untransformed reward space
        delta = reward + state_q_val - avg_state_q_value
        # Update q values
        self.q_values[state][action][opp_action] = (1 - self.alpha) * q_value + self.alpha * delta

    # Deprecated from the q value convergence metric i was talking about
    # im not removing it because I am attached to it, but its not hurting anything
    def get_q_values(self):
        q_values = np.zeros((self.action_size, self.opp_action_size))

        total_visits = sum(sum(v) for v in self.state_visit_counter.values())

        if total_visits == 0:
            return q_values

        for state, q_vals in self.q_values.items():
            num_visits = sum(self.state_visit_counter.get(state, [0] * self.action_size))

            if num_visits == 0:
                continue

            weight = num_visits / total_visits

            q_values += weight * q_vals

        return q_values

    def get_avg_beliefs(self):
        avg_beliefs = np.zeros(self.opp_action_size)

        total_visits = sum(sum(v) for v in self.state_visit_counter.values())
  
        if total_visits == 0:
            return avg_beliefs


        for state, value in self.beliefs.items():
            num_visits = sum(self.state_visit_counter.get(state, [0] * self.action_size))
            weight = num_visits / total_visits

            avg_beliefs += weight * value

        return avg_beliefs
            




