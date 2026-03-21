from .ProspectTheory import ProspectTheory
import numpy as np
import random
from scipy.special import softmax

class AwareHumanPTAgent:
    """
    Sophisticated Aware Human PT Agent
    Knows the game structure and uses PT to compute best responses

    Specifically, has access to opp reference point and uses that to calculate opp best reply to each of its actions
    Then computes the best reply to the opp's best replies. 

    Compared to learning human, no beliefs and no RL, just a best reply agent. 
    """

    def __init__(self, payoff_matrix, pt_params, action_size, state_size, agent_id=0, opp_params=None, ref_setting='Fixed', lambda_ref=0.95, B=5, tit_for_tat=False):
        self.payoff_matrix = payoff_matrix
        self.pt = ProspectTheory(**pt_params)

        self.agent_id = agent_id  # 0 for row player, 1 for column player
        # Defaulted to 0.95, the number here is pretty arbitrary just the reference update parameter

        self.init_lam_ref = lambda_ref
        self.lam_r = self.init_lam_ref
        self.ref_k = 0.9

        self.ref_update_mode = ref_setting

        self.B = B

        self.max_payoff, self.min_payoff = payoff_matrix[:, :, agent_id].max(), payoff_matrix[:, :, agent_id].min()

        self.tit_for_tat = tit_for_tat

        self.ref_point = pt_params['r']
        self.tau = 0.1 # tie break threshold
        self.temperature = 1.3 # High value to encourage randomness

        # env parameters
        self.action_size = action_size
        self.opp_action_size = opp_params['opponent_action_size']
        self.state_size = state_size * B - 1

        # Important to know whether to apply pt transformation
        self.opponent_type = opp_params['opponent_type']

        # Flag whether to apply pt transformation
        if self.opponent_type == "AI":
            self.opp_cpt = False

        else:
            self.opp_cpt = True
            # We are literally constructing the opponent's entire pt function, 
            # and replace it at every environment step
            self.opp_pt = ProspectTheory(**opp_params['opp_pt'])

        # track ties
        self.softmax_counter = 0 

        # Global step counter
        self.global_steps = 0

        self.pt_l2_dists = []
        self.action_changed_flags = []

    def transform_state(self, state):
        # Transform the states from the s(H) to s(H)B format
        # First, normalize for simplicity
        low = self.min_payoff
        high = self.max_payoff

        denom = high - low
        if denom == 0:
            norm_ref_point = 0
        else:
            norm_ref_point = (self.ref_point - low) / denom

        # Clip to 0, 1 for binning
        norm_ref_point = max(0.0, min(1.0, norm_ref_point))

        # Get its bin
        ref_bin = min(int(norm_ref_point * self.B), self.B - 1)

        # Use the bin to get the transformed state
        pt_state = state * self.B + ref_bin
        return pt_state

    def get_opp_br(self, matrix):
        '''
        We start by tracking the opponent replies conditioned on each of our actions. 
        That is, if we play action 0, what will the opponent player, or if we play action 1, what will the opp play
        the return is a len(action_size) array of opp reply indices
        '''

        # Track the indices of the best reply to each of OUR actions
        opp_best_responses = np.zeros(self.action_size, dtype=int)

        # Iterate over our own actions
        for i in range(self.action_size):
            # Temp variable tracks the best response with our each set over OPP actions
            opp_best_value = float("-inf") # value for comparison
            opp_best_response = 0 # index to return
            # now we look at apponent replies to each of our actions
            for j in range(self.opp_action_size):
                opp_value = matrix[i, j, 1 - self.agent_id] # 1 - agent_id keeps this robust to both col/row

                # apply pt transformation if the opp is a pt player
                if self.opp_cpt:
                    opp_value = self.opp_pt.value_function(opp_value)

                # Maximizing action values (eps helps prevent near ties)
                if opp_value > opp_best_value + 1e-8:
                    opp_best_value = opp_value
                    opp_best_response = j

                # Tie Break logic (Maybe random is wrong here?)
                elif np.abs(opp_value - opp_best_value) <= 1e-8:
                    if random.random() < 0.5:
                        opp_best_value = opp_value
                        opp_best_response = j

            # Index in the best response we foound for action i
            opp_best_responses[i] = opp_best_response

        return opp_best_responses

    def get_best_response(self, matrix, opp_best_responses):
        '''
        Now we are looking through our responses and indexing into them with the opponent BRs. 
        The point is that we are assuming that the opp plays that best response, and then we select
        our best response based on the payoffs generated by the opp BR
        '''
        # One value for each of our actions
        best_vals = np.zeros(self.action_size)
        
        # To track the effect of CPT Transformation
        EU_best_vals = np.zeros(self.action_size)

        for i in range(self.action_size):
            # Use precalculated opp response
            opp_response = opp_best_responses[i]
            value = matrix[i, opp_response, self.agent_id] # agent id indexes row/col
            EU_best_vals[i] = value # EU
            # Always PT transforming here — it is degenerate so no need for full lottery (please confirm this)
            # My thinking is that we are not randomizing over our actions ever, and now we have certainty
            # over the opp action, so probabilities are degenerate and value transform is all that matters
            value = self.pt.value_function(value)
            best_vals[i] = value

        pt_l2_dist = np.linalg.norm(best_vals - EU_best_vals)
        self.pt_l2_dists.append(pt_l2_dist)

        # Get max value and second max val for tie breaks
        opt_a = np.argmax(best_vals) # best
        # Check if the CPT transformation influences action decision
        EU_opt_a = np.argmax(EU_best_vals)

        tol = 1e-8

        EU_diff = EU_best_vals[0] - EU_best_vals[1]
        PT_diff = best_vals[0] - best_vals[1]

        if abs(EU_diff) < tol or abs(PT_diff) < tol:
            action_changed = 0
        else:
            action_changed = int(np.sign(EU_diff) != np.sign(PT_diff))

        self.action_changed_flags.append(action_changed)

        subopt_vals = best_vals.copy() # copy to prevent in place mutilation
        subopt_vals[opt_a] = float("-inf") # maksk best option
        subopt_a = np.argmax(subopt_vals) # get max of masked best option list
        # Tie breaks 
        gap = best_vals[opt_a] - best_vals[subopt_a] # find difference
        if gap < self.tau: # if difference is tiny, we call it a tie
            # Log tie break
            self.softmax_counter += 1

            # As defined in the paper's algorithm, normalized here to prevent logit explosions
            vals = best_vals - best_vals.max()
            probs = softmax(vals / self.temperature, axis = 0) # increase randomness
            best_response = np.random.choice(self.action_size, p=probs) # sample

        # if no tie, just return the best response
        else:
            best_response = opt_a

        return best_response

    def act(self, last_opp_action=None):
        if not self.tit_for_tat:
            self.global_steps += 1
            self.lam_ref_update()

            matrix = self.payoff_matrix
            # Make robust to col/row designation
            if self.agent_id == 1:
                matrix = matrix.transpose(1, 0, 2)


            # First we need to get the opp best responses to our actions 
            opp_best_responses = self.get_opp_br(matrix)

            # Now the decision matrix has gone from 2x2 -> 2x1. We plug in each opp response and 
            # argmax the best action we can take conditioned on how the opponent will reply
            player_best_response = self.get_best_response(matrix, opp_best_responses)

        else:
            # tit for tat
            player_best_response = last_opp_action

        return player_best_response

    def lam_ref_update(self):
        self.lam_r = self.init_lam_ref / self.global_steps ** self.ref_k        

    def ref_update(self, payoff, state, opp_payoff):
        '''
        EMA handles gradual updates, Q says based on our knowledge whats the **best** that we can do,
        EMAOR just sets reference point conditioned on opp rewards (we had a conversation talking about the psychology
        of how good it *could* be)
        Fixed doesnt get updated, its fixed
        Still made sense to update the reference points for the AH, 
        ***technically this is learning though so maybe you want to handle it another way***
        '''
        # here it made sense to keep the reference point 
        if self.ref_update_mode == "EMA":
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoff

        # Here we dont need to maximize over Q values, we have the payoff matrix
        # We just select the agent id from the 3rd dimension of the payoff table
        # for reference the payoff tables are structured like this:
        #
        # np.array([
        #        [[-1, -1], [-3, 0]],   # C/C, C/D
        #        [[0, -3], [-2, -2]]    # D/C, D/D
        #    ])
        #
        # We take the pt transformation just like with LH
        elif self.ref_update_mode == 'V':
            opp_br = self.get_opp_br(self.payoff_matrix)
            player_best_response = self.get_best_response(self.payoff_matrix, opp_br)
            if self.agent_id == 0:
                payoffs = self.payoff_matrix[player_best_response,opp_br,0]

            else:
                payoffs = self.payoff_matrix[opp_br, player_best_response, 1]
            
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * payoffs.max()

        # same deal just over opp rewards
        elif self.ref_update_mode == 'EMAOR':
            self.ref_point = self.lam_r * self.ref_point + (1 - self.lam_r) * opp_payoff
        # Make sure to actually update the pt function
        self.pt.r = self.ref_point
