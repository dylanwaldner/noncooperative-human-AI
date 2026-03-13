
# RL Nash with AIs (EUT) and Humans (CPT)

## Primary Author: Dylan Waldner
## Contributor: Mitchelle Ashimosi
## Supervised by Vyacheslav Kungurtsev

## Running the experiments:

### Run static games with python3 main_static.py

### Run repeated games with python3 main_repeated.py

#### Expect a series of prompts to customize the set up of the experiment, e.g. num episodes, game selection, agent selection, etc. 

## File Summaries:

### repeated\_games/

#### Agents

AI: **ai_agent.py** implements epsilon greedy EUT learning, i.e. vanilla Q-learning

Learning Human (LH): **learning_human.py** implements epsilon greedy Q(s, a_i, a_-i) q learning, where each joint action pair is estimated. Beliefs about opponent actions are tracked with EMA. The update bootstraps with q(s', a') by integrating out the opponent actions using the beliefs, and then taking the value of the max action. Formally: q(s, a_i, a_-i) += alpha(reward + $\gamma$ * max(E_b{q(s', a'_i, a'\_-i)}) - q(s, a_i, a_-i)). Reference points are either fixed, or updated via EMA, max Q value, or opponent EMA (EMA over opponent received rewards). 

Aware Human (AH): **aware_human.py** implements a best reply agent *with no reinforcement learning*, although notably it does still update its reference poins the same way that LH does. The best reply algorithm first selects the opp best reply to each of the AH actions, then maximizes its action value over the opp BR to generate a response. 

All agents use a softmax tie break for when action probabilities are within $\tau$ of each other to randomize when there is no decisive action. 

#### Environment, Analysis, and Training

Game Env: **game_env.py** is where the 2x2 matrix games are stored. States are labeled with a base 4 encoding of joint actions, added together to turn a full history into a unique integer. Specifically, (0,0)->0 (0,1)->1 (1,0)->2 (1,1)->3. These joint action pair encodings are added into a state integer variable that is weighted by the place of the joint action in the history. 
          **double_action.py** uses a similar base n^2 state encoding system, where n is the action size for each player. The double auction game uses midpoint pricing if a trade goes through, and rewards of 0 otherwise (could do -1 to encourage exploration?)

Training: **train.py** train_agents() initializes the environment, tracks rewards, actions, q values and ref points for each agent, and steps through the environment episodically. Each round played updates reference points, beliefs, and q values. Importantly, the AH gets the opponent reference point updated each round for its calculus. Epsilon is updated at the end of each episode with a minimum of 0.01.
                       run_complete_experiment() is a wrapper for train_agents, automating the full cycle through the loop. I currently a not using this function, which is why some of the code is a little dubious (e.g. the matchups dont account for the agent typs being on the other side, that is row/col and also col/row). It's probably not worth checking this, I am going to stay in the custom matchup logic and manually recombine. 

Analysis: **analyze.py** analyze_matchup() includes a cumulative rewards/time plot, a ref point over time plot, an action heat map, a Q-value convergence plot (how are q values converging to the real payoffs), a learning convergence plot (how much are q values changing), and then action frequency charts for each player. 
                         compare_all_results() is essentially deprecated and a part of the run complete experiment wrapper. 

#### Prospect Theory and Utils

Prospect Theory: **prospect_theory.py** implements cumulative prospect theory by first taking in outcomes and probabilities, sorting them by the same index, splitting them into gains and losses via the reference point, and then taking the cumulative and decumulative sums of the probabilities respectively, findingt he difference between successive lottery cum. probs, and then feeding the lotteries into the probability weighting function and value function to output a final value. 
Importantly, note that the value function uses the internal representation of the reference point to calculate the value internally, so its important to be 100% sure that each PT agent in train.py is updating their reference point in place in their instance of ProspectTheory(). 

**utils.py** stores the payoff matrices and a smoothing function for plotting. 

### static_games/

#### Solvers

Nash Equilibrium: **ne_solver.py** computes the classic nash equilibrium. First it searches for corner cases with a pure best reply for loop, checking whether the action pair (i, j) motivates both playes to stay, and if so saves the pair. For mixed equilibria, we compute the indifference equation (optimality condition?) directly, the derivation is commented inline. 

CPT-EB: **eb_solver.py** uses the BR solver for the pure equilibria, once again justified because there is no uncertainty over their own strategy, so each players strategy space is linear, evenif their beliefs and influence of opponent actions is not. The semismooth newton method is used to find the mixed equilibria, computing x = J^-1 -F(z), where z is the p, q tuple. F is defined once again with the indifference equation (optimality condition?) which is justified here because we are choosing between pure strategies and randomizing over opponent uncertainty, so our mapping from probability to indifference space does not condition uncertainty over a player's own actions. Formally, if p is the probability that player 1 plays action 0 and q is the probability player 2 plays action 0, then we get F(player 1) = CPT_value (Action 1 | q) - CPT_value (Action 2 | q) and we try to find 0. Player 2 uses p instead of q. 
Worth noting: 
- the semismooth method initializes from many different starting point for p, q to serch for different basin.
- The Jacobian is calculated with finite difference (maybe there was an analytic way to do this?)
- The solver is flexible for any combination of EU nd PT agents
- The jacobian is off diagonal because dF1/dp and dF2/dq are both immediately evaluated to 0.
- The lotteries are evaluated as length two lotteries over each action, so action 0 (split across opp responses) gets q, 1-q, for example. 

CPT-NE: **ptne_solver.py** Implements a semismooth newton method for the mixed equilibria, and a grid search best reply loop for the pure equilibria. 
The thinking here for the pure equilibria is that now that we are randomizing over our own strategy, so our own value space is no longer linear and we cant assume that just because one corner is larger than the other that everything in between will also be smaller. So, instead of just comparing each pure strategy, we compare each corner to a grid search (201) of the interior between the pure strategies. More formally, p = 1 gets compared to p = 0.995, p=0.99, ... 0.005, 0. 
The semismooth method makes some important deviations from EB while retaining its theoretical structure:

- instead of length 2 lotteries, we now get length 4 lotteries (each quadrant of 2x2, then pq, 1-p * q, p 1-q, 1-p 1-q)
- F is now the derivative of the value function, looking for derivative = 0 (i.e. at a balance point). Importantly, we then check for all of the other interior and corner values in a grid search like the pure search, so we arent getting fooled by minima and saddle points.
- Jacobian is now full because F depends on P and Q
- Finite difference still used
       **fp_ptne_solver.py** implements a fixed point iteration (gauss seidel). Here, the pure and mixed equilibria search is collapsed into 1, we do a grid search for values and immediate update p when a new max is found, and at the end of each iteration check if p, q are the same as before. If so, we return it as a mixed equilibria. 
