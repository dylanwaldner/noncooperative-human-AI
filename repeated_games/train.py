import matplotlib.pyplot as plt
from .aware_human import AwareHumanPTAgent 
from .learning_human import LearningHumanPTAgent
from .ai_agent import AIAgent
from .game_env import RepeatedGameEnv 
from .double_auction import DoubleAuction 
from .utils import get_all_games
from .analyze import analyze_matchup
from .da_analyze import analyze_matchup_da
import numpy as np
import pandas as pd
import time
import copy
import random
BASE_SEED = 42

from matplotlib.ticker import FuncFormatter

import os
import time

def train_agents(agent1, agent2, env, episodes=500,
                 exploration_decay=0.99, verbose=True, game_name=''):
    """
    Train two agents against each other

    This loop steps through the environment and tracks/updates:
    - rewards
    - actions
    - beliefs
    - reference points
    - q values
    - states
    - opp PT functions (AH tracks this)

    Importantly, it is EPISODIC and updates gamma **EVERY EPISODE**, so please let me know if it should be continuous
    When I looked on the internet, episodic seemed to be more tractable for RL, but obviously I could be wrong. 
    """

    results = {
        'rewards1': [],
        'rewards2': [],
        'raw_rewards1': [],
        'raw_rewards2': [],
        'actions1': [],
        'actions2': [],
        'avg_rewards1': [],
        'avg_rewards2': [],
        'strategies1': [],  # For aware PT agent
        'strategies2': [],
        'q_values1': [],
        'q_values2':[],
        'ref_points1': [],
        'ref_points2': [],
        'best_responses1': [],
        'best_responses2': [],
        'best_rewards1': [], 
        'best_rewards2': [],
        'states': [],
    }
    joint_counts = np.zeros((agent1.action_size,agent2.action_size), dtype=int)

    start_time = time.time()
    last_time = start_time

    log_every = 100
    global_step = 1

    # Reset metrics
    agent1.pt_l2_dists, agent2.pt_l2_dists = [], []
    agent1.action_changed_flags, agent2.action_changed_flags = [], []

    # Defined to initialize BR agents, agent1 and agent 2 actions sizes always the same for this setup
    action_size = agent1.action_size
    if game_name.startswith('Double Auction Game'):
        payoff_matrix = env.build_payoff_matrix()

    else:
        payoff_matrix = env.payoff_matrix

    last_action1, last_action2 = 0, 0

    for episode in range(episodes):

        state = env.reset()
        episode_rewards1 = []
        episode_rewards2 = []
        episode_actions1 = []
        episode_actions2 = []
        episode_q_values1 = []
        episode_q_values2 = []
        
        # Tracks the real best response that agents could have played wrt raw payoffs
        best_response1 = []
        best_response2 = []

        # Now we track how the best responder would be rewarded
        # 1 is br1(a2) rewards (so what would rewards look like if agent1 made the br to agent2) and vice versa
        best_reward1 = []
        best_reward2 = []

        for _ in range(env.horizon):
            results['states'].append(state)

            # We transform the PT agents states to include ref bins
            if not isinstance(agent1, AIAgent):
                if isinstance(agent1, LearningHumanPTAgent):
                    pt_state1 = agent1.transform_state(state)
                    action1 = agent1.act(pt_state1)
                else: # Aware Human
                    pt_state1 = None
                    action1 = agent1.act(last_action2)

            else: # AI Agent
                pt_state1 = None
                # Agent 1 chooses action
                action1 = agent1.act(state)

            if not isinstance(agent2, AIAgent):
                if isinstance(agent2, LearningHumanPTAgent):
                    pt_state2 = agent2.transform_state(state)
                    action2 = agent2.act(pt_state2)
                else: # AH 
                    pt_state2 = None
                    action2 = agent2.act(last_action1)

            else:
                pt_state2 = None
                # Agent 1 chooses action
                action2 = agent2.act(state)

            last_action1, last_action2 = action1, action2

            # Execute step
            next_state, reward1, reward2, done, _ = env.step(action1, action2)

            if isinstance(agent1, LearningHumanPTAgent):
                agent1.ref_update(payoff=reward1, state=pt_state1, opp_payoff=reward2)
                pt_next_state1 = agent1.transform_state(next_state)
            else:
                pt_next_state1 = None

            if isinstance(agent2, LearningHumanPTAgent):
                agent2.ref_update(payoff=reward2, state=pt_state2, opp_payoff=reward1)
                pt_next_state2 = agent2.transform_state(next_state)

            else:
                pt_next_state2 = None

            if isinstance(agent1, LearningHumanPTAgent):
                # Updates
                agent1.belief_update(pt_state1, action2)
                agent1.avg_rew += (reward1 - agent1.avg_rew) / global_step 
                agent1.q_value_update(pt_state1, pt_next_state1, action1, action2, reward1, done)

                q_vals = agent1.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)  
                #print("Q vals in Train 1: ", q_vals)

                # Tracking
                results['q_values1'].append(q_vals)
                results['ref_points1'].append(agent1.ref_point)

                del q_vals

            elif isinstance(agent1, AIAgent):
                # Update code here
                agent1.avg_rew += (reward1 - agent1.avg_rew) / global_step
                agent1.update(state, action1, next_state, reward1, done)

                # Get q vals, normalize by multiplying by 1 - gamma to remove future discounting
                q_vals = agent1.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)

                # Tracking
                results['q_values1'].append(q_vals)
                
                del q_vals

            else: # Aware Human
                results['ref_points1'].append(agent1.ref_point)
                
            if isinstance(agent2, LearningHumanPTAgent):
                # Update LH variables
                agent2.belief_update(pt_state2, action1)
                agent2.avg_rew += (reward2 - agent2.avg_rew) / global_step
                agent2.q_value_update(pt_state2, pt_next_state2, action2, action1, reward2, done)

                # Get q vals
                q_vals = agent2.get_q_values()
                #print("Q vals in Train: ", q_vals)
                q_vals = np.asarray(q_vals, dtype=np.float32)

                # Tracking
                results['q_values2'].append(q_vals)
                results['ref_points2'].append(agent2.ref_point)
                del q_vals

            elif isinstance(agent2, AIAgent):
                # Update code here
                agent2.avg_rew += (reward2 - agent2.avg_rew) / global_step
                agent2.update(state, action2, next_state, reward2, done)
                q_vals = agent2.get_q_values()
                q_vals = np.asarray(q_vals, dtype=np.float32)

                # Tracking
                results['q_values2'].append(q_vals)

                del q_vals

            else: # Aware Human
                results['ref_points2'].append(agent2.ref_point)
                # Pass agent 1 pt func to agent2
                if not isinstance(agent1, AIAgent):
                    agent2.opp_pt = agent1.pt
            # We needed agent 2 to be fully calculated before passing the agent 2 pt values to agent 1
            if isinstance(agent1, AwareHumanPTAgent):
                if not isinstance(agent2, AIAgent):
                    agent1.opp_pt = agent2.pt

            global_step += 1

            # Store results
            episode_rewards1.append(reward1)
            episode_rewards2.append(reward2)

            episode_actions1.append(action1)
            episode_actions2.append(action2)

            # Agent 1 best response to agent 2's realized action
            agent1_rewards = payoff_matrix[:, action2, 0]
            br1 = np.argmax(agent1_rewards)
            best_response1.append(br1)
            best_reward1.append(agent1_rewards[br1])

            # Agent 2 best response to agent 1's realized action
            agent2_rewards = payoff_matrix[action1, :, 1]
            br2 = np.argmax(agent2_rewards)
            best_response2.append(br2)
            best_reward2.append(agent2_rewards[br2])

            # For action heat map
            joint_counts[action1, action2] += 1

            # Step through
            state = next_state
 

            if done:
                break

        # Store episode results
        print(f"\rEpisode {episode+1} of {episodes}", end='')
        steps = env.horizon
        avg_reward1 = sum(episode_rewards1) / steps
        avg_reward2 = sum(episode_rewards2) / steps

        results['rewards1'].append(sum(episode_rewards1))
        results['rewards2'].append(sum(episode_rewards2))

        results['raw_rewards1'].append(episode_rewards1)
        results['raw_rewards2'].append(episode_rewards2)

        results['actions1'].append(episode_actions1)
        results['actions2'].append(episode_actions2)

        results['avg_rewards1'].append(avg_reward1)
        results['avg_rewards2'].append(avg_reward2)

        results['best_responses1'].append(best_response1)
        results['best_responses2'].append(best_response2)

        results['best_rewards1'].append(best_reward1)
        results['best_rewards2'].append(best_reward2)

        
        # Decay exploration
        if isinstance(agent1, (LearningHumanPTAgent, AIAgent)):
            agent1.epsilon = agent1.epsilon * exploration_decay

        if isinstance(agent2, (LearningHumanPTAgent, AIAgent)):
            agent2.epsilon = agent2.epsilon * exploration_decay

        # Progress update
        if verbose and (episode + 1) % 100 == 0:
            print(f"\r  Episode {episode + 1}/{episodes}: "
                  f"Avg rewards = {avg_reward1:.3f}, {avg_reward2:.3f}"
                  f"\n Time since start = {time.time() - start_time}, Time this 100 episodes = {time.time() - last_time}")

            last_time = time.time()

    results['joint_actions'] = joint_counts
    results['pt_l2_dists1'], results['pt_l2_dists2'] = [], []
    results['action_changed_flags1'], results['action_changed_flags2'] = [], []

    if hasattr(agent1, "pt_l2_dists"):
        results['pt_l2_dists1'] = agent1.pt_l2_dists.copy()

    if hasattr(agent1, "action_changed_flags"):
        results['action_changed_flags1'] = agent1.action_changed_flags.copy()

    if hasattr(agent2, "pt_l2_dists"):
        results['pt_l2_dists2'] = agent2.pt_l2_dists.copy()

    if hasattr(agent2, "action_changed_flags"):
        results['action_changed_flags2'] = agent2.action_changed_flags.copy()

    return results

def run_complete_experiment(game_name, payoff_matrix, episodes=300, ref_setting='Fixed', pt_params={}, ref_point=0, state_history=2, num_experiments=30, action_size=2, env=None):
    """
    Run all agent matchups for a game
    This is pretty much deprecated, I intend to run via custom game or I will edit this.
    
    This is just a wrapper for train agents, but since it spits out results back to back to back and the level of 
    control we need is high, I'm just happy to use the custom matchup setting and specify hyperparameters with each run. 

    Of course, if that changes I can always update this, but the code in here isn't really modeling choices its just
    regular initialization and data collecting, review probably doesn't need to focus here. 
    """

    print("\n" + "="*80)
    print(f"COMPLETE EXPERIMENT: {game_name}")
    print("="*80)
    if isinstance(ref_point, list):
        ref_point1, ref_point2 = ref_point
        pt_params1, pt_params2 = pt_params.copy(), pt_params.copy()
        pt_params1['r'], pt_params2['r'] = ref_point1, ref_point2
    else:
        ref_point1, ref_point2 = ref_point, ref_point
        pt_params1, pt_params2 = pt_params.copy(), pt_params.copy()

    # Reference point setting
    # Options = Fixed, EMA, Q, 'EMAOR': EMA of Opp rewards
    ref_lambda = 0.95

    # Define all matchups to test
    if game_name in ["PrisonersDilemma","StagHunt", "Chicken"]:
        matchups = [
        ('AH1', 'AI'),
        ('AH2', 'AI'),

        ('LH', 'AI'),

        ('AH1', 'LH'),
        ('AH2', 'LH'),
        ('AH1', 'AH1'), # Baseline
        ('AH2', 'AH2'),

        ('LH', 'LH'),
        ('AI', 'AI')  
        ]
    else:
        matchups = [
        ('AH1', 'AI'),
        ('AI', 'AH1'),
        ('AH2', 'AI'),
        ('AI', 'AH2'),

        ('LH', 'AI'),
        ('AI', 'LH'),

        ('AH1', 'LH'),
        ('LH', 'AH1'),
        ('AH2', 'LH'),
        ('LH', 'AH2'),

        ('AH1', 'AH1'), # Baseline
        ('AH2', 'AH2'), # Baseline
        ('LH', 'LH'),
        ('AI', 'AI')  
        ]   

    all_results = dict()
    for agent1_type, agent2_type in matchups:
        print(f"\n{'='*70}")
        print(f"MATCHUP: {agent1_type} vs {agent2_type}")
        print('='*70)

        matchup_key = f"{agent1_type}_vs_{agent2_type}"
        all_results[matchup_key] = dict()
        results = dict()

        # AH2 is the Tit for Tat agent, we dont run that in memoryless games
        if state_history == 0:
            if agent1_type == "AH2" or agent2_type == "AH2":
                continue

        for idx in range(num_experiments):
            print(f'\nRun {idx + 1} / {num_experiments}')

            seed = BASE_SEED + idx
            
            np.random.seed(seed)
            random.seed(seed)
            # Reset environment
            if env is None: # 2x2 matrix game
                env = RepeatedGameEnv(payoff_matrix, horizon=100, state_history=state_history)
            else: # Env passed from main_repeated, just need to reset it
                _ = env.reset()

            # Create agents based on type
            if agent1_type == 'LH':
                agent1 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params1, agent_id=0, ref_setting=ref_setting, lambda_ref = ref_lambda, payoff_matrix=payoff_matrix)
            elif agent1_type == 'AI':  # AI
                agent1 = AIAgent(env.state_size, action_size, action_size, agent_id=0)

            if agent2_type == 'LH':
                agent2 = LearningHumanPTAgent(env.state_size, action_size, action_size, pt_params2, agent_id=1, ref_setting=ref_setting, lambda_ref=ref_lambda, payoff_matrix=payoff_matrix)
            elif agent2_type == 'AI':  # AI
                agent2 = AIAgent(env.state_size, action_size, action_size, agent_id=1)

            if agent1_type == 'AH1':
                opp_params = dict()
                opp_params['opponent_type'] = agent2_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None

                if agent2_type != "AI": # PT agent
                    opp_params['opp_ref'] = ref_point2 
                    opp_params['opp_pt'] = pt_params2

                agent1 = AwareHumanPTAgent(payoff_matrix, pt_params1, action_size, env.state_size, agent_id=0, opp_params=opp_params,ref_setting=ref_setting)

            if agent1_type == 'AH2':
                opp_params = dict()
                opp_params['opponent_type'] = agent2_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None

                if agent2_type != "AI": # PT agent
                    opp_params['opp_ref'] = ref_point2
                    opp_params['opp_pt'] = pt_params2

                agent1 = AwareHumanPTAgent(payoff_matrix, pt_params1, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting, tit_for_tat=True)

            if agent2_type == 'AH1':
                opp_params = dict()
                opp_params['opponent_type'] = agent1_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None

                if agent1_type != "AI": # PT agent
                    opp_params['opp_ref'] = ref_point1
                    opp_params['opp_pt'] = pt_params1

                agent2 = AwareHumanPTAgent(payoff_matrix, pt_params2, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting)

            if agent2_type == 'AH2':
                opp_params = dict()
                opp_params['opponent_type'] = agent1_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None

                if agent1_type != "AI": # PT agent
                    opp_params['opp_ref'] = ref_point1
                    opp_params['opp_pt'] = pt_params1

                agent2 = AwareHumanPTAgent(payoff_matrix, pt_params2, action_size, env.state_size, agent_id=1, opp_params=opp_params, ref_setting=ref_setting, tit_for_tat=True)


            # Train the matchup
            results[f"{idx}"] = train_agents(agent1, agent2, env, episodes=episodes, verbose=True, game_name=game_name)

        # Store results
        matchup_key = f"{agent1_type}_vs_{agent2_type}"
        all_results[matchup_key] = results

        # Analyze this matchup
        if not game_name.startswith('Double Auction Game'):
            games_dict = get_all_games()
            analyze_matchup(results, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params, ref_setting, env)
        else:
            analyze_matchup_da(results, agent1_type, agent2_type, game_name, payoff_matrix, pt_params, ref_setting, env)

    return all_results

    
