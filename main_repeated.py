import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from repeated_games.train import run_complete_experiment, train_agents 
from repeated_games.analyze import compare_all_results
from repeated_games.da_analyze import compare_all_da_results

from repeated_games import (
    RepeatedGameEnv,
    AIAgent,
    ProspectTheory,
    LearningHumanPTAgent,
    AwareHumanPTAgent,
    DoubleAuction,
    train_agents,
    analyze_matchup,
    compare_all_results,
    get_all_games,
)
     

def interactive_experiment():

    print("="*80)
    print("PROSPECT THEORY REPEATED GAMES - ENHANCED EXPERIMENTS")
    print("="*80)

    # Get games
    games = get_all_games()

    while True:
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\nOptions:")
        print("1. Run complete experiment for a specific game")
        print("2. Run custom matchup")
        print("3. Run Double Auction Game (Heavy Memory)")
        print("4. Run Complete Experiment End to End")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        print("\n" + "="*80)
        print("Set Number of Experiment Runs")
        print("="*80)

        num_experiments = int(input("\nEnter Number of Experiment Runs (Rec. For Prod Run: 30): ").strip())

        if choice not in ['3', '4']:

            print("\n" + "="*80)
            print("Set State History (Recommended: Choose 0 or 2)")
            print("="*80)
            state_history = int(input("\nEnter Choice (< 3): ").strip())

            print("\n" + "="*80)
            print("SET REFERENCE POINT SETTING")
            print("="*80)
            print("\nOptions:")
            print("1. Fixed (Set custom ref point)")
            print("2. EMA")
            print("3. Q (updates ref point based on max(Q(S, A)) at the current state. Normalized with (1-gamma) to move from expected discounted return scale to reward scale. ")
            print("4. EMAOR (Exponential Moving Average over Opponent Rewards, basically tracks how well the opponent is doing not how well the player is doing")
          
            ref_setting = ""
            viable_options = ['Fixed', 'EMA', 'Q', 'EMAOR']

            while ref_setting not in viable_options:        
                ref_setting = int(input("\nEnter choice (type the digit): ").strip())
                ref_setting = viable_options[ref_setting-1]

                if ref_setting not in viable_options:
                    print("Failed, please try again.")

        else:
            episodes = input(f"Episodes to run (default 200): ").strip()
            episodes = int(episodes) if episodes.isdigit() else 200

        r = None
        while r == None:
            r = float(input("Input Initial Reference Point Value: ").strip()) 

        print("\n" + "="*80)
        print("SET PT PARAMETERS")
        print("="*80)
        print("\nOptions:")
        print("1. Kahneman and Tversky Params (gamma = 0.61, delta=0.69)")
        print("2. Custom")
        
        pt_choice = 0
        while pt_choice not in [1, 2]:
            pt_choice = int(input("Enter 1 or 2 for your choice: ").strip())
            if pt_choice not in [1, 2]:
                print("Invalid input, try again")

        if pt_choice == 1:
            pt_params = {'lambd': 2.25, 'alpha': 0.88, 'gamma': 0.61, 'r': r, 'delta': 0.69}

        elif pt_choice == 2:
            delta, gamma = -1, -1
            while not 0 <= delta <= 1 or not 0 <= gamma <= 1:
                delta = float(input("Enter delta value (must be in range [0, 1] continuous): ")) 
                gamma = float(input("Enter gamma value (must be in range [0, 1] continuous): "))
 
                if not 0 <= delta <= 1 or not 0 <= gamma <= 1:
                    print("invalid input, try again")

            pt_params = {'lambd': 2.25, 'alpha': 0.88, 'gamma': gamma, 'r': r, 'delta':delta}

        if choice == '1':
            # Complete experiment for specific game
            print("\nAvailable games:")
            for i, (name, data) in enumerate(games.items(), 1):
                print(f"{i}. {name}: {data['description']}")

            game_choice = input("\nEnter game number: ").strip()
            game_names = list(games.keys())

            if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
                game_name = game_names[int(game_choice)-1]
                payoff_matrix = games[game_name]['payoffs']

                episodes = input(f"Episodes to run (default 200): ").strip()
                episodes = int(episodes) if episodes.isdigit() else 200

                print(f"\nStarting complete experiment for {game_name}...")
                all_results = run_complete_experiment(game_name, payoff_matrix, episodes=episodes, ref_setting=ref_setting, pt_params=pt_params, ref_point=r, state_history=state_history, num_experiments=num_experiments)

                # Compare results
                compare_all_results(all_results, game_name, state_history, num_experiments)
            else:
                print("Invalid choice, using Prisoner's Dilemma")
                raise ValueError
                game_name = 'PrisonersDilemma'
                payoff_matrix = games[game_name]['payoffs']
                all_results = run_complete_experiment(game_name, payoff_matrix, ref_setting=ref_setting, pt_params=pt_params)
                compare_all_results(all_results, game_name)

        elif choice == '2':
            # Custom matchup
            print("\nCustom Matchup Configuration")
            print("-"*40)

            # Select game
            print("\nSelect game:")
            for i, (name, data) in enumerate(games.items(), 1):
                print(f"{i}. {name}")

            game_choice = input("\nGame number: ").strip()
            game_names = list(games.keys())

            if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
                game_name = game_names[int(game_choice)-1]
                payoff_matrix = games[game_name]['payoffs']
            else:
                game_name = 'PrisonersDilemma'
                payoff_matrix = games[game_name]['payoffs']

            # Select agent types
            print("\nAgent types: 1) AH, 2) LH, 3) AI")
            valid_types = ['AH', 'LH', 'AI']
            agent1_type, agent2_type = "", ""
          
            while agent1_type not in valid_types or agent2_type not in valid_types:
                agent1_type = int(input("Agent 1 type (Enter Number): ").strip())
                if agent1_type in [1, 2, 3]:
                    agent1_type = valid_types[agent1_type-1]

                agent2_type = int(input("Agent 2 type (Enter number): ").strip())
                if agent2_type in [1, 2, 3]:
                    agent2_type = valid_types[agent2_type-1]

                if agent1_type not in valid_types or agent2_type not in valid_types:
                    print("Agents input incorrectly, try again")

            # Get parameters
            episodes = input("Episodes (default 200): ").strip()
            episodes = int(episodes) if episodes.isdigit() else 200

            # Run custom matchup
            print(f"\nRunning {agent1_type} vs {agent2_type} in {game_name}...")

            env = RepeatedGameEnv(payoff_matrix, horizon=100, state_history=state_history)

            # Reference point setting
            # Options = Fixed, EMA, Q, EMAOR
            ref_lambda = 0.95 

            # Create agents
            action_size = 2
            if agent1_type == 'LH':
                agent1 = LearningHumanPTAgent(env.state_size, 2, 2, pt_params, 0, ref_setting=ref_setting, lambda_ref = ref_lambda, payoff_matrix=payoff_matrix)
            elif agent1_type == "AI":
                agent1 = AIAgent(env.state_size, 2, 2, 0)

            if agent2_type == 'LH':
                agent2 = LearningHumanPTAgent(env.state_size, 2, 2, pt_params, 1, ref_setting=ref_setting, lambda_ref = ref_lambda, payoff_matrix=payoff_matrix)
            elif agent2_type == 'AI':
                agent2 = AIAgent(env.state_size, 2, 2, 1)

            if agent1_type == 'AH':
                opp_params = dict()
                opp_params['opponent_type'] = agent2_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None
                if agent2_type != "AI":
                    opp_params['opp_ref'] = r 
                    opp_params['opp_pt'] = pt_params
                     
                agent1 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=0, opp_params=opp_params, ref_setting=ref_setting, lambda_ref = ref_lambda)

            if agent2_type == 'AH':
                opp_params = dict()
                opp_params['opponent_type'] = agent1_type
                opp_params['opponent_action_size'] = action_size
                opp_params['opp_ref'] = None
                if agent1_type != "AI":
                    opp_params['opp_ref'] = r  
                    opp_params['opp_pt'] = pt_params
                agent2 = AwareHumanPTAgent(payoff_matrix, pt_params, action_size, env.state_size, agent_id=1,opp_params=opp_params, ref_setting=ref_setting, lambda_ref = ref_lambda)


            # Train
            results = dict() 
            for idx in range(num_experiments):
                 print(f'Run {idx + 1} / {num_experiments}')
                 results[f"{idx}"] = train_agents(agent1, agent2, env, episodes=episodes, verbose=True)

            # Analyze
            games_dict = get_all_games()
            print('Payoff Matrix:', payoff_matrix)
            print('Agent 1 Softmax triggers: ', agent1.softmax_counter)
            print('Agent 2 Softmax triggers: ', agent2.softmax_counter)
            
            if agent1_type != 'AH':
                agent1_q_vals = agent1.get_q_values()
                print(f"Agent 1 state visits: {agent1.state_visit_counter}")
                print(f"Agent 1 raw q values = {agent1_q_vals}")

            if agent2_type != 'AH':
                agent2_q_vals = agent2.get_q_values()
                print(f"Agent 2 raw q values = {agent2_q_vals}")

                print(f"Agent 2 state visits: {agent2.state_visit_counter}")


            if hasattr(agent1, "beliefs"):
                print(f"Agent 1 beliefs: {agent1.beliefs}")

            if hasattr(agent2, "beliefs"):
                print(f"Agent 2 beliefs: {agent2.beliefs}")

            analyze_matchup(results, agent1, agent2, agent1_type, agent2_type, game_name, games_dict, payoff_matrix, pt_params, ref_setting, env)


        elif choice == '3':
            # Custom matchup
            print("\nDouble Auction Game")
            print("-"*40)
            # FOr double auction player 1 is the buyer, player 2 is the seller
            games_dict = ""

            # Range of prices the double auction game will operate in
            price_range = 0
            while price_range <= 0 or price_range >= 100:
                price_range = int(input("\n Set Price Range (from 1 -> n, Default = n = 10, Max = n = 99): ").strip())
                if price_range <= 0 or price_range >= 100:
                    print("Invalid input, try again")

            # The true value that the buy thinks the product has (so buying above = loss, below = profit)
            valuation = 0
            while valuation <= 0 or valuation > price_range:
                valuation = int(input(f"\n Set Valuation (Default = 6, Max = {price_range}): ").strip())
                if valuation <= 0 or valuation > price_range:
                    print("Invalid input, try again")

            # The true cost to make the product (selling above = profit, below = loss)
            cost = 0
            while cost <= 0 or cost > price_range:
                cost = int(input(f"\n Set Cost (Default = 4, Max = {price_range}, Valuation = {valuation}): ").strip())
                if cost <= 0 or cost > price_range:
                    print("Invalid input, try again")

            game_name = f"Double Auction Game {price_range}"


            state_histories = [2, 0] 
            for state_history in state_histories: 
                env = DoubleAuction(k=price_range, valuation=valuation, cost=cost, horizon=100, state_history=state_history)
                payoff_matrix = env.build_payoff_matrix()
                ref_settings = ["Fixed", "EMA", "V", "EMAOR"] 
                for ref_setting in ref_settings: 
                    # FOr double auction player 1 is the buyer, player 2 is the seller
                    payoff_matrix = env.build_payoff_matrix()
                    if ref_setting == "Fixed": 
                        ref_point1 = valuation 
                        ref_point2 = cost 

                        r = [ref_point1, ref_point2] 

                    print(f"\nStarting complete experiment for {game_name}...") 
                    # Store results 
                    print(f"\nStarting complete experiment for {game_name}...") 
                    all_results = run_complete_experiment(game_name, payoff_matrix, episodes=episodes, ref_setting=ref_setting, pt_params=pt_params, ref_point=r, state_history=state_history, num_experiments=num_experiments, action_size=price_range, env=env) 

                    # Compare results 
                  
                    data = compare_all_da_results(all_results, game_name, state_history, num_experiments, ref_setting, payoff_matrix) 




        elif choice == '4':
            games_list = list(games.keys())
            all_results = dict()
            for game_name in games_list:
                state_histories = [0, 2]
                for state_history in state_histories:
                    ref_settings = ["Fixed", "EMA", "V", "EMAOR"]
                    for ref_setting in ref_settings:
                        payoff_matrix = games[game_name]['payoffs']

                        if ref_setting == "Fixed":
                            ref_point1 = (payoff_matrix[:, :, 0].max() + payoff_matrix[:, :, 0].min()) / 2
                            ref_point2 = (payoff_matrix[:, :, 1].max() + payoff_matrix[:, :, 1].min()) / 2

                            r = [ref_point1, ref_point2]

                        print(f"\nStarting complete experiment for {game_name}...")
                        # Store results
                        print(f"\nStarting complete experiment for {game_name}...")
                        all_results = run_complete_experiment(game_name, payoff_matrix, episodes=episodes, ref_setting=ref_setting, pt_params=pt_params, ref_point=r, state_history=state_history, num_experiments=num_experiments)

                        # Compare results
                        data = compare_all_results(all_results, game_name, state_history, num_experiments, ref_setting, games)


        elif choice == '5':
            print("\nExiting...")
            break

        else:
            print("\nInvalid choice. Please try again.")

        # Ask to continue
        cont = input("\nContinue? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nExiting...")
            break


if __name__ == "__main__":
    # Run the interactive interface
    interactive_experiment()
