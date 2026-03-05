from static_games.ne_solver import compute_nash_equil
from static_games.eb_solver import compute_eb_equilibrium 
from static_games.ptne_solver import compute_ptne_equilibrium
from static_games.fp_ptne_solver import compute_fp_ptne_equilibrium
from static_games.ProspectTheory import ProspectTheory
from static_games.utils import get_all_games

def main():
	"""Main execution function"""
	print('*' * 80)
	print("Static Equilibrium Solver")
	print("Option 1: Compute NE (EU agents only)\n")
	print("Option 2: Compute CPT-EB Equilibrium\n")
	print("Option 3: Compute CPT-NE Equilibrium with SemiSmooth Newton Method\n")
	print("Option 4: Compute CPT-NE Equilibrium with Gauss Seidel Fixed Point Iteration\n")

	choice = 0
	while choice not in [1, 2, 3, 4]:
		choice = int(input("Choose 1, 2, 3, or 4 enter digit here: ").strip())
		
		if choice not in [1, 2, 3, 4]:
			print("Bad Input, try again")

	if choice in [2, 3, 4]:

		# Select agent types 
		print("\nPreference Types:\n")
		print("Option 1: PT\n")
		print("Option 2: EU\n")
		valid_types = ['PT', 'EU']
		agent1_type, agent2_type = "", ""

		while agent1_type not in valid_types or agent2_type not in valid_types:
			agent1_type = int(input("Player 1 type (Enter Number): ").strip())
			if agent1_type in [1, 2]:
				agent1_type = valid_types[agent1_type-1]

			agent2_type = int(input("Player 2 type (Enter number): ").strip())
			if agent2_type in [1, 2]:
				agent2_type = valid_types[agent2_type-1]

			if agent1_type not in valid_types or agent2_type not in valid_types:
				print("Agents input incorrectly, try again")

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

		pt = ProspectTheory(**pt_params)

	# Select game
	games = get_all_games()
	print("\nSelect game:")
	for i, (name, data) in enumerate(games.items(), 1):
		print(f"{i}. {name}")

	game_choice = 0
	while not (1 <= int(game_choice) <= len(games)):
		game_choice = input("\nGame number: ").strip()
		game_names = list(games.keys())

		if game_choice.isdigit() and 1 <= int(game_choice) <= len(games):
			game_name = game_names[int(game_choice)-1]
			payoff_matrix = games[game_name]['payoffs']
		else:
			print("Bad Input, try again")	


	if choice == 1:
		'''Nash Equilibrium EU Logic'''
		pure_equil, mixed_equil = compute_nash_equil(payoff_matrix)
		print(f'Game: {game_name}, Pure Equilibrium: {pure_equil}, Mixed Equilibrium: {mixed_equil}')


	elif choice == 2:
		'''CPT-EB Equilibrium Logic'''

		equil = compute_eb_equilibrium(payoff_matrix, pt, agent1_type, agent2_type)

		print(f'Game: {game_name}, Equilibrium: {equil}')

		print(f'Equil. Summary:\n')
		print(f'Number of Unique Equilibria: {len(equil.keys())}\n')
		print('Number of Init Seeds that Converged to Each Equilibria:')
		for k, v in equil.items():
			print(f'{k} Num Seeds: {len(v)}\n')

	elif choice == 3:
		''' Semismooth Newton CPT-NE Logic '''
		pure_equil, mixed_equil = compute_ptne_equilibrium(payoff_matrix, pt, agent1_type, agent2_type)

		print(f'Game: {game_name}, Pure Equilibrium: {pure_equil}')
		print(f'Game: {game_name}, Mixed Equilibrium (Key = equilibrium, value = starting points): {mixed_equil}')


		print(f'Equil. Summary:\n')
		print(f'Number of Unique Mixed Equilibria: {len(mixed_equil.keys())}\n')
		print('Number of Init Seeds that Converged to Each Equilibria:')
		for k, v in mixed_equil.items():
			print(f'{k} Num Seeds: {len(v)}\n')

	elif choice == 4:
		''' Fixed Point CPT-NE Logic '''
		equil = compute_fp_ptne_equilibrium(payoff_matrix, pt, agent1_type, agent2_type)

		print(f'Game: {game_name}, Mixed Equilibrium (Key = equilibrium, value = starting points): {equil}')


		print(f'Equil. Summary:\n')
		print(f'Number of Unique Mixed Equilibria: {len(equil.keys())}\n')
		print('Number of Init Seeds that Converged to Each Equilibria:')
		for k, v in equil.items():
			print(f'{k} Num Seeds: {len(v)}\n')

    


if __name__ == "__main__":
    main()
