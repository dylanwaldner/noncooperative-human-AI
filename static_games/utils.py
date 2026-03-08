import numpy as np

def get_all_games():
    """Return all games for testing"""
    return {
        'PrisonersDilemma': {
            'payoffs': np.array([
                [[-1, -1], [-3, 0]],   # C/C, C/D
                [[0, -3], [-2, -2]]    # D/C, D/D
            ]),
            'description': 'Classic Prisoner\'s Dilemma',
            'actions': ['Cooperate', 'Defect']
        },
        'MatchingPennies': {
            'payoffs': np.array([
                [[1, 0], [0, 1]],   # H/H, H/T
                [[0, 1], [1, 0]]    # T/H, T/T
            ]),
            'description': 'Zero-sum Matching Pennies',
            'actions': ['Heads', 'Tails']
        },
        'BattleOfSexes': {
            'payoffs': np.array([
                [[3, 2], [0, 0]],    # Opera/Opera, Opera/Football
                [[0, 0], [2, 3]]     # Football/Opera, Football/Football
            ]),
            'description': 'Battle of the Sexes',
            'actions': ['Opera', 'Football']
        },
        'StagHunt': {
            'payoffs': np.array([
                [[3, 3], [0, 2]],    # Stag/Stag, Stag/Hare
                [[2, 0], [1, 1]]     # Hare/Stag, Hare/Hare
            ]),
            'description': 'Stag Hunt Coordination Game',
            'actions': ['Stag', 'Hare']
        },
        'Chicken': {
            'payoffs': np.array([
                [[0, 0], [-1, 1]],   # Swerve/Swerve, Swerve/Straight
                [[1, -1], [-10, -10]] # Straight/Swerve, Straight/Straight
            ]),
            'description': 'Chicken Game',
            'actions': ['Swerve', 'Straight']
        },
        'OchsGame': {
            'payoffs': np.array([
                [[4, 0], [0, 1]],    # Top/Left, Top/Right
                [[0, 1], [1, 0]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Ochs Game - PT equilibrium pathology',
        },
        'CrawfordGame': {
            'payoffs': np.array([
                [[2, -2], [0, 0]],    # Top/Left, Top/Right
                [[0, 0], [-1, 1]]     # Bottom/Left, Bottom/Right
            ]),
            'description': 'Crawford Game - convex preference effects',
            'actions': ['Top', 'Bottom']
        }
    }
