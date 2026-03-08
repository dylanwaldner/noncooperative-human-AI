import numpy as np

class RepeatedGameEnv:
    """Environment for repeated 2x2 games"""

    def __init__(self, payoff_matrix, horizon=100, state_history=2):
        self.payoff_matrix = payoff_matrix
        self.horizon = horizon
        self.state_history = state_history
        self.state_size = 4 ** state_history

    def reset(self):
        self.round = 0
        self.history = []
        return self._get_state()

    def _get_state(self, ref_bin=None):
        """
        Tabular state: base-4 encoding of last `state_history` joint actions.
        Most recent pair is least significant digit.
        (0,0)->0 (0,1)->1 (1,0)->2 (1,1)->3
        """
        k = self.state_history
        state = 0
        for i in range(k):
            if i < len(self.history):
                a1, a2 = self.history[-(i+1)]
                pair = a1 * 2 + a2
            else:
                pair = 0  # padding for missing history
            state += pair * (4 ** i)

        return int(state)

    def step(self, action1, action2):
        reward1 = float(self.payoff_matrix[action1, action2, 0])
        reward2 = float(self.payoff_matrix[action1, action2, 1])

        self.history.append((action1, action2))
        self.round += 1

        done = self.round >= self.horizon

        return self._get_state(), reward1, reward2, done, {}

