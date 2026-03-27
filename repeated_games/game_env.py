import numpy as np

class RepeatedGameEnv:
    """Environment for repeated 2x2 games"""

    def __init__(self, payoff_matrix, horizon=100, state_history=2):
        self.payoff_matrix = payoff_matrix
        self.horizon = horizon
        self.state_history = state_history
        self.state_size = 4 ** state_history
        self.history = []

    def reset(self):
        self.round = 0
        return self._get_state()

    def _get_state(self, ref_bin=None):
        """
        Tabular state: base-4 encoding of last `state_history` joint actions.
        Most recent pair is least significant digit.
        (0,0)->0 (0,1)->1 (1,0)->2 (1,1)->3
        """
        if self.state_history == 0:
            return 0

        base = self.k * self.k
        state = 0

        recent = self.history[-self.state_history:]

        for i, (a1, a2) in enumerate(reversed(recent)):
            pair = (a1 - 1) * self.k + (a2 - 1)
            state += pair * (base ** i)

        return state

    def step(self, action1, action2):
        reward1 = float(self.payoff_matrix[action1, action2, 0])
        reward2 = float(self.payoff_matrix[action1, action2, 1])

        if self.state_history > 0:
            self.history.append((action1, action2))
            self.history = self.history[-self.state_history:]

        self.round += 1

        done = self.round >= self.horizon

        return self._get_state(), reward1, reward2, done, {}

