import numpy as np

class ProspectTheory:
    """Complete PT implementation for game analysis"""

    def __init__(self, lambd=2.25, alpha=0.88, gamma=0.61, r=0, delta=0.69):
        self.lambd = lambd  # Loss aversion
        self.alpha = alpha  # Diminishing sensitivity
        self.gamma = gamma  # Probability weighting
        self.delta = delta
        self.r = r          # Reference point

    def value_function(self, x):
        """PT value function v(x)"""
        # I chose to handle the reference point subtraction in the method to prevent the need for tracking the correct
        # reference point in the training loop, seemed more modular and self contained this way
        x = np.asarray(x) - self.r

        eps = 1e-10

        gains = (x + eps) ** self.alpha
        losses = -self.lambd * ((-x + eps) ** self.alpha)

        return np.where(x >= 0, gains, losses)

    def w_plus(self, p):
        if p <= 0.0:
            return 0.0
        elif p >= 1.0:
            return 1.0
        else:
            return p ** self.gamma / (p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma) 

    def w_minus(self, p):
        if p <= 0.0:
            return 0.0
        elif p >= 1.0:
            return 1.0
        else:
            return p ** self.delta / (p ** self.delta + (1 - p) ** self.delta) ** (1 / self.delta)

    def cpt_gains(self, outcomes, probabilities):
        # keep gains only, outcomes == self.r won't matter anyways (v = 0)
        mask = outcomes > self.r
        x = outcomes[mask]
        p = probabilities[mask]

        if len(x) == 0:
            return 0.0

        # sort gains ascending, keeping probs and outcomes aligned (before transformation)
        order = np.argsort(x)
        x = x[order]
        p = p[order]

        # tail probabilities
        # This keeps the same dimension but applies a cumulative sum along the vector
        # So if raw probs are [0.3, 0.2, 0.1, 0.4],
        # This will reverse it: [0.4, 0.1, 0.2, 0.3]
        # Cumulatively sum it: [0.4, 0.5, 0.7, 1]
        # and then reverse it again: [1, 0.7, 0.5, 0.4]
        # Formally:
        # If p = [p0, p1, p2, p3] are probabilities of GAINS
        # ordered by increasing outcome value,
        # then:
        #   tail = [p0+p1+p2+p3, p1+p2+p3, p2+p3, p3]
        tail = np.cumsum(p[::-1])[::-1]

        # decision weights
        w_tail = np.array([self.w_plus(t) for t in tail], dtype=float)
        # value function v
        v_x = np.array([self.value_function(xi) for xi in x], dtype=float)
        
        # We already have the cdf for each outcome, 
        # now we just need to shift it and add a 0 to preserve dimensions
        w_tail_next = np.concatenate([w_tail[1:], [0.0]])
        # And then find the difference (right after equation 1 in the CPT paper)
        pi = w_tail - w_tail_next

        return float(np.sum(pi * v_x))

    def cpt_losses(self, outcomes, probabilities):
        # Get losses
        mask = outcomes < self.r
        x = outcomes[mask]
        p = probabilities[mask]

        if len(x) == 0:
            return 0.0

        # Sort losses
        sorted_indices = np.argsort(x)      
        x = x[sorted_indices]
        p = p[sorted_indices]

        # cum sum
        head = np.cumsum(p)

        # decision weighting
        w_head = np.array([self.w_minus(h) for h in head], dtype=float)
        v_x = np.array([self.value_function(xi) for xi in x], dtype=float)

        # previous outcome
        # Here we right shift everything by 1, so 
        # at difference time (indices to follow) we get -m - -m-1, -m +1 - -m, ..., i - i-1
        w_prev_head = np.concatenate([[0.0], w_head[:-1]])

        # difference
        pi = w_head - w_prev_head

        # Rewritten from sum (pi * v_x) to .dot() for speed
        return float(pi.dot(v_x))

    def expected_pt_value(self, outcomes, probabilities):
        outcomes, probabilities = np.array(outcomes), np.array(probabilities,dtype=float)
        return float(self.cpt_gains(outcomes, probabilities) + self.cpt_losses(outcomes, probabilities))






  
