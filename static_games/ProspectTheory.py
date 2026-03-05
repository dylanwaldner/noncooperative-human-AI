import jax.numpy as np

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
        x = x - self.r
        return np.where(
            x >= 0,
            (x + 1e-4)**self.alpha,
            -self.lambd * ((-x + 1e-4)**self.alpha)
        )
    def w_plus(self, p):
        old_p = p
        p = np.clip(p, 1e-4, 1 - 1e-4)
        raw = p**self.gamma / (p**self.gamma + (1 - p)**self.gamma)**(1 / self.gamma)
        return np.where(old_p <= 0.0, 0.0, np.where(old_p >= 1.0, 1.0, raw))
    def w_minus(self, p):
        old_p = p
        p = np.clip(p, 1e-4, 1 - 1e-4)
        raw = p**self.delta / (p**self.delta + (1 - p)**self.delta)**(1 / self.delta)
        return np.where(old_p <= 0.0, 0.0, np.where(old_p >= 1.0, 1.0, raw))

    def cpt_gains(self, outcomes, probabilities, order):
        # keep gains only
        x = np.where(outcomes > self.r, outcomes, self.r)
        p = np.where(outcomes > self.r, probabilities, 0.0)

        # sort gains ascending
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
        w_tail = self.w_plus(tail)
        v_x = self.value_function(x)
        
        # We already have the cdf for each outcome, 
        # now we just need to shift it and add a 0 to preserve dimensions
        w_tail_next = np.concatenate([w_tail[1:], np.array([0.0])])
        pi = w_tail - w_tail_next

        return np.array(np.sum(pi * v_x), float)

    def cpt_losses(self, outcomes, probabilities, order):
        # Get losses
        x = np.where(outcomes < self.r, outcomes, self.r)
        p = np.where(outcomes < self.r, probabilities, 0.0)

        # Sort losses
        x = x[order]
        p = p[order]

        # cum sum
        head = np.cumsum(p)

        # decision weighting
        w_head = self.w_minus(head)
        v_x = self.value_function(x)

        # previous outcome
        # Here we right shift everything by 1, so 
        # at difference time (indices to follow) we get -m - -m-1, -m +1 - -m, ..., i - i-1
        w_prev_head = np.concatenate([np.array([0.0]), w_head[:-1]])

        # difference
        pi = w_head - w_prev_head

        # Rewritten with .dot for speed
        return np.array(pi.dot(v_x), float)

    def expected_pt_value(self, outcomes, probabilities, order):
        outcomes, probabilities = np.array(outcomes), np.array(probabilities,dtype=float)
        return self.cpt_gains(outcomes, probabilities, order) + self.cpt_losses(outcomes, probabilities, order)






  
