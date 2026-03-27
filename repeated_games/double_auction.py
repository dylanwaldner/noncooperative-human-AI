import numpy as np
import time
# Double Auction Game
class DoubleAuction:
  '''
  A double auction game with midpoint pricing. Please let me know if you think a different pricing mechanism is better.
  other than that, very similar to the game env, using state indexing with base n^2 where n is the action size. 

  Another important consideration: should we have no trade be 0 reward or -1?
  -1 might inspire greater exploration early on
  '''
  def __init__(self, k=10, valuation=6, cost=4, horizon=100, state_history=2):
    self.A = set(range(1, k+1))
    self.B = set(range(1, k+1))
    self.k = k
    print("k: ", self.k)

    self.valuation = valuation
    self.cost = cost
    
    self.history = []
    self.state_history = state_history
    self.state_size = (self.k**2) ** self.state_history
    print("in state size: ", self.state_size)
    self.round = 0

    self.horizon = horizon

  # We need a ground truth payoff table for analysis
  def build_payoff_matrix(self):
    payoff_matrix = np.zeros((self.k, self.k, 2))

    for i in range(self.k):
      for j in range(self.k):
        # bids and asks are 1 indexed
        bid = i + 1
        ask = j + 1
        if bid >= ask: # trade confirmed
          price = (bid + ask) / 2 # midpoint pricing
          buyer_payoff = self.valuation - price # reward is distance of valiuation over price
          seller_payoff = price - self.cost # reward is distance from cost under price (profit)
          payoff_matrix[i, j] = (buyer_payoff, seller_payoff)
        else:
          payoff_matrix[i, j] = (0, 0)

    return payoff_matrix

  def reset(self):
    self.round = 0
    return self._get_state()

  def _get_state(self):
    if self.state_history == 0:
        return 0
    base = self.k * self.k
    state = 0

    recent = self.history[-self.state_history:]

    for i, (a1, a2) in enumerate(reversed(recent)):
        # Convert price space (1 indexed) to action space (0 indexed)
        pair = (a1 - 1) * self.k + (a2 - 1)
        state += pair * (base ** i)

    return state

  def step(self, bid, ask):
    # The actions are 0 indexed, bids and asks are 1 indexed
    # We do this to make sure prices and rewards are calculated in price space
    bid += 1
    ask += 1

    assert bid in self.B and ask in self.A, f"bid: {bid}, ask: {ask}"

    if self.state_history > 0:
      self.history.append((bid, ask))
      self.history = self.history[-self.state_history:]

    self.round += 1
    done = self.round >= self.horizon

    # If a successful trade
    if bid >= ask:
      price = (bid + ask) / 2 # midpoint pricing
      buyer_payoff = self.valuation - price # how good a buy
      seller_payoff = price - self.cost # profit

      return self._get_state(), buyer_payoff, seller_payoff, done, {}

    else:
      return self._get_state(), 0, 0, done, {} # Should no trade rewards be -1?
