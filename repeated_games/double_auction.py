import numpy as np
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

    self.valuation = valuation
    self.cost = cost
    
    self.history = []
    self.state_history = state_history
    self.state_size = (self.k**2) ** self.state_history
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
    '''
    Part of the problem with the double auction game is the way states explode, causing memory problems. 
    Maybe there's a better way to handle this? Games are also converging on just a few states, predictably. 
    '''

    state = 0
    for i in range(len(self.history)):
        if i < self.state_history:
            a1, a2 = self.history[-(i+1)]
            pair = (a1-1) * self.k + (a2-1)
        else:
            pair = 0
 
        state += pair * ((self.k ** 2) ** i) 

    return int(state)

  def step(self, bid, ask):
    # The actions are 0 indexed, bids and asks are 1 indexed
    bid += 1
    ask += 1

    assert bid in self.B and ask in self.A, f"bid: {bid}, ask: {ask}"
  
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
