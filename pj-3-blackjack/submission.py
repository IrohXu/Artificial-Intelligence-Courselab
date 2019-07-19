import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0:
            return [-1, 1]
        else:
            return [state]        
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        solution = []
        if state == 0:
            solution.append((1,0.1,100))
            solution.append((-1,0.9,1))
        else:
            solution.append((state,1,0))
        return solution
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1.0
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        totalValue, peekedIndex, deckCardNum = state
        solution = []
        if deckCardNum == None:
            return []
        if action == 'Take':
            if peekedIndex == None:
                for i in range(len(deckCardNum)):
                    tempdeckCardNum = list(deckCardNum)
                    T = float(deckCardNum[i])/sum(deckCardNum)
                    if tempdeckCardNum[i] >= 1:
                        if totalValue + self.cardValues[i] > self.threshold:
                            solution.append(((totalValue + self.cardValues[i], None, None), T, 0))
                            continue
                        tempdeckCardNum[i] = tempdeckCardNum[i] - 1
                        if sum(tempdeckCardNum) == 0:
                            solution.append(((totalValue + self.cardValues[i], None, None), T, totalValue + self.cardValues[i]))
                        else:
                            solution.append(((totalValue + self.cardValues[i], None, tuple(tempdeckCardNum)), T, 0))
                    else:
                        continue
            else:
                if self.cardValues[peekedIndex] + totalValue > self.threshold:
                    solution.append(((self.cardValues[peekedIndex] + totalValue, None, None), 1, 0))
                else:
                    tempdeckCardNum = list(deckCardNum)
                    tempdeckCardNum[peekedIndex] = tempdeckCardNum[peekedIndex] - 1
                    if sum(tempdeckCardNum) == 0:
                        solution.append(((totalValue + self.cardValues[peekedIndex], None, None), 1, totalValue + self.cardValues[peekedIndex]))
                    else:
                        solution.append(((totalValue + self.cardValues[peekedIndex], None, tuple(tempdeckCardNum)), 1, 0))
        
        elif action == 'Peek':
            if peekedIndex == None:
                for i in range(len(deckCardNum)):
                    if deckCardNum[i] >= 1:
                        T = float(deckCardNum[i])/sum(deckCardNum)
                        solution.append(((totalValue, i, deckCardNum) ,T , -self.peekCost))
                    else:
                        continue
            else:
                return []   # peek twice return []
        else:
            solution.append(((totalValue, None, None), 1, totalValue))       
        return solution
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    solution = BlackjackMDP(cardValues=[1,2,3,4,5,6,21], multiplicity=1, threshold=20, peekCost=1)
    return solution
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        Q_next = 0.0
        Q_next = float(max((self.getQ(newState, action), action) for action in self.actions(newState))[0])
        if newState != None:
            difference = reward + self.discount*Q_next - self.getQ(state, action)
        else:
            difference = reward - self.getQ(state, action)
        alpha = self.getStepSize()
        for key, value in self.featureExtractor(state, action):
            self.weights[key] += alpha * difference * value
        # raise Exception("Not implemented yet")
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case

smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

smallMDP.computeStates()
ValueIterationSolution = util.ValueIteration()
ValueIterationSolution.solve(smallMDP)

rl = QLearningAlgorithm(smallMDP.actions, smallMDP.discount(), identityFeatureExtractor)
util.explorationProb=0
util.simulate(smallMDP, rl, numTrials=30000, maxIterations=1000, verbose=False, sort=False)

similar = 0.0
total = 0.0
for s in smallMDP.states:
    total = total + 1.0
    if ValueIterationSolution.pi[s] == rl.getAction(s):
        similar = similar + 1.0

print "The error rate of smallMDP is:" + str( 1.0-float(similar)/float(total))


# Large test case
'''
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP.computeStates()

ValueIterationSolution = util.ValueIteration()
ValueIterationSolution.solve(largeMDP)

rl = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(), identityFeatureExtractor)
util.explorationProb=0
util.simulate(largeMDP, rl, numTrials=30000, maxIterations=1000, verbose=False, sort=False)

similar = 0
total = 0
for s in largeMDP.states:
    total = total + 1
    if ValueIterationSolution.pi[s] == rl.getAction(s):
        similar = similar + 1

print "The error rate of largeMDP is:" + str( 1.0-float(similar)/float(total))
'''


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    feature = []
    feature.append(((total,action),1))
    presence = []
    if counts != None:
        for i in range(len(counts)):
            if counts[i] == 0:
                presence.append(0)
            else:
                presence.append(1)
            feature.append(((counts[i], i, action),1))           
        feature.append(((tuple(presence), action),1))
    return feature

    # raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)


ValueIterationSolution = util.ValueIteration()
ValueIterationSolution.solve(originalMDP)
policy = ValueIterationSolution.pi

rl1 = util.FixedRLAlgorithm(policy)
print policy
total = 10000
print float(sum(util.simulate(newThresholdMDP, rl1, numTrials=total, maxIterations=1000, verbose=False, sort=False)))/total

rl2 = QLearningAlgorithm(originalMDP.actions, newThresholdMDP.discount(), identityFeatureExtractor)
print float(sum(util.simulate(newThresholdMDP, rl2, numTrials=total, maxIterations=1000, verbose=False, sort=False)))/total

