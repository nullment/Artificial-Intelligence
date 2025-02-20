# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodSuccessor = 0

        if successorGameState.isWin():
            return float("inf")

        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float("-inf")

        foodDist = [util.manhattanDistance(food, newPos) for food in list(newFood.asList())]

        if currentGameState.getNumFood() > successorGameState.getNumFood():
            foodSuccessor = 300

        endScore = successorGameState.getScore() - 5 * min(foodDist, default=0) + foodSuccessor

        return endScore
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(state, depth, agentIndex):
        
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)

            return minValue(state, depth, agentIndex)
        
        def minValue(state, depth, agentIndex):
           
            minEval = float("inf")
            numAgents = state.getNumAgents()
            for move in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, move)
                if agentIndex == numAgents - 1:
                    minEval = min(minEval, minimax(successorState, depth + 1, 0))
                else:
                    minEval = min(minEval, minimax(successorState, depth, agentIndex + 1))
            return minEval

        def maxValue(state, depth, agentIndex):

            maxEval = float("-inf")
            for move in state.getLegalActions(agentIndex):
                successorState = state.generateSuccessor(agentIndex, move)
                maxEval = max(maxEval, minimax(successorState, depth, 1))  
            return maxEval

        bestScore = float("-inf")
        bestMove = Directions.STOP
        for move in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, move)
            moveScore = minimax(successorState, 0, 1)
            if moveScore > bestScore:
                bestScore = moveScore
                bestMove = move

        return bestMove
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestScore = float("-inf")
        bestAction = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            actionScore = self.alphabeta(successorState, 0, 1, alpha, beta)
            
            if actionScore > bestScore:
                bestScore = actionScore
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

    def alphabeta(self, state, depth, agentIndex, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agentIndex == 0:
            return self.maxValue(state, depth, alpha, beta)
        else:
            return self.minValue(state, depth, agentIndex, alpha, beta)

    def maxValue(self, state, depth, alpha, beta):

        maxEval = float("-inf")
        for action in state.getLegalActions(0):
            successorState = state.generateSuccessor(0, action)
            maxEval = max(maxEval, self.alphabeta(successorState, depth, 1, alpha, beta))
            
            if maxEval > beta:#prune
                return maxEval  
            alpha = max(alpha, maxEval)

        return maxEval

    def minValue(self, state, depth, agentIndex, alpha, beta):

        minEval = float("inf")
        numAgents = state.getNumAgents()

        for action in state.getLegalActions(agentIndex):
            successorState = state.generateSuccessor(agentIndex, action)
            if agentIndex == numAgents - 1: 
                minEval = min(minEval, self.alphabeta(state.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta))
            else:
                minEval = min(minEval, self.alphabeta(successorState, depth, agentIndex + 1, alpha, beta))
            
            if minEval < alpha: #prune
                return minEval 
            beta = min(beta, minEval)

        return minEval
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestScore = float("-inf")
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            actionScore = self.expectimaxVal(successorState, 0, 1)
            
            if actionScore > bestScore:
                bestScore = actionScore
                bestAction = action

        return bestAction
    
    def expectimaxVal(self, state, depth, agentIndex):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        if agentIndex == 0:
            return self.maxValue(state, depth)
        return self.average(state, depth, agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxEval = float("-inf")
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            maxEval = max(maxEval, self.expectimaxVal(successorState, currentDepth, 1))
        return maxEval

    def average(self, gameState, currentDepth, agentIndex):
        average = 0
        legalActions = gameState.getLegalActions(agentIndex)
        numAgents = gameState.getNumAgents()

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numAgents - 1:
                average += self.expectimaxVal(successor, currentDepth + 1, 0)
            else:
                average += self.expectimaxVal(successor, currentDepth, agentIndex + 1)
        
        return average
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
     This evaluation function calculates the current game state score based on: Pacman's position relative to food and ghosts, the distance to the nearest food and ghost, 
     and ghosts' scared status and the penalty for being near non-scared ghosts
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    pacmanPos = currentGameState.getPacmanPosition()

    foodGrid = currentGameState.getFood()
    foodPositions = foodGrid.asList()

    ghostStates = currentGameState.getGhostStates()
    ghostScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    ghostPenaltyScore = sum(
        (2 / (distance + 0.1) if ghostScaredTimes[i] == 0 else -10 / (distance + 0.1)) 
        for i, distance in enumerate(ghostDistances)
    )

    foodAttractionScore = sum(5 / (manhattanDistance(pacmanPos, food) + 1) for food in foodPositions)

    baseScore = currentGameState.getScore()

    totalScore = baseScore + foodAttractionScore - ghostPenaltyScore

    return totalScore


# Abbreviation
better = betterEvaluationFunction
