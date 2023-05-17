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
        if successorGameState.isWin():
            return float('inf')
        if action == Directions.STOP:
            return -float('inf')
        return successorGameState.getScore() - currentGameState.getScore()

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

        numberOfGhosts = gameState.getNumAgents() - 1

        def maxAction(gameState, prevDepth):
            currentDepth = prevDepth + 1
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(0)
            maxValue = -float('inf')
            for posAction in posActions:
                succ = gameState.generateSuccessor(0, posAction)
                maxValue = max(maxValue, minAction(succ, currentDepth, 1))
            return maxValue

        def minAction(gameState, prevDepth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(agentIndex)
            minValue = float('inf')
            for posAction in posActions:
                successor = gameState.generateSuccessor(agentIndex, posAction)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minValue = min(minValue, maxAction(successor, prevDepth))
                else:
                    minValue = min(minValue, minAction(successor, prevDepth, agentIndex + 1))
            return minValue

        posActions = gameState.getLegalActions(0)
        currentScore = -float('inf')
        result = str()
        for posAction in posActions:
            nextState = gameState.generateSuccessor(0, posAction)
            posScore = minAction(nextState, 0, 1)
            if posScore > currentScore:
                result = posAction
                currentScore = posScore
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def maxAction(gameState, prevDepth, alphaInput, betaInput):
            curretnDepth = prevDepth + 1
            if gameState.isWin() or gameState.isLose() or curretnDepth == self.depth:
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(0)
            maxValue = -float('inf')
            alpha = alphaInput
            for posAction in posActions:
                succ = gameState.generateSuccessor(0, posAction)
                maxValue = max(maxValue, minAction(succ, curretnDepth, 1, alpha, betaInput))
                if maxValue > betaInput:
                    return maxValue
                alpha = max(alpha, maxValue)
            return maxValue

        def minAction(gameState, prevDepth, agentIndex, alphaInput, betaInput):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(agentIndex)
            minValue = float('inf')
            beta = betaInput
            for posAction in posActions:
                succ = gameState.generateSuccessor(agentIndex, posAction)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minValue = min(minValue, maxAction(succ, prevDepth, alphaInput, beta))
                    if minValue < alphaInput:
                        return minValue
                    beta = min(beta, minValue)
                else:
                    minValue = min(minValue, minAction(succ, prevDepth, agentIndex + 1, alphaInput, beta))
                    if minValue < alphaInput:
                        return minValue
                    beta = min(beta, minValue)
            return minValue

        posActions = gameState.getLegalActions(0)
        currentScore = -float('inf')
        result = str()
        alpha = -float('inf')
        beta = float('inf')
        for posAction in posActions:
            newState = gameState.generateSuccessor(0, posAction)
            posScore = minAction(newState, 0, 1, alpha, beta)
            if posScore > currentScore:
                result = posAction
                currentScore = posScore
            if posScore > beta:
                return result
            alpha = max(alpha, posScore)
        return result

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
        def maxAction(gameState, prevDepth): # Exact hetzelfde als bij minimax zonder alpha-beta pruning
            currentDepth = prevDepth + 1
            if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(0)
            maxValue = -float('inf')
            for posAction in posActions:
                succ = gameState.generateSuccessor(0, posAction)
                maxValue = max(maxValue, expectedAction(succ, currentDepth, 1))
            return maxValue

        def expectedAction(gameState, prevDepth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            posActions = gameState.getLegalActions(agentIndex)
            sumExpectedValues = 0.0
            numOfPosActions = float(len(posActions))
            for posAction in posActions:
                succ = gameState.generateSuccessor(agentIndex,posAction)
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedValue = maxAction(succ, prevDepth)
                else:
                    expectedValue = expectedAction(succ, prevDepth, agentIndex + 1)
                sumExpectedValues = sumExpectedValues + expectedValue
            if numOfPosActions == 0:
                return 0
            return sumExpectedValues/numOfPosActions

        posActions = gameState.getLegalActions(0)
        currentScore = -float('inf')
        result = str()
        for posAction in posActions:
            newState = gameState.generateSuccessor(0, posAction)
            posScore = expectedAction(newState, 0, 1)
            if posScore > currentScore:
                result = posAction
                currentScore = posScore
        return result


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # TODO beter maken
    return currentGameState.getScore() - len(currentGameState.getFood().asList())

# Abbreviation
better = betterEvaluationFunction
