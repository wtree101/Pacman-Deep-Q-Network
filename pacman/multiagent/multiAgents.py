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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        oldFoodlist = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        #print(newGhostStates)
        #print(newScaredTimes)
        #print(newFood)
        #print(newPos)
       ## print(ghostState.configuration.getPosition())
        
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        for ghostState in newGhostStates:
            ghost_pos = ghostState.getPosition()
            ghost_scared = ghostState.scaredTimer
            if (ghost_scared==0 and (util.manhattanDistance(ghost_pos,newPos)<=1)):
                score -= 250
            if (ghost_scared>2 and (util.manhattanDistance(ghost_pos,newPos)<=1)):
                score += 10
            
        min_distance = -1

        foodlist = newFood.asList()
        for food in foodlist:
            d = util.manhattanDistance(food,newPos)
            if (min_distance==-1):
                min_distance = d
            else:
                min_distance = min(min_distance,d)

        if (newPos not in oldFoodlist):
            #print(True)
            score -= min_distance

        return score

def scoreEvaluationFunction(currentGameState):
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
    def max_value(self,state,depth_now,agentIndex,fn):
        v = -float('inf')
        path = []
        next_agent = (agentIndex + 1)% state.getNumAgents()
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v_s,path_s = self.value(successor,depth_now,next_agent,fn)
            if (v_s>v):
                v = v_s
                path_s.append(action)
                path = path_s


        return v,path
    
    def min_value(self,state,depth_now,agentIndex,fn):
        v = float('inf')
        next_agent = (agentIndex + 1)% state.getNumAgents()
        path = []
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v_s,path_s = self.value(successor,depth_now,next_agent,fn) #min与max里面不用考虑加一层的问题
            if (v_s<v):
                v = v_s
                path_s.append(action)
                path = path_s

        return v,path
    

    def value(self,state,depth_old,agentIndex,fn):
        """value of current state; depth_now is old, However"""
        #depthnow初始为0
        #path存路径
        if (agentIndex==0):
            depth_now = depth_old + 1
        else:
            depth_now = depth_old
    
        #terminalS
        if (state.isLose()):
            return fn(state),[]
        if (state.isWin()):
            return fn(state),[]
        if (depth_now>self.depth):
            return fn(state),[]
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
       # if len(state.getLegalActions(agentIndex))==0:
        #    return fn(state),[] 这种情况应该是不可能的。


        if agentIndex==0:
            v,path = self.max_value(state,depth_now,agentIndex,fn)
        else:
            v,path = self.min_value(state,depth_now,agentIndex,fn)
        
        #print(path)
        return v,path

    def getAction(self, gameState):
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
        path = []
        _,path = self.value(gameState,0,0,self.evaluationFunction)
       # print(path.reverse())
        path.reverse()
        return path[0]

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    a MAX下界 b MIN上界
    """
    def max_value(self,state,depth_now,agentIndex,fn,a,b):
        v = -float('inf')
        path = []
        next_agent = (agentIndex + 1)% state.getNumAgents()
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v_s,path_s = self.value(successor,depth_now,next_agent,fn,a,b)
            if(v_s>b):
                path_s.append(action)
                return v_s,path_s
            if (v_s>v):
                v = v_s
                path_s.append(action)
                path = path_s
                a = max(a,v_s) #在里面更新问题应该也不是很大
        

        return v,path
    
    def min_value(self,state,depth_now,agentIndex,fn,a,b):
        v = float('inf')
        next_agent = (agentIndex + 1)% state.getNumAgents()
        path = []
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v_s,path_s = self.value(successor,depth_now,next_agent,fn,a,b) #min与max里面不用考虑加一层的问题
            if (v_s<a):
                path_s.append(action)
                return v_s,path_s
            if (v_s<v):
                v = v_s
                path_s.append(action)
                path = path_s
                b = min(b,v_s)

        return v,path
    

    def value(self,state,depth_old,agentIndex,fn,a,b):
        """value of current state; depth_now is old, However"""
        #depthnow初始为0
        #path存路径
        if (agentIndex==0):
            depth_now = depth_old + 1
        else:
            depth_now = depth_old
    
        #terminalS
        if (state.isLose()):
            return fn(state),[]
        if (state.isWin()):
            return fn(state),[]
        if (depth_now>self.depth):
            return fn(state),[]
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
       # if len(state.getLegalActions(agentIndex))==0:
        #    return fn(state),[] 这种情况应该是不可能的。


        if agentIndex==0:
            v,path = self.max_value(state,depth_now,agentIndex,fn,a,b)
        else:
            v,path = self.min_value(state,depth_now,agentIndex,fn,a,b)
        
        #print(path)
        return v,path

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _,path = self.value(gameState,0,0,self.evaluationFunction,-float('inf'),float('inf'))
       # print(path.reverse())
        path.reverse()
        return path[0]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self,state,depth_now,agentIndex,fn):
        v = -float('inf')
        a = ''
        next_agent = (agentIndex + 1)% state.getNumAgents()
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v_s,_ = self.value(successor,depth_now,next_agent,fn)
            if (v_s>v):
                v = v_s
                a = action


        return v,a
    
    def Expected_value(self,state,depth_now,agentIndex,fn):
        
        next_agent = (agentIndex + 1)% state.getNumAgents()
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
        Exp = 0 
        n = len(state.getLegalActions(agentIndex))
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex,action)
            v,_ = self.value(successor,depth_now,next_agent,fn) #min与max里面不用考虑加一层的问题
            Exp += v
        Exp = float(Exp)/n

        return Exp,''
    

    def value(self,state,depth_old,agentIndex,fn):
        """value of current state; depth_now is old, However"""
        #depthnow初始为0
        #path存路径
        if (agentIndex==0):
            depth_now = depth_old + 1
        else:
            depth_now = depth_old
    
        #terminalS
        if (state.isLose()):
            return fn(state),[]
        if (state.isWin()):
            return fn(state),[]
        if (depth_now>self.depth):
            return fn(state),[]
        # if (len(state.getLegalActions(agentIndex))==0):
        #     print('???????????????')
       # if len(state.getLegalActions(agentIndex))==0:
        #    return fn(state),[] 这种情况应该是不可能的。


        if agentIndex==0:
            v,a = self.max_value(state,depth_now,agentIndex,fn)
        else:
            v,a = self.Expected_value(state,depth_now,agentIndex,fn)
        
        #print(path)
        return v,a

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _,a = self.value(gameState,0,0,self.evaluationFunction)
        return a
    
        util.raiseNotDefined()

import math
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
  #  oldFoodlist = currentGameState.getFood().asList()
   
    score = currentGameState.getScore()
    check_trapped = True
    for action in currentGameState.getLegalActions():
        successor = currentGameState.generateSuccessor(0,action)
        check_trapped = check_trapped and successor.isLose()

    
    if (check_trapped):
        score -= 500
        
    min_distance = -1

    foodlist = newFood.asList()
    for food in foodlist:
        d = util.manhattanDistance(food,newPos)
        if (min_distance==-1):
            min_distance = d
        else:
            min_distance = min(min_distance,d)
    
    score += 9*math.exp(-(min_distance-1)) #防止被卡住

   

    return score


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
