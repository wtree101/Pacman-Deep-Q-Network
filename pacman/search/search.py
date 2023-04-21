# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
#complete but not optimal
    path = []
    q = util.Stack()
    track = {} #create a set, to judge if a node is visited and record its father
    q.push((problem.getStartState(),0,0))
    track[problem.getStartState()] = (problem.getStartState(),0)
    # key 就是单重节点，但是father存储信息是 father节点+到达此节点的动作
    total_cost = 0
    while (not q.isEmpty()):
        top  = q.pop()
        state_now = top[0]
        cost_now = top[1]
        
        if problem.isGoalState(state_now): #track_chase_back
            v = state_now
            
            while (track[v][0]!=v):
                path.append(track[v][1])
                v = track[v][0]  
            path.reverse()
            print(cost_now)
            return path

        successors = problem.getSuccessors(state_now)
        for son,action,cost in successors:
            if (son in track.keys()):
                continue
            track[son] = (state_now,action)
            q.push((son,cost_now+cost))

    print("No finding solution!")
        
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    path = []
    q = util.Queue()
    track = {} #create a set, to judge if a node is visited and record its father
    q.push((problem.getStartState(),0,0))
    track[problem.getStartState()] = (problem.getStartState(),0)
    # key 就是单重节点，但是father存储信息是 father节点+到达此节点的动作
    total_cost = 0
    while (not q.isEmpty()):
        top  = q.pop()
        state_now = top[0]
        cost_now = top[1]
        
        if problem.isGoalState(state_now): #track_chase_back
            v = state_now
            
            while (track[v][0]!=v):
                path.append(track[v][1])
                v = track[v][0]  
            path.reverse()
            print(cost_now)
            return path

        successors = problem.getSuccessors(state_now)
        for son,action,cost in successors:
            if (son in track.keys()):
                continue
            track[son] = (state_now,action)
            q.push((son,cost_now+cost))

    print("No finding solution!")
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    path = []
    q = util.PriorityQueue()
    track = {} #create a set, to judge if a node is visited and record its father
    track_cost = {}
    track[problem.getStartState()] = (problem.getStartState(),0)
    track_cost[problem.getStartState()] = 0

    q.push((problem.getStartState(),0),0)
   
    expanded_nodes = set()
    # key 就是单重节点，但是father存储信息是 father节点+到达此节点的动作
    total_cost = 0
    while (not q.isEmpty()):
        top  = q.pop()
        state_now = top[0]
        cost_now = top[1]
        if state_now in expanded_nodes: #这里按照expanded = visited的观点，也是USC的观点处理
            continue
        
        expanded_nodes.add(state_now)

        if problem.isGoalState(state_now): #track_chase_back，出队时检验
            v = state_now
            
            while (track[v][0]!=v):
                path.append(track[v][1])
                v = track[v][0]  
            path.reverse()
            print(cost_now)
            return path

        successors = problem.getSuccessors(state_now)
        for son,action,cost in successors:
            if (son not in track_cost.keys()) or ((son in track_cost.keys()) and (cost_now+cost< track_cost[son])):
                track_cost[son] = cost_now + cost
                track[son] = (state_now,action)
                q.push((son,cost_now+cost),cost_now+cost)
            #这里算是一个剪枝，以及记录路径。我不想把每个状态的路径全部记录下来，这样空间效率太低了。只想记录一个前驱

    print("No finding solution!")
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    path = []
    q = util.PriorityQueue()
    
    track = {}
    track_cost = {}
    track[problem.getStartState()] = (problem.getStartState(),0)
    track_cost[problem.getStartState()] = 0
    
    q.push((problem.getStartState(),0),0)
   
    expanded_nodes = set()
    # key 就是单重节点，但是father存储信息是 father节点+到达此节点的动作
    total_cost = 0
    while (not q.isEmpty()):
        top  = q.pop()
        state_now = top[0]
        cost_now = top[1]

        if state_now in expanded_nodes: #这里按照expanded = visited的观点，也是USC的观点处理
            continue
        expanded_nodes.add(state_now)

        if problem.isGoalState(state_now): #track_chase_back
            v = state_now
            
            while (track[v][0]!=v):
                path.append(track[v][1])
                v = track[v][0]  
            path.reverse()
            print(cost_now)
            return path

        successors = problem.getSuccessors(state_now)
        for son,action,cost in successors:
            if (son not in track_cost.keys()) or ((son in track_cost.keys()) and (cost_now+cost< track_cost[son])):
                track_cost[son] = cost_now + cost
                track[son] = (state_now,action)
                q.push((son,cost_now+cost),cost_now+cost+heuristic(son,problem))

    print("No finding solution!")

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
