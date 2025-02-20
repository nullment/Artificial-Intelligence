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
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
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

    visited = []
    goalWalk =[]
    stack = util.Stack()
    #start = SearchProblem.getStartState()
    #goalBoolean = SearchProblem.isGoalState() # TRUE IF GOAL STATE RETURN

    stack.push((problem.getStartState(), goalWalk))

    while not stack.isEmpty():
        popped = stack.pop()
        currentPosition = popped[0]
        currentPath = popped[1]

        
        if currentPosition not in visited:
            visited.append(currentPosition)
        if problem.isGoalState(currentPosition):
            return currentPath
    
        for successorNodes in  problem.getSuccessors(currentPosition):
            if successorNodes[0] not in visited:
                    newPath = currentPath + [successorNodes[1]]
                    stack.push((successorNodes[0], newPath)) 
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = []
    goalWalk =[]
    queue = util.Queue()
    #start = SearchProblem.getStartState()
    #goalBoolean = SearchProblem.isGoalState() # TRUE IF GOAL STATE RETURN

    queue.push((problem.getStartState(), goalWalk))


    while not queue.isEmpty():
        popped = queue.pop()

        currentPosition = popped[0]
        currentPath = popped[1]

        if currentPosition not in visited:
            visited.append(currentPosition)

        if problem.isGoalState(currentPosition):
            return currentPath
    
        for successorNodes in  problem.getSuccessors(currentPosition):#expand nodes q2 error in double expanding    
            if successorNodes[0] not in visited:
                    if not any(successorNodes[0] == expanded[0] for expanded in queue.list):
                        newPath = currentPath + [successorNodes[1]]
                        queue.push((successorNodes[0], newPath)) 
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = []
    goalWalk =[]
    priorityQueue = util.PriorityQueue()
    costWalk = problem.getCostOfActions(goalWalk)
    #start = SearchProblem.getStartState()
    #goalBoolean = SearchProblem.isGoalState() # TRUE IF GOAL STATE RETURN

    priorityQueue.push((problem.getStartState(), goalWalk), costWalk)

    while not priorityQueue.isEmpty():
        popped = priorityQueue.pop()

        currentPosition = popped[0]
        currentPath = popped[1]
        
        if currentPosition not in visited:
            visited.append(currentPosition)

            if problem.isGoalState(currentPosition):
                return currentPath
        
            for childNodes in  problem.getSuccessors(currentPosition):#expand nodes
                if childNodes[0] not in visited:
                        
                        newPath = currentPath + [childNodes[1]]
                        totalCostofAction = problem.getCostOfActions(newPath)
                        priorityQueue.push((childNodes[0], newPath), totalCostofAction) 

    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = {}  
    priorityQueue = util.PriorityQueue()  

    startState = problem.getStartState()  
    priorityQueue.push((startState, []), heuristic(startState, problem))  
    visited[str(startState)] = 0  

    while not priorityQueue.isEmpty():
        popped = priorityQueue.pop()  
        currentPosition = popped[0]
        currentPath = popped[1]

        if problem.isGoalState(currentPosition): 
            return currentPath  

        currentCost = problem.getCostOfActions(currentPath) 

        if visited[str(currentPosition)] < currentCost:
            continue

        for childNode, successorAction, tempCost in problem.getSuccessors(currentPosition):
            newPath = currentPath + [successorAction]  
            newCost = problem.getCostOfActions(newPath)  
            totalCost = newCost + heuristic(childNode, problem) 

   
            if str(childNode) not in visited or newCost < visited[str(childNode)]:
                visited[str(childNode)] = newCost
                priorityQueue.push((childNode, newPath), totalCost)  

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch