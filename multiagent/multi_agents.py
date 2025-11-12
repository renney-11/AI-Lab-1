# multi_agents.py
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
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


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]

        # for each ghost in the successor game state do the following:
        for state in new_ghost_states:
            
            # empty food distance list
            food_dist = []

            # get the manhattan distance of the new position of pacman to the ghost
            ghost_dist = util.manhattan_distance(new_pos, state.get_position())

            # if the ghost is closer than 3 distance units away, correct the score of this successor state by
            # adding the ghost distance to it (the closer the ghost, lower the score)
            if ghost_dist < 3:
                ghost_corrected_score = successor_game_state.get_score() + ghost_dist
                return ghost_corrected_score
            
            # if the ghost is further away than 3 units, consider the distance to food.
            else: 
                # add the distance to all food items in the successor game state to the list. 
                for food in new_food.as_list():
                    food_dist.append(util.manhattan_distance(new_pos, food))
                
                # if there is indeed food, correct the score by adding a function of the closest food.
                # the relative reward is much bigger for food items that are one distance unit away.
                if food_dist:
                    food_corrected_score = successor_game_state.get_score() + 1/(min(food_dist))
                    return food_corrected_score
        
        return successor_game_state.get_score()

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        """
        For each legal Pacman action:
        generate the successor state
        call helper(successor, depth=0, agent=1)
        store (score, action)
        Return the action with the highest score
        """

        # this helper function creates the recursion needed to traverse the tree
        def helper(state, depth, agent):

            # if there are no legal actions possible, return the score
            if not state.get_legal_actions(agent):
                return self.evaluation_function(state)
            
            # if the current state is win/lose or the maximum depth allowed is reached, return the score
            if state.is_win() or state.is_lose() or depth == self.depth:
                return self.evaluation_function(state)
           
            # successor scores will be stored in this
            scores = []

            # for all legal actions of the agent (pacman or ghost/s)
            for action in state.get_legal_actions(agent):
                # get the successor state
                successor = state.generate_successor(agent, action)
                # if the current agent is not the last one in the game:
                if agent + 1 < state.get_num_agents():
                    # append the score of its successors to the list.
                    # this calls the helper function recursively for the next agent
                    scores.append(helper(successor, depth, agent + 1))
                # if the current agent is the last one, go back to pacman and increase the depth by one
                else: scores.append(helper(successor, depth + 1, 0))
            
            # if the current agent is pacman get the maximum value out of the successor game states
            # if ghost: minimum
            if agent == 0: 
                return max(scores)
            else: return min(scores)

        # store (score, action) tuples
        moves = []

        # for each (from the initial state) legal action of pacman, 
        # generate successors and run the helper function for each.
        # return the highest valued move.
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            moves.append((helper(successor, 0, 1), action))
        return max(moves)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
