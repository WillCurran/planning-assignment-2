import numpy as np
import queue
import heapq
from game import BoardState, GameSimulator, Rules

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg="bfs"):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """
        alg_by_name = {
            "bfs": self.bfs_search,
            "a*_count_pieces_heuristic": self.a_star_search_meta(_count_diff_pieces_heuristic),
            "a*_manhattan_heuristic": self.a_star_search_meta(_knight_manhattan_heuristic),
        }
        # Default to best current A* implementation
        if alg not in alg_by_name:
            self.search_alg_fnc = alg_by_name["a*_manhattan_heuristic"]
            return
        self.search_alg_fnc = alg_by_name[alg]

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    """ Here is an example:
    
    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
    """

    def bfs_search(self):
        """
        Just BFS with no modifications or uniform-cost planning enhancements
        """
        if self.initial_state in self.goal_state_set:
            return [(self.initial_state, None)]
        # Using queue versus non-optimized usage of list as FIFO ~30% speed improvement for BFS on 6-length path
        frontier = queue.Queue()
        frontier.put(self.initial_state)
        # Back-pointers for visited states
        # Using hash demonstrated slight edge ~2% improvement overall for BFS on 6-length path
        parent_state_action_by_state = {hash(self.initial_state): None}
        while not frontier.empty():
            curr = frontier.get()
            for action in self.get_actions(curr):
                neighbor = self.execute(curr, action)
                if hash(neighbor) in parent_state_action_by_state:
                    continue
                parent_state_action_by_state[hash(neighbor)] = (curr, action)  # TODO - could we prune from these states to save memory on bad paths?
                if neighbor in self.goal_state_set:
                    return _path_from_parents(parent_state_action_by_state, neighbor)
                frontier.put(neighbor)
        raise Exception("BFS failed to find a solution")

    def a_star_search_meta(self, heuristic_func):
        """
        Decorator func to return A* with no modifications or uniform-cost planning enhancements.
        Heuristic options:
        - Count number of pieces not in their final position.
            Admissibility: Yes because need at least one move for each case.
            Ignores ball moves, obstacles, knight's move.
        - Count number of knight's moves required for each piece to move to its
          target. Add one if ball not in final position.
            Admissibility: Yes because obstacles may increase cost.
            Ignores ball moves and obstacles.
        """
        
        def _a_star_search():
            if self.initial_state in self.goal_state_set:
                return [(self.initial_state, None)]
            # (priority, cost to get to this state, state)
            frontier = [(0, 0, self.initial_state)]
            # Back-pointers for visited states
            # Using hash demonstrated slight edge ~2% improvement overall for BFS on 6-length path
            parent_state_action_by_state = {hash(self.initial_state): None}
            while len(frontier) > 0:
                _, cost_to_curr, curr = heapq.heappop(frontier)
                for action in self.get_actions(curr):
                    neighbor = self.execute(curr, action)
                    if hash(neighbor) in parent_state_action_by_state:
                        continue
                    parent_state_action_by_state[hash(neighbor)] = (curr, action)  # TODO - could we prune from these states to save memory on bad paths?
                    if neighbor in self.goal_state_set:
                        return _path_from_parents(parent_state_action_by_state, neighbor)
                    # Cost to go to neighbor is 1
                    cost = 1 + cost_to_curr + min(
                        (heuristic_func(self.sim.game_state, neighbor, g) 
                         for g in self.goal_state_set)
                    )
                    heapq.heappush(frontier, (cost, 1 + cost_to_curr, neighbor))
            raise Exception("A* failed to find a solution")
        
        return _a_star_search
  

def _count_diff_pieces_heuristic(game_state, state_a, state_b):
    """
    Return heuristic value estimated between a and b.
    state_a and state_b are encoded state tuples.
    """
    count = 0
    for a, b in zip(state_a[0], state_b[0]):
        count += int(a != b)
    return count


def _knight_manhattan_heuristic(game_state, state_a: BoardState, state_b):
    """
    Return heuristic value estimated between a and b.
    state_a and state_b are encoded state tuples.
    """
    count = 0
    for i, (a, b) in enumerate(zip(state_a[0], state_b[0])):
        # Ball indices
        if i == 5 or i == 11:
            continue
        a_x, a_y = game_state.decode_single_pos(a)
        b_x, b_y = game_state.decode_single_pos(b)
        # Quick way to approximate knight move, even though smaller often means more
        # moves
        count += (abs(b_x - a_x) + abs(b_y - a_y)) / 3
    # Add one for each ball out of place
    count += int(state_a[0][5] != state_b[0][5])
    count += int(state_a[0][11] != state_b[0][11])
    return count


# TODO - unit test and type annotations
def _path_from_parents(parent_state_action_by_state, final_state):
    assert hash(final_state) in parent_state_action_by_state
    assert None in parent_state_action_by_state.values()
    path = [(final_state, None)]
    curr = final_state
    while curr is not None:
        state_action = parent_state_action_by_state[hash(curr)]
        if state_action is None:
            break
        next_state, next_action = state_action
        path.append((next_state, next_action))
        curr = next_state
    return list(reversed(path))
