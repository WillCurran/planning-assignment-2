import copy
import enum
import numpy as np


# Ordered clockwise
KNIGHT_MOVE_POS_DELTA = [
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
]


class Player(enum.IntEnum):
    WHITE = 0
    """White player's index"""
    BLACK = 1
    """Black player's index"""


class Graph:
    def __init__(self, n) -> None:
        self.adj = np.empty((n, n), dtype=bool)
        self.adj.fill(False)  # Do not include self-loops by default
        
    def add_edge(self, i, j) -> None:
        self.adj[i, j] = True
        self.adj[j, i] = True
    
    def neighbors(self, i) -> np.ndarray:
        return np.argwhere(self.adj[i].squeeze())
    
    # Add tours?


class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]
    
    # TODO - consider only using player_pieces or player_pieces_idx
    def player_pieces(self, player_idx):
        """
        Return copy of player's pieces state.
        """
        assert player_idx in (Player.WHITE, Player.BLACK)
        return self.state.copy()[(player_idx * 6):(5 + 6 * player_idx)]

    def player_pieces_idx(self, player_idx):
        """
        Return ndarray of indices of player's pieces in self.state.
        """
        assert player_idx in (Player.WHITE, Player.BLACK)
        return np.arange(player_idx * 6, 5 + 6 * player_idx)
    
    def ball_idx_by_player_idx(self, player_idx):
        """
        Return index of player's ball in self.state.
        """
        assert player_idx in (Player.WHITE, Player.BLACK)
        return 5 + 6 * player_idx
    
    def _all_pieces(self):
        """
        Return white and black pieces, and no ball positions.
        """
        return np.concatenate((self.player_pieces(Player.WHITE), self.player_pieces(Player.BLACK)))

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def valid_decoded_pos(self, cr: tuple):
        return all((
            cr[0] >= 0,
            cr[0] < self.N_COLS,
            cr[1] >= 0,
            cr[1] < self.N_ROWS,
        ))
    
    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
        assert self.valid_decoded_pos(cr)
        return cr[0] + cr[1] * self.N_COLS 

    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)
        """
        row = n // self.N_COLS
        col = n % self.N_COLS
        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a terminal board state, and return True or False.
        """
        _, white_ball_row = self.decode_single_pos(self.state[self.ball_idx_by_player_idx(Player.WHITE)])
        _, black_ball_row = self.decode_single_pos(self.state[self.ball_idx_by_player_idx(Player.BLACK)])
        return all((
            self.is_valid(),
            white_ball_row == (self.N_ROWS - 1) or black_ball_row == 0,
        ))

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constraints that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        """
        return all((
            # Pieces and balls inside board
            np.all(self.state >= 0),
            np.all(self.state < (self.N_COLS * self.N_ROWS)),
            # Pieces are on their own squares
            self._all_pieces().size == np.unique(self._all_pieces()).size,
            # Balls are on a piece of their color
            self.state[self.ball_idx_by_player_idx(Player.WHITE)] in self.player_pieces(Player.WHITE),
            self.state[self.ball_idx_by_player_idx(Player.BLACK)] in self.player_pieces(Player.BLACK),
        ))

class Rules:

    @staticmethod
    def single_piece_actions(board_state: BoardState, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        """
        assert board_state.is_valid()
        assert piece_idx >= 0
        assert piece_idx < board_state.state.size
        # Not a ball index
        assert piece_idx != 5
        assert piece_idx != 11
        col, row = board_state.decode_single_pos(board_state.state[piece_idx])
        valid_moves = []
        for d_col, d_row in KNIGHT_MOVE_POS_DELTA:
            # Try this update
            if not board_state.valid_decoded_pos((col + d_col, row + d_row)):
                continue
            new_idx = board_state.encode_single_pos((col + d_col, row + d_row))
            candidate_board = copy.deepcopy(board_state)
            candidate_board.update(piece_idx, new_idx)
            if candidate_board.is_valid():
                valid_moves.append(new_idx)
        return valid_moves

    @staticmethod
    def single_ball_actions(board_state: BoardState, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for player_idx in the board_state.

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        """
        # TODO - consider visualization and breaking out building connectivity graph
        # Build an adjacency matrix graph for all pieces of player_idx
        # - get decoded vector, then check diag/horiz/vert
        # - check if blocked by other player
        # Do BFS on graph and return all visited
        piece_connectivity = Graph(5)  # 5 states
        pieces = board_state.player_pieces(player_idx)
        # Check every piece against every other piece
        for i, piece in enumerate(pieces):
            source_col, source_row = board_state.decode_single_pos(piece)
            for j in range(i + 1, len(pieces)):
                dest_col, dest_row = board_state.decode_single_pos(pieces[j])
                min_col = min(source_col, dest_col)
                min_row = min(source_row, dest_row)
                max_col = max(source_col, dest_col)
                max_row = max(source_row, dest_row)
                connected = False
                # Horizontal
                if dest_row == source_row:
                    connected = True
                    # Collision-checking
                    for k in range(min_col + 1, max_col):
                        if (board_state.encode_single_pos((k, dest_row)) 
                            in board_state.player_pieces(1 - player_idx)):
                            connected = False
                            break
                # Vertical
                elif dest_col == source_col:
                    connected = True
                    # Collision-checking
                    for k in range(min_row + 1, max_row):
                        if (board_state.encode_single_pos((dest_col, k)) 
                            in board_state.player_pieces(1 - player_idx)):
                            connected = False
                            break
                # Diagonal
                elif abs(dest_col - source_col) == abs(dest_row - source_row):
                    connected = True
                    # Collision-checking
                    for cr in zip(range(min_col + 1, max_col), range(min_row + 1, max_row)):
                        if (board_state.encode_single_pos(cr) 
                            in board_state.player_pieces(1 - player_idx)):
                            connected = False
                            break
                if connected:
                    piece_connectivity.add_edge(i, j)
        # Do BFS on the graph
        visited = np.empty_like(pieces, dtype=bool)
        visited.fill(False)
        # TODO - make indexing the correct ball by player easier
        # Start with the ball state
        state_idx_with_ball = np.argwhere(pieces == board_state.state[5 + 6 * player_idx])[0]
        frontier = [state_idx_with_ball]
        while len(frontier) > 0:
            curr = frontier[0]
            frontier = frontier[1:]
            if len(piece_connectivity.neighbors(curr)) == 0:
                continue
            for i in piece_connectivity.neighbors(curr):
                # TODO - fix array squeeze / resizing
                if np.all(visited[i]):
                    continue
                visited[i] = True
                frontier.append(i)
        # Convert all visited to positions that may be visited by ball this turn
        # (ball may not stay at its current location)
        return set(pieces[visited].tolist()) - set((pieces[state_idx_with_ball]))

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that move like knights in chess, and
              relative index 5 is the player's ball piece.
        """
        actions = []
        # add block actions
        for i in range(5):
            actions.extend(
                (
                    (i, board_pos) 
                    for board_pos in Rules.single_piece_actions(
                        self.game_state, self.game_state.player_pieces_idx(player_idx)[i]
                    )
                )
            )
        # add ball actions for this player
        actions.extend(
            (
                (5, board_pos)
                for board_pos in Rules.single_ball_actions(self.game_state, player_idx)
            )
        )
        return actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        """
        if action in self.generate_valid_actions(player_idx):
            return True
        # TODO - how to determine cause for why each action is invalid? Is there a good reason to analyze each case?
        relative_idx, board_pos = action
        if relative_idx < 0 or relative_idx > 5:
            raise ValueError(f"Action relative index {relative_idx} is not one of the 5 possible pieces in [0,5].")
        if board_pos < 0 or board_pos >= self.game_state.N_COLS * self.game_state.N_ROWS:
            raise ValueError(
                f"Board position {board_pos} is not one of the {self.game_state.N_COLS * self.game_state.N_ROWS}"
                "possible pieces."
            )
        if relative_idx == 5:
            raise ValueError(f"Action to move ball to position {board_pos} is invalid.")
        raise ValueError(f"Action to move piece to {board_pos} is invalid.")
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
