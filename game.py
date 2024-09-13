import copy
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
        self.IDX_BALL_WHITE = 5
        self.IDX_BALL_BLACK = 11
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def _white_pieces(self):
        """
        Return copy of white pieces in state
        """
        return self.state.copy()[:5]
    
    def _black_pieces(self):
        """
        Return copy of black pieces in state
        """
        return self.state.copy()[6:11]
    
    def _all_pieces(self):
        """
        Return white and black pieces, and no ball positions.
        """
        return np.concatenate((self._white_pieces(), self._black_pieces()))

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

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive
        """
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
        _, white_ball_row = self.decode_single_pos(self.state[self.IDX_BALL_WHITE])
        _, black_ball_row = self.decode_single_pos(self.state[self.IDX_BALL_BLACK])
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
            self.state[self.IDX_BALL_WHITE] in self._white_pieces(),
            self.state[self.IDX_BALL_BLACK] in self._black_pieces(),
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
        col, row = board_state.decode_single_pos(board_state.state[piece_idx])
        valid_moves = []
        for d_col, d_row in KNIGHT_MOVE_POS_DELTA:
            # Try this update
            new_idx = board_state.encode_single_pos(col + d_col, row + d_row)
            candidate_board = copy.deepcopy(board_state)
            candidate_board.update(piece_idx, new_idx)
            if candidate_board.is_valid():
                valid_moves.append(new_idx)
        return valid_moves

    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        TODO: You need to implement this.
        """
        raise NotImplementedError("TODO: Implement this function")

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
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        raise NotImplementedError("TODO: Implement this function")

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
        
        TODO: You need to implement this.
        """
        if False:
            raise ValueError("For each case that an action is not valid, specify the reason that the action is not valid in this ValueError.")
        if True:
            return True
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
