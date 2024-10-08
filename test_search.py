import numpy as np
import queue
import pytest
from game import BoardState, GameSimulator, Rules
from search import GameStateProblem

class TestSearch:

    def test_game_state_goal_state(self):
        b1 = BoardState()
        gsp = GameStateProblem(b1, b1, 0)

        sln = gsp.search_alg_fnc()
        ref = [(tuple((tuple(b1.state), 0)), None)]

        assert sln == ref

    ## NOTE: If you'd like to test multiple variants of your algorithms, enter their keys below
    ## in the parametrize function. Your set_search_alg should then set the correct method to
    ## use.
    # Options:
    # - bfs
    # - a*_count_pieces_heuristic (Better)
    # - a*_manhattan_heuristic (Best)
    @pytest.mark.parametrize("alg", ["a*_count_pieces_heuristic"])
    def test_game_state_problem(self, alg):
        """
        Tests search based planning
        """
        ## Single Step no ball
        b1 = BoardState()
        b2 = BoardState()
        b2.update(0, 14)
        sln = _get_solution(b1, b2, alg)
        ref = [(tuple((tuple(b1.state), 0)), (0, 14)), (tuple((tuple(b2.state), 1)), None)]
        assert sln == ref

        ## Single Step with ball move
        b2 = BoardState()
        b2.update(5, 5)
        sln = _get_solution(b1, b2, alg)
        ref = [(tuple((tuple(b1.state), 0)), (5, 5)), (tuple((tuple(b2.state), 1)), None)]
        assert sln == ref

        ## Two Step no ball
        b2 = BoardState()
        b2.update(0, 23)
        sln = _get_solution(b1, b2, alg)
        ## Two Step:
        ## (0, 14) or (0, 10) -> (any) -> (0, 23) -> (undo any) -> (None, goal state)
        #print(gsp.goal_state_set)
        #print(sln)
        assert len(sln) == 5 ## Player 1 needs to move once, then move the piece back
        assert sln[0] == (tuple((tuple(b1.state), 0)), (0, 14)) or sln[0] == (tuple((tuple(b1.state), 0)), (0, 10))
        assert sln[1][0][1] == 1
        assert sln[2][1] == (0, 23)
        assert sln[4] == (tuple((tuple(b2.state), 0)), None)
        
        ## Force ball to move and move back to same piece
        # NOTE: With unaltered BFS, time taken very long because we explore all piece moves first before ball
        b2 = BoardState()
        b2.update(2, 12)
        b2.update(5, 12)
        sln = _get_solution(b1, b2, alg)
        ## 
        ## (5, 1/2/4/5) -> (any) -> (2, 12) -> (undo any) -> (5, 12) -> (None, goal state)
        assert len(sln) == 6 ## Player 1 needs to move once, then move the piece back
        assert sln[0][0] == tuple((tuple(b1.state), 0))
        assert sln[0][1][0] == 5
        assert sln[0][1][1] in (1, 2, 4, 5)
        assert sln[1][0][1] == 1
        assert sln[2][1] == (2, 12)
        assert sln[3][0][1] == 1
        assert sln[4][1] == (5, 12)
        assert sln[5] == (tuple((tuple(b2.state), 1)), None)
        
        # TODO - moves involving both player
        # TODO - set more complex state to start w/ obstacles
        # TODO - several move case 
        # TODO - check that intermediate states are correct
        # TODO - add piece-heavy and ball-heavy cases

    def test_initial_state(self):
        """
        Confirms the initial state of the game board
        """
        board = BoardState()
        assert board.decode_state == board.make_state()

        ref_state = [(1,0),(2,0),(3,0),(4,0),(5,0),(3,0),(1,7),(2,7),(3,7),(4,7),(5,7),(3,7)]

        assert board.decode_state == ref_state

    def test_generate_actions(self):
        sim = GameSimulator(None)
        generated_actions = sim.generate_valid_actions(0)
        assert (0,6) not in generated_actions
        assert (4,0) not in generated_actions

    ## NOTE: You are highly encouraged to add failing test cases here
    ## in order to test your validate_action implementation. To add an
    ## invalid action, fill in the action tuple, the player_idx, the
    ## validity boolean (would be False for invalid actions), and a
    ## unique portion of the descriptive error message that your raised
    ## ValueError should return. For example, if you raised:
    ## ValueError("Cannot divide by zero"), then you would pass some substring
    ## of that description for val_msg.
    @pytest.mark.parametrize("action,player,is_valid,val_msg", [
        ((0,14), 0, True, ""),
        ((0,16), 0, True, ""),
        ((0,10), 0, True, ""),
        ((5,1), 0, True, ""),
        ((5,2), 0, True, ""),
        ((5,4), 0, True, ""),
        ((5,5), 0, True, ""),
        ((-1,5), 0, False, "not one of the 5 possible pieces"),
        ((6,5), 0, False, "not one of the 5 possible pieces"),
        ((5,-1), 0, False, "Board position -1 is not one of the"),
        ((5,56), 0, False, "Board position 56 is not one of the"),
        ((5,8), 0, False, "Action to move ball to position 8 is invalid."),
        ((0,8), 0, False, "Action to move piece to position 8 is invalid."),
    ])
    def test_validate_action(self, action, player, is_valid, val_msg):
        sim = GameSimulator(None)
        if is_valid:
            assert sim.validate_action(action, player) == is_valid
        else:
            with pytest.raises(ValueError) as exinfo:
                result = sim.validate_action(action, player)
            assert val_msg in str(exinfo.value)
        

    @pytest.mark.parametrize("state,is_term", [
        ([1,2,3,4,5,3,50,51,52,53,54,52], False), ## Initial State
        ([1,2,3,4,5,55,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,0], False), ## Invalid State
        ([1,2,3,4,49,49,50,51,52,53,54,54], True), ## Player 1 wins
        ([1,2,3,4,5,5,50,51,52,53,6,6], True), ## Player 2 wins
        ([1,2,3,4,5,5,50,4,52,53,6,6], False), ## Invalid State
    ])
    def test_termination_state(self, state, is_term):
        board = BoardState()
        board.state = np.array(state)
        board.decode_state = board.make_state()

        assert board.is_termination_state() == is_term

    def test_encoded_decode(self):
        board = BoardState()
        assert board.decode_state  == [board.decode_single_pos(x) for x in board.state]

        enc = np.array([board.encode_single_pos(x) for x in board.decode_state])
        assert np.all(enc == board.state)

    def test_is_valid(self):
        board = BoardState()
        assert board.is_valid()

        ## Out of bounds test
        board.update(0,-1)
        assert not board.is_valid()
        
        board.update(0,0)
        assert board.is_valid()
        
        ## Out of bounds test
        board.update(0,-1)
        board.update(6,56)
        assert not board.is_valid()
        
        ## Overlap test
        board.update(0,0)
        board.update(6,0)
        assert not board.is_valid()

        ## Ball is on index 0
        board.update(5,1)
        board.update(0,1)
        board.update(6,50)
        assert board.is_valid()

        ## Player is not holding the ball
        board.update(5,0)
        assert not board.is_valid()
        
        board.update(5,10)
        assert not board.is_valid()

    @pytest.mark.parametrize("state,reachable,player", [
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(0,1),(2,1),(1,2),(1,0)]),
            0
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(3,3)
            ],
            set([(2,2)]),
            1
        ),
        (
            [
                (1,1),(0,1),(2,1),(1,2),(1,0),(1,1),
                (0,0),(2,0),(0,2),(2,2),(3,3),(0,0)
            ],
            set(),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(2,3)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(2,3),(2,3)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(1,2)
            ],
            set([(0,1),(2,1),(3,1),(3,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(0,1)
            ],
            set([(2,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(2,1)
            ],
            set([(0,1),(3,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,1)
            ],
            set([(0,1),(2,1),(3,2),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,1),(2,1),(3,1),(1,2)]),
            1
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(2,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,0),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(0,2),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(2,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(2,2),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(0,3)]),
            0
        ),
        (
            [
                (0,0),(2,0),(0,2),(2,2),(0,3),(0,3),
                (0,1),(2,1),(3,1),(3,2),(1,2),(3,2)
            ],
            set([(0,0),(2,0),(0,2),(2,2)]),
            0
        ),
    ]) 
    def test_ball_reachability(self, state, reachable, player):
        board = BoardState()
        board.state = np.array(list(board.encode_single_pos(cr) for cr in state))
        board.decode_state = board.make_state()
        predicted_reachable_encoded = Rules.single_ball_actions(board, player)
        encoded_reachable = set(board.encode_single_pos(cr) for cr in reachable)
        assert predicted_reachable_encoded == encoded_reachable


class TestRules:
    
    def test_single_piece_actions(self):
        # Default state
        board_state = BoardState()
        assert np.all(board_state.state == np.array([1,2,3,4,5,3,50,51,52,53,54,52]))
        # Knight's moves should be available
        assert set(Rules.single_piece_actions(board_state, 0)) == set((10, 14, 16))
        assert set(Rules.single_piece_actions(board_state, 1)) == set((7, 15, 17, 11))
        assert set(Rules.single_piece_actions(board_state, 3)) == set((9, 17, 19, 13))
        assert set(Rules.single_piece_actions(board_state, 9)) == set((44, 38, 40, 48))
        # Piece with ball cannot move
        assert set(Rules.single_piece_actions(board_state, 2)) == set()
        assert set(Rules.single_piece_actions(board_state, 8)) == set()

        # Construct example with same team and other team blocking possible moves
        board_state.state = np.array([24,11,19,4,5,11,50,51,37,33,54,54])
        assert set(Rules.single_piece_actions(board_state, 0)) == set((39, 29, 15, 9))

    def test_single_ball_actions(self):
        """Warning: Not updating decoded state"""
        board_state = BoardState()
        # Default state
        assert np.all(board_state.state == np.array([1,2,3,4,5,3,50,51,52,53,54,52]))
        # May pass to all non-ball pieces
        assert set(Rules.single_ball_actions(board_state, 0)) == set((1, 2, 4, 5))
        assert set(Rules.single_ball_actions(board_state, 1)) == set((50, 51, 53, 54))

        # Construct example with same team and other team blocking possible moves
        # TODO - offer viz option?
        board_state.state = np.array([49,52,39,37,23,23,50,53,47,44,30,50])
        assert set(Rules.single_ball_actions(board_state, 0)) == set((49, 37, 39))
        assert set(Rules.single_ball_actions(board_state, 1)) == set((44, 47, 53))
        
        # Place ball on an island for both teams
        board_state.state[board_state.ball_idx_by_player_idx(0)] = 52
        board_state.state[board_state.ball_idx_by_player_idx(1)] = 30
        assert set(Rules.single_ball_actions(board_state, 0)) == set(())
        assert set(Rules.single_ball_actions(board_state, 1)) == set(())

        # TODO - probably need another edge case or two


def _get_solution(start_board, goal_board, alg):
    gsp = GameStateProblem(start_board, goal_board, 0)
    gsp.set_search_alg(alg)
    return gsp.search_alg_fnc()
