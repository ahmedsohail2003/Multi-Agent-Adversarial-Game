# tests/test_algorithms.py
import functions

# Importing algorithms transitively imports google.genai (a project dependency);
# the tests below exercise only the pure search logic (minimax / alpha-beta), not the Gemini API.
import algorithms


def test_get_utility_winning_board_returns_1():
    # Player 1 has completed the top row.
    board = [
        [1, 1, 1],
        [2, 2, 0],
        [0, 0, 0],
    ]
    assert algorithms.get_utility(board, 1) == 1


def test_get_utility_losing_board_returns_minus_1():
    # Player 2 has completed the top row, so player 1 has lost.
    board = [
        [2, 2, 2],
        [1, 1, 0],
        [0, 0, 0],
    ]
    assert algorithms.get_utility(board, 1) == -1


def test_get_utility_draw_board_returns_0():
    board = [
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
    ]
    assert algorithms.get_utility(board, 1) == 0


def test_get_utility_in_progress_returns_none():
    board = [
        [1, 2, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert algorithms.get_utility(board, 1) is None


def test_simple_algo_returns_first_available_move():
    board = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected = functions.get_possible_moves(board)[0]
    assert expected == (0, 1)
    assert algorithms.simple_algo(board, 2) == expected


def test_minimax_takes_immediate_winning_move():
    # Player 1 has two in the top row; (0, 2) completes the win.
    board = [
        [1, 1, 0],
        [2, 2, 0],
        [0, 0, 0],
    ]
    assert algorithms.minimax_algo(board, 1) == (0, 2)


def test_minimax_blocks_opponent_winning_threat():
    # Player 2 threatens to complete the top row at (0, 2). Player 1 has no
    # immediate win, so the only non-losing move is to block at (0, 2).
    board = [
        [2, 2, 0],
        [1, 0, 0],
        [0, 0, 0],
    ]
    assert algorithms.minimax_algo(board, 1) == (0, 2)


def test_alpha_beta_matches_minimax_on_midgame_board():
    # A representative mid-game position; alpha-beta pruning must not change
    # the chosen move relative to plain minimax.
    board = [
        [1, 2, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]
    assert algorithms.alpha_beta_algo(board, 1) == algorithms.minimax_algo(board, 1)
