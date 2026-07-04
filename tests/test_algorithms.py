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


# --- Depth-limited search on boards larger than 3x3 -------------------------


def test_get_search_depth_exhaustive_up_to_3x3_limited_above(monkeypatch):
    monkeypatch.setattr(algorithms, "SEARCH_DEPTH_LIMIT", 4)
    assert algorithms.get_search_depth(3) is None  # exhaustive
    assert algorithms.get_search_depth(4) == 4
    assert algorithms.get_search_depth(5) == 4


def test_evaluate_heuristic_is_zero_sum_and_bounded():
    board = [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    score_p1 = algorithms.evaluate_heuristic(board, 1)
    score_p2 = algorithms.evaluate_heuristic(board, 2)
    # Symmetric by construction, and strictly inside (-1, 1) so a heuristic
    # estimate can never outrank a true win/loss utility of +/-1.
    assert score_p1 == -score_p2
    assert -1 < score_p1 < 1


def test_evaluate_heuristic_prefers_uncontested_progress():
    empty = [[0] * 4 for _ in range(4)]
    # Player 1 owns two marks on an uncontested row; player 2 has one mark.
    board = [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert algorithms.evaluate_heuristic(empty, 1) == 0
    assert algorithms.evaluate_heuristic(board, 1) > 0
    assert algorithms.evaluate_heuristic(board, 2) < 0


def test_minimax_4x4_first_move_is_tractable(monkeypatch):
    # Exhaustive minimax on an empty 4x4 board would expand on the order of
    # 10^13 nodes; the depth-4 cutoff keeps the whole first-move search small
    # enough to count. This test simply cannot pass without the depth limit.
    monkeypatch.setattr(algorithms, "SEARCH_DEPTH_LIMIT", 4)
    algorithms.reset_node_counters()
    board = functions.create_board(4)
    move = algorithms.minimax_algo(board, 1)
    assert move in functions.get_possible_moves(board)
    assert algorithms.get_minimax_nodes() < 60_000


def test_minimax_4x4_takes_immediate_winning_move(monkeypatch):
    monkeypatch.setattr(algorithms, "SEARCH_DEPTH_LIMIT", 4)
    # Player 1 completes the top row at (0, 3); a true utility of 1 must beat
    # every heuristic score at the cutoff.
    board = [
        [1, 1, 1, 0],
        [0, 2, 0, 2],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]
    assert algorithms.minimax_algo(board, 1) == (0, 3)


def test_alpha_beta_4x4_blocks_immediate_winning_threat(monkeypatch):
    monkeypatch.setattr(algorithms, "SEARCH_DEPTH_LIMIT", 4)
    # Player 2 threatens to complete the top row at (0, 3). Player 1 has no
    # win of its own, so every non-blocking move loses within the search
    # horizon; (0, 3) is the only move that avoids a -1 utility.
    board = [
        [2, 2, 2, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    assert algorithms.alpha_beta_algo(board, 1) == (0, 3)


def test_minimax_4x4_matches_alpha_beta_on_forced_block(monkeypatch):
    monkeypatch.setattr(algorithms, "SEARCH_DEPTH_LIMIT", 4)
    board = [
        [2, 2, 2, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    assert algorithms.minimax_algo(board, 1) == (0, 3)
