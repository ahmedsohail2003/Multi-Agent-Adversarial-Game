# tests/test_functions.py
import pytest

import functions


def test_create_board_size_and_all_zeros():
    board = functions.create_board(3)
    assert len(board) == 3
    assert all(len(row) == 3 for row in board)
    assert all(cell == 0 for row in board for cell in row)

    board4 = functions.create_board(4)
    assert len(board4) == 4
    assert all(len(row) == 4 for row in board4)
    assert all(cell == 0 for row in board4 for cell in row)


def test_is_valid_move_valid_cell():
    board = functions.create_board(3)
    assert functions.is_valid_move(board, 1, 1) is True


def test_is_valid_move_out_of_bounds():
    board = functions.create_board(3)
    assert functions.is_valid_move(board, -1, 0) is False
    assert functions.is_valid_move(board, 0, 3) is False
    assert functions.is_valid_move(board, 3, 3) is False


def test_is_valid_move_occupied_cell():
    board = functions.create_board(3)
    board[0][0] = 1
    assert functions.is_valid_move(board, 0, 0) is False


def test_get_possible_moves_count_and_excludes_occupied():
    board = functions.create_board(3)
    assert len(functions.get_possible_moves(board)) == 9

    board[0][0] = 1
    board[2][2] = 2
    moves = functions.get_possible_moves(board)
    assert len(moves) == 7
    assert (0, 0) not in moves
    assert (2, 2) not in moves
    assert (1, 1) in moves


def test_make_move_valid_returns_true_and_mutates():
    board = functions.create_board(3)
    assert functions.make_move(board, 1, 2, 1) is True
    assert board[1][2] == 1


def test_make_move_invalid_returns_false_and_no_mutation():
    board = functions.create_board(3)
    board[0][0] = 2
    assert functions.make_move(board, 0, 0, 1) is False
    assert board[0][0] == 2

    # Out of bounds: no cell should change.
    assert functions.make_move(board, 5, 5, 1) is False
    assert all(
        cell == (2 if (r, c) == (0, 0) else 0)
        for r, row in enumerate(board)
        for c, cell in enumerate(row)
    )


def test_check_game_status_row_win():
    board = [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert functions.check_game_status(board) == (True, 1)


def test_check_game_status_column_win():
    board = [
        [2, 0, 0],
        [2, 0, 0],
        [2, 0, 0],
    ]
    assert functions.check_game_status(board) == (True, 2)


def test_check_game_status_main_diagonal_win():
    board = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    assert functions.check_game_status(board) == (True, 1)


def test_check_game_status_anti_diagonal_win():
    board = [
        [0, 0, 2],
        [0, 2, 0],
        [2, 0, 0],
    ]
    assert functions.check_game_status(board) == (True, 2)


def test_check_game_status_full_board_draw():
    board = [
        [1, 2, 1],
        [1, 2, 2],
        [2, 1, 1],
    ]
    assert functions.check_game_status(board) == (True, None)


def test_check_game_status_in_progress():
    board = [
        [1, 2, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert functions.check_game_status(board) == (False, None)


def test_check_input_raises_on_invalid():
    with pytest.raises(ValueError):
        functions.check_input(5, [1, 2, 3, 4])


def test_check_input_valid_returns_none():
    assert functions.check_input(2, [1, 2, 3, 4]) is None
