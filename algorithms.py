#algorithims.py
import functions
from google import genai
import re
import os

# Add global counters for nodes
minimax_nodes = 0
alpha_beta_nodes = 0
_gemini_calls = 0
_gemini_decisions = 0
_gemini_retries = 0            # total corrective re-prompts issued across all calls
_gemini_first_try_valid = 0    # moves accepted on the very first attempt (no retry needed)

def reset_node_counters():
    global minimax_nodes, alpha_beta_nodes, _gemini_calls, _gemini_decisions
    global _gemini_retries, _gemini_first_try_valid
    minimax_nodes = 0
    alpha_beta_nodes = 0
    _gemini_calls  = 0
    _gemini_decisions = 0
    _gemini_retries = 0
    _gemini_first_try_valid = 0

def get_minimax_nodes():
    return minimax_nodes

def get_alpha_beta_nodes():
    return alpha_beta_nodes

def get_gemini_calls():
    return _gemini_calls

def get_gemini_decisions():
    return _gemini_decisions

def get_gemini_retries():
    return _gemini_retries

def get_gemini_first_try_valid():
    return _gemini_first_try_valid

def simple_algo(board_state, player):
    possible_moves = functions.get_possible_moves(board_state)
    return possible_moves[0]


def get_utility(board, player):
    """
    Calculates the utility of a given board state for a player.

    Args:
        board (list): The list of lists representing the board.
        player (int): The player to calculate the utility for (1 or 2).

    Returns:
        int: The utility of the board state (1 for win, -1 for loss, 0 for draw).
    """
    game_finished, winner = functions.check_game_status(board)
    if game_finished:
        if winner == player:
            return 1
        elif winner is None:
            return 0
        else:
            return -1
    return None  # Game is not over


def minimax(board, player, maximizing_player):
    """
    Minimax algorithm to find the best move.

    Args:
        board (list): The current board state.
        player (int): The current player.
        maximizing_player (int): The player for whom we are maximizing.

    Returns:
        int: The best score for the maximizing player.
    """
    global minimax_nodes
    minimax_nodes += 1

    utility = get_utility(board, maximizing_player)
    if utility is not None:
        return utility

    possible_moves = functions.get_possible_moves(board)
    if player == maximizing_player:
        best_score = -float('inf')
        for row, col in possible_moves:
            new_board = [row[:] for row in board]
            functions.make_move(new_board, row, col, player)
            score = minimax(new_board, 2 if player == 1 else 1, maximizing_player)
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for row, col in possible_moves:
            new_board = [row[:] for row in board]  # Create a copy
            functions.make_move(new_board, row, col, player)
            score = minimax(new_board, 2 if player == 1 else 1, maximizing_player)
            best_score = min(best_score, score)
        return best_score


def minimax_algo(board, player):
    """
    Algorithm that uses minimax to choose the best move.

    Args:
        board (list): The current board state.
        player(int): the current player

    Returns:
        tuple: The best move (row, col).
    """

    possible_moves = functions.get_possible_moves(board)
    best_move = None
    best_score = -float('inf')
    for row, col in possible_moves:
        new_board = [row[:] for row in board]
        functions.make_move(new_board, row, col, player)
        score = minimax(new_board, 2 if player == 1 else 1, player)
        if score > best_score:
            best_score = score
            best_move = (row, col)
    return best_move


def alpha_beta_minimax(board, player, maximizing_player, alpha, beta):
    """
    Alpha-Beta Pruning Minimax algorithm.

    Args:
        board (list): The current board state.
        player (int): The current player.
        maximizing_player (int): The player for whom we are maximizing.
        alpha (int): The best value that the maximizing player can guarantee.
        beta (int): The best value that the minimizing player can guarantee.

    Returns:
        int: The best score for the maximizing player.
    """
    global alpha_beta_nodes
    alpha_beta_nodes += 1

    utility = get_utility(board, maximizing_player)
    if utility is not None:
        return utility

    possible_moves = functions.get_possible_moves(board)
    if player == maximizing_player:
        best_score = -float('inf')
        for row, col in possible_moves:
            new_board = [row[:] for row in board]
            functions.make_move(new_board, row, col, player)
            score = alpha_beta_minimax(new_board, 2 if player == 1 else 1, maximizing_player, alpha, beta)
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Beta cutoff
        return best_score
    else:
        best_score = float('inf')
        for row, col in possible_moves:
            new_board = [row[:] for row in board]
            functions.make_move(new_board, row, col, player)
            score = alpha_beta_minimax(new_board, 2 if player == 1 else 1, maximizing_player, alpha, beta)
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return best_score


def alpha_beta_algo(board, player):
    """
    Algorithm that uses Alpha-Beta Pruning to choose the best move.

    Args:
        board (list): The current board state.
        player (int): the current player

    Returns:
        tuple: The best move (row, col).
    """

    possible_moves = functions.get_possible_moves(board)
    best_move = None
    best_score = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    for row, col in possible_moves:
        new_board = [row[:] for row in board]
        functions.make_move(new_board, row, col, player)
        score = alpha_beta_minimax(new_board, 2 if player == 1 else 1, player, alpha, beta)
        if score > best_score:
            best_score = score
            best_move = (row, col)
    return best_move


def _parse_gemini_response(text):
    """
    Parses Gemini's raw text into a (row, col) tuple of ints.

    Raises:
        ValueError: if fewer than two digits can be recovered from the text.
    """
    clean_text = re.sub(r'[^0-9,]', '', text)
    matches = re.findall(r'\d', clean_text)

    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    raise ValueError("Invalid response format")


# Maximum number of corrective re-prompts before falling back to a safe move.
GEMINI_MAX_RETRIES = 3
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def gemini_algo(board, player):
    """
    Gemini-powered AI player that selects a move in a Tic-Tac-Toe game.

    This is an *agentic* player: when Gemini returns an unparseable, out-of-bounds
    or illegal (occupied) move, the model is re-prompted up to GEMINI_MAX_RETRIES
    times. Each retry appends a short corrective message explaining exactly why the
    previous answer was rejected, so the model can self-correct. Only after the
    retries are exhausted does it fall back to a deterministic safe move.

    Args:
        board (list): The current board state.
        player (int): The current player.

    Returns:
        tuple: The chosen move (row, col).
    """
    global _gemini_calls, _gemini_decisions, _gemini_retries, _gemini_first_try_valid
    _gemini_decisions += 1

    possible_moves = functions.get_possible_moves(board)
    best_move = possible_moves[0]  # deterministic safe fallback
    board_size = len(board)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Using fallback move.")
        return best_move

    symbols = {0: " ", 1: "X", 2: "O"}
    board_desc = "\n".join(
        "|".join(symbols[cell] for cell in row) + "\n" + "-" * (board_size * 2 - 1)
        for row in board
    )

    base_prompt = f"""You are Player {symbols[player]} in a {board_size}x{board_size} Tic-Tac-Toe game.
Current Board (0-based indices):
{board_desc}
Valid moves: {possible_moves}
Return ONLY the zero-based row and column as two numbers between 0-{board_size-1},
formatted exactly like: 'row,column' with no other text.
Examples of valid responses: '0,1' or '{board_size-1},{board_size-1}'"""

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        # Model could not even be constructed (e.g. no/invalid key) -> safe fallback.
        print(f"Gemini error: {str(e)[:50]}... Using fallback move.")
        return best_move

    feedback = ""  # corrective text appended on each retry
    # attempt 0 is the first try; attempts 1..GEMINI_MAX_RETRIES are corrective retries.
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        if attempt > 0:
            _gemini_retries += 1  # count this as a corrective re-prompt

        prompt = base_prompt + feedback
        try:
            _gemini_calls += 1
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            row, col = _parse_gemini_response(response.text)

            if not (0 <= row < board_size and 0 <= col < board_size):
                raise ValueError(
                    f"REJECTED: ({row},{col}) is OUT OF BOUNDS. "
                    f"Both numbers must be between 0 and {board_size - 1}."
                )

            if not functions.is_valid_move(board, row, col):
                raise ValueError(
                    f"REJECTED: cell ({row},{col}) is already OCCUPIED. "
                    f"Choose one of the still-empty Valid moves listed above."
                )

            # Accepted. Record whether this was a clean first-try success.
            if attempt == 0:
                _gemini_first_try_valid += 1
            return (row, col)

        except Exception as e:
            reason = str(e)
            # Classify unparseable responses for a clearer corrective message.
            if "Invalid response format" in reason:
                reason = (
                    "REJECTED: response was UNPARSEABLE. Reply with exactly two "
                    "digits in 'row,column' form and nothing else."
                )
            # If retries remain, append the reason and let the model try again.
            if attempt < GEMINI_MAX_RETRIES:
                feedback = (
                    f"\n\nYour previous answer was rejected. {reason} "
                    f"Try again and return ONLY 'row,column'."
                )
                continue
            # Retries exhausted -> deterministic safe fallback.
            print(
                f"Gemini error after {GEMINI_MAX_RETRIES} retries: "
                f"{reason[:60]}... Using fallback move."
            )
            return best_move

    # Defensive: loop always returns above, but keep a guaranteed return.
    return best_move
