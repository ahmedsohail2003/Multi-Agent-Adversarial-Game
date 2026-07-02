# eval_gemini.py
"""
Small evaluation utility for the Gemini agentic player.

Runs the Gemini agent over several games versus another algorithm and reports:
  - First-try valid-move rate   (how often Gemini produced a legal move with no retry)
  - Average retries per move     (mean corrective re-prompts issued)
  - Win / Draw / Loss counts     (from Gemini's perspective)

Design notes
------------
* The Gemini API key is read inside ``algorithms`` from ``os.environ['GEMINI_API_KEY']``.
  This module performs NO live calls at import time, so it imports and
  ``py_compile``s cleanly without a key. Live calls only happen when ``main()``
  (or ``evaluate``) is actually invoked, and ``main()`` refuses to run without a
  key unless explicitly overridden.
* All metric plumbing reuses the counters already exposed by ``algorithms``:
  ``get_gemini_decisions``, ``get_gemini_calls``, ``get_gemini_retries`` and
  ``get_gemini_first_try_valid``.
"""

import os

import algorithms
import functions


def evaluate(opponent_algo=None, num_games=3, board_size=3, gemini_is_player1=True):
    """
    Play ``num_games`` of Gemini vs. ``opponent_algo`` and aggregate metrics.

    Args:
        opponent_algo (callable): A ``(board, player) -> (row, col)`` algorithm.
            Defaults to ``algorithms.alpha_beta_algo`` (a strong, fast opponent).
        num_games (int): Number of games to play.
        board_size (int): Board dimension (3 = standard).
        gemini_is_player1 (bool): If True Gemini is player 1 (moves first), else
            player 2.

    Returns:
        dict: Aggregated metrics.

    Note:
        This function issues live Gemini API calls and therefore requires a valid
        ``GEMINI_API_KEY``. It is the caller's responsibility to ensure one is set.
    """
    if opponent_algo is None:
        opponent_algo = algorithms.alpha_beta_algo

    gemini = algorithms.gemini_algo

    if gemini_is_player1:
        player1, player2 = gemini, opponent_algo
        gemini_player_num = 1
    else:
        player1, player2 = opponent_algo, gemini
        gemini_player_num = 2

    wins = draws = losses = 0
    total_moves = 0          # number of Gemini decisions (moves) made
    total_calls = 0          # actual generate_content requests, including retries
    total_retries = 0        # number of corrective re-prompts across those moves
    total_first_try_valid = 0  # decisions accepted on the first attempt

    for game_index in range(num_games):
        # run_game resets the counters internally before play begins, so the
        # post-game counter values reflect exactly this game's Gemini activity.
        _, winner, _ = functions.run_game(
            player1, player2, board_size=board_size, visualize=False
        )

        if winner == gemini_player_num:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        moves = algorithms.get_gemini_decisions()
        calls = algorithms.get_gemini_calls()
        total_moves += moves
        total_calls += calls
        total_retries += algorithms.get_gemini_retries()
        total_first_try_valid += algorithms.get_gemini_first_try_valid()

        print(
            f"  Game {game_index + 1}/{num_games}: "
            f"winner={winner!r}, gemini_moves={moves}, api_calls={calls}, "
            f"retries={algorithms.get_gemini_retries()}, "
            f"first_try_valid={algorithms.get_gemini_first_try_valid()}"
        )

    first_try_rate = (total_first_try_valid / total_moves) if total_moves else 0.0
    avg_retries = (total_retries / total_moves) if total_moves else 0.0

    return {
        "num_games": num_games,
        "board_size": board_size,
        "gemini_player_num": gemini_player_num,
        "opponent": getattr(opponent_algo, "__name__", str(opponent_algo)),
        "total_moves": total_moves,
        "total_api_calls": total_calls,
        "total_retries": total_retries,
        "first_try_valid": total_first_try_valid,
        "first_try_valid_rate": first_try_rate,
        "avg_retries_per_move": avg_retries,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


def print_report(metrics):
    """Pretty-print the metrics dict returned by ``evaluate``."""
    print("\n=== Gemini Agent Evaluation ===")
    print(f"Opponent:            {metrics['opponent']}")
    print(
        f"Games:               {metrics['num_games']} "
        f"(board {metrics['board_size']}x{metrics['board_size']}, "
        f"Gemini = Player {metrics['gemini_player_num']})"
    )
    print(f"Gemini moves:        {metrics['total_moves']}")
    print(f"Gemini API calls:    {metrics['total_api_calls']}")
    print(f"First-try valid:     {metrics['first_try_valid']} / {metrics['total_moves']}")
    print(f"First-try valid rate:{metrics['first_try_valid_rate'] * 100:6.1f}%")
    print(f"Avg retries / move:  {metrics['avg_retries_per_move']:.3f}")
    print(f"Total retries:       {metrics['total_retries']}")
    print(
        f"Win / Draw / Loss:   {metrics['wins']} / "
        f"{metrics['draws']} / {metrics['losses']}  (Gemini's perspective)"
    )
    print("===============================\n")


def main():
    """
    CLI entry point. Guards live calls behind the presence of a real API key so
    that importing / compiling this module never triggers network activity.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. Set it in your environment to run a live "
            "evaluation, e.g.\n"
            "  export GEMINI_API_KEY=your_key_here   (bash)\n"
            "  $env:GEMINI_API_KEY='your_key_here'    (PowerShell)\n"
            "Skipping live evaluation."
        )
        return

    metrics = evaluate(
        opponent_algo=algorithms.alpha_beta_algo,
        num_games=3,
        board_size=3,
        gemini_is_player1=True,
    )
    print_report(metrics)


if __name__ == "__main__":
    main()
