# Tic-Tac-Toe AI Showdown

This program allows you to run and visualize Tic-Tac-Toe games between different AI algorithms. You can also run performance tests to compare the algorithms.

> Academic team project (4 members).

## Features

* **Standardized Tests:** Run multiple games to compare the performance of different AI algorithms.
* **Custom Game Play:** Play games with selected algorithms and visualize the gameplay in real-time.
* **Algorithm Selection:** Choose from various AI algorithms, including:
    * `simple_algo`: A basic algorithm that makes the first available move.
    * `minimax_algo`: An AI using the Minimax algorithm.
    * `alpha_beta_algo`: An AI using the Alpha-Beta Pruning Minimax algorithm.
    * `gemini_algo`: An AI powered by Google's Gemini API.
* **Board Size Configuration:** Play on boards of different sizes (e.g., 3x3, 4x4).
* **Performance Metrics:** Track time, search-node counts, Gemini decisions,
  actual API requests, first-try validity, and corrective retries.

### Search depth on larger boards

On the standard 3x3 board, `minimax_algo` and `alpha_beta_algo` search the game
tree exhaustively, so their play is optimal. Exhaustive search is intractable on
larger boards (a 4x4 game has on the order of 10^13 move sequences), so on any
board larger than 3x3 both searchers cut off after a fixed number of plies —
4 by default, overridable with the `SEARCH_DEPTH_LIMIT` environment variable —
and score the cutoff position with a line-counting heuristic
(`evaluate_heuristic` in `algorithms.py`). Heuristic scores are normalized into
(-1, 1) so they can never outrank a true win or loss found within the search
horizon. Play on larger boards is therefore strong but not provably optimal,
and the program prints a note saying so when you pick a board larger than 3x3.

## Getting Started

### Prerequisites

* Python 3.x installed on your system.
* Required Python packages:
    * `google-genai` (the current Google GenAI SDK)
* Google Gemini API Key (for the Gemini algorithm)

### Installation

1.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set your Google Gemini API key** as an environment variable (the key is never stored in the code):

    ```bash
    # macOS / Linux
    export GEMINI_API_KEY="your-gemini-api-key"

    # Windows (PowerShell)
    $env:GEMINI_API_KEY="your-gemini-api-key"
    ```

    The program reads the key from `GEMINI_API_KEY` at runtime (see `.env.example`).
    You can optionally override the default model (`gemini-2.5-flash`) with the
    `GEMINI_MODEL` environment variable. The `simple`, `minimax`, and `alpha-beta`
    algorithms run without a key.

### Running the Program

1.  Open a terminal and navigate to the directory.
2.  Run the script using:

    ```bash
    python main.py
    ```

### Running the Tests

The Gemini agent is tested against a mocked client, so the suite needs no API
key or network access:

```bash
pip install -r requirements-dev.txt
pytest
```

The same suite runs in GitHub Actions (see `.github/workflows/ci.yml`).

### Usage

The program will present you with a menu:

1.  **Standardized Test:** Runs a batch of games over fixed algorithm pairings
    (Minimax vs. Alpha-Beta, Minimax vs. Gemini, Alpha-Beta vs. Gemini) with a
    user-chosen game count and board size, then prints win/draw counts and
    average time and operations per move.
2.  **Standardized Test with Selected Algorithm:** Runs a single game between two user-selected algorithms.
3.  **Watch Game in Real Time:** Runs a game between two selected algorithms and displays the board and metrics.
4.  **End Program:** Exits the program.

### File Structure

* `main.py`: The main script that runs the program.
* `functions.py`: Contains functions for board manipulation and game logic.
* `algorithms.py`: Contains the AI algorithms (Minimax, Alpha-Beta, Gemini) and the depth-limit/heuristic logic for larger boards.
* `testing.py`: Contains functions for running performance tests.
* `eval_gemini.py`: Standalone evaluation script for the Gemini agent (win/draw/loss, first-try valid-move rate, average retries per move). Makes live API calls, so it requires `GEMINI_API_KEY`.
* `tests/`: Pytest suite covering the game logic, both searchers (including the depth-limited behavior), and the Gemini agent's retry loop.
* `.env.example`: Template for the environment variables the program reads at runtime.
