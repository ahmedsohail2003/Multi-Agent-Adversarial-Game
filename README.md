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
* **Performance Metrics:** Track time and node/API call metrics for each move.

## Getting Started

### Prerequisites

* Python 3.x installed on your system.
* Required Python packages:
    * `google-generativeai` (for the Gemini algorithm)
* Google Gemini API Key (for the Gemini algorithm)

### Installation

1.  **Install the required packages:**

    ```bash
    pip install google-generativeai
    ```

2.  **Set your Google Gemini API key** as an environment variable (the key is never stored in the code):

    ```bash
    # macOS / Linux
    export GEMINI_API_KEY="your-gemini-api-key"

    # Windows (PowerShell)
    $env:GEMINI_API_KEY="your-gemini-api-key"
    ```

    The program reads the key from `GEMINI_API_KEY` at runtime (see `.env.example`). The `simple`, `minimax`, and `alpha-beta` algorithms run without a key.

### Running the Program

1.  Open a terminal and navigate to the directory.
2.  Run the script using:

    ```bash
    python main.py
    ```

### Usage

The program will present you with a menu:

1.  **Standardized Test:** Runs performance tests between selected algorithms.
2.  **Standardized Test with Selected Algorithm:** Runs a single game between two user selected algorithms.
3.  **Watch Game in Real Time:** Runs a game between two selected algorithms and displays the board and metrics.
4.  **End Program:** Exits the program.

### File Structure

* `main.py`: The main script that runs the program.
* `functions.py`: Contains functions for board manipulation and game logic.
* `algorithms.py`: Contains the AI algorithms (Minimax, Alpha-Beta, Gemini).
* `testing.py`: Contains functions for running performance tests.
