from types import SimpleNamespace

import algorithms


def _client_with_responses(monkeypatch, responses):
    calls = []
    remaining = iter(responses)

    def generate_content(**kwargs):
        calls.append(kwargs)
        value = next(remaining)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value)

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    monkeypatch.setattr(algorithms.genai, "Client", lambda **_: client)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return calls


def test_gemini_accepts_valid_first_response(monkeypatch):
    calls = _client_with_responses(monkeypatch, ["1,2"])
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    algorithms.reset_node_counters()

    move = algorithms.gemini_algo(board, 1)

    assert move == (1, 2)
    assert algorithms.get_gemini_decisions() == 1
    assert algorithms.get_gemini_calls() == 1
    assert algorithms.get_gemini_retries() == 0
    assert algorithms.get_gemini_first_try_valid() == 1
    assert calls[0]["model"] == algorithms.GEMINI_MODEL


def test_gemini_reprompts_with_feedback_after_occupied_move(monkeypatch):
    calls = _client_with_responses(monkeypatch, ["0,0", "2,1"])
    board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    algorithms.reset_node_counters()

    move = algorithms.gemini_algo(board, 2)

    assert move == (2, 1)
    assert algorithms.get_gemini_decisions() == 1
    assert algorithms.get_gemini_calls() == 2
    assert algorithms.get_gemini_retries() == 1
    assert algorithms.get_gemini_first_try_valid() == 0
    assert "already OCCUPIED" in calls[1]["contents"]


def test_gemini_falls_back_after_retries(monkeypatch):
    calls = _client_with_responses(
        monkeypatch,
        ["not a move"] * (algorithms.GEMINI_MAX_RETRIES + 1),
    )
    board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    algorithms.reset_node_counters()

    move = algorithms.gemini_algo(board, 2)

    assert move == (0, 1)
    assert algorithms.get_gemini_decisions() == 1
    assert algorithms.get_gemini_calls() == algorithms.GEMINI_MAX_RETRIES + 1
    assert algorithms.get_gemini_retries() == algorithms.GEMINI_MAX_RETRIES
    assert len(calls) == algorithms.GEMINI_MAX_RETRIES + 1


def test_gemini_without_key_uses_fallback_without_api_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        algorithms.genai,
        "Client",
        lambda **_: (_ for _ in ()).throw(AssertionError("Client should not be created")),
    )
    board = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    algorithms.reset_node_counters()

    move = algorithms.gemini_algo(board, 2)

    assert move == (0, 1)
    assert algorithms.get_gemini_decisions() == 1
    assert algorithms.get_gemini_calls() == 0
