"""Tests for coding adapter + sandbox verification."""
from __future__ import annotations

from adapters.coding import load_coding_questions, verify_solution
from adapters import get_adapter
from harness.sandbox import extract_python_code, execution_accuracy, run_tests


class TestSandbox:
    def test_extract_python_fence(self):
        text = "Sure.\n```python\ndef add(a, b):\n    return a + b\n```\n"
        assert "def add" in extract_python_code(text)

    def test_gold_passes(self):
        code = "def add(a, b):\n    return a + b\n"
        acc, valid, err = execution_accuracy(
            code, "add", [{"args": [1, 2], "expected": 3}]
        )
        assert acc == 1.0 and valid and err == ""

    def test_wrong_answer(self):
        code = "def add(a, b):\n    return a - b\n"
        acc, valid, _ = execution_accuracy(
            code, "add", [{"args": [1, 2], "expected": 3}]
        )
        assert acc == 0.0 and valid

    def test_syntax_invalid(self):
        code = "def add(a, b)\n    return a + b\n"
        acc, valid, _ = execution_accuracy(
            code, "add", [{"args": [1, 2], "expected": 3}]
        )
        assert acc == 0.0 and valid is False

    def test_timeout(self):
        code = "def loop():\n    while True:\n        pass\n"
        result = run_tests(code, "loop", [{"args": [], "expected": None}], timeout_s=0.3)
        assert result["timed_out"] is True
        assert result["ok"] is False


class TestCodingFixture:
    def test_load_enough_problems(self):
        qs = load_coding_questions()
        assert len(qs) >= 60
        diffs = {q["difficulty"] for q in qs}
        assert "easy" in diffs and "hard" in diffs
        topics = {q["db_id"] for q in qs}
        assert len(topics) >= 3

    def test_all_gold_solutions_pass(self):
        from adapters.coding import _index

        failed = []
        for p in _index().values():
            acc, valid, err = execution_accuracy(
                p["gold_solution"], p["function_name"], p["tests"]
            )
            if acc != 1.0:
                failed.append((p["id"], valid, err))
        assert failed == [], failed[:5]

    def test_get_adapter(self):
        adapter = get_adapter("coding")
        assert adapter.name == "coding"
        items = adapter.build_feed(n=5, full=False, seed=1)
        phases = {i.phase for i in items}
        assert phases == {"baseline", "degraded", "recovery"}

    def test_verify_solution_wrapper(self):
        from adapters.coding import _index

        p = next(iter(_index().values()))
        acc, valid, _ = verify_solution(f"```python\n{p['gold_solution']}\n```", p)
        assert acc == 1.0 and valid
