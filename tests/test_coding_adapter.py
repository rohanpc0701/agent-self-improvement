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


class TestCodingRulesAndTeacher:
    def test_strip_think_before_extract(self):
        from adapters.coding import _strip_think
        from harness.sandbox import extract_python_code

        raw = "<think>plan</think>\n```python\ndef f():\n    return 1\n```\n"
        code = extract_python_code(_strip_think(raw))
        assert "def f" in code

    def test_rules_block_empty_without_graph(self, tmp_path, monkeypatch):
        import correction.graph as g
        from adapters.coding import _rules_block

        monkeypatch.setattr(g, "_STORE_PATH", tmp_path / "graph_store.json")
        g.reload()
        block, n = _rules_block("dp", "climb stairs")
        assert block == "" and n == 0

    def test_write_graph_rules_fallback(self, tmp_path, monkeypatch):
        import correction.graph as g
        from adapters.coding import write_graph_rules, _index
        from contracts.schemas import DriftEvent, FailureMode
        from correction.learner import FailingCase

        monkeypatch.setattr(g, "_STORE_PATH", tmp_path / "graph_store.json")
        g.reload()
        # Force distill fallback (no API)
        monkeypatch.setattr(
            "correction.distill._call_model",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api")),
        )
        p = _index()["h_climb_stairs"]
        case = FailingCase(
            run_id=f"{p['id']}_deadbeef",
            question=p["question"],
            broken_output="def climb_stairs(n):\n    return n\n",
            gold_output=p["gold_solution"],
            domain_id=p["topic"],
            difficulty="hard",
        )
        ev = DriftEvent(
            detected_at=1.0,
            channel="execution_accuracy",
            severity=0.4,
            baseline_mean=1.0,
            window_mean=0.6,
            failing_run_ids=[case.run_id],
            failure_mode=FailureMode.INVALID_OUTPUT,
        )
        n = write_graph_rules(ev, [case])
        assert n == 1
        rules = g.get_rules(p["topic"], f"Implement climb_stairs please")
        assert rules, "expected trigger hit on function name"

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


class TestPromptAssembly:
    def _config(self):
        from contracts.schemas import AgentConfig, FewShotExample

        return AgentConfig(
            config_id="t",
            model="m",
            few_shot_examples=[
                FewShotExample(
                    question="q1", correct_output="def a(): pass", domain_id="dp"
                ),
                FewShotExample(
                    question="q2", correct_output="def b(): pass", domain_id="graphs"
                ),
            ],
        )

    def test_injects_same_topic_examples_with_stats(self, monkeypatch):
        from adapters.coding import build_user_prompt

        monkeypatch.delenv("AGENT_USE_EXAMPLES", raising=False)
        prompt, stats = build_user_prompt(
            "solve it", self._config(), topic="dp", use_rules=False
        )
        assert "q1" in prompt and "q2" not in prompt
        assert stats["examples_available"] == 2
        assert stats["examples_injected"] == 1
        assert stats["example_ids"] == ["q1"]
        assert stats["rules_injected"] == 0

    def test_examples_flag_off(self, monkeypatch):
        from adapters.coding import build_user_prompt

        monkeypatch.setenv("AGENT_USE_EXAMPLES", "0")
        prompt, stats = build_user_prompt(
            "solve it", self._config(), topic="dp", use_rules=False
        )
        assert "q1" not in prompt
        assert stats["examples_injected"] == 0
        assert stats["examples_available"] == 2

    def test_rules_counted(self, tmp_path, monkeypatch):
        import correction.graph as g
        from adapters.coding import build_user_prompt
        from correction.contracts import CorrectionRule
        from correction.graph import add_rule

        monkeypatch.setattr(g, "_STORE_PATH", tmp_path / "graph_store.json")
        g.reload()
        add_rule(
            CorrectionRule(
                id="r_dp_base",
                scope="db",
                db_id="dp",
                trap="off-by-one in base case",
                fix="start dp table at index 0",
                trigger="solve",
                source="seed",
            )
        )
        monkeypatch.delenv("AGENT_USE_EXAMPLES", raising=False)
        _, stats = build_user_prompt(
            "solve it", self._config(), topic="dp", use_rules=True
        )
        assert stats["rules_injected"] >= 1


class TestHardCurriculumFeed:
    def test_phase_counts_and_heldout_disjoint(self):
        from adapters.coding import CodingAdapter

        adapter = CodingAdapter()
        items = adapter.build_hard_curriculum_feed(
            seed=42, n_baseline=40, n_learn=100, n_heldout=40
        )
        phases = [it.phase for it in items]
        assert phases.count("baseline") == 40
        assert phases.count("degraded") == 100
        assert phases.count("recovery") == 40
        assert all(it.difficulty == "easy" for it in items if it.phase == "baseline")
        assert all(
            it.difficulty in ("hard", "extra")
            for it in items
            if it.phase in ("degraded", "recovery")
        )
        learn_ids = {it.question_id for it in items if it.phase == "degraded"}
        held_ids = {it.question_id for it in items if it.phase == "recovery"}
        assert learn_ids.isdisjoint(held_ids), "held-out must not appear in LEARN"

    def test_heldout_frac_controls_unique_split(self):
        from adapters import get_adapter

        a = get_adapter("coding")
        items = a.build_hard_curriculum_feed(seed=42, db_heldout_frac=0.5)
        learn_ids = {i.question_id for i in items if i.phase == "degraded"}
        held_ids = {i.question_id for i in items if i.phase == "recovery"}
        assert learn_ids.isdisjoint(held_ids)
        assert len(held_ids) >= 20  # ~half of ≥60 hard pool at frac=0.5
        items2 = a.build_hard_curriculum_feed(seed=42, db_heldout_frac=0.5)
        assert {i.question_id for i in items2 if i.phase == "recovery"} == held_ids


class TestChatRetry:
    class _Err429(Exception):
        status_code = 429

    class _Err404(Exception):
        status_code = 404

    def _client(self, fail_times, err):
        calls = {"n": 0}

        class FakeCompletions:
            def create(self, **kw):
                calls["n"] += 1
                if calls["n"] <= fail_times:
                    raise err("boom")
                return "resp"

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        return FakeClient(), calls

    def test_retries_429_then_succeeds(self, monkeypatch):
        from adapters import coding

        monkeypatch.setattr(coding.time, "sleep", lambda s: None)
        client, calls = self._client(2, self._Err429)
        assert coding._chat_with_retry(client, model="m", messages=[]) == "resp"
        assert calls["n"] == 3

    def test_non_retryable_raises_immediately(self):
        from adapters import coding

        client, calls = self._client(99, self._Err404)
        import pytest

        with pytest.raises(self._Err404):
            coding._chat_with_retry(client, model="m", messages=[])
        assert calls["n"] == 1

    def test_gives_up_after_max_retries(self, monkeypatch):
        from adapters import coding

        monkeypatch.setattr(coding.time, "sleep", lambda s: None)
        client, calls = self._client(99, self._Err429)
        import pytest

        with pytest.raises(self._Err429):
            coding._chat_with_retry(client, model="m", messages=[], max_retries=3)
        assert calls["n"] == 4
