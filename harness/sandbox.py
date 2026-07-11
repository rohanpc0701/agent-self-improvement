"""Sandboxed Python execution for coding-problem verification.

Runs student code in a subprocess with timeout, no network intent, and
stdin/stdout isolation. Not a security boundary for hostile tenants — enough
to keep flaky/hanging student code from wedging the harness.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_S = 2.0


def extract_python_code(text: str) -> str:
    """Pull the first ```python ... ``` block, else treat whole text as code."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    for i, chunk in enumerate(parts):
        if i % 2 == 0:
            continue
        body = chunk.strip()
        if body.lower().startswith("python"):
            body = body[6:].lstrip("\n")
        return body.strip()
    return text.strip()


def _runner_source(function_name: str, tests: list[dict[str, Any]]) -> str:
    """Python source that imports student module and runs tests, printing JSON."""
    tests_literal = json.dumps(tests)
    return textwrap.dedent(
        f"""\
        import json
        import sys
        import traceback
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        tests = json.loads({tests_literal!r})
        results = {{"ok": False, "passed": 0, "total": len(tests), "error": None, "valid": True}}

        try:
            import student
        except Exception as e:
            results["valid"] = False
            results["error"] = f"import: {{type(e).__name__}}: {{e}}"
            print(json.dumps(results))
            sys.exit(0)

        fn = getattr(student, {function_name!r}, None)
        if fn is None or not callable(fn):
            results["valid"] = False
            results["error"] = "missing function {function_name}"
            print(json.dumps(results))
            sys.exit(0)

        for t in tests:
            args = t.get("args", [])
            kwargs = t.get("kwargs", {{}})
            expected = t.get("expected")
            try:
                got = fn(*args, **kwargs)
            except Exception as e:
                results["valid"] = True  # ran, crashed on a case
                results["error"] = f"runtime: {{type(e).__name__}}: {{e}}"
                print(json.dumps(results))
                sys.exit(0)
            if got != expected:
                results["error"] = f"assert {{got!r}} != {{expected!r}}"
                print(json.dumps(results))
                sys.exit(0)
            results["passed"] += 1

        results["ok"] = results["passed"] == results["total"]
        print(json.dumps(results))
        """
    )


def run_tests(
    code: str,
    function_name: str,
    tests: list[dict[str, Any]],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute `code` and check `function_name` against `tests`.

    Returns dict with keys: ok (bool), passed, total, error, valid, timed_out.
    """
    if not tests:
        return {
            "ok": False,
            "passed": 0,
            "total": 0,
            "error": "no tests",
            "valid": False,
            "timed_out": False,
        }

    with tempfile.TemporaryDirectory(prefix="coding_sbx_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "student.py").write_text(code, encoding="utf-8")
        (tmp_path / "runner.py").write_text(
            _runner_source(function_name, tests), encoding="utf-8"
        )
        try:
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": str(tmp_path),
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": tmp,
            }
            proc = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "passed": 0,
                "total": len(tests),
                "error": f"timeout after {timeout_s}s",
                "valid": False,
                "timed_out": True,
            }

        stdout = (proc.stdout or "").strip().splitlines()
        if not stdout:
            return {
                "ok": False,
                "passed": 0,
                "total": len(tests),
                "error": (proc.stderr or "no output")[:200],
                "valid": False,
                "timed_out": False,
            }
        try:
            payload = json.loads(stdout[-1])
        except json.JSONDecodeError:
            return {
                "ok": False,
                "passed": 0,
                "total": len(tests),
                "error": f"bad runner output: {stdout[-1][:120]}",
                "valid": False,
                "timed_out": False,
            }
        payload.setdefault("timed_out", False)
        return payload


def execution_accuracy(
    code: str,
    function_name: str,
    tests: list[dict[str, Any]],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[float, bool, str]:
    """Return (accuracy 0/1, query_valid, error_detail)."""
    result = run_tests(code, function_name, tests, timeout_s=timeout_s)
    if result.get("timed_out"):
        return 0.0, False, result.get("error") or "timeout"
    valid = bool(result.get("valid", False))
    ok = bool(result.get("ok", False))
    return (1.0 if ok else 0.0), valid, result.get("error") or ""
