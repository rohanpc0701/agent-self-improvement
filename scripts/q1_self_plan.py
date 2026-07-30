#!/usr/bin/env python3
"""Q1 SELF-PLAN — plan STRUCTURE vs teacher KNOWLEDGE (docs/prereg/PREREG_Q1_SELF_PLAN.md).

Arms (paired, same questions):
  A NOPLAN-MATCHED  student restates the question (~250w, no strategy), then answers
  B SELF-PLAN       student writes its own <=250w plan, then answers
  C TEACHER-PLAN    GLM 5.2 writes <=250w plan (question only, no rubric), student answers

Commands:
  draw         write runs/q1_question_ids.json (40 stratified from frozen 80, seed 42)
  run --pilot  8 questions x 3 arms x k=3, 3 judge passes
  pilot-check  mechanical gate report (no per-arm score means — no peeking)
  run          full 40 (requires committed prereg + committed id list)
  analyze      paired deltas + bootstrap CIs + branch rule (only when all cells done)

State: runs/q1_state.json — single JSON, atomic-rewritten after EVERY sub-call
(preamble / answer / grade separately). Resume = rerun the same command.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs"
STATE_PATH = RUNS / "q1_state.json"
IDS_PATH = RUNS / "q1_question_ids.json"
ANSWER_MAX_TOKENS = 8192  # 2048 truncated 55/72 pilot answers, skewed against B/C
PREREG = ROOT / "docs" / "prereg" / "PREREG_Q1_SELF_PLAN.md"

EXPECTED_TEACHER = "z-ai/glm-5.2"
EXPECTED_STUDENT = "qwen/qwen3.6-27b"
ARMS = ("A", "B", "C")
K_REPS = 3
JUDGE_PASSES = 3
PILOT_N = 8
FULL_N = 40
PREAMBLE_WORDS = 250

RESTATE_PROMPT = (
    "Restate the question below in your own words, in roughly 250 words. Describe only "
    "what is given and what is being asked. Do not outline an approach, do not list "
    "steps, and do not give strategy, hints, or analysis — a neutral restatement only.\n\n"
    "Question:\n{q}"
)
SELFPLAN_PROMPT = (
    "Before answering, write a brief plan (at most 250 words) for the question below: "
    "what is being asked, what determines the answer, what steps you will take, and "
    "what mistakes to avoid. Do not write the answer itself yet.\n\nQuestion:\n{q}"
)
TEACHER_PLAN_SYSTEM = (
    "You are a senior finance expert coaching a capable analyst. Write a brief plan "
    "(at most 250 words) for the problem: what is being asked, what determines the "
    "answer, the ordered steps to take, and the mistakes to avoid. Do NOT state the "
    "final answer, numeric results, or conclusions — the analyst must do the work."
)
ANSWER_PROMPT = (
    "{q}\n\n---\nPreliminary note (written before answering):\n{preamble}\n\n"
    "Now write your complete answer to the question."
)


# ---------- env / startup asserts (checklist §1) ----------

def load_dotenv_explicit() -> None:
    env = ROOT / ".env"
    if not env.exists():
        raise SystemExit("FATAL: .env not found — refusing implicit environment")
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def startup_asserts() -> None:
    teacher = (os.environ.get("TEACHER_MODEL") or "").strip()
    print(f"[startup] TEACHER_MODEL = {teacher!r}", flush=True)
    if teacher != EXPECTED_TEACHER:
        raise SystemExit(
            f"FATAL: TEACHER_MODEL={teacher!r} != {EXPECTED_TEACHER!r} "
            "(MiniMax silent-routing guard) — set it in .env"
        )
    student = (os.environ.get("STUDENT_MODEL") or os.environ.get("AGENT_MODEL") or "").strip()
    print(f"[startup] student model  = {student!r}", flush=True)
    if student != EXPECTED_STUDENT:
        raise SystemExit(f"FATAL: student model {student!r} != {EXPECTED_STUDENT!r}")
    base = (os.environ.get("AGENT_BASE_URL") or "").strip()
    print(f"[startup] AGENT_BASE_URL = {base!r}", flush=True)
    if "pinference" not in base.lower() and "primeintellect" not in base.lower():
        raise SystemExit("FATAL: prereg pins platform=Prime; AGENT_BASE_URL is not Prime")
    if os.environ.get("AGENT_ENABLE_THINKING", "").strip() in ("1", "true", "yes"):
        raise SystemExit("FATAL: AGENT_ENABLE_THINKING set — prereg requires reasoning off")
    # Teacher venue: OpenRouter (GLM empty-content behavior calibrated there — prereg).
    from correction.provider import teacher_client_and_model

    t_client, t_model = teacher_client_and_model()
    t_base = str(t_client.base_url)
    print(f"[startup] teacher base   = {t_base!r} model = {t_model!r}", flush=True)
    if "openrouter.ai" not in t_base.lower():
        raise SystemExit("FATAL: prereg pins teacher=OpenRouter; teacher base is not OpenRouter")
    if t_model != EXPECTED_TEACHER:
        raise SystemExit(f"FATAL: teacher resolved to {t_model!r} != {EXPECTED_TEACHER!r}")
    # Importing correction.judge runs the judge!=teacher assert at module load.
    from correction.judge import JUDGE_MODEL  # noqa: F401
    print(f"[startup] JUDGE_MODEL   = {JUDGE_MODEL!r} (!= teacher asserted)", flush=True)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def assert_preregistered(*, need_ids: bool) -> None:
    if "____" in PREREG.read_text():
        raise SystemExit("FATAL: prereg confidence %% not filled in")
    if not _git("log", "--oneline", "-1", "--", str(PREREG.relative_to(ROOT))):
        raise SystemExit("FATAL: prereg file not committed — commit it before any API call")
    if _git("status", "--porcelain", "--", str(PREREG.relative_to(ROOT))):
        raise SystemExit("FATAL: prereg file has uncommitted changes")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "feat/q1-self-plan":
        raise SystemExit(f"FATAL: on branch {branch!r}, prereg pins feat/q1-self-plan")
    if need_ids:
        rel = str(IDS_PATH.relative_to(ROOT))
        if not _git("log", "--oneline", "-1", "--", rel) or _git("status", "--porcelain", "--", rel):
            raise SystemExit("FATAL: question-id list not committed (checklist §3)")


# ---------- state (single JSON, atomic, per-sub-call checkpoints) ----------

def _state_path(shard: int | None) -> Path:
    return STATE_PATH if shard is None else RUNS / f"q1_state_shard{shard}.json"


def load_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"teacher_plans": {}, "units": {}}


def load_merged_state() -> dict:
    """Merge the legacy state with all shard states (disjoint questions per shard)."""
    merged = {"teacher_plans": {}, "units": {}}
    for path in [STATE_PATH, *sorted(RUNS.glob("q1_state_shard*.json"))]:
        s = load_state(path)
        merged["teacher_plans"].update(s["teacher_plans"])
        merged["units"].update(s["units"])
    return merged


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    RUNS.mkdir(exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _key(qid: str, arm: str, rep: int) -> str:
    return f"{qid}|{arm}|{rep}"


# ---------- calls ----------

def _word_trim(text: str, max_words: int = PREAMBLE_WORDS) -> str:
    toks = text.split()
    return text.strip() if len(toks) <= max_words else " ".join(toks[:max_words]).strip()


def student_chat(user_content: str, *, max_tokens: int) -> str:
    """One student call: Prime, reasoning.enabled=false, empty content = hard error."""
    from adapters.finance import _SYSTEM
    from harness.agent import _chat_with_retry, _get_client

    resp = _chat_with_retry(
        _get_client(),
        model=EXPECTED_STUDENT,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=int(os.environ.get("AGENT_MAX_RETRIES", "5")),
        extra_body={"reasoning": {"enabled": False}},
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("student returned empty content")
    return text


def teacher_plan(question: str) -> str:
    from correction.provider import teacher_client_and_model

    client, model = teacher_client_and_model()
    if model != EXPECTED_TEACHER:
        raise SystemExit(f"FATAL: teacher resolved to {model!r} != {EXPECTED_TEACHER!r}")
    max_tokens = max(12000, int(os.environ.get("TEACHER_MAX_TOKENS", "12000")))
    for attempt in (1, 2):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TEACHER_PLAN_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    raise SystemExit("FATAL: teacher returned EMPTY content twice — hard exit (checklist §1)")


# ---------- commands ----------

def cmd_draw() -> None:
    from adapters.finance import get_problem, load_manifest
    from correction.judge import JudgeParseError, rubric_max_points

    m = load_manifest()
    usable, excluded = [], []
    for qid in m["validation_ids"]:
        try:
            rubric_max_points(get_problem(qid)["rubric"])
            usable.append(qid)
        except JudgeParseError as exc:
            excluded.append({"id": qid, "reason": str(exc)})
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid in usable:
        by_cat[get_problem(qid)["category"]].append(qid)
    rng = random.Random(42)
    for v in by_cat.values():
        v.sort()
        rng.shuffle(v)
    ordered: list[str] = []
    pools = sorted(by_cat.items())
    while any(p for _, p in pools) and len(ordered) < FULL_N:
        for _, pool in pools:
            if pool and len(ordered) < FULL_N:
                ordered.append(pool.pop())
    RUNS.mkdir(exist_ok=True)
    IDS_PATH.write_text(json.dumps(
        {"seed": 42, "n": len(ordered), "pilot_ids": ordered[:PILOT_N],
         "question_ids": ordered, "excluded_unscoreable": excluded}, indent=1))
    print(f"[draw] {len(ordered)} ids ({len(excluded)} unscoreable excluded) -> {IDS_PATH}")
    print("[draw] COMMIT runs/q1_question_ids.json before `run`")


def cmd_run(pilot: bool, shard: int | None = None, nshards: int = 1) -> None:
    from adapters.finance import get_problem
    from correction.judge import grade

    startup_asserts()
    assert_preregistered(need_ids=True)
    ids_doc = json.loads(IDS_PATH.read_text())
    ids = ids_doc["pilot_ids"] if pilot else ids_doc["question_ids"]
    if shard is not None:
        ids = ids[shard::nshards]
    spath = _state_path(shard)
    state = load_state(spath)
    # Teacher plans already built (e.g. by the pilot) are cap-independent; seed them.
    if not state["teacher_plans"]:
        state["teacher_plans"].update(
            {q: p for q, p in load_merged_state()["teacher_plans"].items() if q in ids}
        )
    units = state["units"]
    total = len(ids) * len(ARMS) * K_REPS
    done = sum(1 for k, u in units.items() if u.get("grade") and k.split("|")[0] in ids)
    print(f"[run] {'pilot' if pilot else 'full'}: {len(ids)} questions, "
          f"{total} units, {done} already complete", flush=True)

    for qid in ids:
        p = get_problem(qid)
        # Teacher plan cached once per question (temp 0).
        if qid not in state["teacher_plans"]:
            state["teacher_plans"][qid] = _word_trim(teacher_plan(p["question"]))
            save_state(state, spath)
            print(f"  [plan] {qid} teacher plan ok", flush=True)
        for arm in ARMS:
            for rep in range(K_REPS):
                key = _key(qid, arm, rep)
                unit = units.setdefault(key, {})
                if unit.get("grade"):
                    continue
                try:
                    if unit.get("preamble") is None or not unit.get("preamble", "").strip():
                        if arm == "A":
                            pre = student_chat(RESTATE_PROMPT.format(q=p["question"]),
                                               max_tokens=1024)
                        elif arm == "B":
                            pre = student_chat(SELFPLAN_PROMPT.format(q=p["question"]),
                                               max_tokens=1024)
                        else:
                            pre = state["teacher_plans"][qid]
                        pre = _word_trim(pre)
                        if not pre.strip():
                            raise RuntimeError("empty preamble — malformed unit")
                        unit["preamble"] = pre
                        save_state(state, spath)
                    if not unit.get("answer"):
                        # Preamble must exist before the answer call (checklist §1).
                        assert unit["preamble"].strip(), "empty preamble before answer"
                        unit["answer"] = student_chat(
                            ANSWER_PROMPT.format(q=p["question"], preamble=unit["preamble"]),
                            max_tokens=ANSWER_MAX_TOKENS)
                        save_state(state, spath)
                    unit["grade"] = grade(question=p["question"], rubric=p["rubric"],
                                          answer=unit["answer"], passes=JUDGE_PASSES)
                    unit["ts"] = time.time()
                    save_state(state, spath)
                    print(f"  [unit] {key} ok", flush=True)  # no scores printed: no peeking
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    unit["error"] = str(exc)
                    save_state(state, spath)
                    print(f"  [unit] {key} FAILED ({exc})", flush=True)
                    low = str(exc).lower()
                    if any(t in low for t in ("insufficient", "401", "403")):
                        raise SystemExit(f"FATAL provider error: {exc}") from exc
    remaining = sum(1 for qid in ids for a in ARMS for r in range(K_REPS)
                    if not units.get(_key(qid, a, r), {}).get("grade"))
    print(f"[run] done; {remaining} units still incomplete (rerun to resume)", flush=True)


def cmd_pilot_check() -> None:
    """Mechanical gate only — never prints per-arm score means."""
    state = load_merged_state()
    ids = json.loads(IDS_PATH.read_text())["pilot_ids"]
    empties, sigmas, pre_lens = [], defaultdict(list), defaultdict(list)
    incomplete = []
    for qid in ids:
        for arm in ARMS:
            per_rep = []
            for rep in range(K_REPS):
                u = state["units"].get(_key(qid, arm, rep), {})
                if not u.get("grade"):
                    incomplete.append(_key(qid, arm, rep))
                    continue
                if not u.get("answer", "").strip() or not u.get("preamble", "").strip():
                    empties.append(_key(qid, arm, rep))
                per_rep.append(float(u["grade"]["normalized"]))
                pre_lens[arm].append(len(u["preamble"].split()))
            if len(per_rep) >= 2:
                sigmas[arm].append(statistics.stdev(per_rep))
    print(f"incomplete units : {len(incomplete)} {incomplete[:6]}")
    print(f"empty outputs    : {len(empties)} {empties[:6]}")
    for arm in ARMS:
        s, w = sigmas[arm], pre_lens[arm]
        print(f"arm {arm}: within-question sd (noise floor) mean="
              f"{statistics.mean(s):.1f} max={max(s):.1f} | "
              f"preamble words mean={statistics.mean(w):.0f} "
              f"range=[{min(w)},{max(w)}]" if s else f"arm {arm}: no complete pairs")
    print("\n--- teacher plans (spot-read for rubric vocabulary) ---")
    for qid in ids:
        print(f"\n### {qid}\n{state['teacher_plans'].get(qid, '(missing)')}")
    print("\n--- arm B self-plans, rep 0 (flag answer attempts) ---")
    num_re = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
    for qid in ids:
        pre = state["units"].get(_key(qid, "B", 0), {}).get("preamble", "(missing)")
        n_nums = len(num_re.findall(pre))
        flags = []
        if n_nums > 3:
            flags.append(f"{n_nums} numeric tokens")
        if "=" in pre:
            flags.append("contains '='")
        if any(w in pre.lower() for w in ("therefore", "the answer is", "in conclusion")):
            flags.append("conclusion language")
        tag = f"  ⚠ ANSWER-LIKE: {', '.join(flags)}" if flags else ""
        print(f"\n### {qid}{tag}\n{pre}")


def cmd_analyze() -> None:
    from analysis.bootstrap import paired_bootstrap

    state = load_merged_state()
    ids = json.loads(IDS_PATH.read_text())["question_ids"]
    per_q: dict[str, dict[str, float]] = {}
    for qid in ids:
        means = {}
        for arm in ARMS:
            scores = [float(state["units"][_key(qid, arm, r)]["grade"]["normalized"])
                      for r in range(K_REPS)
                      if state["units"].get(_key(qid, arm, r), {}).get("grade")]
            if len(scores) < K_REPS:
                raise SystemExit(f"FATAL: incomplete cells for {qid} arm {arm} — "
                                 "analysis only after ALL cells complete (checklist §4)")
            means[arm] = sum(scores) / len(scores)
        per_q[qid] = means
    a = [per_q[q]["A"] for q in ids]
    b = [per_q[q]["B"] for q in ids]
    c = [per_q[q]["C"] for q in ids]
    ba = paired_bootstrap(a, b, n_boot=10_000, seed=42)
    ca = paired_bootstrap(a, c, n_boot=10_000, seed=42)
    out = {
        "label": "Prime, matched-baseline, n=%d, preregistered" % len(ids),
        "mean_A_noplan": statistics.mean(a),
        "mean_B_selfplan": statistics.mean(b),
        "mean_C_teacherplan": statistics.mean(c),
        "B_minus_A": ba,
        "C_minus_A": ca,
    }
    teacher_effect_real = not (ca["ci_low"] <= 0.0 <= ca["ci_high"])
    if teacher_effect_real:
        out["recovery"] = ba["delta"] / ca["delta"]
        r = out["recovery"]
        out["branch"] = ("STRUCTURE -> Q2 platform check" if r >= 2 / 3 else
                         "KNOWLEDGE -> plan-caching probe" if r <= 1 / 3 else
                         "HYBRID -> plan-caching + self-plan fallback layer")
    else:
        out["recovery"] = None
        out["branch"] = ("C-A CI includes 0: teacher-plan effect does not replicate "
                         "under matched conditions on this stack; STOP, report, re-scope. "
                         "recovery undefined — do not quote it.")
    print(json.dumps(out, indent=2))
    (RUNS / "q1_summary.json").write_text(json.dumps(out, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("draw", "run", "pilot-check", "analyze"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    load_dotenv_explicit()
    if args.cmd == "draw":
        cmd_draw()
    elif args.cmd == "run":
        cmd_run(pilot=args.pilot, shard=args.shard, nshards=args.nshards)
    elif args.cmd == "pilot-check":
        cmd_pilot_check()
    else:
        cmd_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
