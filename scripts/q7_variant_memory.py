#!/usr/bin/env python3
"""Q7 — similarity-matched memory: variant generation + 4-cell eval.

Prereg: docs/prereg/Q7_similarity_matched_memory.md (committed before any generation).
CTO constraint (2026-07-28): small steps only — phases are gated:
    --phase provenance   $0   reconstruct lesson-source qids from runs/build_mem.log
    --phase samples      ~$0.30  generate 3 variants -> runs/q7/variant_samples.md, PAUSE
    --phase pilot        ~$4  4 variants + 4 unrelated, all cells -> sigma report, PAUSE
    --phase full         ~$15-20  20+20 (requires --confirm-full; blocked by default)
    --phase report       $0   compute the mechanical verdict from state

Resumable: every unit -> one record in runs/q7/state.json (JSONL despite the name asked
for in the brief; one record per line, last-write-wins per key — same convention as every
other resumable state file in this repo). Budget ceiling $25 tracked from usage tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    """Load ROOT/.env (existing env wins). The teacher-client resolution reads
    OPENROUTER_API_KEY from the environment; without this, resolution silently
    falls through to whatever key IS present (observed: MiniMax, which does not
    serve glm-5.2)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
# Force the teacher onto OpenRouter regardless of stray shell credentials.
os.environ["TEACHER_USE_OPENROUTER"] = "1"
os.environ.pop("TEACHER_BASE_URL", None)
os.environ.pop("TEACHER_API_KEY", None)

from adapters.finance import (  # noqa: E402
    generate_answer,
    get_problem,
    load_manifest,
)
from analysis.similarity import pair_cosine, pairwise_max  # noqa: E402
from contracts.schemas import AgentConfig, FewShotExample  # noqa: E402
from correction.judge import grade, rubric_max_points  # noqa: E402

Q7 = ROOT / "runs" / "q7"
STATE = Q7 / "state.json"
STORE = ROOT / "runs" / "finance_memory_good.json"
BUILD_LOG = ROOT / "runs" / "build_mem.log"
STUDENT = os.environ.get("Q7_STUDENT", "qwen/qwen3.6-27b")
TEACHER = os.environ.get("Q7_TEACHER", "z-ai/glm-5.2")
BUDGET_USD = float(os.environ.get("Q7_BUDGET_USD", "25"))
COSINE_LO, COSINE_HI = 0.30, 0.90

# rough OpenRouter prices (USD/Mtok in, out) for budget accounting
_PRICES = {"qwen/qwen3.6-27b": (0.10, 0.35), "z-ai/glm-5.2": (0.55, 2.0),
           "openai/gpt-5.2": (1.25, 10.0)}
_spend = {"usd": 0.0}


def _track(model: str, usage) -> None:
    if usage is None:
        return
    pin, pout = _PRICES.get(model, (1.0, 3.0))
    _spend["usd"] += (getattr(usage, "prompt_tokens", 0) or 0) / 1e6 * pin \
                   + (getattr(usage, "completion_tokens", 0) or 0) / 1e6 * pout
    if _spend["usd"] > BUDGET_USD:
        raise RuntimeError(f"Q7 budget ${BUDGET_USD} exceeded (${_spend['usd']:.2f}) — aborting")


# ── state ─────────────────────────────────────────────────────────────────────
def _load_state() -> dict:
    rows = {}
    if STATE.exists():
        for line in STATE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    return rows


def _put(row: dict) -> None:
    Q7.mkdir(parents=True, exist_ok=True)
    with STATE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── provenance (mismatch repair: store has no source ids) ────────────────────
def reconstruct_provenance() -> dict[str, list[dict]]:
    """Map lesson-source qid -> store items, by matching build_mem.log entries
    (qid, category, kind, char-length) against the frozen store. Asserts every
    store item is claimed exactly once; raises otherwise (fallback per prereg)."""
    store = json.loads(STORE.read_text())["items"]
    log_entries = []
    pat = re.compile(r"^(fpb-\d+) \[(\w+)\] (playbook|trap|skeleton): (\d+) chars")
    for line in BUILD_LOG.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            log_entries.append({"qid": m.group(1), "category": m.group(2),
                                "kind": m.group(3), "chars": int(m.group(4))})
    claimed = [False] * len(store)
    mapping: dict[str, list[dict]] = {}
    for e in log_entries:
        prefix = f"[FINANCE_{e['kind'].upper()}] {e['category']}"
        for i, it in enumerate(store):
            if claimed[i] or it["question"] != prefix:
                continue
            if abs(len(it["correct_output"]) - e["chars"]) <= 2:  # trim tolerance
                claimed[i] = True
                mapping.setdefault(e["qid"], []).append(it)
                break
    n_claimed = sum(claimed)
    if n_claimed != len(store):
        raise RuntimeError(
            f"provenance reconstruction incomplete: {n_claimed}/{len(store)} store items "
            f"matched to build_mem.log — use the category-matched fallback (prereg) and "
            f"say so in the report"
        )
    return mapping


def _background() -> list[dict]:
    """IDF background pool: all benchmark questions (cached)."""
    from adapters.finance import load_finance_questions
    global _BG
    try:
        return _BG
    except NameError:
        _BG = load_finance_questions()
        return _BG


# ── variant generation (blind — asserted) ─────────────────────────────────────
_VARIANT_SYS = (
    "You write exam-quality variant questions for expert finance benchmarks. Given a source "
    "question and its grading rubric, produce ONE variant with the SAME reasoning structure "
    "and SAME governing standards, but different company names, numbers, dates, and surface "
    "framing. The variant rubric must keep the exact same 'Item R<n>(max <m>)' structure and "
    "the same point maxima, with numeric expectations recomputed for the new numbers.\n"
    "Return EXACTLY this delimited format (no JSON, no code fences):\n"
    "=== VARIANT QUESTION ===\n<the variant question>\n=== VARIANT RUBRIC ===\n<the variant rubric>\n=== END ==="
)


def _teacher_client():
    from correction.provider import teacher_client_and_model
    os.environ.setdefault("TEACHER_USE_OPENROUTER", "1")
    os.environ.setdefault("TEACHER_MODEL", TEACHER)
    return teacher_client_and_model()


def _assert_blind(context: str) -> None:
    """Prereg hard rule: no memory text may enter the variant-generation context."""
    store = json.loads(STORE.read_text())["items"]
    for it in store:
        probe = it["correct_output"][:80]
        if probe and probe in context:
            raise RuntimeError("blind-generation violated: memory text found in variant context")


def generate_variant(qid: str, state: dict, idx: int = 0) -> dict | None:
    key = f"variant|{qid}|{idx}"
    if key in state and state[key].get("ok"):
        return state[key]
    p = get_problem(qid)
    prompt = (f"SOURCE QUESTION:\n{p['question']}\n\nSOURCE RUBRIC:\n{p['rubric']}\n\n"
              f"Category: {p['category']}. Produce variant #{idx + 1} now — if this is not the "
              f"first variant, make it materially different from plausible earlier ones "
              f"(different industry, different numeric regime). JSON only.")
    _assert_blind(prompt)
    client, model = _teacher_client()
    from harness.agent import _chat_with_retry
    resp = _chat_with_retry(client, model=model,
                            messages=[{"role": "system", "content": _VARIANT_SYS},
                                      {"role": "user", "content": prompt}],
                            temperature=0.3, max_tokens=int(os.environ.get('Q7_VARIANT_MAX_TOKENS', '16000')))
    _track(model, getattr(resp, "usage", None))
    raw = (resp.choices[0].message.content or "").strip()
    row = {"key": key, "qid": qid, "ok": False}
    if not raw:
        row["error"] = "empty teacher content (reasoning-token exhaustion — raise Q7_VARIANT_MAX_TOKENS)"
        _put(row); state[key] = row
        return row
    try:
        # Delimited format: JSON forced GLM to embed multi-line rubrics in quoted
        # strings, which it emits with literal newlines -> unterminated-string errors
        # on 3/3 attempts. Delimiters have no escaping problem.
        m = re.search(r"=== VARIANT QUESTION ===\s*(.*?)\s*=== VARIANT RUBRIC ===\s*(.*?)\s*(?:=== END ===|\Z)",
                      raw, flags=re.DOTALL)
        if not m:
            raise KeyError("delimiters not found")
        vq, vr = m.group(1).strip(), m.group(2).strip()
        # rubric must parse and preserve the max-point total
        src_max, var_max = rubric_max_points(p["rubric"]), rubric_max_points(vr)
        if abs(src_max - var_max) > 1e-6:
            row["error"] = f"max points {var_max} != source {src_max}"
        else:
            bg = {q["id"]: q["question"] for q in _background()}
            sim = pair_cosine(vq, p["question"], bg)
            row.update(ok=COSINE_LO <= sim <= COSINE_HI, question=vq, rubric=vr,
                       cosine_to_source=round(sim, 4), category=p["category"])
            if not row["ok"]:
                row["error"] = f"cosine {sim:.3f} outside [{COSINE_LO},{COSINE_HI}]"
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        row["error"] = f"parse: {e}"
    _put(row)
    state[key] = row
    return row


# ── evaluation cells ──────────────────────────────────────────────────────────
def _memory_items() -> list[FewShotExample]:
    return [FewShotExample(**it) for it in json.loads(STORE.read_text())["items"]]


def run_cell(tag: str, question: str, rubric: str, category: str, mem: bool,
             rep: int, state: dict) -> None:
    key = f"cell|{tag}|{'M' if mem else 'B'}|{rep}"
    if key in state and state[key].get("ok"):
        return
    os.environ["AGENT_USE_EXAMPLES"] = "1" if mem else "0"   # matches prior arms' mechanism
    cfg = AgentConfig(config_id=f"q7-{'mem' if mem else 'bare'}", model=STUDENT,
                      few_shot_examples=_memory_items() if mem else [])
    try:
        answer, stats = generate_answer(question, cfg, category)
        g = grade(question, rubric, answer, passes=3)
        _put({"key": key, "ok": True, "tag": tag, "mem": mem, "rep": rep,
              "normalized": g["normalized"], "injected": stats.get("examples_injected", 0),
              "answer_chars": len(answer)})
    except Exception as e:  # recorded, never fabricated
        _put({"key": key, "ok": False, "tag": tag, "mem": mem, "rep": rep,
              "error": f"{type(e).__name__}: {e}"[:200]})
    state[key] = {"ok": True}


def unrelated_ids(sources: list[str], n: int) -> list[str]:
    man = load_manifest()
    held = man["heldout_ids"]
    src_docs = {s: get_problem(s)["question"] for s in sources}
    tgt_docs = {h: get_problem(h)["question"] for h in held}
    sims = pairwise_max(tgt_docs, src_docs)
    ranked = sorted(sims, key=lambda h: sims[h])          # least similar first
    return [h for h in ranked if rubric_max_points_safe(get_problem(h)["rubric"])][:n]


def rubric_max_points_safe(rubric: str) -> bool:
    try:
        return rubric_max_points(rubric) > 0
    except Exception:
        return False


# ── phases ────────────────────────────────────────────────────────────────────
def phase_provenance() -> dict:
    m = reconstruct_provenance()
    print(f"provenance OK: {len(m)} source questions -> "
          f"{sum(len(v) for v in m.values())} store items")
    for qid, items in m.items():
        kinds = [i["question"].split("]")[0].strip("[") for i in items]
        print(f"  {qid}  {get_problem(qid)['category']:8} {kinds}")
    (Q7 / "provenance.json").parent.mkdir(parents=True, exist_ok=True)
    (Q7 / "provenance.json").write_text(json.dumps(
        {q: [i["question"] for i in items] for q, items in m.items()}, indent=1))
    return m


def phase_samples(n: int = 3) -> None:
    state = _load_state()
    sources = list(phase_provenance().keys())[:n]
    lines = ["# Q7 variant samples — human review gate\n"]
    for qid in sources:
        row = generate_variant(qid, state, idx=0)
        p = get_problem(qid)
        lines += [f"\n## {qid} ({p['category']}) — cosine {row.get('cosine_to_source')} "
                  f"{'OK' if row.get('ok') else 'REJECTED: ' + str(row.get('error'))}",
                  "\n### Source question\n", p["question"][:1200],
                  "\n### Variant question\n", str(row.get("question", ""))[:1200],
                  "\n### Variant rubric\n", str(row.get("rubric", ""))[:1200]]
    out = Q7 / "variant_samples.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} · spend ${_spend['usd']:.2f} — PAUSED for human review")


def phase_eval(n_v: int, n_u: int, reps: int = 3) -> None:
    state = _load_state()
    prov = reconstruct_provenance()
    sources = list(prov.keys())
    variants = []
    per_source = -(-n_v // max(1, len(sources)))          # ceil division
    for idx in range(per_source):
        for qid in sources:
            if len(variants) >= n_v:
                break
            row = generate_variant(qid, state, idx=idx)
            if row.get("ok"):
                variants.append(row)
    unrel = unrelated_ids(sources, n_u)
    print(f"eval: {len(variants)} variants, {len(unrel)} unrelated, reps={reps}")
    for row in variants:
        for rep in range(reps):
            for mem in (False, True):
                run_cell(f"V:{row['qid']}", row["question"], row["rubric"],
                         row["category"], mem, rep, state)
    for qid in unrel:
        p = get_problem(qid)
        for rep in range(reps):
            for mem in (False, True):
                run_cell(f"U:{qid}", p["question"], p["rubric"], p["category"],
                         mem, rep, state)
    print(f"phase done · spend ${_spend['usd']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["provenance", "samples", "pilot", "full", "report"])
    ap.add_argument("--confirm-full", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.phase == "provenance":
        phase_provenance()
    elif args.phase == "samples":
        phase_samples()
    elif args.phase == "pilot":
        phase_eval(4, 4)
    elif args.phase == "full":
        if not args.confirm_full:
            raise SystemExit("full run blocked: CTO small-batch constraint — pass "
                             "--confirm-full only after explicit approval")
        phase_eval(20, 20)
    elif args.phase == "report":
        from analysis.q7_report import report
        report()
    print(f"[{args.phase}] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
