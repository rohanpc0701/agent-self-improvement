"""FinancePro-Bench adapter — rubric-graded free-text reasoning (RSI-Mem v2).

Rubric access policy (docs/RSI_MEM_V2_FINANCE.md §1):
- Student never sees rubrics.
- Teacher may see rubrics for TRAIN-STREAM ids only.
- Validation / held-out rubrics: judge-only.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

from contracts.schemas import AgentConfig, Difficulty, FewShotExample, TelemetryRecord
from correction.learner import FailingCase
from correction.provider import teacher_client_and_model
from harness import agent
from harness.feed import FeedItem

_ROOT = Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "fixtures" / "finance_pro_bench.json"
_MANIFEST = _ROOT / "fixtures" / "finance_manifest.json"
_log = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert finance professional. Answer the question with clear, "
    "structured reasoning. Cite standards and show steps where relevant. "
    "Do not invent rubric scores — produce the substantive answer only."
)

_MEMORY_KINDS = ("playbook", "trap", "skeleton")
_KIND_PREFIX = {
    "playbook": "[FINANCE_PLAYBOOK]",
    "trap": "[FINANCE_TRAP]",
    "skeleton": "[FINANCE_SKELETON]",
}

_BY_ID: dict[str, dict] | None = None
_MANIFEST_CACHE: dict | None = None


def _load_raw() -> list[dict]:
    raw = json.loads(_DATASET.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        return list(raw["items"])
    if isinstance(raw, list):
        return raw
    raise ValueError(f"unexpected dataset shape in {_DATASET}")


def _index() -> dict[str, dict]:
    global _BY_ID
    if _BY_ID is None:
        _BY_ID = {p["id"]: p for p in _load_raw()}
    return _BY_ID


def load_manifest(path: Path | None = None) -> dict:
    global _MANIFEST_CACHE
    p = Path(path) if path is not None else Path(
        os.environ.get("FINANCE_MANIFEST", str(_MANIFEST))
    )
    p = p.resolve()
    if _MANIFEST_CACHE is not None and _MANIFEST_CACHE.get("_path") == str(p):
        return _MANIFEST_CACHE["data"]
    data = json.loads(p.read_text(encoding="utf-8"))
    _MANIFEST_CACHE = {"_path": str(p), "data": data}
    return data


def get_problem(qid: str) -> dict:
    return _index()[qid]


def rubric_for(
    qid: str, *, role: str = "judge", manifest: dict | None = None
) -> str:
    """Return rubric text with role ACL: student never; teacher train-only; judge any."""
    if role == "student":
        raise PermissionError("rubric firewall: student may never read rubrics")
    if role == "teacher":
        assert_rubric_allowed_for_teacher(qid, manifest)
    elif role != "judge":
        raise ValueError(f"unknown rubric role {role!r}")
    return get_problem(qid)["rubric"]


def split_of(qid: str, manifest: dict | None = None) -> str:
    """Return 'train' | 'validation' | 'heldout'."""
    m = manifest or load_manifest()
    if qid in m["train_ids"]:
        return "train"
    if qid in m["validation_ids"]:
        return "validation"
    if qid in m["heldout_ids"]:
        return "heldout"
    raise KeyError(f"{qid} not in finance manifest")


def assert_rubric_allowed_for_teacher(qid: str, manifest: dict | None = None) -> None:
    """Raise if teacher is not allowed to see this question's rubric."""
    if split_of(qid, manifest) != "train":
        raise PermissionError(
            f"rubric firewall: teacher may not see rubric for {qid} "
            f"(split={split_of(qid, manifest)})"
        )


def memory_kind_of(ex: FewShotExample) -> str | None:
    """Return playbook|trap|skeleton from question prefix, else None."""
    q = (ex.question or "").lstrip()
    for kind, prefix in _KIND_PREFIX.items():
        if q.startswith(prefix):
            return kind
    return None


def select_category_memory(
    examples: list[FewShotExample],
    category: str,
    *,
    max_playbooks: int = 1,
    max_traps: int = 2,
    max_skeletons: int = 1,
) -> list[FewShotExample]:
    """Category-keyed retrieval with TraceLift compaction caps (≤4 items)."""
    same = [e for e in examples if not e.domain_id or e.domain_id == category]
    playbooks: list[FewShotExample] = []
    traps: list[FewShotExample] = []
    skeletons: list[FewShotExample] = []
    other: list[FewShotExample] = []
    for e in same:
        k = memory_kind_of(e)
        if k == "playbook":
            playbooks.append(e)
        elif k == "trap":
            traps.append(e)
        elif k == "skeleton":
            skeletons.append(e)
        else:
            other.append(e)
    picked: list[FewShotExample] = []
    picked.extend(playbooks[:max_playbooks])
    picked.extend(traps[:max_traps])
    picked.extend(skeletons[:max_skeletons])
    # Fill remaining slots (up to 4) with untyped same-category exemplars if needed.
    remaining = 4 - len(picked)
    if remaining > 0:
        picked.extend(other[:remaining])
    return picked[:4]


def build_student_prompt(
    question: str,
    config: AgentConfig,
    category: str,
    *,
    forbidden_rubric_stems: list[str] | None = None,
) -> tuple[str, dict]:
    """Assemble student prompt from question + memory only — never rubric.

    Memory injection (AGENT_USE_EXAMPLES): ≤4 items by category match —
    1 playbook + ≤2 traps + ≤1 skeleton (RA-RFT adapted; no embeddings).
    """
    stats = {
        "examples_available": len(config.few_shot_examples),
        "examples_injected": 0,
        "example_ids": [],
        "rules_injected": 0,
        "by_kind": {"playbook": 0, "trap": 0, "skeleton": 0, "other": 0},
    }
    parts: list[str] = []
    stems = forbidden_rubric_stems or []
    if os.environ.get("AGENT_USE_EXAMPLES", "1") != "0":
        selected = select_category_memory(config.few_shot_examples, category)
        if selected:
            lines = ["Category memory (TraceLift):"]
            for i, ex in enumerate(selected):
                blob = f"{ex.question}\n{ex.correct_output}"
                _assert_no_rubric_smuggle(blob, stems, where="few-shot example")
                kind = memory_kind_of(ex) or "other"
                stats["by_kind"][kind] = stats["by_kind"].get(kind, 0) + 1
                lines.append(
                    f"Memory {i + 1} [{kind}]:\n{ex.question}\n{ex.correct_output}"
                )
                stats["example_ids"].append(
                    f"{kind}:{getattr(ex, 'source', 'ex')}"
                )
            stats["examples_injected"] = len(selected)
            parts.append("\n\n".join(lines) + "\n\n")
    parts.append(question)
    prompt = "".join(parts)
    _assert_no_rubric_smuggle(prompt, stems, where="student prompt")
    return prompt, stats


# Alias used in the TraceLift work order.
build_user_prompt = build_student_prompt


def _assert_no_rubric_smuggle(
    text: str, stems: list[str], *, where: str
) -> None:
    """Reject text that carries known non-train rubric content or rubric markers."""
    if "OFFICIAL RUBRIC" in text:
        raise PermissionError(f"rubric firewall: OFFICIAL RUBRIC marker in {where}")
    for stem in stems:
        if stem and stem in text:
            raise PermissionError(
                f"rubric firewall: non-train rubric stem leaked into {where}"
            )


def all_rubric_stems(n: int = 120) -> list[str]:
    """Stable prefixes of ALL rubrics — student must never see any of them."""
    stems: list[str] = []
    for p in _load_raw():
        stem = p["rubric"].strip()[:n]
        if len(stem) >= 40:
            stems.append(stem)
    return stems


def non_train_rubric_stems(manifest: dict | None = None, n: int = 120) -> list[str]:
    """Deprecated alias — student firewall uses all stems."""
    del manifest
    return all_rubric_stems(n=n)


def build_teacher_prompt(
    question: str,
    *,
    qid: str,
    rubric: str | None = None,
    broken: str | None = None,
    manifest: dict | None = None,
) -> str:
    """Teacher prompt. Rubric only when qid is train-stream.

    Rubric text is always resolved via ``rubric_for(qid, role='teacher')``.
    If ``rubric`` is passed, it must exactly match the fixture text.
    """
    parts = [question]
    if rubric is not None:
        # ACL + identity check (rejects held-out text under a train qid).
        expected = rubric_for(qid, role="teacher", manifest=manifest)
        if rubric != expected:
            raise PermissionError(
                f"rubric firewall: rubric text does not match fixture for {qid}"
            )
        parts.append("\n\n--- OFFICIAL RUBRIC (train-stream only) ---\n")
        parts.append(expected)
    if broken:
        # Refuse pasted rubric markers even when rubric=None.
        if "OFFICIAL RUBRIC" in broken:
            raise PermissionError(
                "rubric firewall: student attempt appears to contain rubric text"
            )
        parts.append("\n\n--- STUDENT ATTEMPT ---\n")
        parts.append(broken)
        parts.append("\n\nProvide a corrected expert answer.")
    return "".join(parts)


def _student_max_tokens(default: int = 8192) -> int:
    raw = (os.environ.get("STUDENT_MAX_TOKENS") or str(default)).strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return default


def generate_answer(
    question: str,
    config: AgentConfig,
    category: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    forbidden_rubric_stems: list[str] | None = None,
) -> tuple[str, dict]:
    prompt, stats = build_student_prompt(
        question,
        config,
        category,
        forbidden_rubric_stems=forbidden_rubric_stems,
    )
    client = agent._get_client()
    from harness.agent import _chat_with_retry

    # Thinking SKUs (qwen3.6-27b): on OpenRouter, chat_template_kwargs
    # enable_thinking=false is often ignored and still yields content=None
    # while burning the token budget on reasoning. Prefer a large bare
    # max_tokens first (STUDENT_MAX_TOKENS, default 8192). Opt into the
    # Prime-era disable flag via AGENT_FORCE_NO_THINKING=1.
    tok = _student_max_tokens() if max_tokens is None else max_tokens
    kwargs: dict = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": tok,
        "max_retries": int(os.environ.get("AGENT_MAX_RETRIES", "5")),
    }
    # Disable reasoning by DEFAULT for thinking SKUs (qwen3.6-27b). On OpenRouter
    # the unified `reasoning.enabled=false` param works where chat_template_kwargs
    # does not (verified: 4.6s + content vs 26.5s + empty). Opt back in with
    # AGENT_ENABLE_THINKING=1.
    if os.environ.get("AGENT_ENABLE_THINKING", "").strip() not in ("1", "true", "yes"):
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    def _call(kw: dict):
        return _chat_with_retry(client, **kw)

    resp = _call(kwargs)
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        reasoning = getattr(msg, "reasoning", None) or (
            (msg.model_extra or {}).get("reasoning") if hasattr(msg, "model_extra") else None
        )
        if reasoning:
            _log.warning(
                "empty content from %s (reasoning_len=%d); retrying bare max_tokens bumps",
                config.model,
                len(str(reasoning)),
            )
            for bump in (8192, 16384):
                if bump <= tok:
                    continue
                kw2 = dict(kwargs)
                kw2["max_tokens"] = bump
                kw2.pop("extra_body", None)
                _log.warning(
                    "empty content from %s; retry max_tokens=%d no extra_body",
                    config.model,
                    bump,
                )
                resp2 = _call(kw2)
                text = (resp2.choices[0].message.content or "").strip()
                if text:
                    break
    return text, stats


def load_finance_questions(split: str | None = None) -> list[dict]:
    """Map finance items into the shared feed question shape.

    split=None → all; or 'train'/'validation'/'heldout' from manifest.
    """
    m = load_manifest()
    allow: set[str] | None = None
    if split == "train":
        allow = set(m["train_ids"])
    elif split == "validation":
        allow = set(m["validation_ids"])
    elif split == "heldout":
        allow = set(m["heldout_ids"])
    elif split is not None:
        raise ValueError(f"unknown split {split!r}")

    out = []
    for p in _load_raw():
        if allow is not None and p["id"] not in allow:
            continue
        out.append(
            {
                "id": p["id"],
                "question": p["question"],
                # Feed expects expected_sql / db_id keys historically.
                "expected_sql": "",  # no gold free-text in this bench
                "db_id": p["category"],
                "difficulty": "hard",
                "category": p["category"],
                # Rubric stored separately — never copy into feed prompts.
                "_rubric_id": p["id"],
            }
        )
    return out


class FinanceAdapter:
    name = "finance"

    def load_questions(self) -> list[dict]:
        # Default train-only so undifferentiated feeds never mix held-out.
        return load_finance_questions("train")

    def build_feed(self, n: int, full: bool, seed: int) -> list[FeedItem]:
        """Train-stream feed (seeded). Validation/held-out are separate eval paths."""
        import random

        train = load_finance_questions("train")
        rng = random.Random(seed)
        rng.shuffle(train)
        k = len(train) if full else min(n, len(train))
        chosen = train[:k]
        return [
            FeedItem(
                question_id=q["id"],
                question=q["question"],
                gold_output="",
                domain_id=q["db_id"],
                difficulty=q["difficulty"],
                phase="degraded",
            )
            for q in chosen
        ]

    def build_continuous_feed(
        self, n_cycles: int, full: bool, seed: int
    ) -> list[FeedItem]:
        items = self.build_feed(n=80, full=full, seed=seed)
        return items * max(1, n_cycles)

    def run_item(
        self, item: FeedItem, config: AgentConfig, use_rules: bool = True
    ) -> TelemetryRecord | None:
        del use_rules  # KG not wired for finance Phase 0
        from correction.judge import grade

        stems = all_rubric_stems()
        # Firewall before any LLM call.
        build_student_prompt(
            item.question,
            config,
            item.domain_id,
            forbidden_rubric_stems=stems,
        )
        max_tok = int(os.environ.get("STUDENT_MAX_TOKENS", "8192"))
        answer, stats = generate_answer(
            item.question,
            config,
            category=item.domain_id,
            forbidden_rubric_stems=stems,
            max_tokens=max_tok,
        )

        result = grade(
            question=item.question,
            rubric=rubric_for(item.question_id, role="judge"),
            answer=answer,
        )
        normalized = float(result["normalized"])
        return TelemetryRecord(
            run_id=f"{item.question_id}_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            difficulty=Difficulty.HARD,
            execution_accuracy=max(0.0, min(1.0, normalized / 100.0)),
            query_valid=True,
            generated_complexity=0,
            required_complexity=0,
            generated_output=answer,
            gold_output="",
            domain_id=item.domain_id,
            injection_stats=stats,
        )

    def make_examples(
        self,
        failing_cases: list[FailingCase],
        anchor_cases: list[FailingCase],
    ) -> list[FewShotExample]:
        """Phase 0 stub — TraceLift write path uses teacher_repair / distill."""
        del failing_cases, anchor_cases
        return []


# ── TraceLift Task A: teacher repair + memory distillation ───────────────────

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9&'’-]+(?:\s+(?:of|and|the|for|de|du|la|le)){0,1}\s+)"
    r"{1,}[A-Z][a-zA-Z0-9&'’-]+"
    r"(?:\s+(?:Inc|Inc\.|LLC|Ltd|Ltd\.|Corp|Corp\.|Co|Co\.|PLC|LP|LLP))?\b"
)
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_COMMON_CAPS = {
    "THE", "AND", "FOR", "WITH", "FROM", "THAT", "THIS", "WHEN", "WHERE",
    "ASC", "GAAP", "IFRS", "SEC", "FASB", "US", "USA", "CEO", "CFO", "EPS",
    "EBITDA", "ROE", "ROA", "NPV", "IRR", "WACC", "VAT", "FX", "USD", "EUR",
    "ITEM", "TOTAL", "NOTE", "PART", "SECTION", "TABLE", "EXHIBIT", "APPENDIX",
}


def extract_named_entities(text: str) -> list[str]:
    """Heuristic named-entity harvest for leakage scrubbing (not NER-quality)."""
    found: list[str] = []
    for m in _ENTITY_RE.finditer(text):
        s = m.group(0).strip()
        if len(s) >= 6:
            found.append(s)
    for m in _TICKER_RE.finditer(text):
        s = m.group(0)
        if s not in _COMMON_CAPS and len(s) >= 3:
            found.append(s)
    # Stable unique, longest-first so strip replaces rich phrases first.
    uniq = sorted(set(found), key=lambda x: (-len(x), x))
    return uniq


def strip_named_entities(text: str, entities: list[str] | None = None) -> str:
    """Replace named entities with [ENTITY] placeholders."""
    ents = entities if entities is not None else extract_named_entities(text)
    out = text
    for e in ents:
        if e and e in out:
            out = out.replace(e, "[ENTITY]")
    # Collapse repeated placeholders.
    out = re.sub(r"(?:\[ENTITY\]\s*){2,}", "[ENTITY] ", out)
    return out


def _forbidden_question_stems(
    *, splits: tuple[str, ...] = ("validation", "heldout"), n: int = 80
) -> list[str]:
    m = load_manifest()
    ids: list[str] = []
    if "validation" in splits:
        ids.extend(m["validation_ids"])
    if "heldout" in splits:
        ids.extend(m["heldout_ids"])
    stems: list[str] = []
    for qid in ids:
        q = get_problem(qid)["question"].strip()
        stem = q[:n]
        if len(stem) >= 40:
            stems.append(stem)
    return stems


def scrub_leakage(text: str) -> str:
    """Strip entities + remove any long stems that match val/held-out questions."""
    cleaned = strip_named_entities(text)
    for stem in _forbidden_question_stems():
        if stem and stem in cleaned:
            cleaned = cleaned.replace(stem, "[REDACTED_SPLIT_TEXT]")
    return cleaned


def _teacher_max_tokens() -> int:
    raw = (os.environ.get("TEACHER_MAX_TOKENS") or "4000").strip()
    try:
        return max(256, int(raw))
    except ValueError:
        return 4000


_TEACHER_SYSTEM = (
    "You are an expert finance teacher. Produce a corrected expert answer "
    "that would score well against the official rubric. Use the rubric as a "
    "grading guide for completeness — do not paste rubric item IDs into the "
    "answer. Return only the corrected answer."
)


def teacher_repair(
    qid: str,
    student_answer: str,
    *,
    client=None,
    model: str | None = None,
    manifest: dict | None = None,
) -> str:
    """GLM teacher repair on a TRAIN-STREAM failure. Rubric ACL enforced."""
    problem = get_problem(qid)
    rubric = rubric_for(qid, role="teacher", manifest=manifest)
    prompt = build_teacher_prompt(
        problem["question"],
        qid=qid,
        rubric=rubric,
        broken=student_answer,
        manifest=manifest,
    )
    if client is None:
        client, resolved = teacher_client_and_model()
    else:
        resolved = model or os.environ.get("TEACHER_MODEL") or "z-ai/glm-5.2"
    max_tokens = _teacher_max_tokens()
    resp = client.chat.completions.create(
        model=resolved if model is None else model,
        messages=[
            {"role": "system", "content": _TEACHER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        # Thinking models: bump budget once (same pattern as student generate).
        resp2 = client.chat.completions.create(
            model=resolved if model is None else model,
            messages=[
                {"role": "system", "content": _TEACHER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max(max_tokens, 8192),
        )
        text = (resp2.choices[0].message.content or "").strip()
    return text


def _token_trim(text: str, max_tokens: int = 300) -> str:
    toks = text.split()
    if len(toks) <= max_tokens:
        return text.strip()
    return " ".join(toks[:max_tokens]).strip()


def _skeletonize(repaired: str) -> str:
    """Compress a repair into issue → framework → steps → conclusion (≤300 tok)."""
    cleaned = scrub_leakage(repaired)
    # Prefer explicit section headers if present; else take lead paragraphs.
    sections: list[str] = []
    for label in ("Issue", "Framework", "Steps", "Conclusion"):
        m = re.search(
            rf"(?im)^\s*{label}\s*:\s*(.+?)(?=^\s*(?:Issue|Framework|Steps|Conclusion)\s*:|\Z)",
            cleaned,
            flags=re.DOTALL,
        )
        if m:
            sections.append(f"{label}: {m.group(1).strip()}")
    if sections:
        body = "\n".join(sections)
    else:
        # Lead + mid + tail snippets without raw dump.
        paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
        picked = paras[:2]
        if len(paras) > 3:
            picked.append(paras[len(paras) // 2])
        if len(paras) > 1:
            picked.append(paras[-1])
        body = "\n".join(picked) if picked else cleaned
    return _token_trim(body, 300)


def _playbook_from(repaired: str, category: str) -> str:
    # Extract checklist structure BEFORE leakage scrub so entity wipes
    # don't destroy numbering / step lines.
    lines = []
    for raw in repaired.splitlines():
        s = raw.strip()
        if not s:
            continue
        if re.match(r"^(\d+[\).]|[-*•]|\(\d+\))", s) or s.lower().startswith(
            ("step", "check", "gate", "first", "then", "finally", "always", "never")
        ):
            lines.append(s)
    if not lines:
        lines = [_token_trim(repaired, 120)]
    bullets = []
    for x in lines[:12]:
        item = scrub_leakage(x.lstrip("-•* ").strip())
        item = re.sub(r"(?:\[ENTITY\]\s*)+", "[ENTITY] ", item).strip()
        if len(item) >= 8:
            bullets.append(f"- {item}")
    if not bullets:
        bullets = [
            f"- {category}: map facts → governing framework → ordered gates → "
            f"quantitative conclusion; re-check aggregation and single-party rights."
        ]
    body = f"Category playbook ({category}):\n" + "\n".join(bullets)
    return _token_trim(scrub_leakage(body), 300)


def _trap_from(repaired: str, category: str) -> str:
    traps: list[str] = []
    for raw in repaired.splitlines():
        s = raw.strip()
        if re.search(r"(?i)\b(trap|avoid|do not|don't|never|pitfall|mistake)\b", s):
            traps.append(scrub_leakage(s))
    if not traps:
        traps = [
            f"TRAP ({category}): verify framework gates before applying safe harbors "
            f"or shortcuts; re-check aggregation and single-party exercisability."
        ]
    body = "\n".join(f"- {t}" for t in traps[:4] if t.strip())
    return _token_trim(scrub_leakage(body), 300)


_DISTILL_SYSTEM = (
    "You extract reusable, transferable exam lessons for a weaker finance student. "
    "Output ONLY the requested content — no preamble, no meta-commentary."
)


def _teacher_distill(
    category: str,
    repaired: str,
    kind: str,
    *,
    client,
    model: str | None = None,
) -> str:
    """One teacher call → clean, transferable memory content (no question specifics).

    Produces real analytical content instead of the extract-then-scrub path,
    which stripped items down to generic boilerplate. Leak-safe by construction:
    the prompt forbids any entity/number/detail unique to the source problem.
    """
    if kind == "playbook":
        ask = (
            f"Write a reusable analytical PLAYBOOK (4–6 imperative steps) a weaker "
            f"model should follow on ANY {category} problem, generalizing the "
            f"reasoning below. Start with 'Category playbook ({category}):'."
        )
    elif kind == "trap":
        ask = (
            "State the ONE most important named TRAP the analysis below avoids, as "
            "'TRAP: <trigger condition> → <what to do instead>'. One or two sentences."
        )
    else:
        ask = (
            f"Write a compact worked-reasoning SKELETON for {category} problems "
            "(Issue → Framework → 4 ordered steps → Conclusion), generalizing the "
            "approach below. Under 120 words."
        )
    prompt = (
        f"A stronger model produced this correct analysis of a {category} finance "
        f"problem:\n\n{_token_trim(repaired, 900)}\n\n{ask}\n\n"
        "HARD RULES: use ONLY general finance principles, framework names, and "
        "accounting/finance standard citations (e.g. ASC 810, IFRS 10, the VIE "
        "model). Do NOT mention any company name, ticker, person, specific dollar "
        "amount, date, or any detail unique to this particular problem — the lesson "
        "must transfer to other problems in this category."
    )
    resolved = model or os.environ.get("TEACHER_MODEL") or "z-ai/glm-5.2"
    # Reasoning off: we want the distilled content, not the thinking trace.
    resp = client.chat.completions.create(
        model=resolved,
        messages=[
            {"role": "system", "content": _DISTILL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=_teacher_max_tokens(),
        extra_body={"reasoning": {"enabled": False}},
    )
    text = (resp.choices[0].message.content or "").strip()
    return _token_trim(text, 300) if text else ""


def distill_memory_item(
    qid: str,
    repaired: str,
    *,
    kind: str = "skeleton",
    manifest: dict | None = None,
    client=None,
    model: str | None = None,
) -> FewShotExample:
    """Compress a teacher repair into a TraceLift memory item.

    kind: playbook | trap | skeleton. domain_id = category, source = tracelift.
    When `client` is provided, the teacher distills transferable content directly
    (real reasoning). Without a client, falls back to the deterministic extract
    path (used by hermetic tests). Leakage guard strips source-question entities.
    """
    if kind not in _MEMORY_KINDS:
        raise ValueError(f"unknown memory kind {kind!r}")
    # Distillation only from train-stream repairs (firewall).
    assert_rubric_allowed_for_teacher(qid, manifest)
    problem = get_problem(qid)
    category = problem["category"]

    body = ""
    if client is not None:
        body = _teacher_distill(category, repaired, kind, client=client, model=model)

    if not body:
        # Offline / test fallback: deterministic extract from the repair text.
        if kind == "playbook":
            body = _playbook_from(repaired, category)
        elif kind == "trap":
            body = _trap_from(repaired, category)
        else:
            body = _skeletonize(repaired)
        body = scrub_leakage(body)

    # Question side is a category-keyed stub — never the raw train question.
    q_stub = scrub_leakage(f"{_KIND_PREFIX[kind]} {category}")
    # Strip only entities that appear in the SOURCE question (leak protection),
    # not general finance terms — teacher content is already specifics-free.
    src_ents = extract_named_entities(problem["question"])
    body = strip_named_entities(body, src_ents)
    q_stub = strip_named_entities(q_stub, src_ents)

    return FewShotExample(
        question=q_stub,
        correct_output=_token_trim(body, 300),
        domain_id=category,
        source="tracelift",
    )
