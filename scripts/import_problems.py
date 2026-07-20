#!/usr/bin/env python3
"""Import MBPP+/HumanEval+ problems into the coding fixture.

Stages:
  fetch  — pull via evalplus, convert, sandbox-validate golds
           → fixtures/imported_candidates.json
  probe  — 3B student k=2 @ temp 0.7 per candidate; pass-rate <= 0.5 → hard;
           append survivors to fixtures/coding_subset.json

Licenses: MBPP+ (Apache-2.0), HumanEval+ (MIT) via the evalplus package.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.sandbox import execution_accuracy  # noqa: E402

CANDIDATES_PATH = ROOT / "fixtures" / "imported_candidates.json"
FIXTURE_PATH = ROOT / "fixtures" / "coding_subset.json"

_TOPIC_KEYWORDS = [
    (
        "dp",
        [
            "subsequence",
            "knapsack",
            "dynamic",
            "climb",
            "coin",
            "partition",
            "edit distance",
            "longest increasing",
            "minimum cost",
            "ways to",
            "memo",
            "fibonacci",
            "dp[",
            "tabulat",
        ],
    ),
    (
        "graphs",
        [
            "graph",
            "node",
            "edge",
            "tree",
            "path",
            "bfs",
            "dfs",
            "island",
            "connected",
            "adjacency",
            "binary tree",
            "traverse",
            "parent",
            "child",
        ],
    ),
    (
        "strings",
        [
            "string",
            "palindrome",
            "substring",
            "anagram",
            "character",
            "vowel",
            "letter",
            "word",
            "text",
            "str.",
            "lower()",
            "upper()",
            "split(",
            "join(",
            "regex",
            "bracket",
            "parenthes",
        ],
    ),
    (
        "greedy",
        [
            "greedy",
            "interval",
            "minimum number of",
            "maximum profit",
            "jump",
            "schedule",
            "earliest",
            "latest",
        ],
    ),
    (
        "arithmetic",
        [
            "sum of two",
            "digit",
            "prime",
            "factorial",
            "gcd",
            "lcm",
            "power",
            "arithmetic",
            "multiply",
            "divide",
            "modulo",
            "integer",
            "even",
            "odd",
            "median",
            "average",
            "mean",
            "percentage",
            "sqrt",
            "abs(",
        ],
    ),
]


def label_topic(question: str, code: str) -> str:
    text = f"{question} {code}".lower()
    for topic, kws in _TOPIC_KEYWORDS:
        if any(kw in text for kw in kws):
            return topic
    return "arrays"


def _json_roundtrip_safe(value) -> bool:
    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


def _docstring_question(prompt: str, entry_point: str) -> str:
    """Extract the natural-language docstring of entry_point as the question."""
    try:
        tree = ast.parse(prompt + "    pass\n")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                doc = ast.get_docstring(node)
                if doc:
                    return re.split(r"\n\s*>>>", doc)[0].strip()
    except SyntaxError:
        pass
    return prompt.strip()


def convert_evalplus_item(item: dict, source: str) -> dict | None:
    entry = item["entry_point"]
    gold = item["prompt"] + item["canonical_solution"]
    inputs = item.get("base_input") or []
    if not inputs:
        return None

    tests = []
    for args in inputs[:8]:  # cap tests per problem
        if not _json_roundtrip_safe(args):
            return None
        # Normalize nested lists for call; keep fixture args as original structure.
        if not isinstance(args, list):
            return None
        acc_ns: dict = {}
        try:
            exec(gold, acc_ns)  # trusted benchmark gold, local machine only
            expected = acc_ns[entry](*[json.loads(json.dumps(a)) for a in args])
        except Exception:
            return None
        if not _json_roundtrip_safe(expected):
            return None
        tests.append({"args": args, "expected": expected})
    if not tests:
        return None

    slug = re.sub(r"\W+", "_", item["task_id"].lower()).strip("_")
    question = _docstring_question(item["prompt"], entry)
    return {
        "id": f"h2_{slug}",
        "question": question,
        "function_name": entry,
        "tests": tests,
        "topic": label_topic(question, gold),
        "difficulty": "hard",  # provisional; probe stage finalizes
        "gold_solution": gold.rstrip() + "\n",
        "source": source,
    }


def fetch() -> None:
    from collections import Counter

    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    existing_names = set()
    for p in json.loads(FIXTURE_PATH.read_text()):
        existing_names.add(p["function_name"])

    candidates: list[dict] = []
    for source, loader in (
        ("humaneval+", get_human_eval_plus),
        ("mbpp+", get_mbpp_plus),
    ):
        for item in loader().values():
            p = convert_evalplus_item(item, source=source)
            if p is None or p["function_name"] in existing_names:
                continue
            acc, _, err = execution_accuracy(
                p["gold_solution"], p["function_name"], p["tests"]
            )
            if acc != 1.0:
                print(f"  drop {p['id']}: gold fails converted tests ({err!r})")
                continue
            candidates.append(p)

    CANDIDATES_PATH.write_text(json.dumps(candidates, indent=1))
    print(f"{len(candidates)} candidates → {CANDIDATES_PATH}")
    print("topics:", dict(Counter(c["topic"] for c in candidates)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["fetch", "probe"])
    ap.add_argument("--k", type=int, default=2, help="probe samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--max-keep",
        type=int,
        default=400,
        help="cap on new hard problems appended to the fixture (default 400)",
    )
    ap.add_argument("--offset", type=int, default=0, help="probe chunk start index")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="probe at most this many candidates in this invocation",
    )
    ap.add_argument(
        "--no-merge",
        action="store_true",
        help="probe chunk only; do not append to coding_subset.json yet",
    )
    args = ap.parse_args()
    if args.stage == "fetch":
        fetch()
    else:
        from scripts.import_problems_probe import probe

        probe(
            k=args.k,
            temperature=args.temperature,
            max_keep=args.max_keep,
            offset=args.offset,
            limit=args.limit,
            merge=not args.no_merge,
        )


if __name__ == "__main__":
    main()
