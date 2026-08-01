#!/usr/bin/env python3
"""Block credentials from entering the repo. Runs as a pre-commit hook and in CI.

Scans staged content (hook) or tracked files (CI). Exits non-zero on a hit and prints
file:line with the value REDACTED — a scanner that echoes the secret it caught just moves
the leak into your terminal scrollback and CI logs.

    python3 scripts/secret_scan.py --staged   # pre-commit
    python3 scripts/secret_scan.py --tracked  # CI
    python3 scripts/secret_scan.py --history  # every commit, slow

Adding a pattern: put the *shape* of the credential here, never an example of a real one.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (name, compiled pattern). Kept deliberately tight: a scanner nobody trusts gets bypassed.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private key block", re.compile(r"-----BEGIN (?:OPENSSH|RSA|DSA|EC|PGP) PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("OpenRouter key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}\b")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    # Generic assignment: KEY = "long-opaque-string". Requires a credential-ish name AND
    # a value that is long and not obviously a placeholder.
    (
        "credential assignment",
        re.compile(
            r"""(?ix)
            # Name: allow provider prefixes (MINIMAX_API_KEY, PRIME_API_KEY, OR_TOKEN).
            # A leading \b here would be a bug: '_' is a word character, so there is NO
            # boundary between the prefix and the keyword, and every prefixed name --
            # i.e. almost every real one -- would silently never match.
            # Include quotes in the boundary so JSON keys ("MINIMAX_API_KEY": "...") match.
            (?:^|[\s,;{(\["'])(?:[A-Za-z0-9]+[_-])*
            (?:api[_-]?key|secret|token|passwd|password|access[_-]?key|private[_-]?key)
            # Optional closing quote: in JSON the name is quoted ("API_KEY": "value"), so
            # the quote sits between the name and the separator.
            ['"]?\s*[:=]\s*
            # Quotes optional: unquoted `KEY=value` is the dotenv and `export` form, which
            # is exactly how a leaked .env copy looks.
            ['"]?
            # Every alternative below must be non-empty. One that can match the empty
            # string (e.g. \s*) makes this negative lookahead always succeed, which
            # silently disables the whole rule.
            (?!(?:\s+|x{3,}|\.{3,}|<[^>]+>|\$\{|%\(|your[_-]|example|placeholder|redacted|changeme|dummy|fake|none|null))
            ([A-Za-z0-9_\-/+=.]{24,})
            ['"]?
            """
        ),
    ),
    # Long opaque bearer-ish values assigned to a credential-named variable are caught by
    # the rule above regardless of provider, so new providers are covered without needing
    # their prefix enumerated here. Provider-specific rules stay only where the shape is
    # distinctive enough to catch the value even OUTSIDE an assignment (e.g. pasted into
    # prose or JSON).
]

# Files that legitimately describe credential *shapes* — this scanner, its test, and docs
# explaining env setup. Scanning them would make the tool flag itself.
SKIP_PATHS = {
    "scripts/secret_scan.py",
    "tests/test_secret_scan.py",
    ".github/workflows/security.yml",
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".sqlite", ".db", ".ico", ".webm"}
# Real experiment data: model answers can coincidentally match loose patterns.
# They are still scanned for hard credential shapes, just not the generic assignment rule.
DATA_PREFIXES = ("runs/", "fixtures/")


def _run(args: list[str]) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def _redact(match: str) -> str:
    if len(match) <= 8:
        return "*" * len(match)
    return f"{match[:4]}…{match[-2:]} ({len(match)} chars, value redacted)"


def scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    hits = []
    is_data = path.startswith(DATA_PREFIXES)
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(line) > 8000:  # minified/base64 blobs: check only hard patterns
            line = line[:8000]
        for name, pat in PATTERNS:
            if is_data and name == "credential assignment":
                continue
            m = pat.search(line)
            if m:
                val = m.group(m.lastindex or 0)
                hits.append((path, lineno, name, _redact(val)))
    return hits


def _should_skip(path: str) -> bool:
    return path in SKIP_PATHS or Path(path).suffix.lower() in SKIP_SUFFIXES


def scan_staged() -> list[tuple[str, int, str, str]]:
    files = [f for f in _run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]).split("\n") if f]
    hits = []
    for f in files:
        if _should_skip(f):
            continue
        blob = _run(["git", "show", f":{f}"])
        if blob:
            hits += scan_text(f, blob)
    return hits


def scan_tracked() -> list[tuple[str, int, str, str]]:
    files = [f for f in _run(["git", "ls-files"]).split("\n") if f]
    hits = []
    for f in files:
        if _should_skip(f):
            continue
        p = ROOT / f
        try:
            hits += scan_text(f, p.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, UnicodeDecodeError):
            continue
    return hits


def scan_history() -> list[tuple[str, int, str, str]]:
    hits = []
    for name, pat in PATTERNS:
        if name == "credential assignment":
            continue  # too noisy across full history
        out = _run(["git", "log", "--all", "--oneline", "-S", pat.pattern, "--pickaxe-regex"])
        for line in out.splitlines():
            if line.strip():
                hits.append((f"git history: {line.strip()}", 0, name, "see commit"))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    if args.history:
        hits = scan_history()
    elif args.tracked:
        hits = scan_tracked()
    else:
        hits = scan_staged()

    if not hits:
        print("secret scan: clean")
        return 0

    print("SECRET SCAN FAILED — credential-shaped content found:\n", file=sys.stderr)
    for path, lineno, name, redacted in hits:
        loc = f"{path}:{lineno}" if lineno else path
        print(f"  {loc}\n      {name}: {redacted}", file=sys.stderr)
    print(
        "\nIf a real credential reached a commit, rotating it is the only real fix — "
        "removing the line does not un-leak it.\n"
        "False positive? Add the file to SKIP_PATHS in scripts/secret_scan.py.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
