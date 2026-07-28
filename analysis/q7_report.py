"""Q7 report — 4-cell table, paired deltas, mechanical verdict. No API calls."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "q7" / "state.json"
OUT = ROOT / "runs" / "q7" / "RESULTS.md"


def _cells() -> dict:
    rows = {}
    for line in STATE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["key"]] = r          # last write wins
    return {k: r for k, r in rows.items() if k.startswith("cell|") and r.get("ok")}


def _per_question(cells: dict) -> dict[str, dict[str, float]]:
    """tag -> {'M': k-avg, 'B': k-avg}"""
    acc: dict[str, dict[str, list[float]]] = {}
    for r in cells.values():
        acc.setdefault(r["tag"], {}).setdefault("M" if r["mem"] else "B", []).append(r["normalized"])
    return {t: {c: mean(v) for c, v in d.items()} for t, d in acc.items()}


def _paired_delta(per: dict, prefix: str) -> list[float]:
    return [d["M"] - d["B"] for t, d in per.items()
            if t.startswith(prefix) and "M" in d and "B" in d]


def _boot(deltas: list[float], n_boot: int = 10_000, seed: int = 0) -> dict:
    import random
    rng = random.Random(seed)
    n = len(deltas)
    boots = sorted(mean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    point = mean(deltas)
    p = 2 * min(sum(1 for b in boots if b <= 0), sum(1 for b in boots if b >= 0)) / n_boot
    return {"delta": point, "lo": boots[int(0.025 * n_boot)],
            "hi": boots[int(0.975 * n_boot)], "p": min(1.0, p), "n": n}


def report() -> None:
    cells = _cells()
    per = _per_question(cells)
    dv, du = _paired_delta(per, "V:"), _paired_delta(per, "U:")
    lines = ["# Q7 results\n", f"cells: {len(cells)} · questions: {len(per)}"]
    for name, ds in (("delta_V (variants)", dv), ("delta_U (unrelated)", du)):
        if not ds:
            lines.append(f"- {name}: no data")
            continue
        b = _boot(ds)
        lines.append(f"- {name}: {b['delta']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] "
                     f"p={b['p']:.3f} n={b['n']} sd={stdev(ds) if len(ds) > 1 else 0:.1f}")
    if dv and du:
        cross = _boot([mean(dv) - mean(du)] and dv + [-x for x in du])  # pooled crossover
        lines.append(f"- crossover (delta_V − delta_U): {mean(dv) - mean(du):+.2f}")
    # mechanical verdict per prereg
    verdict = "AMBIGUOUS"
    if dv:
        b = _boot(dv)
        if b["delta"] >= 3 and b["lo"] > 0 and (not du or mean(du) <= 0):
            verdict = "WORKS"
        elif b["lo"] <= 0 <= b["hi"]:
            verdict = "DEAD" if b["delta"] < 3 else "AMBIGUOUS"
    lines.append(f"\n**VERDICT (mechanical, per prereg): {verdict}**")
    errs = sum(1 for line in STATE.read_text().splitlines()
               if line.strip() and not json.loads(line).get("ok")
               and json.loads(line)["key"].startswith("cell|"))
    lines.append(f"\nerror cells: {errs} (excluded, never zeros)")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    report()
