#!/usr/bin/env python3
"""Generate docs/demo.gif from a recorded events.jsonl (no browser required).

Reads VIEWER_LOG or fixtures/demo_events.jsonl, plots windowed hard-bucket accuracy,
and animates drift/correction markers. Requires pillow (pip install pillow).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from viewer.app import build_state  # noqa: E402


def main() -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Install pillow: pip install pillow", file=sys.stderr)
        sys.exit(1)

    log = Path(os.environ.get("VIEWER_LOG", REPO / "fixtures" / "demo_events.jsonl"))
    out = REPO / "docs" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    state = build_state(log)
    runs = state["runs"]
    if not runs:
        print(f"No telemetry in {log}", file=sys.stderr)
        sys.exit(1)

    w, h = 900, 420
    pad_l, pad_r, pad_t, pad_b = 56, 24, 36, 48
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def y_scale(v: float) -> int:
        return pad_t + int((1.0 - v) * plot_h)

    def x_scale(i: int) -> int:
        n = len(runs) - 1 or 1
        return pad_l + int(i / n * plot_w)

    frames: list[Image.Image] = []
    step = max(1, len(runs) // 80)
    marks = []
    for d in state.get("drifts") or ([state["drift"]] if state.get("drift") else []):
        if d:
            marks.append((d["at"], "#f85149", "drift"))
    for c in state.get("corrections") or ([state["correction"]] if state.get("correction") else []):
        if c:
            marks.append((c["at"], "#3fb950", "learn"))

    for end in range(0, len(runs), step):
        img = Image.new("RGB", (w, h), "#0b0f14")
        draw = ImageDraw.Draw(img)
        draw.text((pad_l, 8), "Agent Self-Improvement — hard/extra execution accuracy", fill="#e6edf3")
        draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill="#30363d")
        draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill="#30363d")

        pts = []
        for i, r in enumerate(runs[: end + 1]):
            yv = r.get("acc_hard")
            if yv is None:
                continue
            pts.append((x_scale(i), y_scale(yv)))
        if len(pts) >= 2:
            draw.line(pts, fill="#58a6ff", width=2)

        for at, color, _ in marks:
            if at <= end:
                x = x_scale(at)
                draw.line([(x, pad_t), (x, pad_t + plot_h)], fill=color, width=2)

        draw.text((pad_l, h - 28), f"run {end}/{len(runs)-1}", fill="#8b949e")
        frames.append(img)

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
