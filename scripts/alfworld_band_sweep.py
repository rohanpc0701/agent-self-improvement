#!/usr/bin/env python3
"""ALFWorld student band sweep — B1 spec §3.

Runs N bare ReAct episodes per candidate model on valid_unseen; reports
success rate. Required band for the B1 student: 0.30–0.60 unaided.

Usage:
    export ALFWORLD_DATA=~/.cache/alfworld   # after `alfworld-download`
    python3 scripts/alfworld_band_sweep.py --models mistralai/mistral-nemo \
        --episodes 30 --max-steps 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from adapters.coding import _chat_with_retry  # noqa: E402  (shared retry)

_SYSTEM = (
    "You are an agent in a text-based household environment. At each turn you "
    "receive an observation and a list of admissible commands. Reply with "
    "EXACTLY ONE admissible command — no explanation, no quotes, nothing else."
)

# Generic action-grammar exemplar (constant scaffolding for ALL models/arms —
# NOT learned memory; teaches the command format, not any specific task).
_FORMAT_EXEMPLAR = """Example of how to act (a different task):
Task: put a mug in the desk.
> go to shelf 1
On the shelf 1, you see a mug 2.
> take mug 2 from shelf 1
You pick up the mug 2.
> go to desk 1
On the desk 1, you see a laptop 1.
> put mug 2 in/on desk 1
You put the mug 2 on the desk 1. Task completed.

Now solve YOUR task. One admissible command per reply."""


def _make_env(split: str):
    """Build a TextWorld ALFWorld env. Requires ALFWORLD_DATA to be populated."""
    import yaml

    from alfworld.agents.environment import get_environment

    cfg_path = os.environ.get("ALFWORLD_CONFIG")
    if cfg_path:
        config = yaml.safe_load(open(cfg_path))
    else:
        # Minimal config: TextWorld-only (no THOR), goal-desc task strings.
        config = {
            "env": {
                "type": "AlfredTWEnv",
                "regen_game_files": False,
                "domain_randomization": False,
                "task_types": [1, 2, 3, 4, 5, 6],
                "goal_desc_human_anns_prob": 0.0,
                "expert_timeout_steps": 150,
                "expert_type": "handcoded",
            },
            "dataset": {
                "data_path": "$ALFWORLD_DATA/json_2.1.1/train",
                "eval_id_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_seen",
                "eval_ood_data_path": "$ALFWORLD_DATA/json_2.1.1/valid_unseen",
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": "$ALFWORLD_DATA/logic/alfred.pddl",
                "grammar": "$ALFWORLD_DATA/logic/alfred.twl2",
            },
            "general": {
                "random_seed": 42,
                "use_cuda": False,
                "training_method": "dagger",
                "save_path": "/tmp/alfworld_out",
                "training": {"batch_size": 1},
            },
            "dagger": {"training": {"max_nb_steps_per_episode": 50}},
            "rl": {"training": {"max_nb_steps_per_episode": 50}},
        }
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    return env.init_env(batch_size=1)


def _pick_action(client, model: str, history: list[dict]) -> str:
    resp = _chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM + "\n\n" + _FORMAT_EXEMPLAR}
        ] + history[-12:],
        temperature=0.0,
        max_tokens=64,
    )
    return (resp.choices[0].message.content or "").strip().splitlines()[0].strip()


def run_episodes(model: str, n_episodes: int, max_steps: int) -> tuple[int, list[dict]]:
    client = OpenAI(
        api_key=os.environ["PRIME_API_KEY"],
        base_url=os.environ.get("AGENT_BASE_URL", "https://api.pinference.ai/api/v1"),
    )
    env = _make_env("eval_out_of_distribution")
    wins = 0
    stats: list[dict] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ob = obs[0]
        history: list[dict] = []
        won = False
        invalid = 0
        t0 = time.time()

        for step in range(max_steps):
            cmds = info["admissible_commands"][0]
            user = f"Observation: {ob}\nAdmissible commands:\n" + "\n".join(
                f"- {c}" for c in cmds
            )
            history.append({"role": "user", "content": user})
            action = _pick_action(client, model, history)
            history.append({"role": "assistant", "content": action})
            if action not in cmds:
                invalid += 1
                # feed back the error once; count as invalid action
                ob = f"Invalid command: {action!r}. Choose one admissible command."
                continue
            obs, scores, dones, info = env.step([action])
            ob = obs[0]
            if dones[0]:
                won = bool(info["won"][0])
                break

        wins += int(won)
        stats.append(
            {"episode": ep, "won": won, "steps": step + 1, "invalid": invalid,
             "secs": round(time.time() - t0, 1)}
        )
        print(
            f"  [{ep + 1:>2}/{n_episodes}] {'✓' if won else '✗'} "
            f"steps={step + 1} invalid={invalid}",
            flush=True,
        )
    return wins, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    print(f"{'=' * 60}\n  ALFWORLD BAND SWEEP (valid_unseen, bare student)\n{'=' * 60}")
    results = {}
    for m in args.models:
        print(f"\n== {m}")
        wins, _ = run_episodes(m, args.episodes, args.max_steps)
        rate = wins / args.episodes
        results[m] = rate
        verdict = "IN BAND" if 0.3 <= rate <= 0.6 else (
            "too weak" if rate < 0.3 else "too strong"
        )
        print(f"  success={rate:.3f} ({wins}/{args.episodes}) — {verdict}")

    print(f"\n{'=' * 60}")
    for m, r in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {r:.3f}  {m}")
    print("=" * 60)


if __name__ == "__main__":
    main()
