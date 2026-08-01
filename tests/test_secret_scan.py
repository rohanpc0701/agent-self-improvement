"""The secret scanner must catch real credential shapes and stay quiet on research data.

A scanner that cries wolf gets bypassed, and a scanner that misses is worse than none — so
both directions are tested.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from secret_scan import scan_text  # noqa: E402

# Synthetic, structurally valid, never issued. Built at runtime so this file itself
# contains no contiguous credential-shaped literal for the scanner to flag.
FAKE_OPENAI = "sk-" + "A1b2C3d4E5f6G7h8I9j0" + "KLMNOP"
FAKE_OPENROUTER = "sk-or-v1-" + "0" * 32
FAKE_AWS = "AKIA" + "1234567890ABCDEF"
FAKE_GH = "ghp_" + "z" * 36


def _names(hits):
    return {h[2] for h in hits}


def test_catches_common_key_shapes():
    assert _names(scan_text("x.py", f'KEY = "{FAKE_OPENAI}"'))
    assert _names(scan_text("x.py", f'K="{FAKE_OPENROUTER}"'))
    assert "AWS access key id" in _names(scan_text("x.py", f"id = {FAKE_AWS}"))
    assert "GitHub token" in _names(scan_text("x.py", f"tok={FAKE_GH}"))


def test_catches_private_key_block():
    hits = scan_text("id_rsa", "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n")
    assert "private key block" in _names(hits)


def test_catches_generic_credential_assignment():
    hits = scan_text("cfg.py", 'api_key = "' + "q7RtY2wPzX4mNb8vC1sD6fG9hJ0kL3aE" + '"')
    assert "credential assignment" in _names(hits)


def test_never_prints_the_secret_value():
    hits = scan_text("x.py", f'KEY = "{FAKE_OPENAI}"')
    redacted = hits[0][3]
    assert FAKE_OPENAI not in redacted, "the scanner must not echo the credential"
    assert "redacted" in redacted


def test_quiet_on_placeholders_and_env_docs():
    for line in (
        'OPENROUTER_API_KEY=your-key-here',
        'api_key = "<YOUR_KEY>"',
        'export MINIMAX_API_KEY=sk-...',
        'password = "changeme"',
        'token: ${GITHUB_TOKEN}',
    ):
        assert not scan_text("README.md", line), f"false positive on: {line}"


def test_quiet_on_research_data_prose():
    # Model answers are long opaque text; they must not trip the generic rule.
    line = 'answer = "The WACC calculation requires unlevering beta across comparable firms"'
    assert not scan_text("runs/q1_state_shard0.json", line)


def test_hard_patterns_still_apply_to_data_files():
    # A real key pasted into run state must still be caught.
    hits = scan_text("runs/q1_state_shard0.json", f'{{"note": "{FAKE_OPENROUTER}"}}')
    assert "OpenRouter key" in _names(hits)


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print("secret scan tests: OK")
