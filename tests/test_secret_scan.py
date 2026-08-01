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


# --- regressions for the audit findings (2026-08-01) ---------------------------
# The original rule had a leading \b before the credential keyword. '_' is a word
# character, so no boundary exists in MINIMAX_API_KEY= and NO prefixed variable name
# could ever match -- while the only test then present used a bare `api_key`, so the
# suite stayed green over a scanner blind to every real-world name.

FAKE_LONG = "e" * 120   # MiniMax-length opaque value
FAKE_MED = "z" * 64     # Prime-length opaque value


def test_prefixed_variable_names_match():
    """MINIMAX_API_KEY / PRIME_API_KEY etc. -- not just a bare `api_key`."""
    for name in ("MINIMAX_API_KEY", "PRIME_API_KEY", "OPENROUTER_API_KEY",
                 "JUDGE_API_KEY", "SOME_NEW_PROVIDER_TOKEN"):
        assert scan_text("x.env", f"{name}={FAKE_LONG}"), f"missed prefixed name {name}"


def test_unquoted_dotenv_and_export_forms():
    """A leaked .env copy has no quotes around values."""
    assert scan_text(".env.bak", f"MINIMAX_API_KEY={FAKE_LONG}")
    assert scan_text("setup.sh", f"export PRIME_API_KEY={FAKE_MED}")


def test_all_contexts_a_leaked_key_appears_in():
    name, val = "PRIME_API_KEY", FAKE_MED
    for ctx in (f"{name}={val}", f"export {name}={val}", f'{name} = "{val}"',
                f'{{"{name}": "{val}"}}', f"{name}: {val}"):
        assert scan_text("x", ctx), f"missed context: {ctx[:40]}"


def test_non_credential_config_is_not_flagged():
    """Model slugs, URLs and numbers must stay quiet or the hook gets bypassed."""
    for line in ("AGENT_MODEL=qwen/qwen3.6-27b",
                 "AGENT_BASE_URL=https://openrouter.ai/api/v1",
                 "TEACHER_MAX_TOKENS=32000",
                 "OPENROUTER_PROVIDER_ORDER=deepinfra,alibaba"):
        assert not scan_text(".env.example", line), f"false positive: {line}"


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
    print("secret scan tests: OK")
