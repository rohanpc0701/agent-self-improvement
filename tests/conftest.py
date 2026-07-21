"""Shared test fixtures."""
import os
import pytest


@pytest.fixture(autouse=True)
def _reset_finance_state():
    """Finance adapter memoizes dataset/manifest in module globals and reads
    manifest-path env vars; reset both between tests so one test's fixture can't
    leak into another (fixes a rubric-firewall test-ordering flake)."""
    import adapters.finance as fin
    for var in ("FINANCE_MANIFEST", "HELDOUT_MANIFEST"):
        os.environ.pop(var, None)
    fin._BY_ID = None
    fin._MANIFEST_CACHE = None
    yield
    fin._BY_ID = None
    fin._MANIFEST_CACHE = None
