"""Hermetic tests for paired_bootstrap."""
from __future__ import annotations

from analysis.bootstrap import paired_bootstrap


class TestPairedBootstrap:
    def test_all_equal_high_p(self):
        a = [1.0, 0.0, 1.0, 0.0, 1.0] * 10
        out = paired_bootstrap(a, list(a), n_boot=2000, seed=0)
        assert abs(out["delta"]) < 1e-12
        assert out["p_value"] > 0.5

    def test_clear_separation_low_p(self):
        a = [0.0] * 40
        b = [1.0] * 40
        out = paired_bootstrap(a, b, n_boot=2000, seed=1)
        assert out["delta"] == 1.0
        assert out["p_value"] < 0.05
        assert out["ci_low"] > 0.5

    def test_deterministic(self):
        a = [0.0, 1.0, 0.0, 1.0]
        b = [1.0, 1.0, 0.0, 1.0]
        x = paired_bootstrap(a, b, n_boot=500, seed=7)
        y = paired_bootstrap(a, b, n_boot=500, seed=7)
        assert x == y
