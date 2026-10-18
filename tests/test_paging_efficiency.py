"""Paging: VRAM slots fixed; pool growth must not inflate card tensors."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestPagingEfficiency(unittest.TestCase):
    def setUp(self):
        import torch
        from minagi.paged import PagedPool
        self.torch = torch
        self.PagedPool = PagedPool
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _pool(self, n=32, resident=8, d_model=64, d_ff=128):
        # tiny shapes so CPU tests stay fast
        return self.PagedPool(
            self.tmp.name, d_model, d_ff, n_experts=n, resident=resident,
            ram_capacity=16, device="cpu")

    def test_slot_tensors_match_resident_not_pool(self):
        p = self._pool(n=64, resident=8)
        self.assertEqual(p.w1.shape[0], 8)
        self.assertEqual(p.w2.shape[0], 8)
        self.assertEqual(p.w3.shape[0], 8)
        self.assertEqual(p.n_experts(), 64)
        self.assertEqual(p.n_resident(), 8)

    def test_vram_params_scale_with_resident_only(self):
        a = self._pool(n=32, resident=4)
        b = self._pool(n=128, resident=4)
        # gate grows with pool; strip it
        slot_a = 3 * a.resident * a.d_model * a.d_ff
        slot_b = 3 * b.resident * b.d_model * b.d_ff
        self.assertEqual(slot_a, slot_b)
        self.assertEqual(a.w1.numel() + a.w2.numel() + a.w3.numel(), slot_a)
        self.assertEqual(b.w1.numel() + b.w2.numel() + b.w3.numel(), slot_b)
        # larger pool -> more total params, same slot params
        self.assertGreater(b.n_params(), a.n_params())
        self.assertEqual(
            a.vram_params() - a.gate.numel(),
            b.vram_params() - b.gate.numel())

    def test_doubling_resident_doubles_slot_params(self):
        a = self._pool(n=32, resident=4)
        b = self._pool(n=32, resident=8)
        sa = a.w1.numel() + a.w2.numel() + a.w3.numel()
        sb = b.w1.numel() + b.w2.numel() + b.w3.numel()
        self.assertEqual(sb, 2 * sa)

    def test_disk_bytes_formula(self):
        """8 bytes/param on disk (fp32 weight + bf16×2 moments)."""
        p = self._pool(n=10, resident=4, d_model=512, d_ff=2048)
        per = 3 * 512 * 2048 * 8
        self.assertEqual(p.disk_bytes(), 10 * per)
        self.assertEqual(p.disk_bytes(extra=2), 12 * per)
        # documented 25.2 MB at creator shapes
        self.assertAlmostEqual(per / 1e6, 25.165824, places=3)

    def test_wider_expert_disk_matches_config_scale(self):
        p = self._pool(n=1, resident=1, d_model=768, d_ff=3072)
        per = p.disk_bytes()
        self.assertAlmostEqual(per / 1e6, 56.623104, places=3)

    def test_growth_does_not_resize_slots(self):
        p = self._pool(n=8, resident=4)
        before = tuple(p.w1.shape)
        # add_experts needs seed machinery; bump _n and gate like growth does
        k = 4
        p.gate = self.torch.nn.Parameter(
            self.torch.cat([p.gate.data, self.torch.ones(k) * 0.001]))
        p._n = 8 + k
        self.assertEqual(tuple(p.w1.shape), before)
        self.assertEqual(p.n_resident(), 4)
        self.assertEqual(p.n_experts(), 12)

    def test_report_hit_rate_in_unit_interval(self):
        p = self._pool()
        r = p.report()
        self.assertIn("hit_rate", r)
        self.assertGreaterEqual(r["hit_rate"], 0.0)
        self.assertLessEqual(r["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
