"""Tiny forward: activated params stay << pool; step finishes on CPU."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestForwardEfficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from unittest import mock
        from minagi.create import create
        from minagi.precision import set_compute_dtype
        from train import build_paged

        set_compute_dtype("fp32")
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory()
        # tiny model so CPU CI stays under a few seconds
        cfg = {
            "model": {"d_model": 64, "n_head": 4, "d_ff": 128,
                      "n_prelude": 1, "n_recur": 1, "n_coda": 0,
                      "max_steps": 4, "min_steps": 1,
                      "halt_prior": 0.25, "halt_thresh": 0.9,
                      "halt_freeze": True, "ponder_beta": 0.01,
                      "bptt_window": 4, "train_steps_mean": 2.0,
                      "context_end": 64},
            "pool": {"experts": 16, "resident": 4, "width": 64,
                     "depth": 1, "top_k": 2, "balance": 0.0,
                     "select_temperature": 0},
        }
        with mock.patch("minagi.config.load", return_value=cfg):
            create(cls.tmp.name, verbose=False, force=True)
        model, rcfg, pool, man = build_paged(
            cls.tmp.name, torch.device("cpu"), resident=4, ram_capacity=16,
            ceiling=64)
        cls.model = model
        cls.pool = pool
        cls.cfg = rcfg

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_activated_experts_bounded_by_resident(self):
        """VRAM cost tracks resident slots, not pool size."""
        slot = 3 * self.pool.resident * self.pool.d_model * self.pool.d_ff
        pool = self.pool.n_params() - self.pool.gate.numel()
        self.assertLess(slot, pool)
        self.assertEqual(
            self.pool.w1.numel() + self.pool.w2.numel() + self.pool.w3.numel(),
            slot)
        # pool has 16 experts, card 4 → at most 25% of expert weights on card
        self.assertLessEqual(slot / pool, 0.26)

    def test_train_step_completes_fast_on_cpu(self):
        torch = self.torch
        model = self.model
        model.train()
        B, T = 1, 32
        x = torch.randint(0, 265, (B, T))
        y = torch.randint(0, 265, (B, T))
        t0 = time.perf_counter()
        logits, loss = model(x, targets=y)
        loss.backward()
        elapsed = time.perf_counter() - t0
        self.assertEqual(logits.shape[-1], 265)
        self.assertTrue(torch.isfinite(loss))
        # loose budget: tiny model on CPU should finish well under 30s
        self.assertLess(elapsed, 30.0, f"step took {elapsed:.1f}s")

    def test_eval_forward_no_grad_cheaper_graph(self):
        torch = self.torch
        model = self.model
        model.eval()
        x = torch.randint(0, 265, (1, 32))
        with torch.no_grad():
            logits, _ = model(x)
        self.assertIsNone(logits.grad_fn)
        self.assertEqual(logits.shape, (1, 32, 265))

    def test_depth_recorded(self):
        torch = self.torch
        model = self.model
        model.train()
        x = torch.randint(0, 265, (1, 16))
        y = torch.randint(0, 265, (1, 16))
        model(x, targets=y)
        self.assertTrue(hasattr(model, "last_steps"))
        self.assertGreater(float(model.last_steps), 0.0)
        self.assertLessEqual(float(model.last_steps), self.cfg.max_steps)


if __name__ == "__main__":
    unittest.main()
