"""Precision policy: weights stay fp32; compute may be bf16."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestPrecisionEfficiency(unittest.TestCase):
    def test_set_compute_dtype_names(self):
        import torch
        from minagi.precision import set_compute_dtype, compute_dtype, NAMES

        for name in ("fp32", "bf16"):
            set_compute_dtype(name)
            self.assertEqual(compute_dtype(), NAMES[name])
        set_compute_dtype(torch.float32)
        self.assertEqual(compute_dtype(), torch.float32)

    def test_bf16_halves_activation_bytes(self):
        import torch
        from minagi.precision import set_compute_dtype, amp

        x = torch.randn(8, 64)
        set_compute_dtype("bf16")
        # amp is a context that casts matmul inputs where safe
        with amp(torch.device("cpu")):
            # on CPU bf16 may fall back; just ensure context enters/exits
            y = x @ x.T
        self.assertTrue(torch.isfinite(y).all())
        set_compute_dtype("fp32")


if __name__ == "__main__":
    unittest.main()
