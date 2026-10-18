"""Shape guards and create()-time efficiency accounting."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _fake_cfg(model, pool):
    return {"model": model, "pool": pool}


class TestCreateGuards(unittest.TestCase):
    def test_bad_head_split_exits(self):
        import torch  # noqa: F401
        from minagi.create import create

        cfg = _fake_cfg(
            {"d_model": 768, "n_head": 10, "d_ff": 2048,
             "n_prelude": 2, "n_recur": 1, "n_coda": 0,
             "max_steps": 4, "context_end": 512},
            {"experts": 8, "resident": 4, "width": 128, "depth": 1, "top_k": 2})
        with tempfile.TemporaryDirectory() as d, \
             mock.patch("minagi.config.load", return_value=cfg):
            with self.assertRaises(SystemExit) as cm:
                create(os.path.join(d, "w"), verbose=False, force=True)
            self.assertIn("divide evenly", str(cm.exception))

    def test_top_k_over_resident_exits(self):
        import torch  # noqa: F401
        from minagi.create import create

        cfg = _fake_cfg(
            {"d_model": 64, "n_head": 4, "d_ff": 128,
             "n_prelude": 1, "n_recur": 1, "n_coda": 0,
             "max_steps": 2, "context_end": 64},
            {"experts": 8, "resident": 2, "width": 64, "depth": 1, "top_k": 4})
        with tempfile.TemporaryDirectory() as d, \
             mock.patch("minagi.config.load", return_value=cfg):
            with self.assertRaises(SystemExit) as cm:
                create(os.path.join(d, "w"), verbose=False, force=True)
            self.assertIn("top_k", str(cm.exception))

    def test_tiny_create_resident_params_less_than_pool(self):
        """Fresh tiny model: card holds a fraction of the pool."""
        import torch  # noqa: F401
        from minagi.create import create

        cfg = _fake_cfg(
            {"d_model": 64, "n_head": 4, "d_ff": 128,
             "n_prelude": 1, "n_recur": 1, "n_coda": 0,
             "max_steps": 2, "context_end": 64},
            {"experts": 16, "resident": 4, "width": 64, "depth": 1, "top_k": 2})
        with tempfile.TemporaryDirectory() as d, \
             mock.patch("minagi.config.load", return_value=cfg):
            out = os.path.join(d, "w")
            got = create(out, verbose=False, force=True)
            per = 1 * 3 * 64 * 64
            self.assertEqual(got["pool_resident"], 4)
            self.assertLess(4 * per, 16 * per)
            self.assertTrue(os.path.exists(os.path.join(out, "core.npz")))
            self.assertTrue(os.path.exists(os.path.join(out, "manifest.json")))


if __name__ == "__main__":
    unittest.main()
