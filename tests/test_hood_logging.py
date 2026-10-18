"""Under-the-hood logging stays cheap and informative."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestHoodLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # noqa: F401
        import train as T
        cls.row = staticmethod(T._hood_row)
        cls.line = staticmethod(T._hood_line)
        cls.write = staticmethod(T._write_status)

    def test_line_contains_key_fields(self):
        text = self.line({
            "lane": "chat", "loss": 0.9, "depth": 12.0, "experts": 100,
            "on_card": list(range(40)), "loads_per_chunk": 2.1,
            "hit_rate": 0.88, "dying": 5, "entropy_frac": 0.6,
            "gate_top": [1.0, 0.5, 0.2], "lr": 3e-4, "gap": 0.01,
            "ctx": 4096, "vram_mb": 6000.0,
            "visits": {"chat": 8, "stories": 4, "chess": 1},
        })
        self.assertTrue(text.startswith("  hood  "))
        for needle in ("chat", "loss", "depth", "pool", "RAM hit",
                       "vram", "mix"):
            self.assertIn(needle, text)

    def test_row_from_pool_report(self):
        import torch
        from minagi.paged import PagedPool

        with tempfile.TemporaryDirectory() as d:
            pool = PagedPool(d, 32, 64, n_experts=8, resident=4,
                             ram_capacity=8, device="cpu")
            model = type("M", (), {"last_steps": 7.5})()
            row = self.row(pool, model, step=3, chars=1000, lane="stories",
                           loss=1.1, lr=1e-4, visits={"stories": 2})
        self.assertEqual(row["kind"] if "kind" in row else None, None)
        self.assertEqual(row["experts"], 8)
        self.assertEqual(row["depth"], 7.5)
        self.assertEqual(row["lane"], "stories")
        self.assertGreaterEqual(row["hit_rate"], 0.0)

    def test_status_jsonl_append(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "status.jsonl")
            self.write(path, {"t": 1, "kind": "tick", "loss": 1.2})
            self.write(path, {"t": 2, "kind": "sample", "loss": 1.1})
            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["kind"], "sample")

    def test_empty_status_path_is_noop(self):
        self.write("", {"t": 1})
        self.write(None, {"t": 1})


if __name__ == "__main__":
    unittest.main()
