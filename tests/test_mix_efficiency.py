"""Subject mixing helpers: no dead weights, probs sum to 1, speech bias."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _Lane:
    def __init__(self, name):
        self.name = name


class TestMixEfficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # noqa: F401  — train imports it
        import train as T
        cls.parse = staticmethod(T._parse_subject_weights)
        cls.probs = staticmethod(T._lane_probs)
        cls.aliases = T._WEIGHT_ALIASES

    def test_alias_chat_hermes_into_chat(self):
        w = self.parse({"chat_hermes": 3.0, "chat": 1.0, "stories": 2.0})
        self.assertNotIn("chat_hermes", w)
        self.assertEqual(w["chat"], 4.0)
        self.assertEqual(w["stories"], 2.0)

    def test_cli_string_parse(self):
        w = self.parse("chat=3,stories=2,chess=0.5")
        self.assertEqual(w["chat"], 3.0)
        self.assertEqual(w["chess"], 0.5)

    def test_probs_sum_to_one(self):
        lanes = [_Lane(n) for n in
                 ("chat", "stories", "wikipedia", "chess", "code")]
        w = self.parse({"chat": 3.5, "stories": 3.0, "chess": 0.5})
        p = self.probs(lanes, w, warn=False)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertTrue(np.all(p >= 0))

    def test_zero_weight_drops_subject(self):
        lanes = [_Lane(n) for n in ("chat", "chess")]
        p = self.probs(lanes, {"chat": 1.0, "chess": 0.0}, warn=False)
        self.assertEqual(float(p[1]), 0.0)
        self.assertAlmostEqual(float(p[0]), 1.0)

    def test_all_zero_falls_back_uniform(self):
        lanes = [_Lane(n) for n in ("a", "b", "c")]
        p = self.probs(lanes, {"a": 0, "b": 0, "c": 0}, warn=False)
        np.testing.assert_allclose(p, np.full(3, 1 / 3))

    def test_speech_heavier_than_chess(self):
        lanes = [_Lane(n) for n in
                 ("chat", "stories", "reasoning", "wikipedia", "chess")]
        w = self.parse({
            "chat": 3.5, "stories": 3.0, "reasoning": 2.0,
            "wikipedia": 0.75, "chess": 0.5,
        })
        # apply early boost like the reader
        boost = 2.0
        for name in ("chat", "stories", "reasoning", "self-knowledge"):
            w[name] = w.get(name, 1.0) * boost
        p = self.probs(lanes, w, warn=False)
        by = dict(zip([l.name for l in lanes], p))
        self.assertGreater(by["chat"], by["chess"] * 5)
        self.assertGreater(by["stories"], by["wikipedia"])

    def test_unused_key_warns(self):
        import io
        from contextlib import redirect_stdout
        lanes = [_Lane("chat")]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.probs(lanes, {"chat": 1, "nope": 9}, warn=True)
        self.assertIn("nope", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
