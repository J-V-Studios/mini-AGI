"""Adaptation decode: loop guard without sampling cost."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestDecodeEfficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from minagi.decode import pick_next
        cls.torch = torch
        cls.pick = staticmethod(pick_next)

    def test_greedy_is_deterministic(self):
        torch = self.torch
        logits = torch.randn(1, 265)
        prev = torch.randint(0, 265, (1, 40))
        a = self.pick(logits, prev, temperature=0.0,
                      adapt_strength=3.0, adapt_decay=0.88, adapt_window=128)
        b = self.pick(logits, prev, temperature=0.0,
                      adapt_strength=3.0, adapt_decay=0.88, adapt_window=128)
        self.assertEqual(int(a), int(b))

    def test_adaptation_breaks_self_loop(self):
        """A peaked logit on the last char should lose under adaptation."""
        torch = self.torch
        V = 50
        logits = torch.zeros(1, V)
        # char 7 is strongest raw
        logits[0, 7] = 5.0
        logits[0, 3] = 4.5
        prev = torch.full((1, 20), 7)          # heavy recent use of 7
        raw = int(self.pick(logits.clone(), prev, temperature=0.0,
                            adapt_strength=0.0))
        adapted = int(self.pick(logits.clone(), prev, temperature=0.0,
                                adapt_strength=3.0, adapt_decay=0.88,
                                adapt_window=128))
        self.assertEqual(raw, 7)
        self.assertNotEqual(adapted, 7)

    def test_adapt_window_caps_work(self):
        """Only last adapt_window tokens touch the trace (O(window), not O(T))."""
        torch = self.torch
        logits = torch.randn(1, 100)
        prev = torch.randint(0, 100, (1, 10_000))
        # must not blow up / take forever
        out = self.pick(logits, prev, temperature=0.0,
                        adapt_strength=2.5, adapt_window=64)
        self.assertEqual(out.shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
