"""auto_resident must raise slots with VRAM but never explode."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestSuggestResident(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch  # noqa: F401
        import train as T
        cls.suggest = staticmethod(T._suggest_resident)

    def test_cpu_returns_floor(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(
                self.suggest(768, 3072, 1, floor=40), 40)

    def test_never_below_floor(self):
        props = mock.Mock(total_memory=8 * 10**9)

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_properties", return_value=props):
            got = self.suggest(768, 3072, 1, floor=40, mem_frac=0.5,
                               overhead_gb=6.0)
        self.assertGreaterEqual(got, 40)

    def test_12gb_allows_above_floor(self):
        props = mock.Mock(total_memory=int(12e9))

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_properties", return_value=props):
            got = self.suggest(768, 3072, 1, floor=40, mem_frac=0.90,
                               overhead_gb=4.5)
        self.assertGreaterEqual(got, 40)
        self.assertLessEqual(got, 40 + 32)   # hard cap in helper
        self.assertLessEqual(got, 96)

    def test_tiny_card_stays_at_floor(self):
        props = mock.Mock(total_memory=int(2e9))

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_properties", return_value=props):
            got = self.suggest(768, 3072, 1, floor=40, mem_frac=0.90,
                               overhead_gb=4.5)
        self.assertEqual(got, 40)


if __name__ == "__main__":
    unittest.main()
