"""Config must stay inside a 12 GB card and keep paging economics."""

from __future__ import annotations

import os
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


class TestConfigEfficiency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = _cfg()

    def test_head_dim_is_64(self):
        m = self.c["model"]
        self.assertEqual(m["d_model"] % m["n_head"], 0)
        self.assertEqual(m["d_model"] // m["n_head"], 64)

    def test_top_k_fits_resident(self):
        p = self.c["pool"]
        self.assertLessEqual(p["top_k"], p["resident"])
        # keep ~4:1 resident/top_k so a card is not all one character's picks
        self.assertGreaterEqual(p["resident"] / p["top_k"], 3.0)

    def test_expert_train_state_fits_12gb_card(self):
        """Resident train state + overhead must leave headroom on 12 GB."""
        p, m = self.c["pool"], self.c["model"]
        per = 3 * m["d_model"] * p["width"] * 12          # bytes / expert slot
        experts = p["resident"] * per
        overhead = 4.5e9                                   # trunk + activations
        total = experts + overhead
        self.assertLess(
            total, 12e9 * float(self.c["growth"]["mem_frac"]),
            f"resident train state {total/1e9:.2f} GB exceeds "
            f"{100*self.c['growth']['mem_frac']:.0f}% of 12 GB")

    def test_vram_slots_not_pool_size(self):
        """Disk pool may grow; VRAM cost is slots, not experts on disk."""
        p = self.c["pool"]
        self.assertGreater(p["experts"], p["resident"])
        self.assertGreaterEqual(p["ram_cache"], p["resident"])

    def test_chunk_not_larger_than_context(self):
        self.assertLessEqual(
            self.c["training"]["chunk"], self.c["model"]["context_end"])
        self.assertLessEqual(
            self.c["model"]["context_start"], self.c["model"]["context_end"])

    def test_accum_is_cheap(self):
        """accum averages grads; graph still one chunk — keep it small."""
        a = int(self.c["training"].get("accum", 1))
        self.assertGreaterEqual(a, 1)
        self.assertLessEqual(a, 8)

    def test_speech_mix_dominates_early(self):
        w = dict(self.c["data"]["subject_weights"])
        speech = self.c["data"]["speech_subjects"]
        boost = float(self.c["data"]["speech_boost"])
        # after boost, speech share of named weights
        boosted = {k: (v * boost if k in speech else v) for k, v in w.items()}
        speech_mass = sum(boosted[k] for k in speech if k in boosted)
        total = sum(boosted.values())
        self.assertGreater(
            speech_mass / total, 0.60,
            "early speech boost should put >60% visits on speech lanes")

    def test_disk_ceiling_holds_wider_experts(self):
        m, p, g = self.c["model"], self.c["pool"], self.c["growth"]
        per = 3 * m["d_model"] * p["width"] * 8            # disk bytes / expert
        n = int(g["max_disk_gb"] * 1e9 / per)
        self.assertGreaterEqual(n, 200, f"only ~{n} experts fit on disk ceiling")

    def test_logging_heartbeat_on(self):
        log = self.c.get("logging") or {}
        self.assertGreater(float(log.get("every_min", 0)), 0)
        self.assertTrue(log.get("status"))


if __name__ == "__main__":
    unittest.main()
