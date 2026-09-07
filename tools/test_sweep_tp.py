#!/usr/bin/env python3
"""Unit tests for tools/sweep_tp.py, the parts that do not need a card.

    python3 -m unittest discover -s tools -p 'test_*.py'
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_tp


ARGV = ["reng-tp", "reng-bench", "out", "/models/Llama-3.2-1B"]


class ParseArgs(unittest.TestCase):
    def test_defaults(self):
        tp, bench, prefix, opts, models = sweep_tp.parse_args(ARGV)
        self.assertEqual((tp, bench, prefix), ("reng-tp", "reng-bench", "out"))
        self.assertEqual(models, ["/models/Llama-3.2-1B"])
        self.assertEqual(opts["worlds"], "2,4,8")
        # Three repeats a cell, which is what the README's medians are.
        self.assertEqual(opts["repeats"], "3")

    def test_option_before_and_after_the_models(self):
        _, _, _, opts, models = sweep_tp.parse_args(
            ["a", "b", "c", "--worlds", "2", "/m1", "--repeats", "1", "/m2"]
        )
        self.assertEqual(models, ["/m1", "/m2"])
        self.assertEqual(opts["worlds"], "2")
        self.assertEqual(opts["repeats"], "1")

    def test_a_misspelled_option_is_rejected_not_ignored(self):
        with self.assertRaises(SystemExit) as e:
            sweep_tp.parse_args(["a", "b", "c", "--wrolds", "2,4,8", "/m"])
        self.assertIn("unknown option --wrolds", str(e.exception))

    def test_a_trailing_option_does_not_raise_indexerror(self):
        with self.assertRaises(SystemExit) as e:
            sweep_tp.parse_args(["a", "b", "c", "/m", "--ctx"])
        self.assertIn("--ctx needs a value", str(e.exception))

    def test_a_non_numeric_count_is_rejected(self):
        for bad in (["--repeats", "many"], ["--ctx", "-1"], ["--timeout", ""]):
            with self.assertRaises(SystemExit):
                sweep_tp.parse_args(["a", "b", "c"] + bad + ["/m"])

    def test_no_model_is_rejected(self):
        with self.assertRaises(SystemExit):
            sweep_tp.parse_args(["a", "b", "c", "--worlds", "2"])


class SplitReason(unittest.TestCase):
    def test_the_megatron_gate(self):
        # SmolLM2-135M: 9 heads, 3 kv heads.
        cfg = {"num_attention_heads": 9, "num_key_value_heads": 3,
               "intermediate_size": 1536}
        self.assertIn("9 heads", sweep_tp.split_reason(cfg, 2))
        # Qwen2.5-7B divides four ways and not eight.
        q7 = {"num_attention_heads": 28, "num_key_value_heads": 4,
              "intermediate_size": 18944}
        self.assertIsNone(sweep_tp.split_reason(q7, 4))
        why = sweep_tp.split_reason(q7, 8)
        self.assertIn("28 heads", why)
        self.assertIn("4 kv heads", why)


class MedianCell(unittest.TestCase):
    def test_the_median_repeat_wins_not_the_first_or_the_best(self):
        reps = [{"ok": True, "tok_s": 425.5, "step_ms": 2.35},
                {"ok": True, "tok_s": 13.3, "step_ms": 74.98},
                {"ok": True, "tok_s": 95.1, "step_ms": 10.5}]
        cell = sweep_tp.median_cell(reps)
        self.assertEqual(cell["tok_s"], 95.1)
        self.assertEqual(cell["step_ms"], 10.5)
        self.assertEqual(cell["tok_s_range"], [13.3, 425.5])
        self.assertEqual(len(cell["repeats"]), 3)

    def test_a_failed_repeat_is_dropped_from_the_median(self):
        reps = [{"ok": False, "log": "boom"},
                {"ok": True, "tok_s": 500.0, "step_ms": 2.0},
                {"ok": True, "tok_s": 510.0, "step_ms": 1.96}]
        cell = sweep_tp.median_cell(reps)
        self.assertTrue(cell["ok"])
        self.assertEqual(cell["tok_s"], 510.0)
        self.assertEqual(len(cell["repeats"]), 3)

    def test_all_repeats_failing_stays_failed(self):
        cell = sweep_tp.median_cell([{"ok": False, "log": "x"}] * 3)
        self.assertFalse(cell["ok"])
        self.assertEqual(len(cell["repeats"]), 3)


class Entries(unittest.TestCase):
    def test_the_bench_json_shape_plus_world_and_strategy(self):
        cell = {"ok": True, "tok_s": 512.8, "step_ms": 1.95}
        out = sweep_tp.entries("Llama-3.2-1B", cell, "tensor", 2, 1, 841.0, "x")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["unit"], "tok/s")
        self.assertEqual(out[0]["value"], 512.8)
        self.assertEqual(out[0]["world"], 2)
        self.assertEqual(out[0]["strategy"], "tensor")
        self.assertEqual(out[1]["unit"], "%")
        self.assertAlmostEqual(out[1]["value"], 100.0 * 512.8 / 841.0)

    def test_no_ceiling_means_no_percentage_row(self):
        cell = {"ok": True, "tok_s": 1.0, "step_ms": 1.0}
        self.assertEqual(len(sweep_tp.entries("m", cell, "data", 8, 8, None, "")), 1)


if __name__ == "__main__":
    unittest.main()
