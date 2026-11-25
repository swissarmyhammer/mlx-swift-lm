"""Self-check of the gate in shard_check_compare.py, on synthetic logits.

Run: python3 -I scripts/test_shard_check_compare.py
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parent / "shard_check_compare.py"
_SPEC = importlib.util.spec_from_file_location("shard_check_compare", _MODULE_PATH)
compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare)

MAX_KL = 0.05
VOCABULARY = 8

# The Qwen3.8-27B-mxfp4 prefill tie of ^zyx7p8z: one bf16 step apart in [4, 8).
TIE_TOP = 6.59375
TIE_SECOND = 6.5625


def logits_with(top: dict[int, float]) -> np.ndarray:
    """One step of logits: zero, except the given token logits."""
    row = np.zeros((1, VOCABULARY))
    for token, logit in top.items():
        row[0, token] = logit
    return row


class Bfloat16StepTests(unittest.TestCase):
    """The tie limit rests on the bf16 spacing at the top logit."""

    def test_ulp_in_four_to_eight_is_one_thirty_second(self) -> None:
        """bf16 keeps 7 fraction bits, so in [4, 8) adjacent values are 2**-5 apart."""
        self.assertEqual(compare.bf16_ulp(TIE_TOP), 2.0**-5)

    def test_limit_is_two_ulps_when_larger_than_the_floor(self) -> None:
        """At 6.6 two ulps (0.0625) exceed the 0.05 floor, so two ulps decide."""
        self.assertEqual(compare.tie_margin_limit(TIE_TOP), 2.0**-4)

    def test_limit_is_the_floor_at_small_magnitudes(self) -> None:
        """Near 1.0 two ulps are tiny, so the 0.05 floor keeps real ties skipped."""
        self.assertEqual(compare.tie_margin_limit(1.0), 0.05)


class GateTests(unittest.TestCase):
    """The per-step verdicts and the overall gate."""

    def test_flipped_tie_is_skipped(self) -> None:
        """A one-ulp near tie that sharding flips is noise, not a sharding defect (^zyx7p8z)."""
        reference = logits_with({3: TIE_TOP, 5: TIE_SECOND})
        sharded = logits_with({3: TIE_SECOND, 5: TIE_TOP})
        steps = compare.compare_steps(reference, sharded)
        self.assertTrue(compare.gate_passes(steps, MAX_KL))

    def test_flipped_tie_reports_tie(self) -> None:
        """The skipped step is labelled a tie, so the output says why it did not fail."""
        reference = logits_with({3: TIE_TOP, 5: TIE_SECOND})
        sharded = logits_with({3: TIE_SECOND, 5: TIE_TOP})
        steps = compare.compare_steps(reference, sharded)
        self.assertEqual(steps[0].verdict, compare.Verdict.TIE)

    def test_clear_argmax_mismatch_fails(self) -> None:
        """A flip against a clear reference margin fails even when KL is not gated."""
        reference = logits_with({3: 8.0, 5: 6.0})
        sharded = logits_with({3: 6.0, 5: 8.0})
        steps = compare.compare_steps(reference, sharded)
        self.assertFalse(compare.gate_passes(steps, float("inf")))

    def test_kl_breach_fails_with_argmax_agreeing(self) -> None:
        """A distribution that drifts beyond the KL limit fails though the argmax holds."""
        reference = logits_with({3: 5.0})
        sharded = logits_with({3: 5.0, 5: 4.0, 6: 4.0})
        steps = compare.compare_steps(reference, sharded)
        self.assertFalse(compare.gate_passes(steps, MAX_KL))

    def test_identical_logits_pass(self) -> None:
        """Equal logits are the baseline: zero KL and agreeing argmax pass."""
        reference = logits_with({3: 8.0, 5: 6.0})
        steps = compare.compare_steps(reference, reference.copy())
        self.assertTrue(compare.gate_passes(steps, MAX_KL))


class ForcedInputTests(unittest.TestCase):
    """A teacher-forced run must have been fed the tokens of this reference."""

    def test_inputs_of_the_reference_match(self) -> None:
        """The forced inputs are the reference tokens of every step but the last."""
        reference = {"prompt": [1, 2], "tokens": [7, 8, 9]}
        sharded = {"prompt": [1, 2], "forcedTokens": [7, 8]}
        self.assertTrue(compare.forced_inputs_match(reference, sharded))

    def test_inputs_of_another_reference_do_not_match(self) -> None:
        """A run forced with another reference compares different contexts, so it fails."""
        reference = {"prompt": [1, 2], "tokens": [7, 8, 9]}
        sharded = {"prompt": [1, 2], "forcedTokens": [7, 4]}
        self.assertFalse(compare.forced_inputs_match(reference, sharded))

    def test_another_prompt_does_not_match(self) -> None:
        """Equal forced tokens after another prompt still compare different contexts."""
        reference = {"prompt": [1, 2], "tokens": [7, 8, 9]}
        sharded = {"prompt": [1, 3], "forcedTokens": [7, 8]}
        self.assertFalse(compare.forced_inputs_match(reference, sharded))


if __name__ == "__main__":
    unittest.main()
