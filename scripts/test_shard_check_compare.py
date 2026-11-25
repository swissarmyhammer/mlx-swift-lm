"""Self-check of the gate in shard_check_compare.py, on synthetic logits.

Run: python3 -I scripts/test_shard_check_compare.py
"""

import sys
import types
import unittest
from pathlib import Path

import numpy as np

# The comparator runs from its source, compiled here, so the self-check leaves no
# scripts/__pycache__ behind. A dataclass looks its module up in sys.modules.
_MODULE_PATH = Path(__file__).resolve().parent / "shard_check_compare.py"
compare = types.ModuleType("shard_check_compare")
compare.__file__ = str(_MODULE_PATH)
sys.modules[compare.__name__] = compare
exec(compile(_MODULE_PATH.read_text(), _MODULE_PATH, "exec"), compare.__dict__)

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


# A long run: the prefill plus 1024 decode steps, as in ^17p77c9.
LONG_RUN = 1025
# A short run at the largest length that keeps the strict gate.
SHORT_RUN = 64
# The steady KL of a healthy sharded run.
QUIET_KL = 0.001
# The isolated Inkling outlier of ^17p77c9: above 0.05, below the 0.2 maximum.
OUTLIER_KL = 0.0764
# The KL of the Qwen 3-ulp argmax flip of ^17p77c9: below the 0.02 mismatch floor.
NOISE_FLIP_KL = 0.017
# A KL above the 0.02 floor, so a mismatch at it is confident.
CONFIDENT_KL = 0.03
# A margin outside the tie rule at small logits.
CLEAR_MARGIN = 1.0


def run_of(length: int, changes: dict[int, compare.Step] | None = None) -> list[compare.Step]:
    """A run of quiet agreeing steps, with the given steps replaced."""
    quiet = compare.Step(
        kl=QUIET_KL,
        top_logprob_diff=0.0,
        max_logit_diff=0.0,
        margin=CLEAR_MARGIN,
        verdict=compare.Verdict.AGREE,
    )
    steps = [quiet] * length
    for index, step in (changes or {}).items():
        steps[index] = step
    return steps


def step_with(kl: float, verdict: compare.Verdict = compare.Verdict.AGREE) -> compare.Step:
    """One step at `kl` with the given argmax verdict."""
    return compare.Step(
        kl=kl, top_logprob_diff=0.0, max_logit_diff=0.0, margin=CLEAR_MARGIN, verdict=verdict
    )


def flips_at(count: int, kl: float) -> dict[int, compare.Step]:
    """`count` argmax mismatches at `kl`, spread over a long run."""
    return {index * 100 + 1: step_with(kl, compare.Verdict.MISMATCH) for index in range(count)}


class LongRunGateTests(unittest.TestCase):
    """Runs over 64 steps pass on statistics, so one isolated outlier does not fail them."""

    def test_isolated_outlier_passes_a_long_run(self) -> None:
        """One step of 1025 at KL 0.0764 is noise, not drift (the Inkling run of ^17p77c9)."""
        steps = run_of(LONG_RUN, {500: step_with(OUTLIER_KL)})
        self.assertTrue(compare.run_passes(steps, MAX_KL))

    def test_isolated_outlier_fails_a_short_run(self) -> None:
        """A short run keeps the strict per-step gate, so the same outlier fails it."""
        steps = run_of(SHORT_RUN, {30: step_with(OUTLIER_KL)})
        self.assertFalse(compare.run_passes(steps, MAX_KL))

    def test_steady_rise_fails(self) -> None:
        """KL that climbs to 0.02 over the run is drift: its mean exceeds 0.005."""
        steps = [step_with(float(kl)) for kl in np.linspace(0.0, 0.02, LONG_RUN)]
        self.assertFalse(compare.run_passes(steps, MAX_KL))

    def test_one_step_above_the_maximum_fails(self) -> None:
        """A single step above KL 0.2 is a defect however quiet the rest is."""
        steps = run_of(LONG_RUN, {500: step_with(0.3)})
        self.assertFalse(compare.run_passes(steps, MAX_KL))

    def test_too_many_steps_above_the_outlier_limit_fail(self) -> None:
        """Six steps above 0.05 in 1025 exceed the 0.5% allowance."""
        steps = run_of(LONG_RUN, {index * 100: step_with(OUTLIER_KL) for index in range(6)})
        self.assertFalse(compare.run_passes(steps, MAX_KL))

    def test_confident_mismatch_fails(self) -> None:
        """A mismatch clear of a tie, at KL above 0.02, is a defect even in a long run."""
        steps = run_of(LONG_RUN, flips_at(1, CONFIDENT_KL))
        self.assertFalse(compare.run_passes(steps, MAX_KL))

    def test_noise_flip_passes(self) -> None:
        """A mismatch at KL 0.017 is a noise flip (the Qwen 3-ulp flip of ^17p77c9)."""
        steps = run_of(LONG_RUN, flips_at(1, NOISE_FLIP_KL))
        self.assertTrue(compare.run_passes(steps, MAX_KL))

    def test_low_kl_mismatch_is_a_noise_flip(self) -> None:
        """The output labels the uncounted mismatch, so a reader sees why it did not fail."""
        self.assertTrue(compare.is_noise_flip(step_with(NOISE_FLIP_KL, compare.Verdict.MISMATCH)))

    def test_ties_do_not_count_as_noise_flips(self) -> None:
        """Ties stay skipped in a long run: six in 1025 steps still pass (^17p77c9 had five)."""
        ties = {index * 100: step_with(NOISE_FLIP_KL, compare.Verdict.TIE) for index in range(6)}
        self.assertTrue(compare.run_passes(run_of(LONG_RUN, ties), MAX_KL))

    def test_too_many_noise_flips_fail(self) -> None:
        """Six noise flips in 1025 steps exceed the 0.5% allowance."""
        steps = run_of(LONG_RUN, flips_at(6, NOISE_FLIP_KL))
        self.assertFalse(compare.run_passes(steps, MAX_KL))


class WindowTests(unittest.TestCase):
    """Per-window statistics make drift visible across a long run."""

    def test_windows_of_a_1025_step_run(self) -> None:
        """The prefill and 1024 decode steps fall in the five windows the task names."""
        windows = compare.window_stats(run_of(LONG_RUN))
        bounds = [(window.first, window.last) for window in windows]
        self.assertEqual(bounds, [(0, 32), (33, 128), (129, 256), (257, 512), (513, 1024)])

    def test_window_reports_its_max(self) -> None:
        """The outlier shows as the max of the window that holds it."""
        windows = compare.window_stats(run_of(LONG_RUN, {200: step_with(OUTLIER_KL)}))
        self.assertEqual(windows[2].max_kl, OUTLIER_KL)

    def test_short_run_stops_at_its_last_step(self) -> None:
        """A 40-step run ends inside the second window, which ends at its last step."""
        windows = compare.window_stats(run_of(40))
        self.assertEqual([(window.first, window.last) for window in windows], [(0, 32), (33, 39)])


# The load peaks of ^ffn1stg for Qwen3.5-2B: the whole model, and one of two ranks.
WHOLE_PEAK = 1_009_000_000
RANK_PEAK = 633_000_000


class LoadPeakTests(unittest.TestCase):
    """A sharded load holds about its share, so its peak stays well below the whole model's."""

    def test_layer_by_layer_peak_passes(self) -> None:
        """The 2B rank peak of a layer-by-layer load is 63% of the whole load."""
        self.assertTrue(
            compare.load_peak_passes({"loadPeakBytes": WHOLE_PEAK}, {"loadPeakBytes": RANK_PEAK})
        )

    def test_whole_model_peak_fails(self) -> None:
        """A rank that held the whole model before it split peaks as high as the reference."""
        self.assertFalse(
            compare.load_peak_passes({"loadPeakBytes": WHOLE_PEAK}, {"loadPeakBytes": WHOLE_PEAK})
        )


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
