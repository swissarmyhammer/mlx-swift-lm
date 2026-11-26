"""Self-check of the gate in shard_check_compare.py, on synthetic logits, and of the
family list of `shard-check.sh --tiny`, against a fake ShardCheck.

Run: python3 -I scripts/test_shard_check_compare.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The comparator runs from its source, compiled here, so the self-check leaves no
# scripts/__pycache__ behind. A dataclass looks its module up in sys.modules.
_MODULE_PATH = Path(__file__).resolve().parent / "shard_check_compare.py"
compare = types.ModuleType("shard_check_compare")
compare.__file__ = str(_MODULE_PATH)
sys.modules[compare.__name__] = compare
exec(compile(_MODULE_PATH.read_text(), _MODULE_PATH, "exec"), compare.__dict__)

VOCABULARY = 8

# The Qwen3.8-27B-mxfp4 prefill tie of ^zyx7p8z: one bf16 step apart in [4, 8).
TIE_TOP = 6.59375
TIE_SECOND = 6.5625

# A sharded row whose top-5 is tokens 0, 1, 2, 4 and 6, with the 5th-highest logit 6.0.
# In [4, 8) a bf16 step is 2**-5, so the near-tie tolerance there is 2**-4.
FIFTH = 6.0
SHARDED_TOP5 = {0: 10.0, 1: 9.0, 2: 8.0, 4: 7.0, 6: FIFTH}
# The reference argmax, token 3; the reference row also holds the sharded argmax in its top-5.
REFERENCE_ROW = {3: 10.0, 0: 9.0}


def logits_with(top: dict[int, float]) -> np.ndarray:
    """One step of logits: zero, except the given token logits."""
    row = np.zeros((1, VOCABULARY))
    for token, logit in top.items():
        row[0, token] = logit
    return row


def sharded_with_token_3_at(logit: float) -> np.ndarray:
    """The sharded row of SHARDED_TOP5, with the reference argmax (token 3) at `logit`."""
    return logits_with({**SHARDED_TOP5, 3: logit})


def run_passes(reference: np.ndarray, sharded: np.ndarray) -> bool:
    """Whether the run of these logits passes the gate."""
    return compare.run_passes(compare.compare_steps(reference, sharded))


class Bfloat16StepTests(unittest.TestCase):
    """The near-tie tolerance rests on the bf16 spacing at the other side's 5th logit."""

    def test_ulp_in_four_to_eight_is_one_thirty_second(self) -> None:
        """bf16 keeps 7 fraction bits, so in [4, 8) adjacent values are 2**-5 apart."""
        self.assertEqual(compare.bf16_ulp(TIE_TOP), 2.0**-5)

    def test_tolerance_is_two_ulps_at_the_fifth_logit(self) -> None:
        """At a 5th logit of 6.0 two bf16 steps are 2**-4."""
        self.assertEqual(compare.near_tie_tol(FIFTH), 2.0**-4)

    def test_tolerance_scales_with_magnitude(self) -> None:
        """At a 5th logit of 20.0, in [16, 32), two bf16 steps are 2**-2: no fixed constant."""
        self.assertEqual(compare.near_tie_tol(20.0), 2.0**-2)


class GateTests(unittest.TestCase):
    """The top-5 check of vLLM's check_logprobs_close, applied at every step."""

    def test_equal_argmax_passes(self) -> None:
        """Equal argmax is the baseline: the step agrees."""
        reference = logits_with({3: 8.0, 5: 6.0})
        sharded = logits_with({3: 7.5, 5: 6.5})
        steps = compare.compare_steps(reference, sharded)
        self.assertEqual(steps[0].verdict, compare.Verdict.AGREE)
        self.assertTrue(compare.run_passes(steps))

    def test_differing_argmax_within_the_top5_passes(self) -> None:
        """A flip whose argmaxes are each in the other's top-5 passes, even clear of a tie."""
        reference = logits_with({3: 8.0, 5: 6.0})
        sharded = logits_with({3: 6.0, 5: 8.0})
        steps = compare.compare_steps(reference, sharded)
        self.assertEqual(steps[0].verdict, compare.Verdict.IN_TOP_K)
        self.assertTrue(compare.run_passes(steps))

    def test_flipped_bf16_tie_passes(self) -> None:
        """A one-ulp near tie that sharding flips is in the top-5 of both (^zyx7p8z)."""
        reference = logits_with({3: TIE_TOP, 5: TIE_SECOND})
        sharded = logits_with({3: TIE_SECOND, 5: TIE_TOP})
        self.assertTrue(run_passes(reference, sharded))

    def test_outside_the_top5_within_the_near_tie_tolerance_passes(self) -> None:
        """The reference argmax one bf16 step below the sharded 5th logit is a near tie."""
        reference = logits_with(REFERENCE_ROW)
        sharded = sharded_with_token_3_at(FIFTH - 2.0**-5)
        steps = compare.compare_steps(reference, sharded)
        self.assertEqual(steps[0].verdict, compare.Verdict.NEAR_TIE)
        self.assertTrue(compare.run_passes(steps))

    def test_tie_with_the_fifth_logit_passes(self) -> None:
        """Equal to the 5th logit but sorted after it into 6th place, the token is a near tie."""
        # Token 7 sorts after token 6, the 5th, at the same logit.
        reference = logits_with({7: 10.0, 0: 9.0})
        sharded = logits_with({**SHARDED_TOP5, 7: FIFTH})
        self.assertEqual(
            compare.compare_steps(reference, sharded)[0].verdict, compare.Verdict.NEAR_TIE
        )

    def test_just_beyond_the_near_tie_tolerance_fails(self) -> None:
        """Three bf16 steps below the 5th logit is beyond the two-step tolerance."""
        reference = logits_with(REFERENCE_ROW)
        sharded = sharded_with_token_3_at(FIFTH - 3 * 2.0**-5)
        self.assertFalse(run_passes(reference, sharded))

    def test_clearly_outside_the_top5_fails(self) -> None:
        """The reference argmax far below the sharded top-5 is a mismatch."""
        reference = logits_with(REFERENCE_ROW)
        sharded = sharded_with_token_3_at(1.0)
        steps = compare.compare_steps(reference, sharded)
        self.assertEqual(steps[0].verdict, compare.Verdict.MISMATCH)
        self.assertFalse(compare.run_passes(steps))

    def test_sharded_argmax_outside_the_reference_top5_fails(self) -> None:
        """The check runs both ways: the sharded argmax must be in the reference's top-5."""
        reference = logits_with({3: 10.0, 0: 9.0, 1: 8.0, 2: 7.0, 4: 6.0})
        sharded = logits_with({7: 10.0, 3: 9.0})
        self.assertFalse(run_passes(reference, sharded))

    def test_large_kl_with_equal_argmax_passes(self) -> None:
        """KL is reported, not gated: an agreeing step at KL far above 0.05 passes."""
        reference = logits_with({3: 5.0})
        sharded = logits_with({3: 5.0, 5: 4.0, 6: 4.0})
        steps = compare.compare_steps(reference, sharded)
        self.assertGreater(steps[0].kl, 0.4)
        self.assertTrue(compare.run_passes(steps))

    def test_one_failing_step_fails_the_run(self) -> None:
        """The run passes only when every step passes."""
        reference = np.vstack([logits_with({3: 8.0}), logits_with(REFERENCE_ROW)])
        sharded = np.vstack([logits_with({3: 8.0}), sharded_with_token_3_at(1.0)])
        self.assertFalse(run_passes(reference, sharded))


# A long run: the prefill plus 1024 decode steps, as in ^17p77c9.
LONG_RUN = 1025
# The steady KL of a healthy sharded run.
QUIET_KL = 0.001
# The isolated Inkling outlier of ^17p77c9.
OUTLIER_KL = 0.0764


def run_of(length: int, changes: dict[int, compare.Step] | None = None) -> list[compare.Step]:
    """A run of quiet agreeing steps, with the given steps replaced."""
    steps = [step_with(QUIET_KL)] * length
    for index, step in (changes or {}).items():
        steps[index] = step
    return steps


def step_with(kl: float) -> compare.Step:
    """One agreeing step at `kl`."""
    return compare.Step(
        kl=kl,
        top_logprob_diff=0.0,
        max_logit_diff=0.0,
        reference_argmax=0,
        sharded_argmax=0,
        verdict=compare.Verdict.AGREE,
    )


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

    def test_ungated_peak_passes_whatever_it_reads(self) -> None:
        """The tiny checkpoints of `shard-check.sh --tiny` load a few megabytes, so both peaks
        read about 0.00 GiB and the 75% gate says nothing about the layer-by-layer load; the
        tiny mode turns the gate off, and a peak as high as the reference's then passes."""
        self.assertTrue(
            compare.peak_pass(
                {"loadPeakBytes": WHOLE_PEAK}, {"loadPeakBytes": WHOLE_PEAK}, gated=False
            )
        )

    def test_gated_peak_fails_a_whole_model_peak(self) -> None:
        """Without --no-load-peak-gate the gate holds, so a whole-model peak still fails."""
        self.assertFalse(
            compare.peak_pass(
                {"loadPeakBytes": WHOLE_PEAK}, {"loadPeakBytes": WHOLE_PEAK}, gated=True
            )
        )

    def test_command_line_gates_the_peak_by_default(self) -> None:
        """A run of a real checkpoint keeps the gate unless the command line turns it off."""
        self.assertTrue(compare.parse_arguments(["reference.npy", "sharded.npy"]).load_peak_gate)

    def test_command_line_turns_the_gate_off(self) -> None:
        """--no-load-peak-gate is the explicit switch of the tiny mode."""
        arguments = compare.parse_arguments(["--no-load-peak-gate", "reference.npy", "sharded.npy"])
        self.assertFalse(arguments.load_peak_gate)


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


_SCRIPT_PATH = Path(__file__).resolve().parent / "shard-check.sh"
# Where shard-check.sh looks for ShardCheck, relative to the repository root.
_BINARY_PATH = Path(".build/shard-check/Build/Products/Release/ShardCheck")
# The mode of a fake tool: executable, as a build product is.
_EXECUTABLE_MODE = 0o755
# The file the fake ShardCheck creates beside itself each time it runs.
_RAN_MARKER_SUFFIX = ".ran"
# The last stderr line of a run whose build logged an `error:` line.
_BUILD_ERROR_MESSAGE = "the ShardCheck build logged an error: line, so no check ran"
# The fake xcodebuild of ^8pkmyp2: it logs a failed compile and still exits 0.
_SPURIOUS_FAILURE_BUILD = (
    'echo "error: the following command failed with exit code 0 but produced no further'
    ' output"\nexit 0'
)


@dataclass(frozen=True)
class TinyRun:
    """What one `shard-check.sh --tiny` run against fake tools did."""

    status: int
    last_error_line: str
    shard_check_ran: bool


def write_tool(path: Path, body: str) -> None:
    """Writes an executable bash script with `body` at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(_EXECUTABLE_MODE)


def tiny_run(fake_binary: str, fake_xcodebuild: str | None = None) -> TinyRun:
    """Runs `shard-check.sh --tiny` against a fake ShardCheck whose body is `fake_binary`.

    The script runs from a copy in a temporary tree that holds the fake binary where the
    script looks for ShardCheck. With no `fake_xcodebuild`, the build is skipped; with one, an
    `xcodebuild` of that body comes first on PATH, so no Swift build is touched either way.
    The fake ShardCheck creates a marker beside itself, so the run reports whether it ran.
    """
    with tempfile.TemporaryDirectory() as root:
        script = Path(root) / "scripts" / _SCRIPT_PATH.name
        script.parent.mkdir()
        shutil.copy(_SCRIPT_PATH, script)
        binary = Path(root) / _BINARY_PATH
        marker = binary.with_name(binary.name + _RAN_MARKER_SUFFIX)
        write_tool(binary, f'touch "{marker}"\n{fake_binary}')
        environment = {**os.environ, "SHARD_CHECK_WORK": str(Path(root) / "work")}
        if fake_xcodebuild is None:
            environment["SHARD_CHECK_SKIP_BUILD"] = "1"
        else:
            tools = Path(root) / "bin"
            write_tool(tools / "xcodebuild", fake_xcodebuild)
            environment["PATH"] = f"{tools}{os.pathsep}{environment['PATH']}"
        result = subprocess.run(
            ["/bin/bash", str(script), "--tiny"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        stderr_lines = result.stderr.splitlines() or [""]
        return TinyRun(result.returncode, stderr_lines[-1], marker.exists())


def tiny_run_outcome(fake_binary: str) -> tuple[int, str]:
    """The exit status and the last stderr line of a skipped-build `--tiny` run."""
    run = tiny_run(fake_binary)
    return run.status, run.last_error_line


class TinyFamilyListTests(unittest.TestCase):
    """`--tiny` with no family named fails, and says why, when it cannot check any family."""

    def test_failing_family_list_fails_the_run(self) -> None:
        """When `ShardCheck tiny-families` fails, no family was checked, so the run fails."""
        self.assertEqual(
            tiny_run_outcome("exit 1"),
            (1, "could not list the tiny families (ShardCheck tiny-families)"),
        )

    def test_empty_family_list_fails_the_run(self) -> None:
        """An empty family list would check nothing, so the run fails rather than pass."""
        self.assertEqual(
            tiny_run_outcome("exit 0"), (1, "ShardCheck tiny-families listed no family to check")
        )


class BuildErrorTests(unittest.TestCase):
    """A ShardCheck build that logs `error:` but exits 0 stops the script (^8pkmyp2).

    xcodebuild logged a failed MLXVLM compile, exited 0 and kept the old objects, so the
    script ran a stale binary and a source mutation looked uncaught.
    """

    def test_logged_build_error_fails_the_run(self) -> None:
        """The exit status of xcodebuild is not enough: an `error:` line fails the run."""
        run = tiny_run("exit 1", _SPURIOUS_FAILURE_BUILD)
        self.assertEqual((run.status, run.last_error_line), (1, _BUILD_ERROR_MESSAGE))

    def test_logged_build_error_runs_no_check(self) -> None:
        """A possibly stale ShardCheck must not run at all, not even to list the families."""
        self.assertFalse(tiny_run("exit 1", _SPURIOUS_FAILURE_BUILD).shard_check_ran)

    def test_failing_build_runs_no_check(self) -> None:
        """The log goes through `tee`, whose status must not hide a failed xcodebuild."""
        self.assertFalse(tiny_run("exit 1", "exit 65").shard_check_ran)

    def test_clean_build_runs_the_checks(self) -> None:
        """A build that logs no `error:` line goes on to ShardCheck, as before."""
        self.assertTrue(tiny_run("exit 1", "echo 'warning: a warning is no error'").shard_check_ran)


if __name__ == "__main__":
    unittest.main()
