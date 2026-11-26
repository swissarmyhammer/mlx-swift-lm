"""Compare the logits and greedy tokens of a sharded ShardCheck run with a reference run.

Usage: shard_check_compare.py [--no-load-peak-gate] <reference.npy> <sharded.npy>
       shard_check_compare.py --tokens-only [--no-load-peak-gate] <reference.json> <sharded.json>

Each .npy holds the last-position logits of each step, and the .json beside it holds the
greedy tokens. The gate is the top-k check of vLLM's `check_logprobs_close` with
`always_check_logprobs=True`, applied at every step (vllm-project/vllm,
tests/models/utils.py, with its helpers `_topk_token_sets_match` and `_in_topk_or_near_tie`):
the reference argmax must be in the sharded run's top-5 tokens, and the sharded argmax in the
reference's top-5. A token outside the other side's top-5 is still accepted at a near tie, when
its logit on that side is within 2 bf16 ulps (at the magnitude of that side's 5th-highest logit)
of that side's 5th-highest logit. A step passes or fails on that alone, and the run passes when
every step passes.

KL divergence, the top-10 log-probability difference and the largest logit difference are
computed and printed as reported values, per step, per window, and as the mean, p99 and max
over the run. None of them is gated.

A sharded run that was teacher-forced (its report has `forcedTokens`) was fed the reference
tokens, so its own tokens need not equal them; its inputs must be those of the reference.
A free-running sharded run must produce the reference tokens.

The load of the sharded run must also peak at most 75% of the reference's load peak: each rank
loads its share layer by layer, so it never holds the whole model. With --no-load-peak-gate the
peaks are printed and not gated: `shard-check.sh --tiny` passes it, because a checkpoint of a
few megabytes loads in about 0.00 GiB on both sides, where the gate says nothing.

With --tokens-only, the two reports of a generation are compared, and the check passes when
their tokens are equal.

Self-check: python3 -I scripts/test_shard_check_compare.py
"""

import argparse
import enum
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bf16 keeps 7 fraction bits.
BF16_FRACTION_BITS = 7
# The near-tie tolerance, in bf16 steps at the magnitude of the other side's k-th logit.
NEAR_TIE_ULPS = 2
# Each side's argmax must be in the other side's top this many tokens.
TOP_K = 5
# The tokens whose log-probabilities are compared and printed.
TOP_TOKENS = 10
# The percentile of the reported p99 KL.
P99 = 99
# The largest load peak of a sharded rank, as a fraction of the whole model's load peak.
LOAD_PEAK_FRACTION = 0.75
# The last step of each printed window; past the last one, windows keep doubling.
WINDOW_ENDS = (32, 128, 256, 512, 1024)


class Verdict(enum.Enum):
    """The top-k outcome of one step, or of one argmax against the other side.

    The members are ordered from best to worst; a step takes the worse of its two directions.
    """

    AGREE = "agree"
    IN_TOP_K = "in top-k"
    NEAR_TIE = "near tie"
    MISMATCH = "mismatch"

    @property
    def passes(self) -> bool:
        """Whether a step with this outcome passes the gate."""
        return self is not Verdict.MISMATCH


VERDICT_ORDER = list(Verdict)


@dataclass(frozen=True)
class Step:
    """The comparison of one step. Only `verdict` is gated; the rest is reported."""

    kl: float
    top_logprob_diff: float
    max_logit_diff: float
    reference_argmax: int
    sharded_argmax: int
    verdict: Verdict


def bf16_ulp(magnitude: float) -> float:
    """The spacing of bf16 values at `magnitude`."""
    exponent = math.floor(math.log2(abs(magnitude))) if magnitude else 0
    return 2.0 ** (exponent - BF16_FRACTION_BITS)


def near_tie_tol(kth_logit: float) -> float:
    """The near-tie tolerance at a side whose k-th highest logit is `kth_logit`."""
    return NEAR_TIE_ULPS * bf16_ulp(kth_logit)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log-probabilities along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def top_k_membership(token: int, other: np.ndarray, k: int = TOP_K) -> Verdict:
    """Whether `token` is in the top-k of the logits row `other`, as `_in_topk_or_near_tie` reads it.

    Outside the top-k, the token is a near tie when its logit in `other` is within
    `near_tie_tol` of the k-th highest logit of `other`; otherwise it is a mismatch.
    """
    k = min(k, other.shape[-1])
    top = np.argsort(-other, kind="stable")[:k]
    if token in top:
        return Verdict.IN_TOP_K
    kth = float(other[top[-1]])
    if float(other[token]) >= kth - near_tie_tol(kth):
        return Verdict.NEAR_TIE
    return Verdict.MISMATCH


def step_verdict(reference: np.ndarray, sharded: np.ndarray) -> Verdict:
    """The top-k outcome of one step: each argmax checked against the other side's top-k."""
    reference_argmax = int(reference.argmax())
    sharded_argmax = int(sharded.argmax())
    if reference_argmax == sharded_argmax:
        return Verdict.AGREE
    return max(
        top_k_membership(reference_argmax, sharded),
        top_k_membership(sharded_argmax, reference),
        key=VERDICT_ORDER.index,
    )


def compare_steps(reference: np.ndarray, sharded: np.ndarray) -> list[Step]:
    """Compare the `(steps, vocabulary)` logits of two runs, step by step."""
    reference_lp = log_softmax(reference)
    sharded_lp = log_softmax(sharded)
    # KL(reference || sharded) per step, reported only.
    kl = (np.exp(reference_lp) * (reference_lp - sharded_lp)).sum(axis=-1)
    top = np.argsort(-reference, axis=-1)[:, :TOP_TOKENS]
    top_lp_diff = np.abs(
        np.take_along_axis(reference_lp, top, -1) - np.take_along_axis(sharded_lp, top, -1)
    ).max(axis=-1)
    max_logit_diff = np.abs(reference - sharded).max(axis=-1)
    return [
        Step(
            kl=float(kl[index]),
            top_logprob_diff=float(top_lp_diff[index]),
            max_logit_diff=float(max_logit_diff[index]),
            reference_argmax=int(reference[index].argmax()),
            sharded_argmax=int(sharded[index].argmax()),
            verdict=step_verdict(reference[index], sharded[index]),
        )
        for index in range(len(kl))
    ]


def run_passes(steps: list[Step]) -> bool:
    """Whether every step passes the top-k check."""
    return all(step.verdict.passes for step in steps)


@dataclass(frozen=True)
class Window:
    """The KL statistics of the steps `first` through `last`."""

    first: int
    last: int
    max_kl: float
    mean_kl: float


def window_bounds(count: int) -> list[tuple[int, int]]:
    """The first and last step of each window that holds some of `count` steps."""
    ends = list(WINDOW_ENDS)
    while ends[-1] < count - 1:
        ends.append(ends[-1] * 2)
    starts = [0] + [end + 1 for end in ends[:-1]]
    return [(start, min(end, count - 1)) for start, end in zip(starts, ends) if start < count]


def window_stats(steps: list[Step]) -> list[Window]:
    """The max and mean KL of each window of the run."""
    kl = np.array([step.kl for step in steps])
    return [
        Window(first, last, float(kl[first : last + 1].max()), float(kl[first : last + 1].mean()))
        for first, last in window_bounds(len(steps))
    ]


def forced_inputs_match(reference_report: dict, sharded_report: dict) -> bool:
    """Whether a teacher-forced run was fed the prompt and tokens of this reference."""
    forced = sharded_report["forcedTokens"]
    return (
        sharded_report["prompt"] == reference_report["prompt"]
        and forced == reference_report["tokens"][: len(forced)]
    )


def load_run(logits_path: Path) -> tuple[np.ndarray, dict]:
    """Return the logits and the report of one run."""
    report = json.loads(logits_path.with_suffix(".json").read_text())
    return np.load(logits_path).astype(np.float64), report


def argmax_label(step: Step) -> str:
    """The top-k outcome of one step."""
    tokens = f"reference {step.reference_argmax}, sharded {step.sharded_argmax}"
    return {
        Verdict.AGREE: "argmax agrees",
        Verdict.IN_TOP_K: f"argmax differs ({tokens}), each in the other's top-{TOP_K}",
        Verdict.NEAR_TIE: f"argmax differs ({tokens}), near tie at the other's top-{TOP_K}",
        Verdict.MISMATCH: f"argmax MISMATCH ({tokens}), outside the other's top-{TOP_K}",
    }[step.verdict]


def print_steps(steps: list[Step]) -> None:
    """Print one line per step."""
    for index, step in enumerate(steps):
        label = "prefill" if index == 0 else f"decode {index}"
        print(
            f"{label:>9}: KL {step.kl:.5f}, "
            f"top-{TOP_TOKENS} logprob diff {step.top_logprob_diff:.4f}, "
            f"max abs logit diff {step.max_logit_diff:.4f}, {argmax_label(step)}"
        )


def print_kl_report(steps: list[Step]) -> None:
    """Print the KL of each window and of the run, reported and not gated."""
    for window in window_stats(steps):
        print(
            f"steps {window.first}-{window.last}: "
            f"max KL {window.max_kl:.5f}, mean KL {window.mean_kl:.5f} (reported, not gated)"
        )
    kl = np.array([step.kl for step in steps])
    print(
        f"KL over {len(steps)} steps (reported, not gated): mean {kl.mean():.5f}, "
        f"p99 {np.percentile(kl, P99):.5f}, max {kl.max():.5f} at step {int(kl.argmax())}"
    )


def print_gate(steps: list[Step]) -> None:
    """Print the count of each top-k outcome."""
    counts = ", ".join(
        f"{verdict.value} {sum(step.verdict is verdict for step in steps)}" for verdict in Verdict
    )
    print(f"top-{TOP_K} gate over {len(steps)} steps: {counts}")


def print_tokens(reference_report: dict, sharded_report: dict) -> None:
    """Print the tokens of both runs."""
    print(f"reference tokens: {reference_report['tokens']}")
    print(f"sharded tokens:   {sharded_report['tokens']}")


def tokens_pass(reference_report: dict, sharded_report: dict) -> bool:
    """Print the tokens, and whether they meet the gate of the sharded run's mode."""
    print_tokens(reference_report, sharded_report)
    if sharded_report.get("forcedTokens") is None:
        print("free running: the tokens must be equal")
        return reference_report["tokens"] == sharded_report["tokens"]
    matched = forced_inputs_match(reference_report, sharded_report)
    print(f"teacher-forced with the reference tokens: inputs match {matched}")
    return matched


def main(reference_path: Path, sharded_path: Path, load_peak_gate: bool) -> int:
    """Print the comparison and return the exit status."""
    reference, reference_report = load_run(reference_path)
    sharded, sharded_report = load_run(sharded_path)
    if reference.shape != sharded.shape:
        print(f"FAIL: shapes differ, {reference.shape} vs {sharded.shape}")
        return 1

    steps = compare_steps(reference, sharded)
    print_steps(steps)
    tokens_ok = tokens_pass(reference_report, sharded_report)
    print_kl_report(steps)
    print_gate(steps)
    print_timings(reference_report, sharded_report)

    peak_ok = peak_pass(reference_report, sharded_report, gated=load_peak_gate)
    return verdict(tokens_ok and peak_ok and run_passes(steps))


def load_peak_passes(reference_report: dict, sharded_report: dict) -> bool:
    """Whether the sharded load peaked well below the load of the whole model."""
    limit = LOAD_PEAK_FRACTION * reference_report["loadPeakBytes"]
    return sharded_report["loadPeakBytes"] <= limit


def peak_pass(reference_report: dict, sharded_report: dict, gated: bool) -> bool:
    """Print the load peaks, and whether the sharded one is within its limit.

    An ungated peak passes: the tiny checkpoints of `shard-check.sh --tiny` are too small for
    the limit to say anything.
    """
    gib = float(1 << 30)
    limit = (
        f"limit {LOAD_PEAK_FRACTION:.0%} of the reference"
        if gated
        else "not gated: --no-load-peak-gate"
    )
    print(
        f"load peak: reference {reference_report['loadPeakBytes'] / gib:.2f} GiB, "
        f"sharded rank 0 {sharded_report['loadPeakBytes'] / gib:.2f} GiB ({limit})"
    )
    return not gated or load_peak_passes(reference_report, sharded_report)


def print_timings(reference_report: dict, sharded_report: dict) -> None:
    """Print the group size and the times of both runs."""
    for name, report in (("reference", reference_report), ("sharded", sharded_report)):
        print_timing(name, report)


def verdict(passed: bool) -> int:
    """Print the verdict and return the exit status."""
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


def print_timing(name: str, report: dict) -> None:
    """Print the group size and the times of one run."""
    print(
        f"{name}: group {report['groupSize']}, prefill {report['prefillSeconds']:.3f}s, "
        f"decode {report['decodeSecondsPerToken'] * 1000:.1f} ms/token"
    )


def compare_tokens(reference_path: Path, sharded_path: Path, load_peak_gate: bool) -> int:
    """Print the tokens of two generation reports and return the exit status."""
    reference_report = json.loads(reference_path.read_text())
    sharded_report = json.loads(sharded_path.read_text())
    print_tokens(reference_report, sharded_report)
    print_timings(reference_report, sharded_report)
    peak_ok = peak_pass(reference_report, sharded_report, gated=load_peak_gate)
    return verdict(peak_ok and reference_report["tokens"] == sharded_report["tokens"])


def parse_arguments(arguments: list[str]) -> argparse.Namespace:
    """The command line: the two runs, and the switches of the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare a sharded ShardCheck run with its reference."
    )
    parser.add_argument(
        "--tokens-only", action="store_true", help="compare the tokens of two generation reports"
    )
    parser.add_argument(
        "--no-load-peak-gate",
        dest="load_peak_gate",
        action="store_false",
        help="print the load peaks without gating them, for checkpoints of a few megabytes",
    )
    parser.add_argument(
        "reference", type=Path, help="the reference .npy, or .json with --tokens-only"
    )
    parser.add_argument("sharded", type=Path, help="the sharded .npy, or .json with --tokens-only")
    return parser.parse_args(arguments)


if __name__ == "__main__":
    ARGUMENTS = parse_arguments(sys.argv[1:])
    COMPARE = compare_tokens if ARGUMENTS.tokens_only else main
    sys.exit(COMPARE(ARGUMENTS.reference, ARGUMENTS.sharded, ARGUMENTS.load_peak_gate))
