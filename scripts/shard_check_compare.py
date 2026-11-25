"""Compare the logits and greedy tokens of a sharded ShardCheck run with a reference run.

Usage: shard_check_compare.py <reference.npy> <sharded.npy> <max-kl>
       shard_check_compare.py --tokens-only <reference.json> <sharded.json>

Each .npy holds the last-position logits of each step, and the .json beside it holds the
greedy tokens. The check passes when the KL divergence of each step's distribution is at
most max-kl and every argmax matches, except at a near tie: an argmax mismatch fails only
when the reference margin (top-1 minus top-2 logit) exceeds max(0.05, 2 bf16 ulps at the
top-1 logit). Raw logit differences are printed but not gated: over a large vocabulary they
peak on tokens with negligible probability.

A sharded run that was teacher-forced (its report has `forcedTokens`) was fed the reference
tokens, so its own tokens need not equal them; its inputs must be those of the reference.
A free-running sharded run must produce the reference tokens.

With --tokens-only, the two reports of a generation are compared, and the check passes when
their tokens are equal.

Self-check: python3 -I scripts/test_shard_check_compare.py
"""

import enum
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# bf16 keeps 7 fraction bits.
BF16_FRACTION_BITS = 7
# A mismatch within this many bf16 steps of the top-1 logit is a tie.
TIE_ULPS = 2
# The smallest margin that still counts as a tie, in logits.
TIE_MARGIN_FLOOR = 0.05
# The tokens whose log-probabilities are compared and printed.
TOP_TOKENS = 10


class Verdict(enum.Enum):
    """The argmax outcome of one step."""

    AGREE = "agree"
    TIE = "tie"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class Step:
    """The comparison of one step."""

    kl: float
    top_logprob_diff: float
    max_logit_diff: float
    margin: float
    verdict: Verdict


def bf16_ulp(magnitude: float) -> float:
    """The spacing of bf16 values at `magnitude`."""
    exponent = math.floor(math.log2(abs(magnitude))) if magnitude else 0
    return 2.0 ** (exponent - BF16_FRACTION_BITS)


def tie_margin_limit(top: float) -> float:
    """The largest reference margin at which an argmax mismatch is a tie."""
    return max(TIE_MARGIN_FLOOR, TIE_ULPS * bf16_ulp(top))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log-probabilities along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def compare_steps(reference: np.ndarray, sharded: np.ndarray) -> list[Step]:
    """Compare the `(steps, vocabulary)` logits of two runs, step by step."""
    reference_lp = log_softmax(reference)
    sharded_lp = log_softmax(sharded)
    # KL(reference || sharded) per step.
    kl = (np.exp(reference_lp) * (reference_lp - sharded_lp)).sum(axis=-1)
    order = np.argsort(-reference, axis=-1)
    top = order[:, :TOP_TOKENS]
    top_lp_diff = np.abs(
        np.take_along_axis(reference_lp, top, -1) - np.take_along_axis(sharded_lp, top, -1)
    ).max(axis=-1)
    max_logit_diff = np.abs(reference - sharded).max(axis=-1)
    top_two = np.take_along_axis(reference, order[:, :2], -1)
    agrees = reference.argmax(axis=-1) == sharded.argmax(axis=-1)

    steps = []
    for index in range(len(kl)):
        top1, top2 = (float(value) for value in top_two[index])
        margin = top1 - top2
        if agrees[index]:
            verdict = Verdict.AGREE
        elif margin <= tie_margin_limit(top1):
            verdict = Verdict.TIE
        else:
            verdict = Verdict.MISMATCH
        steps.append(
            Step(
                kl=float(kl[index]),
                top_logprob_diff=float(top_lp_diff[index]),
                max_logit_diff=float(max_logit_diff[index]),
                margin=margin,
                verdict=verdict,
            )
        )
    return steps


def gate_passes(steps: list[Step], max_kl: float) -> bool:
    """Whether every step is within `max_kl` and no argmax mismatch is clear of a tie."""
    return all(step.kl <= max_kl and step.verdict is not Verdict.MISMATCH for step in steps)


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


def print_steps(steps: list[Step]) -> None:
    """Print one line per step."""
    for index, step in enumerate(steps):
        label = "prefill" if index == 0 else f"decode {index}"
        argmax = {
            Verdict.AGREE: "argmax agrees",
            Verdict.TIE: f"argmax tie (margin {step.margin:.5f}), skipped",
            Verdict.MISMATCH: f"argmax MISMATCH (margin {step.margin:.5f})",
        }[step.verdict]
        print(
            f"{label:>9}: KL {step.kl:.5f}, "
            f"top-{TOP_TOKENS} logprob diff {step.top_logprob_diff:.4f}, "
            f"max abs logit diff {step.max_logit_diff:.4f}, {argmax}"
        )


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


def main(reference_path: Path, sharded_path: Path, max_kl: float) -> int:
    """Print the comparison and return the exit status."""
    reference, reference_report = load_run(reference_path)
    sharded, sharded_report = load_run(sharded_path)
    if reference.shape != sharded.shape:
        print(f"FAIL: shapes differ, {reference.shape} vs {sharded.shape}")
        return 1

    steps = compare_steps(reference, sharded)
    print_steps(steps)
    tokens_ok = tokens_pass(reference_report, sharded_report)
    kl = [step.kl for step in steps]
    worst = int(np.argmax(kl))
    print(f"max KL {kl[worst]:.5f} at step {worst} (limit {max_kl})")
    agreeing = sum(step.verdict is Verdict.AGREE for step in steps)
    ties = sum(step.verdict is Verdict.TIE for step in steps)
    print(f"argmax agreement {agreeing}/{len(steps)}, ties skipped {ties}")
    for name, report in (("reference", reference_report), ("sharded", sharded_report)):
        print_timing(name, report)

    passed = tokens_ok and gate_passes(steps, max_kl)
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


def print_timing(name: str, report: dict) -> None:
    """Print the group size and the times of one run."""
    print(
        f"{name}: group {report['groupSize']}, prefill {report['prefillSeconds']:.3f}s, "
        f"decode {report['decodeSecondsPerToken'] * 1000:.1f} ms/token"
    )


def compare_tokens(reference_path: Path, sharded_path: Path) -> int:
    """Print the tokens of two generation reports and return the exit status."""
    reference_report = json.loads(reference_path.read_text())
    sharded_report = json.loads(sharded_path.read_text())
    print_tokens(reference_report, sharded_report)
    for name, report in (("reference", reference_report), ("sharded", sharded_report)):
        print_timing(name, report)
    passed = reference_report["tokens"] == sharded_report["tokens"]
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1] == "--tokens-only":
        sys.exit(compare_tokens(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])))
