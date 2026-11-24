"""Compare the logits and greedy tokens of a sharded ShardCheck run with a reference run.

Usage: shard_check_compare.py <reference.npy> <sharded.npy> <max-kl>

Each .npy holds the last-position logits of each step, and the .json beside it holds the
greedy tokens. The check passes when every greedy token and every argmax match, and the
KL divergence of each step's distribution is at most max-kl. Raw logit differences are
printed but not gated: over a large vocabulary they peak on tokens with negligible
probability.
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_run(logits_path: Path) -> tuple[np.ndarray, dict]:
    """Return the logits and the report of one run."""
    report = json.loads(logits_path.with_suffix(".json").read_text())
    return np.load(logits_path).astype(np.float64), report


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log-probabilities along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main(reference_path: Path, sharded_path: Path, max_kl: float) -> int:
    """Print the comparison and return the exit status."""
    reference, reference_report = load_run(reference_path)
    sharded, sharded_report = load_run(sharded_path)
    if reference.shape != sharded.shape:
        print(f"FAIL: shapes differ, {reference.shape} vs {sharded.shape}")
        return 1

    reference_lp = log_softmax(reference)
    sharded_lp = log_softmax(sharded)
    # KL(reference || sharded) per step.
    kl = (np.exp(reference_lp) * (reference_lp - sharded_lp)).sum(axis=-1)
    top = np.argsort(-reference, axis=-1)[:, :10]
    top_lp_diff = np.abs(
        np.take_along_axis(reference_lp, top, -1) - np.take_along_axis(sharded_lp, top, -1)
    ).max(axis=-1)
    difference = np.abs(reference - sharded).max(axis=-1)
    argmax_agree = reference.argmax(axis=-1) == sharded.argmax(axis=-1)
    tokens_agree = reference_report["tokens"] == sharded_report["tokens"]

    for step in range(len(kl)):
        label = "prefill" if step == 0 else f"decode {step}"
        print(
            f"{label:>9}: KL {kl[step]:.5f}, top-10 logprob diff {top_lp_diff[step]:.4f}, "
            f"max abs logit diff {difference[step]:.4f}, argmax agrees {bool(argmax_agree[step])}"
        )
    print(f"reference tokens: {reference_report['tokens']}")
    print(f"sharded tokens:   {sharded_report['tokens']}")
    print(f"max KL {kl.max():.5f} at step {int(kl.argmax())} (limit {max_kl})")
    print(f"argmax agreement {int(argmax_agree.sum())}/{len(argmax_agree)}")
    for name, report in (("reference", reference_report), ("sharded", sharded_report)):
        print(
            f"{name}: group {report['groupSize']}, prefill {report['prefillSeconds']:.3f}s, "
            f"decode {report['decodeSecondsPerToken'] * 1000:.1f} ms/token"
        )

    passed = tokens_agree and bool(argmax_agree.all()) and kl.max() <= max_kl
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])))
