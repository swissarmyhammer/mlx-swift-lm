"""Compare the logits and greedy tokens of a sharded ShardCheck run with a reference run.

Usage: shard_check_compare.py <reference.npy> <sharded.npy> <tolerance>

Each .npy holds the last-position logits of each step, and the .json beside it holds the
greedy tokens. The check passes when every greedy token matches and the largest absolute
logit difference is at most the tolerance.
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_run(logits_path: Path) -> tuple[np.ndarray, dict]:
    """Return the logits and the report of one run."""
    report = json.loads(logits_path.with_suffix(".json").read_text())
    return np.load(logits_path), report


def main(reference_path: Path, sharded_path: Path, tolerance: float) -> int:
    """Print the comparison and return the exit status."""
    reference, reference_report = load_run(reference_path)
    sharded, sharded_report = load_run(sharded_path)
    if reference.shape != sharded.shape:
        print(f"FAIL: shapes differ, {reference.shape} vs {sharded.shape}")
        return 1

    difference = np.abs(reference - sharded)
    max_logits_agree = reference.argmax(axis=-1) == sharded.argmax(axis=-1)
    tokens_agree = reference_report["tokens"] == sharded_report["tokens"]

    for step, (row, agrees) in enumerate(zip(difference, max_logits_agree)):
        label = "prefill" if step == 0 else f"decode {step}"
        print(f"{label:>9}: max abs diff {row.max():.5f}, argmax agrees {bool(agrees)}")
    print(f"reference tokens: {reference_report['tokens']}")
    print(f"sharded tokens:   {sharded_report['tokens']}")
    print(f"max abs logit diff {difference.max():.5f} (tolerance {tolerance})")
    print(f"max-logit agreement {int(max_logits_agree.sum())}/{len(max_logits_agree)}")
    for name, report in (("reference", reference_report), ("sharded", sharded_report)):
        print(
            f"{name}: group {report['groupSize']}, prefill {report['prefillSeconds']:.3f}s, "
            f"decode {report['decodeSecondsPerToken'] * 1000:.1f} ms/token"
        )

    passed = tokens_agree and difference.max() <= tolerance
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])))
