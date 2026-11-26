#!/bin/bash
# Checks tensor-parallel sharding on one Mac: runs ShardCheck unsharded as the reference,
# then as two ring processes on localhost, and compares the logits of each step.
#
# A step passes when each run's argmax is in the other's top-5 tokens, or within 2 bf16 ulps of
# the other's 5th-highest logit: the top-k check of vLLM's check_logprobs_close
# (https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py). KL is printed, not gated.
#
# By default the sharded run is teacher-forced: it feeds the reference's greedy tokens as its
# next inputs, so both runs see the same contexts and one flipped near tie cannot cascade.
# With --free-running, the sharded run feeds its own argmax, and its tokens must equal those
# of the reference.
#
# Each rank loads its share layer by layer, and the load peak of rank 0 must stay at most 75%
# of the unsharded load peak.
#
# With --announce, rank 0 generates through a ClusterLanguageModel that announces each op to
# rank 1 with an all-sum, as a cluster worker does, and the tokens must equal those of a local
# generation.
#
# With --tiny, no model directory is needed: for each model family that shards (or each family
# named), ShardCheck writes a tiny checkpoint on random weights (a few MB) into the output
# directory, and the script runs the teacher-forced check and the --announce check on it, with
# the prompt ids the tiny vocabulary holds. Both load peaks of a tiny checkpoint read about
# 0.00 GiB, so the 75% load-peak gate says nothing there, and the tiny mode alone turns it off
# (--no-load-peak-gate of shard_check_compare.py). The script prints one PASS or FAIL line per
# family and check, and fails when any check fails, or when ShardCheck lists no family to check.
#
# The script builds ShardCheck first, keeps the build output in build.log of the output
# directory, and fails before any check when that output holds an `error:` line, even when
# xcodebuild exits 0.
#
# Usage: scripts/shard-check.sh [--free-running | --announce] <model-directory>
#        scripts/shard-check.sh --tiny [<family>...]
# Env: SHARD_CHECK_WORK (output directory), SHARD_CHECK_PORT (first ring port),
#      SHARD_CHECK_TOKENS (comma-separated prompt ids; default: an English sentence
#      pre-tokenized per model family; ignored by --tiny), SHARD_CHECK_DECODE_STEPS (decode
#      steps after the prefill; default: ShardCheck's 8), SHARD_CHECK_SKIP_BUILD=1.

set -euo pipefail

USAGE="usage: $0 [--free-running | --announce] <model-directory>
       $0 --tiny [<family>...]"
MODE=teacher-forced
case "${1:-}" in
--free-running)
    MODE=free-running
    shift
    ;;
--announce)
    MODE=announce
    shift
    ;;
--tiny)
    MODE=tiny
    shift
    ;;
esac

if [[ "$MODE" != tiny && $# -ne 1 ]]; then
    echo "$USAGE" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${SHARD_CHECK_WORK:-$(mktemp -d -t shard-check)}"
PORT="${SHARD_CHECK_PORT:-50211}"
DERIVED="$ROOT/.build/shard-check"
BINARY="$DERIVED/Build/Products/Release/ShardCheck"
# The ports of one check: one per ring process.
PORTS_PER_CHECK=2
# The checks of each family in the tiny mode.
TINY_MODES=(teacher-forced announce)

# The output of the ShardCheck build, kept for the failure report.
BUILD_LOG="$WORK/build.log"
# A diagnostic line of an error, from xcodebuild or from a compiler it ran.
BUILD_ERROR_PATTERN='(^|[[:space:]])error:'

mkdir -p "$WORK"

# Builds ShardCheck, shows the output, and keeps it in BUILD_LOG. Exits with status 1 when the
# output holds an `error:` line, even when xcodebuild exits 0: xcodebuild has logged `error:
# the following command failed with exit code 0 but produced no further output` for an MLXVLM
# compile, exited 0 and kept the old objects, so the checks ran a stale binary (^8pkmyp2).
build_shard_check() {
    # xcodebuild puts mlx-swift_Cmlx.bundle, which holds the metallib, beside the binary.
    # -quiet still prints each warning and error; `pipefail` keeps the status of xcodebuild.
    xcodebuild build -scheme ShardCheck -configuration Release \
        -destination 'platform=macOS' -skipPackagePluginValidation \
        -derivedDataPath "$DERIVED" -quiet 2>&1 | tee "$BUILD_LOG"
    local errors
    if errors="$(grep -E "$BUILD_ERROR_PATTERN" "$BUILD_LOG")"; then
        echo "the ShardCheck build exited 0 but logged (full log: $BUILD_LOG):" >&2
        echo "$errors" >&2
        echo "the ShardCheck build logged an error: line, so no check ran" >&2
        exit 1
    fi
}

if [[ "${SHARD_CHECK_SKIP_BUILD:-0}" != "1" ]]; then
    build_shard_check
fi

# Runs one check, and returns its status.
# Usage: check_model <mode> <model-directory> <work-directory> <first-port> <tokens>
#        [<compare option>...]
# <tokens> is a comma-separated prompt, or empty for ShardCheck's default prompt.
check_model() {
    local mode="$1" model="$2" work="$3" port="$4" tokens="$5"
    shift 5
    local compare_options=("$@")
    # check_tiny calls this function as an `if` condition, where `set -e` is off, so each step
    # whose failure would otherwise go on returns its status explicitly.
    mkdir -p "$work" || return

    local common_args=()
    if [[ "$mode" == announce ]]; then
        common_args+=(--announce)
    fi
    if [[ -n "$tokens" ]]; then
        common_args+=(--tokens "$tokens")
    fi
    if [[ -n "${SHARD_CHECK_DECODE_STEPS:-}" ]]; then
        common_args+=(--decode-steps "$SHARD_CHECK_DECODE_STEPS")
    fi
    local sharded_args=("${common_args[@]+"${common_args[@]}"}")
    if [[ "$mode" == teacher-forced ]]; then
        sharded_args+=(--teacher-forcing "$work/reference.json")
    fi

    cat > "$work/hostfile.json" <<EOF || return
[["127.0.0.1:$port"], ["127.0.0.1:$((port + 1))"]]
EOF

    echo "== unsharded reference"
    "$BINARY" "$model" "$work/reference.npy" "${common_args[@]+"${common_args[@]}"}" || return

    echo "== sharded ($mode): 2 ring processes on localhost"
    launch_rank() {
        local rank="$1" output="$2"
        MLX_RANK="$rank" MLX_HOSTFILE="$work/hostfile.json" \
            "$BINARY" "$model" "$work/$output" "${sharded_args[@]+"${sharded_args[@]}"}" \
            > "$work/rank$rank.log" 2>&1 &
    }
    launch_rank 0 sharded.npy
    local rank0=$!
    launch_rank 1 rank1.npy
    local rank1=$!

    local status=0
    wait "$rank0" || status=$?
    wait "$rank1" || status=$?
    cat "$work/rank0.log"
    if [[ "$mode" == announce ]]; then
        cat "$work/rank1.log"
    fi
    if [[ $status -ne 0 ]]; then
        echo "a ring process failed ($status); rank 1 log:" >&2
        cat "$work/rank1.log" >&2
        return "$status"
    fi

    echo "== compare"
    if [[ "$mode" == announce ]]; then
        python3 -I "$ROOT/scripts/shard_check_compare.py" --tokens-only \
            "${compare_options[@]+"${compare_options[@]}"}" \
            "$work/reference.json" "$work/sharded.json"
    else
        python3 -I "$ROOT/scripts/shard_check_compare.py" \
            "${compare_options[@]+"${compare_options[@]}"}" \
            "$work/reference.npy" "$work/sharded.npy"
    fi
}

# Writes the tiny checkpoint of each family, runs each check of TINY_MODES on it with its own
# ports, prints one line per family and check, and exits with status 1 when any check fails,
# or when the family list cannot be read or is empty.
# Usage: check_tiny [<family>...]; no family means every family that shards.
check_tiny() {
    local families=("$@") family
    if [[ ${#families[@]} -eq 0 ]]; then
        # A command substitution, not `done < <(...)`: a process substitution discards the
        # status of ShardCheck, so a failed listing would check no family and pass.
        local families_text
        families_text="$("$BINARY" tiny-families)" || {
            echo "could not list the tiny families (ShardCheck tiny-families)" >&2
            exit 1
        }
        while IFS= read -r family; do
            if [[ -n "$family" ]]; then
                families+=("$family")
            fi
        done <<< "$families_text"
        if [[ ${#families[@]} -eq 0 ]]; then
            echo "ShardCheck tiny-families listed no family to check" >&2
            exit 1
        fi
    fi

    local results=() failures=0 next_port="$PORT" mode tokens
    for family in "${families[@]}"; do
        local checkpoint="$WORK/$family/checkpoint"
        echo "==== $family: writing a tiny checkpoint"
        if ! tokens="$("$BINARY" tiny-checkpoint "$family" "$checkpoint")"; then
            results+=("FAIL $family: the tiny checkpoint was not written")
            failures=$((failures + 1))
            continue
        fi
        for mode in "${TINY_MODES[@]}"; do
            echo "==== $family: $mode"
            if check_model "$mode" "$checkpoint" "$WORK/$family/$mode" "$next_port" \
                "$tokens" --no-load-peak-gate; then
                results+=("PASS $family $mode")
            else
                results+=("FAIL $family $mode")
                failures=$((failures + 1))
            fi
            next_port=$((next_port + PORTS_PER_CHECK))
        done
    done

    echo "==== tiny checkpoints in $WORK"
    printf '%s\n' "${results[@]}"
    if [[ $failures -ne 0 ]]; then
        echo "$failures check(s) failed" >&2
        exit 1
    fi
}

if [[ "$MODE" == tiny ]]; then
    check_tiny "$@"
else
    check_model "$MODE" "$1" "$WORK" "$PORT" "${SHARD_CHECK_TOKENS:-}"
fi
