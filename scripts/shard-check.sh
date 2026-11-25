#!/bin/bash
# Checks tensor-parallel sharding on one Mac: runs ShardCheck unsharded as the reference,
# then as two ring processes on localhost, and compares the logits of each step.
#
# By default the sharded run is teacher-forced: it feeds the reference's greedy tokens as its
# next inputs, so both runs see the same contexts and one flipped near tie cannot cascade.
# With --free-running, the sharded run feeds its own argmax, and its tokens must equal those
# of the reference.
#
# With --announce, rank 0 generates through a ClusterLanguageModel that announces each op to
# rank 1 with an all-sum, as a cluster worker does, and the tokens must equal those of a local
# generation.
#
# Usage: scripts/shard-check.sh [--free-running | --announce] <model-directory> [max-kl]
# Env: SHARD_CHECK_WORK (output directory), SHARD_CHECK_PORT (first ring port),
#      SHARD_CHECK_TOKENS (comma-separated prompt ids; default: an English sentence
#      pre-tokenized per model family), SHARD_CHECK_SKIP_BUILD=1.

set -euo pipefail

USAGE="usage: $0 [--free-running | --announce] <model-directory> [max-kl]"
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
esac

if [[ $# -lt 1 ]]; then
    echo "$USAGE" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$1"
MAX_KL="${2:-0.05}"
WORK="${SHARD_CHECK_WORK:-$(mktemp -d -t shard-check)}"
PORT="${SHARD_CHECK_PORT:-50211}"
DERIVED="$ROOT/.build/shard-check"
BINARY="$DERIVED/Build/Products/Release/ShardCheck"

mkdir -p "$WORK"

# xcodebuild puts mlx-swift_Cmlx.bundle, which holds the metallib, beside the binary.
if [[ "${SHARD_CHECK_SKIP_BUILD:-0}" != "1" ]]; then
    xcodebuild build -scheme ShardCheck -configuration Release \
        -destination 'platform=macOS' -skipPackagePluginValidation \
        -derivedDataPath "$DERIVED" -quiet
fi

COMMON_ARGS=()
if [[ "$MODE" == announce ]]; then
    COMMON_ARGS+=(--announce)
fi
if [[ -n "${SHARD_CHECK_TOKENS:-}" ]]; then
    COMMON_ARGS+=(--tokens "$SHARD_CHECK_TOKENS")
fi
SHARDED_ARGS=("${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}")
if [[ "$MODE" == teacher-forced ]]; then
    SHARDED_ARGS+=(--teacher-forcing "$WORK/reference.json")
fi

cat > "$WORK/hostfile.json" <<EOF
[["127.0.0.1:$PORT"], ["127.0.0.1:$((PORT + 1))"]]
EOF

echo "== unsharded reference"
"$BINARY" "$MODEL" "$WORK/reference.npy" "${COMMON_ARGS[@]+"${COMMON_ARGS[@]}"}"

echo "== sharded ($MODE): 2 ring processes on localhost"
MLX_RANK=0 MLX_HOSTFILE="$WORK/hostfile.json" \
    "$BINARY" "$MODEL" "$WORK/sharded.npy" "${SHARDED_ARGS[@]+"${SHARDED_ARGS[@]}"}" \
    > "$WORK/rank0.log" 2>&1 &
RANK0=$!
MLX_RANK=1 MLX_HOSTFILE="$WORK/hostfile.json" \
    "$BINARY" "$MODEL" "$WORK/rank1.npy" "${SHARDED_ARGS[@]+"${SHARDED_ARGS[@]}"}" \
    > "$WORK/rank1.log" 2>&1 &
RANK1=$!

STATUS=0
wait "$RANK0" || STATUS=$?
wait "$RANK1" || STATUS=$?
cat "$WORK/rank0.log"
if [[ "$MODE" == announce ]]; then
    cat "$WORK/rank1.log"
fi
if [[ $STATUS -ne 0 ]]; then
    echo "a ring process failed ($STATUS); rank 1 log:" >&2
    cat "$WORK/rank1.log" >&2
    exit "$STATUS"
fi

echo "== compare"
if [[ "$MODE" == announce ]]; then
    python3 -I "$ROOT/scripts/shard_check_compare.py" --tokens-only \
        "$WORK/reference.json" "$WORK/sharded.json"
else
    python3 -I "$ROOT/scripts/shard_check_compare.py" \
        "$WORK/reference.npy" "$WORK/sharded.npy" "$MAX_KL"
fi
