#!/bin/bash
# Checks tensor-parallel sharding on one Mac: runs ShardCheck as two ring processes on
# localhost, then unsharded, and compares the greedy tokens and logits.
#
# Usage: scripts/shard-check.sh <model-directory> [max-kl]
# Env: SHARD_CHECK_WORK (output directory), SHARD_CHECK_PORT (first ring port),
#      SHARD_CHECK_TOKENS (comma-separated prompt ids), SHARD_CHECK_SKIP_BUILD=1.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <model-directory> [max-kl]" >&2
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

TOKEN_ARGS=()
if [[ -n "${SHARD_CHECK_TOKENS:-}" ]]; then
    TOKEN_ARGS=(--tokens "$SHARD_CHECK_TOKENS")
fi

cat > "$WORK/hostfile.json" <<EOF
[["127.0.0.1:$PORT"], ["127.0.0.1:$((PORT + 1))"]]
EOF

echo "== sharded: 2 ring processes on localhost"
MLX_RANK=0 MLX_HOSTFILE="$WORK/hostfile.json" \
    "$BINARY" "$MODEL" "$WORK/sharded.npy" "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}" \
    > "$WORK/rank0.log" 2>&1 &
RANK0=$!
MLX_RANK=1 MLX_HOSTFILE="$WORK/hostfile.json" \
    "$BINARY" "$MODEL" "$WORK/rank1.npy" "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}" \
    > "$WORK/rank1.log" 2>&1 &
RANK1=$!

STATUS=0
wait "$RANK0" || STATUS=$?
wait "$RANK1" || STATUS=$?
cat "$WORK/rank0.log"
if [[ $STATUS -ne 0 ]]; then
    echo "a ring process failed ($STATUS); rank 1 log:" >&2
    cat "$WORK/rank1.log" >&2
    exit "$STATUS"
fi

echo "== unsharded reference"
"$BINARY" "$MODEL" "$WORK/reference.npy" "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}"

echo "== compare"
python3 -I "$ROOT/scripts/shard_check_compare.py" \
    "$WORK/reference.npy" "$WORK/sharded.npy" "$MAX_KL"
