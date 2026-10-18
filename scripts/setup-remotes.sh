#!/bin/bash
# Point the `upstream` remote at ml-explore/mlx-swift-lm.
#
# GitHub records this fork's parent as ctymoszek/mlx-swift-lm, a stale fork of
# ml-explore, so `gh repo clone` adds that fork as `upstream`. Run this once
# after you clone.

set -euo pipefail
cd "$(dirname "$0")/.."

readonly UPSTREAM_URL="git@github.com:ml-explore/mlx-swift-lm.git"

if git remote get-url upstream >/dev/null 2>&1; then
    git remote set-url upstream "$UPSTREAM_URL"
else
    git remote add upstream "$UPSTREAM_URL"
fi
git fetch upstream
git remote -v
