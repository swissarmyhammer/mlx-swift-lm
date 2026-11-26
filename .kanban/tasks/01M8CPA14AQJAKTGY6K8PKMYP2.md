---
assignees:
- claude-code
position_column: todo
position_ordinal: '8380'
title: 'shard-check.sh: a ShardCheck build that logs "error:" but exits 0 leaves a stale binary'
---
Found in ^727pxfp. Twice, `xcodebuild build -scheme ShardCheck -configuration Release ... -quiet` (the build step of `scripts/shard-check.sh`) printed `error: the following command failed with exit code 0 but produced no further output` for `SwiftCompile normal arm64 (in target 'MLXVLM')`, exited 0, and kept the old MLXVLM objects. The script went on with a stale binary: a temporary mutation of `Libraries/MLXVLM/Models/Inkling2MoE.swift` produced sharded logits bit-identical to the unmutated run. A rebuild after that worked.

- [ ] Make the build step fail when the xcodebuild output holds an `error:` line, even when xcodebuild exits 0 (or find the cause of the spurious failure and remove it).
- [ ] Show it: a build whose log holds such a line stops the script before any check runs. #cluster