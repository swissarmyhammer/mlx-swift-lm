---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8crbhwmp1j8jbahck7czkaa
  text: |-
    Research on the cause of the spurious `failed with exit code 0` (not reproduced; ruled out three suspects):
    - Concurrent builds on one DerivedData: NOT it. Two xcodebuilds launched together on `.build/shard-check` -> one fails LOUDLY: `error: unable to attach DB ... build.db: database is locked Possibly there are two concurrent builds`, exit 65 (BUILD FAILED). The other builds fine.
    - Stale Metal toolchain cryptex path: NOT it. The only mount is `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.2.47.0.80zEbX` (mounted Oct 14), and the XCBuildData build descriptions of `.build/shard-check` name only that path (568 refs, one variant). Deleting XCBuildData would change nothing, so it was not deleted.
    - SwiftPM/xcodebuild sharing build dirs: NOT it. The script uses its own `-derivedDataPath .build/shard-check`; SwiftPM uses `.build/out`.
    - The failing builds' xcactivitylogs are gone (Logs/Build keeps only recent ones), so the failing task cannot be inspected. Remaining guess: a swift-frontend lost to resource pressure (the box runs big models); unconfirmed.
    - DISCOVERY (why no mtime freshness check): Xcode 27.2's Swift compile does not rewrite an object whose content is unchanged, so the link does not rerun and the ShardCheck binary keeps its old mtime. Measured: after `touch Libraries/MLXVLM/Models/Inkling2MoE.swift` and a successful build (it ran SwiftCompile for MLXVLM), Inkling2MoE.o stayed at 13:45 and ShardCheck at 13:48; four sources rewritten at 13:58 (format-only) are also newer than the correct binary. A real content change (appending a func) did rebuild: .o 14:19:38, binary 14:19:46 (reverted, rebuilt). So "binary newer than newest Libraries/**/*.swift" would fail correct builds after any touch/format/checkout; not added. The `error:` gate is the guard.
    - Implementation: `build_shard_check` in scripts/shard-check.sh pipes `xcodebuild ... -quiet 2>&1 | tee "$WORK/build.log"` (`-quiet` still prints every warning and error, so it stays; `pipefail` keeps xcodebuild's status), then greps `(^|[[:space:]])error:`; on a match prints the full-log path, the matching lines, and `the ShardCheck build logged an error: line, so no check ran`, exit 1, before ShardCheck runs at all (not even tiny-families).
    - Tests (BuildErrorTests): fake `xcodebuild` first on PATH. RED: `error:` + exit 0 -> got status 1 'could not list the tiny families' and the fake ShardCheck ran. GREEN after the change. Also guards: a failing xcodebuild (exit 65) runs no check — verified it FAILS with `set -eu` (no pipefail), i.e. it catches a tee swallowing the status; a build that logs only a warning still runs ShardCheck.
  timestamp: 2026-11-26T20:22:07.508453+00:00
- actor: claude-code
  id: 01m8crbpgaajgkp0t1sbyk7m92
  text: |-
    ### implement — changed
    - evidence: 2 files — scripts/shard-check.sh (build_shard_check: tee to $WORK/build.log, fail on `error:` line before any check; header doc), scripts/test_shard_check_compare.py (tiny_run helper with optional fake xcodebuild on PATH + ran-marker; BuildErrorTests: 4 tests). `python3 -I scripts/test_shard_check_compare.py`: 31 tests OK. Real `scripts/shard-check.sh --tiny` (with build): exit 0, 18/18 PASS, build.log has no `error:` line. bash -n ok; ruff format applied. No Swift change; no build state deleted.
    - next: review
  timestamp: 2026-11-26T20:22:12.234564+00:00
- actor: claude-code
  id: 01m8crfs5c9j22a89eagv2haw5
  text: |-
    ### review — clean
    - evidence: review sha HEAD~1..HEAD (a8dfe3d2) — 0 findings (9 validator pairs attempted, 0 failed); reviewed scripts/shard-check.sh, scripts/test_shard_check_compare.py; 4 .kanban files matched no validator
    - next: none — moved to done
  timestamp: 2026-11-26T20:24:26.028219+00:00
- actor: claude-code
  id: 01m8crg4csdpffa753jhfymfxp
  text: |-
    ### finish iteration 1 — clean
    - implement: changed — scripts/shard-check.sh, scripts/test_shard_check_compare.py
    - test: green — python 31 OK, bash -n ok, --tiny 18/18 PASS, build.log 0 error: lines, usage paths exit 2
    - commit: a8dfe3d2
    - review: clean — 0 findings; task done
  timestamp: 2026-11-26T20:24:37.529441+00:00
position_column: done
position_ordinal: ffffff8280
title: 'shard-check.sh: a ShardCheck build that logs "error:" but exits 0 leaves a stale binary'
---
Found in ^727pxfp. Twice, `xcodebuild build -scheme ShardCheck -configuration Release ... -quiet` (the build step of `scripts/shard-check.sh`) printed `error: the following command failed with exit code 0 but produced no further output` for `SwiftCompile normal arm64 (in target 'MLXVLM')`, exited 0, and kept the old MLXVLM objects. The script went on with a stale binary: a temporary mutation of `Libraries/MLXVLM/Models/Inkling2MoE.swift` produced sharded logits bit-identical to the unmutated run. A rebuild after that worked.

- [x] Make the build step fail when the xcodebuild output holds an `error:` line, even when xcodebuild exits 0 (or find the cause of the spurious failure and remove it).
- [x] Show it: a build whose log holds such a line stops the script before any check runs. #cluster