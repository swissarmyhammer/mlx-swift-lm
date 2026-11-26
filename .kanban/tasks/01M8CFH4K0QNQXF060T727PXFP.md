---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m8cp9q622ep6ta8vk2v0fz0r
  text: |-
    Research + implementation notes.
    - Builder lives in MLXLMTestSupport (`TinyShardCheckpoint`), not a new library: ShardCheck (executable target) can depend on that regular target, and nothing in it uses @testable/Testing, so the xcodebuild Release build works. ShardCheck's own copy of `PseudoWordTokenizerLoader` was a duplicate of the support one and is removed; ShardCheck no longer depends on MLXScriptedLM directly.
    - Reuse: Qwen configs start from `Qwen35FixtureJSON.text(mtpLayers: 0)` and override a JSON shape (hidden 256, 4 layers / interval 2, heads 4/2, GDN heads 2/4 dim 64, MLP 512, 8 experts x 256 + shared 256, vocab 512, 4-bit affine g64). Inkling uses `Inkling2TinyModels.target(json:)` with the tiny target JSON moved from `Inkling2ModelTests.tinyJSON` into `Inkling2TinyModels.targetJSON`; `inkling2` drops `loop_config`. Prism writes its manifest from a JSON table (blocks 32/64/128) over the dense Qwen text config and randomizes rotated weights+signs. `RemoteModelExecutorTests`' private per-family `ShardableFamilyConfiguration` table is replaced by `TinyShardCheckpoint.makeModel(of:)`.
    - Saving `model.parameters()` round-trips through the loader (`quantizationParameters` quantizes the layers whose `.scales` the file holds), so the writer quantizes with the config's `perLayerQuantization` and needs no sanitize mapping.
    - The default ShardCheck prompt is pre-tokenized for real vocabularies (ids up to ~248k), so the tiny mode passes `TinyShardCheckpoint.prompt` (ids < 48, the tiny Inkling unpadded vocabulary), printed by `ShardCheck tiny-checkpoint`.
    - DISCOVERY 1 (xcodebuild): twice the ShardCheck build printed `error: the following command failed with exit code 0 but produced no further output` on the MLXVLM SwiftCompile, exited 0, and kept STALE MLXVLM objects, so shard-check ran an old binary and a mutation looked uncaught. A mutation result is only trustworthy when the build log has no `error:` line. Filed as a new task.
    - DISCOVERY 2 (Inkling tiny target): the MoE branch of the tiny Inkling 2 target does not reach the logits: zeroing every expert down_proj global_scale (in memory) or every expert scale / the router weight (in the checkpoint) leaves logits bit-identical, while zeroing self_attn.o_proj changes them by 2.4. So the tiny Inkling check covers attention sharding but cannot see an Inkling MoE all-sum break (dropping `Inkling2SparseMoE`'s all-sum left the sharded logits bit-identical; ShardCheck's GreedyRun reaches the MoE through `prefill` -> `callAsFunction`, 54 calls with group set, never `decode`). Filed as a new task.
  timestamp: 2026-11-26T19:46:10.242968+00:00
- actor: claude-code
  id: 01m8cq009twmf4v3p7nkwrq90j
  text: |-
    ### implement — changed
    - evidence: 14 files. New: Tests/MLXLMTestSupport/TinyShardCheckpoint.swift (per-family recipe table, config + seeded random weights, write/makeModel/configurationJSON, prompt), tools/ShardCheck/TinyCheckpointCommand.swift (`ShardCheck tiny-families`, `ShardCheck tiny-checkpoint <family> <dir>`), Tests/MLXLMTests/TinyShardCheckpointTests.swift. Changed: Package.swift (ShardCheck depends on MLXLMTestSupport, not MLXScriptedLM), tools/ShardCheck/ShardCheck.swift (dispatch, usage, duplicate PseudoWordTokenizerLoader removed), Tests/MLXLMTestSupport/Inkling2TinyModels.swift (targetJSON moved in), Tests/MLXLMTests/Inkling2Tests.swift (tinyJSON aliases it), Tests/MLXLMTests/RemoteModelExecutorTests.swift (per-family table replaced by TinyShardCheckpoint.makeModel), scripts/shard-check.sh (check_model function; `--tiny [family...]` loops every family x teacher-forced+announce on its own ports, prints PASS/FAIL table), scripts/shard_check_compare.py (argparse, `--no-load-peak-gate`), scripts/test_shard_check_compare.py (4 tests), DocC distributed-inference.md, skills/mlx-swift-lm/references/distributed-inference.md.
    - tiny shard-check, 2 localhost ring processes, final code: 18/18 PASS in 62s incl. build — inkling2, inkling2_loop, prism_hadamard_qwen35, qwen3_5, qwen3_5_moe, qwen3_5_text, qwen3_8, qwen3_8_moe, qwen3_8_text, each teacher-forced (agree 9/9) and --announce (tokens equal). Checkpoints 1.0-5.1 MB. Normal mode (teacher-forced/free-running/announce) still runs with the 75% gate on.
    - Break detection (reverted after): dropping the Qwen35SparseMoeBlock all-sum -> qwen3_5_moe and qwen3_8_moe FAIL teacher-forced (6/9 mismatch) and announce; dropping Inkling 2's attention-output all-sum (makeShardedToAll -> makeInputShard) -> all 4 Inkling checks FAIL (inkling2 1/9, inkling2_loop 6/9 mismatch). Dropping the Inkling MoE all-sum is NOT caught (tiny target's MoE does not reach logits) -> ^4nasszt. xcodebuild stale-binary hazard -> ^8pkmyp2.
    - CI: not wired. Mac jobs of pull_request.yml run only in ml-explore/mlx-swift-lm on its self-hosted runner; documented.
    - Tests: MLXLMTests XCTest 1261/0 failures + Swift Testing 2782 passed (3 pre-existing known issues); MLXFoundationModelsTests 618; MLXGuidedGenerationTests 96; CXGrammarTests 7; MLXHuggingFaceMacrosTests 11 XCTest. python self-check 25 OK. verify-docs exit 0. swift-format applied; swiftlint no_magic_numbers/force rules clean on new Swift files; ruff format on the python.
    - next: review
  timestamp: 2026-11-26T19:58:20.474656+00:00
- actor: claude-code
  id: 01m8cqr4mxzy62rcjqq74anya1
  text: |-
    ### review — findings
    - evidence: review sha HEAD~1..HEAD (14672856): 1 finding (1 confirmed, 0 refuted) — scripts/shard-check.sh:157
    - next: implement the finding in the 2026-11-26 14:07 Review Findings section; task stays in review
  timestamp: 2026-11-26T20:11:31.357642+00:00
- actor: claude-code
  id: 01m8cqrhnzmhw612sx2q2nja6w
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — 13 files (TinyShardCheckpoint, ShardCheck tiny-families/tiny-checkpoint, shard-check.sh --tiny, tests, docs); --tiny 18/18 PASS
    - test: green — MLXLMTests 1261 XCTest + 2782 ST, FM 618, Guided 96, CXGrammar 7, Macros 11; swift build ok; python 25 OK; verify-docs ok; --tiny PASS
    - commit: 14672856
    - review: findings — scripts/shard-check.sh:157
  timestamp: 2026-11-26T20:11:44.703184+00:00
- actor: claude-code
  id: 01m8cqxxtpvt6ewmq37vj4vnrc
  text: |-
    Fix of the shard-check.sh:157 finding.
    - DISCOVERY: on macOS /bin/bash 3.2 the old code did not actually exit 0: an empty `families` hits `set -u` ("families[@]: unbound variable") and exits 1 by accident with no useful message. On bash >= 4.4 (empty arrays are fine under set -u) it would have exited 0 as the finding says. So the regression tests assert the exact status AND message, not just non-zero; a bare non-zero assert passed on the old code (tried first, discarded).
    - check_tiny: the listing is a command substitution with `|| { echo "could not list the tiny families (ShardCheck tiny-families)" >&2; exit 1; }`, split with a here-string (blank lines dropped), and an empty list fails with "ShardCheck tiny-families listed no family to check".
    - Whole-script sweep: no pipes; the other `$(...)` are plain assignments (ROOT, WORK with mktemp - verified `set -e` aborts on `W="${X:-$(false)}"` in bash 3.2) or already guarded (`if ! tokens="$(...)"`). The other swallowed failures: check_model runs as an `if` condition in check_tiny, where set -e is off, so `mkdir -p "$work"` and the hostfile heredoc now `|| return`.
    - Tests: TinyFamilyListTests in scripts/test_shard_check_compare.py copies the script into a temp tree with a fake ShardCheck at .build/shard-check/Build/Products/Release/ShardCheck, SHARD_CHECK_SKIP_BUILD=1. RED: both failed (got the unbound-variable line). GREEN: 27 tests OK. ruff format --line-length 100.
  timestamp: 2026-11-26T20:14:40.982248+00:00
- actor: claude-code
  id: 01m8cqy02hy35p3y82g66za821
  text: |-
    ### implement — changed
    - evidence: 2 files — scripts/shard-check.sh, scripts/test_shard_check_compare.py. Fake ShardCheck `exit 1` -> --tiny status 1 "could not list the tiny families (ShardCheck tiny-families)"; fake `exit 0` (empty list) -> status 1 "ShardCheck tiny-families listed no family to check". Real `scripts/shard-check.sh --tiny` (with build): status 0, 18/18 PASS (9 families x teacher-forced/announce), build log has no `error:` line (^8pkmyp2). python3 -I scripts/test_shard_check_compare.py: 27 tests OK. No Swift change. Finding flipped to [x].
    - next: review
  timestamp: 2026-11-26T20:14:43.281167+00:00
position_column: doing
position_ordinal: '8180'
title: 'ShardCheck: write tiny random checkpoints so shard-check runs with no real checkpoint'
---
Found in ^ze3xffb. `scripts/shard-check.sh` (2 ring processes on localhost vs the unsharded logits) is the only sharded-vs-local logits check that runs the production `shard` path end to end, but it needs a model directory. For ^ze3xffb a throwaway test wrote tiny random `qwen3_5_moe` (4-bit affine, 8 experts) and `prism_hadamard_qwen35` (4-bit rotated, blocks 32/64/128) checkpoints (config.json + model.safetensors, ~2-5 MB) and both passed shard-check (teacher-forced and --announce); a per-shard signs mutation and an MoE missing-all-sum mutation each failed 9/9 steps. The writer is not committed.

- [x] Add a way to write tiny random checkpoints of each shardable family (a ShardCheck subcommand or a script), with the configs that split across two processes.
- [x] Run shard-check over each of them in CI or document the command, so a new shardable family gets a logits check with no download.
- [x] Note that the 75% load-peak gate is not meaningful for megabyte-sized checkpoints (both read 0.00 GiB). #cluster

## Review Findings (2026-11-26 14:07)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 11 file(s) reviewed, 10 not reviewed.

> 10 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M8853Y27H14BR99PEZE3XFFB.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8853Y27H14BR99PEZE3XFFB.md` — no validator matches this file
> - `.kanban/tasks/01M8CFH4K0QNQXF060T727PXFP.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8CFH4K0QNQXF060T727PXFP.md` — no validator matches this file
> - `.kanban/tasks/01M8CP9XNJQWH4DA8GY4NASSZT.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8CP9XNJQWH4DA8GY4NASSZT.md` — no validator matches this file
> - `.kanban/tasks/01M8CPA14AQJAKTGY6K8PKMYP2.jsonl` — no validator matches this file
> - `.kanban/tasks/01M8CPA14AQJAKTGY6K8PKMYP2.md` — no validator matches this file
> - `Libraries/MLXLMCommon/Documentation.docc/distributed-inference.md` — no validator matches this file
> - `skills/mlx-swift-lm/references/distributed-inference.md` — no validator matches this file

- [x] `scripts/shard-check.sh:157` `completeness/public-output-contract` — If `"$BINARY" tiny-families` fails, the exit status is discarded by the `done < <(...)` process substitution. `check_tiny` then runs zero families, prints `==== tiny checkpoints in $WORK` with an empty result list, and exits 0. `--tiny` therefore reports success when no family was checked. The script's stated contract is to print one PASS or FAIL line per family and check, and to fail when any check fails. Capture the family list into a variable, for example `families_text="$("$BINARY" tiny-families)" || { echo "could not list the tiny families" >&2; exit 1; }`, then split it into `families`. Also fail when the resulting list is empty, so that a silent zero-check run cannot pass.
