---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m63pw2vqexmec3gny0aecxfg
  text: |-
    Picked up. Worktree branch was on main (e3d4a20e), reset to mtp (17dd4def) before any work, per the brief.

    Research:
    - `CommittedTurnSplice.suffixStart` (shared by QwenCommittedTurnRule + DSMLCommittedTurnRule) splices AT the commit when uncommitted is [] or [eot], and declines any other uncommitted tokens. Two existing policy tests assert that decline (`declines an uncommitted lookahead the render cannot explain`, Qwen + DSML); the card's order to feed the dropped token supersedes them.
    - `reconcilePromptCache` builds a `PromptCacheTurn` with no uncommitted tokens; `ExecutorPromptCacheEntry` keeps none; `commitOutcome` drops `generatedTokens.dropFirst(committed)` on the floor.
    - The fed input is always `promptTokens[suffixStart...]` (executor `narrowed`, ChatSession `LMInput(tokens:)`), and `PromptCacheCheckpoint.prefillToCheckpoint` feeds `promptTokens[start..<boundary]`. A token outside the render has no way in today, so the decision has to carry it explicitly.
    - ChatSession passes uncommitted tokens, so once the splice feeds them ChatSession must feed them too (else the same bug moves there).
    - Executor usage: total = fed + prefilled + reused. MTPPromptCacheTests asserts MTP recomputed == plain recomputed, and the card pins 26 of 233, so the fed uncommitted token (the tail of the previous answer) is excluded from the prompt usage counts.
    - Spilled entries: ExecutorPromptCacheFile writes tokens/renderTokens only; the uncommitted tokens must round-trip too or a spilled MTP session hits the bug again.

    Plan: decision cases `.appendSuffix/.appendSuffixToMain` gain `uncommittedTokens: [Int] = []` (fed ahead of the suffix); splice feeds `uncommitted` (minus a trailing commit) at the commit; `reconcilePromptCache` takes `previousGenerationUncommittedTokens`; executor entry/plan/checkpoint feed/usage/file carry them; ChatSession feeds them.
  timestamp: 2026-10-29T11:31:38.487861+00:00
- actor: claude-code
  id: 01m63v0w989ep2bg2vrtr02x6s
  text: |-
    Implementation landed (worktree branch worktree-agent-a0c8ccf53844acdcb, on mtp 17dd4def).

    Design:
    - `PromptCacheReuseDecision.appendSuffix/.appendSuffixToMain` gain `uncommittedTokens: [Int] = []`: tokens fed AHEAD of `promptTokens[suffixStart...]`. `PromptCacheReuse` carries them plus `fedTokens(of:)`. `reconcilePromptCache` takes `previousGenerationUncommittedTokens`.
    - `CommittedTurnSplice` (Qwen + DSML): the think-block guard reads cached + uncommitted; the feed is the uncommitted tokens (a trailing commit dropped, because the render's commit stands for it) at the commit index. It declines uncommitted tokens after a ledger that holds the commit, and uncommitted tokens that hold a commit before their end. The old `declines an uncommitted lookahead the render cannot explain` tests (Qwen, DSML) asserted the behavior this card replaces. I removed them; the new `CommittedTurnUncommittedTokenTests` covers both rules.
    - Executor: `ExecutorPromptCacheEntry.uncommittedTokens` (set by commitOutcome from `generatedTokens.dropFirst(committed)`), passed to reconcile. The plan feeds them first, in `input` or in the checkpoint feed (`PromptCacheCheckpoint.prefillToCheckpoint(after:)`). Usage subtracts them (`uncommittedTokenCount`), so usage stays total = render and the MTP recompute equals plain. The plan log line adds ` uncommitted=N`. The spill file writes/reads the `uncommittedTokens` ledger (the spool never outlives a process, so the key is required and there is no compat path).
    - ChatSession now gets the splice for [t] (it used to decline). It feeds the tokens in the suffix input or in the first checkpoint feed, and reports cached = suffixStart - uncommitted.count, so cached + prompt = render (prompt = fed, +1).
    - Test support refactor (to avoid copies): `QwenThinkBlockScalarTokenizer` and `Qwen35MTPFixture.markerFreeDecode` moved to MLXLMTestSupport. `ExecutorReasoningConversation` was extracted from ExecutorPromptCacheReasoningFollowUpTests and is shared with the new suite.

    TDD: policy tests RED (5 behavior tests failed on the decline), executor `aFollowUpReadsEveryToken...` RED (follow-up ledger missed t), then GREEN. ChatSession: its RED was not observed in the intermediate state. I proved it with a mutation instead: dropping the uncommitted tokens from the checkpoint feed makes `ChatSessionUnfedTokenSpliceTests` trip the MTP timeline assertion (stale sharedKV). Reverted.

    Gotcha: the follow-up pass extends the caches of the first entry object in place (same KVCache refs), thus a spill test must prefill caches of its own.

    Release (bench.lock held, load per run):
    - MTPPromptCacheTests (load 7.82→10.98): PASS. Plain ledger 207, MTP ledger 206. Turn 2 plain `reused=207 fed=26 rule=splice`, MTP `reused=207 fed=26 uncommitted=1 rule=splice`. Turn-2 ledger 324 for BOTH (before: MTP one short). Recomputed turn 2: MTP 26 of 233, plain 26 of 233. Drafter resumed (start position 0). 85.7 vs 58.9 tok/s.
    - qwen38FourBitMTPSystemPromptCache (load 6.70→7.69): PASS, edit turn prefilled 123 (MTP and plain). Follow-ups: MTP prefills 38 vs plain 37 (the +1 unfed token), cached+prefill equal.
    - qwen38FourBitMTPGeneration: PASS, but the default (bs4 pMin .75) measured 95.8/85.0/81.2 (load 6.7→7.05, the test's own noise 7.4%), and on a re-run 87.4/70.9/72.0 (load 9.6→44.5, noise 18.1%, plain also fell to 55). Not within 3% of 100.4/87.4/83.4. This cold single-turn decode runs no changed code (no uncommitted tokens, no splice), and the drop follows the external load (EditorKit/Testudo builds), but I could not get a clean confirmation. Re-run when the machine is quiet.
  timestamp: 2026-10-29T12:44:09.896295+00:00
- actor: claude-code
  id: 01m63v15nvypatxjygn0beyga7
  text: |-
    ### implement — changed
    - evidence: branch worktree-agent-a0c8ccf53844acdcb (base mtp 17dd4def); `swift build --build-tests` 0 Swift warnings; xctest MLXLMTests 2020 Swift Testing + 920 XCTest, 0 failures; MLXGuidedGenerationTests 72; MLXFoundationModelsTests 420; CXGrammarTests 7; MLXHuggingFaceMacrosTests 6; 0 failures, 0 skipped. Release MTPPromptCacheTests PASS (turn 2 MTP 26/233 = plain 26/233, `uncommitted=1` splice, ledger 324 both, drafter resumed), qwen38FourBitMTPSystemPromptCache PASS (edit 123), qwen38FourBitMTPGeneration PASS but default 95.8/85.0/81.2 at load ~7 (re-run 87.4/70.9/72.0 at load 9.6→44): perf guard unconfirmed because of external load.
    - files: Libraries/MLXLMCommon/{PromptCacheReusePolicy,PromptCacheCheckpoint,ChatSession}.swift, Libraries/MLXLMCommon/Tool/Parsers/CommittedTurnSplice.swift, Libraries/MLXFoundationModels/{ExecutorPromptCache,ExecutorPromptCacheFile,MLXLanguageModel}.swift; tests: new CommittedTurnUncommittedTokenTests, ChatSessionUnfedTokenSpliceTests, ExecutorPromptCacheUnfedTokenTests, ExecutorReasoningConversation, MLXLMTestSupport/QwenThinkBlockScalarTokenizer; edited ExecutorPromptCacheReasoningFollowUpTests, Qwen/DSMLCommittedTurnRuleTests, MuseGlimmerTests
    - next: /review; re-run qwen38FourBitMTPGeneration on a quiet machine
  timestamp: 2026-10-29T12:44:19.515867+00:00
- actor: claude-code
  id: 01m63vctv4686mby0n399xgmby
  text: |-
    ### review — findings
    - evidence: review sha 0534c446~1..0534c446 — 5 findings (21 confirmed, 6 refuted) — Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift:195, Libraries/MLXLMCommon/ChatSession.swift:1916, Tests/MLXFoundationModelsTests/ExecutorReasoningConversation.swift:26, :29, :32
    - next: implement the 5 open items in the 2026-10-29 07:45 Review Findings section
  timestamp: 2026-10-29T12:50:41.636198+00:00
- actor: claude-code
  id: 01m63vdjr8w0t9906d3dz1tgrs
  text: |-
    ### finish iteration 1 — findings
    - implement: changed — splice feeds uncommitted last tokens; executor keeps/spills them; ChatSession uses the same splice (worktree 2994ef33, applied cleanly onto mtp)
    - test: green in worktree (5 bundles, MTPPromptCacheTests 26/233 both, ledger 324 both, SystemPromptCache edit 123); merged-tree test run in progress; speed guard unconfirmed under load
    - commit: 0534c446
    - review: findings — ExecutorPromptCacheFile.swift:195, ChatSession.swift:1916, ExecutorReasoningConversation.swift:26, :29, :32
  timestamp: 2026-10-29T12:51:06.120689+00:00
- actor: claude-code
  id: 01m63w82bhnytbpm5aqx8sejmp
  text: |-
    Review findings (2026-10-29 07:45) worked, in worktree branch worktree-agent-a50e8aaca8bea5119 (reset to mtp 0534c446 first).

    - ExecutorPromptCacheFile: new `TokenLedger.decodeIfPresent` reads a missing `uncommittedTokens` key as []; type doc now states both compat paths (drafter keys, uncommitted key). TDD: new `ExecutorPromptCacheUnfedTokenTests.aFileWrittenBeforeTheUncommittedTokensRestoresItsEntryAndItsDrafterState` writes the follow-up entry (with drafter state) minus the key, reads with the fixture drafter; RED `.ledgerNotDecodable("uncommittedTokens")`, then GREEN (tokens, renderTokens equal, uncommitted [], drafter state restored and aligned).
    - Gotcha: `PromptCacheSaveInput.metadata` is the FLAT metadata; caller keys sit under `1.` (private `FlatMetadataSection.userMetadata`), so the test strips `1.uncommittedTokens`. Swift Testing filters via `swift test --skip-build --filter` fail here (default metallib not found), so single-test runs go through the whole xctest bundle.
    - ChatSession: `.appendSuffix` and `.appendSuffixToMain` share one case that builds the input once; the main-only reset sits under `if case .appendSuffixToMain = decision`.
    - ExecutorReasoningConversation: prompt token IDs named in a private `PromptToken` enum. swiftlint no_magic_numbers (allowed [0,1,-1,100], test_parent_classes []) reports 19 violations on the old file, 0 on the new one.
  timestamp: 2026-10-29T13:05:34.065078+00:00
- actor: claude-code
  id: 01m63w85wqnym965xav02pvdwz
  text: |-
    ### implement — changed
    - evidence: branch worktree-agent-a50e8aaca8bea5119, commit 1a9300b1 (on mtp 0534c446); `swift build --build-tests` 0 Swift warnings; xcrun xctest: MLXLMTests 2020 Swift Testing + 920 XCTest, MLXFoundationModelsTests 427, MLXGuidedGenerationTests 72, CXGrammarTests 7, MLXHuggingFaceMacrosTests 6 — 0 failures, 0 skipped; swiftlint no_magic_numbers 0 violations
    - files: Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift, Libraries/MLXLMCommon/ChatSession.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheUnfedTokenTests.swift, Tests/MLXFoundationModelsTests/ExecutorReasoningConversation.swift
    - next: /review (all 5 findings of 2026-10-29 07:45 checked)
  timestamp: 2026-10-29T13:05:37.687979+00:00
- actor: claude-code
  id: 01m63wxxpajfvyma5j4hpdh0st
  text: |-
    ### review — clean
    - evidence: review sha 330c77f3~1..330c77f3 — 0 findings (7 attempted, 1 refuted, 0 confirmed); 4 files reviewed; all 5 items in Review Findings (2026-10-29 07:45) checked
    - next: none; task moved to done
  timestamp: 2026-10-29T13:17:30.186381+00:00
- actor: claude-code
  id: 01m63wydd92wapw57d07w6g4fa
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 4 files (decodeIfPresent for old spill files + test, shared splice input, named constants) in worktree, applied to mtp
    - test: green — worktree 5 bundles 0 failures 0 skipped; merged tree at 0534c446 green (Release MTP 95.0/86.0/77.7 vs plain ~58.5 at load ~10–18, edit turn 123)
    - commit: 330c77f3
    - review: clean — task moved to done
  timestamp: 2026-10-29T13:17:46.281309+00:00
position_column: done
position_ordinal: ffff8880
title: 'FoundationModels: an MTP turn that stops on its token budget leaves its last token out of the cache, and the Qwen splice drops it from the next turn'
---
Found in ^k2zn64x (Release MTPPromptCacheTests, Qwen3.8-27B-4bit, `.reasoning`, 128-token budget). The MTP turn 1 ledger is 206 tokens (79 prompt + 127 generated), the plain ledger 207: the MTP iterator returns its final verified token without feeding it. Turn 2 splices at the `<|im_end|>` commit (reused=207 for both). `reconcilePromptCache` gets no `previousGenerationUncommittedTokens` from the executor, thus `CommittedTurnSplice.suffixStart` takes the empty-uncommitted branch and the model reads its answer without that last token before `<|im_end|>`. `ChatSession` tracks `uncommittedTokens` and passes them, so its splice declines (or feeds the token).

- [x] Commit the uncommitted generated tokens on the executor entry (as `ChatSession.Conversation.uncommittedTokens`) and pass them to the reuse policy through `reconcilePromptCache`
- [x] Decide with the splice what the follow-up feeds: the dropped token before the commit, so the next turn reads the whole answer
- [x] Unit test with the Qwen35 MTP fixture: an MTP turn stopped on its budget, follow-up reads every generated token
- [x] Release: MTPPromptCacheTests still recomputes only the new message (26 of 233) and the drafter still resumes

#mtp #foundation-models #prompt-cache

## Review Findings (2026-10-29 07:45)

> Scope: `review sha 0534c446~1..0534c446` — reviewed the diffs only — lines this change added or modified. 16 file(s) reviewed, 2 not reviewed.

> 2 file(s) not reviewed — no validator matched:
> - `.kanban/tasks/01M63JA4CQK6Q4VTYBVG2T7CAA.jsonl` — no validator matches this file
> - `.kanban/tasks/01M63JA4CQK6Q4VTYBVG2T7CAA.md` — no validator matches this file

- [x] `Libraries/MLXFoundationModels/ExecutorPromptCacheFile.swift:195` `completeness/inverse-operation-coverage` — The writer now emits an `uncommittedTokens` metadata key, and the reader requires it. A spill file written before this change has no such key, so `TokenLedger.decode` throws `ledgerNotDecodable` and the whole entry fails to restore. The file's doc comment says the format name is unchanged and that older files read back with the new part empty, but that compatibility holds only for drafter state, not for this key. Decode a missing `uncommittedTokens` key as `[]`, the same way the drafter state is treated when absent, and add a read test using a file without that key. Alternatively, state in the doc comment that files without the key are unsupported and make that explicit in the format check.
- [x] `Libraries/MLXLMCommon/ChatSession.swift:1916` `reuse/reuse` — The `.appendSuffixToMain` case repeats the inline feed construction from `.appendSuffix` (same `LMInput(tokens: MLXArray(uncommittedTokens + promptTokenIds[suffixStart...]))` shape). This is a second copy of a near-identical splice, not an extension of a shared path. Compute the input once for both splice cases, for example by grouping them in one `case` that sets `input` and `unfedGeneratedTokens` and then handles the main-only reset separately. Or share the helper proposed for the `.appendSuffix` finding.
- [x] `Tests/MLXFoundationModelsTests/ExecutorReasoningConversation.swift:26` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/ExecutorReasoningConversation.swift:29` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `Tests/MLXFoundationModelsTests/ExecutorReasoningConversation.swift:32` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.