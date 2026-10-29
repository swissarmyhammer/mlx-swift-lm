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
position_column: doing
position_ordinal: '8180'
title: 'FoundationModels: an MTP turn that stops on its token budget leaves its last token out of the cache, and the Qwen splice drops it from the next turn'
---
Found in ^k2zn64x (Release MTPPromptCacheTests, Qwen3.8-27B-4bit, `.reasoning`, 128-token budget). The MTP turn 1 ledger is 206 tokens (79 prompt + 127 generated), the plain ledger 207: the MTP iterator returns its final verified token without feeding it. Turn 2 splices at the `<|im_end|>` commit (reused=207 for both). `reconcilePromptCache` gets no `previousGenerationUncommittedTokens` from the executor, thus `CommittedTurnSplice.suffixStart` takes the empty-uncommitted branch and the model reads its answer without that last token before `<|im_end|>`. `ChatSession` tracks `uncommittedTokens` and passes them, so its splice declines (or feeds the token).

- [x] Commit the uncommitted generated tokens on the executor entry (as `ChatSession.Conversation.uncommittedTokens`) and pass them to the reuse policy through `reconcilePromptCache`
- [x] Decide with the splice what the follow-up feeds: the dropped token before the commit, so the next turn reads the whole answer
- [x] Unit test with the Qwen35 MTP fixture: an MTP turn stopped on its budget, follow-up reads every generated token
- [x] Release: MTPPromptCacheTests still recomputes only the new message (26 of 233) and the drafter still resumes

#mtp #foundation-models #prompt-cache