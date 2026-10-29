---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m5yw4rkbnnyjg1gye3ykcr52
  text: |-
    Picked up (user directed despite blockers ^achvxy1/^ahnb0j9: this run is a diagnostic of today's behavior). Scope extended by the user: besides system prompt + follow-ups, add (1) a multi-turn chat with thinking ON (re-render drops earlier <think>), (2) an edit-and-regenerate turn (fork(history:) back to an earlier message end). Print one line per turn for MTP and plain sessions; failing expectations become withKnownIssue naming the gap task.

    Research:
    - No public metric for drafter prefill. MTPSpeculativeTokenIterator feeds the drafter exactly `input.text.tokens` (prepareDrafterState cold, resumeDrafterState warm), i.e. the same tokens the target prefilled. So drafter-prefilled = info.promptTokenCount when the session ran the MTP iterator (proposedDraftTokens != nil) and the drafter actually started (not passthrough with 0 proposals); 0 when ChatSession fell back (speculativeDecodingFallbackReason != nil).
    - ChatSession keepsCheckpoints requires speculativeDecoding == nil; a stateful drafter's reuse capability is .appendOnly, so a rewind becomes .trimToCommonPrefixMainOnly (fallback .unavailableContinuation) or .rebuild (GDN cannot trim) with no checkpoint restore.
    - No public transcript accessor on ChatSession; edit-and-regenerate builds history from the prompts and the streamed text and uses fork(history:), which deep-copies the cache incl. the MTP continuation.
    - MTP drafter container: MTPDrafterModelFactory.shared.loadContainer(...) -> MTPDrafterContainer; SpeculativeDecodingConfig(mtpDrafter:blockSize:).
  timestamp: 2026-10-27T14:27:33.611571+00:00
- actor: claude-code
  id: 01m5ywtapz33w8dqdqv8jn1mcv
  text: |-
    Results: Release xcodebuild, Qwen3.8-27B-4bit + Qwen3.8-27B-MTP-4bit, blockSize 2, temperature 0. The system prompt is the math-tutor preamble plus 8000 characters of benchmark text, about 2170 tokens. All 3 runs gave the same outcome. Numbers below are from run 3 (/private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/d4dd68db-1555-410f-b902-6376979e7c8c/scratchpad/run3.log). "cached" is cachedPromptTokenCount and "pre" is prefilled tokens / prefill ms.

    | chat | turn | MTP: cached / pre | drafter | spec | MTP tok/s, accept | plain: cached / pre | plain tok/s | text = plain |
    |---|---|---|---|---|---|---|---|---|
    | follow-ups | 0 | 0 / 2212 tok 1691ms | 2212 | MTP | 79.4, 94.9% | 0 / 2212 tok 2048ms | 56.3 | yes |
    | follow-ups | 1 | 2403 / 38 tok 88ms | 38 | MTP | 78.4, 94.3% | 2404 / 37 tok 125ms | 56.5 | yes |
    | follow-ups | 2 | 2544 / 29 tok 92ms | 29 | MTP | 81.1, 96.2% | 2544 / 29 tok 123ms | 56.5 | NO (after 3 chars) |
    | thinking | 0 | 0 / 2191 tok 1575ms | 2191 | MTP | 85.4, 91.4% | 0 / 2191 tok 1792ms | 56.6 | yes |
    | thinking | 1 | 2302 / 25 tok 68ms | 25 | MTP | 87.4, 92.0% | 2303 / 24 tok 110ms | 56.6 | NO (after 133 chars) |
    | thinking | 2 | 2423 / 25 tok 67ms | 25 | MTP | 86.4, 90.6% | 2425 / 24 tok 110ms | 56.5 | NO (at char 0) |
    | edit | 0 | 0 / 2197 tok 1573ms | 2197 | MTP | 83.2, 91.4% | 0 / 2197 tok 1858ms | 56.7 | yes |
    | edit | 1 | 2264 / 26 tok 68ms | 26 | MTP | 89.2, 97.1% | 2265 / 25 tok 120ms | 56.6 | yes |
    | edit | 2 (edit) | 0 / 2292 tok 1611ms | 2292 | MTP | 84.6, 97.1% | 2169 / 123 tok 185ms | 56.6 | yes |

    Findings:
    - Follow-ups with thinking off: both sessions reuse the cache. Each turn recomputes only the new suffix, 29-38 tokens. MTP stays on, and the drafter prefills only the suffix. MTP decodes 1.39-1.44x faster than plain. Turn 2 text diverges from plain.
    - Follow-ups with thinking on: thinking really ran (the text holds the reasoning and `</think>`). The re-render does NOT break reuse: the cached prefix is about 2300 tokens and only 24-25 tokens are recomputed. The policy keeps the cached generated tokens, so no rewind, no .trimToCommonPrefixMainOnly and no fallback happened. MTP stays on at 1.51-1.54x. Turns 1 and 2 text diverges from plain.
    - Edit and regenerate (fork(history:) back to the end of reply 0): the plain session restores its system-prompt checkpoint and recomputes 123 tokens in 185ms. The MTP session takes `.rebuild` (GDN cannot trim, and there are no checkpoints with MTP on) and recomputes the whole chat, 2292 tokens in 1611ms. MTP did NOT fall back: a rebuild starts the drafter cold over the full prompt, so speculation stays on (1.49x). Text equals plain.
    - The suspected `.trimToCommonPrefixMainOnly` / `.unavailableContinuation` fallback did not occur in any of these chats.
    - NEW: on warm follow-ups, MTP greedy text diverges from plain in 3 of 6 turns. Every cold turn matches. The divergence starts within the first tokens after the warm suffix prefill, even where both sessions hold the same history. Filed as ^chgkh4c.

    Known issues recorded today (withKnownIssue in `mtpChatKnownGaps`):
    - follow-ups t2, thinking t1, thinking t2: matchesPlainText -> ^chgkh4c
    - edit t2: mtpReusesCache and drafterPrefillsOnlyTheSuffix -> ^achvxy1 (checkpoints with MTP), then ^ahnb0j9 (drafter continuation through the checkpoint restore)
  timestamp: 2026-10-27T14:39:20.287173+00:00
- actor: claude-code
  id: 01m5ywthdeee2hqdgx0xz1fs42
  text: |-
    Process notes: run 1 was RED, with 5 expectation failures and no known gaps listed. Run 2 added diagnostics (a text excerpt per turn, and where MTP text leaves plain text) and failed the same 5 checks. Run 3 was GREEN: passed with 5 known issues, TEST SUCCEEDED in 47 s of test time and about 80 s wall. `swift build --build-tests`: Build complete. Its only warning is the existing "missing creator for mutated node ... mlx-swift_Cmlx.bundle", which comes from the mlx-swift dependency. The xcodebuild warnings are all in untouched files (HardReserveStressTests, TokenizerVocabExtractorTests) or in the SDK module cache.

    Mistake on my part: I added the `mathTutorPreamble` constant with a one-off python edit instead of the files tool. The content is correct.

    ### implement — changed
    - evidence: 2 files — Libraries/BenchmarkHelpers/BenchmarkHelpers.swift, IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift. Release xcodebuild `-only-testing:IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache()`: passed with 5 known issues. `swift build --build-tests`: Build complete. Not committed. New task ^chgkh4c filed.
    - next: /review
  timestamp: 2026-10-27T14:39:27.150018+00:00
- actor: claude-code
  id: 01m63segptfj9438byejnp9v29
  text: |-
    Audit at the shipped default (^hbq0f3n: blockSize 5, pMin 0.75). This supersedes the blockSize-2 table in the earlier comment.

    Code audit: every item on the card is already implemented. Nothing left to build.
    - `benchmarkMTPSystemPromptCache(container:drafter:scenarios:systemPrompt:systemPromptCharacterCount:blockSize:)` (Libraries/BenchmarkHelpers/BenchmarkHelpers.swift) builds `SpeculativeDecodingConfig(mtpDrafter:blockSize:)`. blockSize defaults to `defaultMTPBlockSize` = 5. The init's own default sets minimumDraftProbability to `defaultMTPMinimumDraftProbability` = 0.75. So the benchmark runs at the shipped default without overrides. For each turn it reports prefill ms, cached, drafter-prefilled, speculation state or fallback reason, decode tok/s, acceptance, and the plain-session line.
    - `qwen38FourBitMTPSystemPromptCache()` (IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift) is enabled only when the target and `mlx-community/Qwen3.8-27B-MTP-4bit` are local. It runs the three chats: follow-ups, thinking, and edit. For every follow-up turn it checks mtpReusesCache, plainReusesCache, drafterPrefillsOnlyTheSuffix, matchesPlainTokens and decodesFasterThanPlain. It checks matchesPlainTokens on turn 0 too. A token match counts when the tokens are exact, or when MTP leaves plain only at a near-tie of plain's two best tokens. `mtpChatKnownGaps` is now EMPTY: the earlier known issues (^chgkh4c, ^achvxy1, ^ahnb0j9) are all closed, so every check is a hard #expect.
    - `qwen38FourBitMTPGeneration()` is the cold single-prompt block/pMin sweep. It does not cover sessions, and this card's benchmark fills that gap.

    Release run: xcodebuild in IntegrationTesting, `-destination 'platform=macOS,arch=arm64' -configuration Release ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES -only-testing:'IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache()'`. Result: TEST SUCCEEDED. The test passed after 45.8 s with 0 known issues and 0 skipped. The run held bench.lock and no other mlx-swift-lm build or test binary was running. Load average was 8.56 at start and 7.02 at end; the rest of the load came from other projects (EditorKit, Testudo, AgentViewKit tests). Log: /private/tmp/claude-501/-Users-wballard-github-swissarmyhammer-mlx-swift-lm/d4dd68db-1555-410f-b902-6376979e7c8c/scratchpad/hs9h-run.log

    The system prompt is about 2170 tokens (math-tutor preamble plus 8000 characters). Temperature 0. "cached / pre" means cachedPromptTokenCount / prefilled tokens and prefill time.

    | chat | turn | MTP cached / pre | drafter | MTP tok/s, accept | plain cached / pre | plain tok/s | speedup | tokens vs plain |
    |---|---|---|---|---|---|---|---|---|
    | follow-ups | 0 | 0 / 2212 in 1835ms | 2212 | 104.9, 95.4% | 0 / 2212 in 1820ms | 56.5 | 1.86x | exact |
    | follow-ups | 1 | 2403 / 38 in 111ms | 38 | 99.4, 89.7% | 2404 / 37 in 125ms | 56.1 | 1.77x | exact |
    | follow-ups | 2 | 2544 / 29 in 111ms | 29 | 102.9, 90.1% | 2544 / 29 in 124ms | 56.1 | 1.83x | near-tie at token 1 (gap 0.125) |
    | thinking | 0 | 0 / 2191 in 1804ms | 2191 | 86.6, 81.2% | 0 / 2191 in 1793ms | 56.3 | 1.54x | exact |
    | thinking | 1 | 2303 / 24 in 97ms | 24 | 93.5, 87.8% | 2303 / 24 in 112ms | 55.9 | 1.67x | near-tie at token 0 (gap 0.125) |
    | thinking | 2 | 2418 / 24 in 94ms | 24 | 88.9, 77.6% | 2425 / 24 in 113ms | 55.8 | 1.59x | near-tie at token 36 (gap 0.125) |
    | edit | 0 | 0 / 2197 in 1819ms | 2197 | 96.2, 87.5% | 0 / 2197 in 1812ms | 55.9 | 1.72x | exact |
    | edit | 1 | 2264 / 26 in 109ms | 26 | 87.4, 88.3% | 2265 / 25 in 122ms | 55.8 | 1.57x | exact |
    | edit | 2 (edit) | 2169 / 123 in 180ms | 123 | 85.1, 92.7% | 2169 / 123 in 187ms | 55.7 | 1.53x | exact |

    Every turn ran with speculation on (state "MTP") and none fell back. Each follow-up turn reused the cache and prefilled only the new suffix, 24-38 tokens, with the drafter fed the same suffix. The edit turn now keeps the system-prompt prefix (2169 cached, 123 prefilled), the same as plain; the earlier run rebuilt all 2292 tokens. The ChatSession reference numbers (MTP 100.4/87.4/83.4 vs plain 59.5) were measured at load < 5. This run's plain rate of about 56 is lower because the machine was shared (load 7-9). MTP still beats plain on every follow-up turn of the math chats, by 1.53-1.83x.

    Checkbox audit: all 4 subtasks, both acceptance items and the Tests item are already [x]. The run above confirms them at the shipped default. I made no edits.
  timestamp: 2026-10-29T12:16:39.642475+00:00
- actor: claude-code
  id: 01m63setkekrhsscc0h6nkwb8z
  text: |-
    ### implement — no-change
    - evidence: 0 files changed. The audit found every subtask, acceptance item and test already met by Libraries/BenchmarkHelpers/BenchmarkHelpers.swift `benchmarkMTPSystemPromptCache` and IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift `qwen38FourBitMTPSystemPromptCache()`, which run at the shipped default blockSize 5, pMin 0.75. `mtpChatKnownGaps` is empty. The Release xcodebuild `-only-testing:'IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache()'` gave TEST SUCCEEDED: 1 test passed in 45.8 s, 0 known issues, 0 skipped. Load average was 8.56 at start and 7.02 at end, under bench.lock. Follow-up turns ran at 1.53-1.83x plain. Edit turn: 2169 cached, 123 prefilled. No swift build or unit bundles were run, because there was no code change.
    - next: /review (or close; nothing new to review)
  timestamp: 2026-10-29T12:16:49.774116+00:00
depends_on:
- 01M5QWGWNFDVP77YAW9ACHVXY1
- 01M5QWHA34WP7DQMWQ4AHNB0J9
- 01M5SWXS9E09Y97PV3BV51TPYA
position_column: doing
position_ordinal: '80'
title: 'MTP: benchmark ChatSession multi-turn with MTP and prompt-cache reuse'
---
## What
`qwen38FourBitMTPGeneration` (IntegrationTesting/IntegrationTestingTests/GenerationBenchmarkTests.swift) measures MTP only on one cold prompt. Add a session benchmark that shows the real use: a system prompt, then follow-up turns that reuse the main and drafter caches.

- [x] In Libraries/BenchmarkHelpers/BenchmarkHelpers.swift, add `benchmarkMTPSystemPromptCache(container:drafter:...)`, modeled on `benchmarkSystemPromptCache`. Report for each turn: the prefill time, `cachedPromptTokenCount`, the drafter tokens prefilled, decode tok/s, acceptance, and plain-session tok/s for the same turns.
- [x] Add `qwen38FourBitMTPSystemPromptCache()` to GenerationBenchmarkTests. It skips when the target or `mlx-community/Qwen3.8-27B-MTP-4bit` is not local.
- [x] `#expect` that each follow-up turn has `cachedPromptTokenCount > 0`, drafter prefill only over the new suffix, and text equal to the plain session.
- [x] (Scope added by the user) Scenarios: follow-ups with thinking off, follow-ups with thinking ON (the re-render drops earlier reasoning), and edit-and-regenerate (fork back to the end of an earlier reply). Each turn prints one line for the MTP session and one for the plain session, with the speculation state or fallback reason from `.info`. An expectation the current code fails is a `withKnownIssue` that names its gap task.

## Acceptance Criteria
- [x] The benchmark runs with the Release xcodebuild command in the file header and prints one summary line for each turn.
- [x] Follow-up turns show MTP decode tok/s > plain decode tok/s on the math prompt category.

## Tests
- [x] `xcodebuild test ... -only-testing:'IntegrationTestingTests/GenerationBenchmarkTests/qwen38FourBitMTPSystemPromptCache()'` passes (Release, as in the file header).

## Workflow
- Use `/tdd`: write failing tests first, then implement to make them pass. #mtp #qwen38 #benchmark