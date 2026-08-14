---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzywqvrcstkd4r186zmrtp8m
  text: |-
    Job 1 is green. The post-merge port needs no correction.

    `xcodebuild build-for-testing` succeeded. Each test ran alone with `test-without-building`. Note: a Swift Testing selector needs the parentheses — `-only-testing:.../loadsTheRealCheckpointEndToEnd()`. Without them xcodebuild finds the suite and runs 0 tests, and still exits 0.

    | test | result | time |
    |---|---|---|
    | loadsTheRealCheckpointEndToEnd | pass | 22.2 s |
    | wiredMemoryLimitCoversTheWholeCheckpoint | pass | 7.3 s |
    | decodeStepStaysInsideTheLongGenerationBudget | pass | 33.1 s |
    | greedyFirstTokensMatchThePythonFixture | pass | 28.0 s |
    | chatAndThinkingModesBothGenerate | pass | 21.9 s |
    | twoRoundConversationRecallsTheFirstRound | pass | 15.8 s |
    | thinkingModeRoundOneTranscriptIsNotARoundTwoPrefix | pass | 0.001 s |
    | chatModeRoundOneTranscriptIsARoundTwoPrefix | pass | 0.001 s |

    `longGenerationPastTwelveThousandTokensCompletes` was not run; the user skipped it permanently.

    The wired-memory wiring holds. The manager applied the whole 168,662,344,796-byte request, and the median decode step is 0.603 s, near the 0.593 s the file records for a raised limit and far below the 2.124 s of an unraised one.
  timestamp: 2026-08-14T01:03:48.492403+00:00
- actor: claude-code
  id: 01kzyx3gv863pz59pa5e1qwshc
  text: |-
    Research for Job 2.

    How an agentic round reaches the cache:
    - `DeepSeekV4EncodingTokenizer.applyChatTemplate` reads `additionalContext["thinking"]` (default true) and calls `DeepSeekV4ChatEncoder.encode`. Tools reach the encoder through `DeepSeekV4ChatEncoder.Message.messages(from:tools:)`, which attaches them to the first system or developer turn.
    - `DeepSeekV4Model.toolCallFormat` is `.dsml`, thus `DSMLToolCallParser` reads the calls back out of the token stream and `ChatSession` gives them as `Generation.toolCall`.
    - `ChatSession` keeps `Conversation.cachedTokens` — the prompt tokens plus the generated tokens the cache holds. Each turn builds a `PromptCacheTurn` and asks `PromptCacheReusePolicy`. `ExtendCachedPrefixRule` needs `promptTokens.starts(with: cachedTokens)`; `RewindToCommonPrefixRule` needs `PromptCacheState.isTrimmable`, which `canTrimPromptCache(kvCache.cache)` computes.
    - `ChatSession.streamDetails(to:[Chat.Message])` appends tool results and keeps the live cache. With no `toolDispatch` the tool calls reach the caller, thus a test can drive the loop by hand and measure each pass.
    - `Conversation.record` writes `.assistant(content, toolCalls:)` and never writes `reasoning`. DeepSeek-V4 declares no `reasoningConfig`, thus a thinking-mode reply keeps its whole `reasoning</think>answer` text inside `content`.

    One thing to watch: `DeepSeekV4ChatEncoder.encode` turns `dropsEarlierReasoning` OFF when any turn carries tools. Thus the earlier "thinking mode drops earlier reasoning" measurement was taken without tools and may not hold in an agentic run. Measure it.

    Structure: `DeepseekV4IntegrationTests.swift` holds the 141 GiB shared load as file-scope `private`, which a second suite cannot reach. Copying it would be duplication. Thus the load moves to a new `DeepseekV4SharedCheckpoint.swift` that both suites read. The Xcode target uses file-system-synchronized groups, thus a new file needs no project edit.
  timestamp: 2026-08-14T01:10:10.536104+00:00
- actor: claude-code
  id: 01kzyyh53570rp1rw6t8416jh5
  text: |-
    Job 2 is blocked, and the block is not the cache.

    ## The answer the measurement did reach: trimmability

    `everyCacheLayerRewinds` passes.

    ```
    DSV4 CACHE: cache layers = 43
    DSV4 CACHE: trimmable cache layers = 43
    DSV4 CACHE: cache kinds = KVCacheSimple
    ```

    DeepSeek-V4 builds one `KVCacheSimple` for each of its 43 layers, and every one answers `isTrimmable == true`. Thus `RewindToCommonPrefixRule` CAN act on this model. The hybrid Mamba stack of Qwen 3.6 could not rewind, and that is what disqualified it; DeepSeek-V4 does not carry that limit.

    ## Why (a) and (b) have no numbers

    Neither mode reached a tool round, because DeepSeek-V4 writes gibberish on a long prompt.

    | test | prompt tokens | what the model wrote |
    |---|---|---|
    | chatModeToolRoundReusesThePromptCache | 2,6xx with tools | ` for? huh but huh we это это нам это нам huh but huh 意思 ...` |
    | thinkingModeToolRoundReusesThePromptCache | 2,6xx with tools | ` huh but nothing huh we но huh  huh nothing huh 意思 ...` |
    | longPromptWithoutToolsRecallsAPlantedFact | 3,626, NO tools | ` huh? huh? huh? huh  huh  huh  huh  huh  huh  huh  huh  huh ` |

    The third test is the reduction. It carries no tool at all, plants the number `4172` in the body of its own prompt, and asks for it back. Thus the tools are ruled out and the length is the cause.

    Two shapes were tried and neither helped: a real system turn in front of the tools (the encoder otherwise makes an EMPTY system turn to hold them, which the published reference never emits), and an instruction moved out of the user turn.

    ## The cause

    `DeepSeekV4Model` runs plain dense attention on every layer.

    - `DeepSeekV4Configuration.slidingWindow` decodes as 128 and NOTHING reads it.
    - The header of `Libraries/MLXLLM/Models/DeepSeekV4.swift`, item 3, states the other half: the compressor and the indexer both load and neither runs, because sparse attention needs a pooled cache this repository does not record.

    Below 128 tokens a window and no window give the same attention. Every earlier real-weights test feeds a prompt of a few dozen tokens, thus none of them reads this range, and the port looked correct. Card ^3x0krt4 holds the defect and names the reproduction.

    ## What this means for the goal

    An agent's transcript passes 128 tokens at its first turn. Thus DeepSeek-V4 is not usable agentically in this repository today, and the prompt cache is the second problem, not the first.
  timestamp: 2026-08-14T01:35:05.829526+00:00
- actor: claude-code
  id: 01kzyyhddwyqkpdbbr91jdb2qk
  text: |-
    ### implement — stuck
    - evidence: Job 1 green, 7 of 7 tests run one at a time (the >12k endurance test stays skipped). Job 2 gives the trimmability answer (43 of 43 `KVCacheSimple` layers rewind) and cannot give the (a)/(b) numbers: DeepSeek-V4 writes gibberish on a 3,626-token prompt with no tools, thus no tool round completes. Files: IntegrationTesting/IntegrationTestingTests/DeepseekV4SharedCheckpoint.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift, IntegrationTesting/IntegrationTestingTests/DeepseekV4IntegrationTests.swift, IntegrationTesting/IntegrationTestingTests/Qwen36UpstreamPromptCacheAssessmentTests.swift, Libraries/IntegrationTestHelpers/IntegrationTestHelpers.swift
    - next: card ^3x0krt4 holds the blocker — apply `slidingWindow` and run the sparse path. The agentic cache cannot be measured until a long prompt answers.
  timestamp: 2026-08-14T01:35:14.364328+00:00
position_column: doing
position_ordinal: '8180'
title: Prove DeepSeek-V4 after the upstream catch-up, then measure the agentic prompt cache
---
The merge of `ml-explore/mlx-swift-lm` used `-X theirs`, which reverted the DeepSeek-V4 hooks in shared files. A person re-applied them by hand. Nobody has run the real-weights suite since.

The goal: DeepSeek-V4 must work agentically, in multi-tool loops. Many rounds with tool calls need a prompt cache that works.

## Job 1 — prove the port still works

- [ ] Build `IntegrationTesting` with `xcodebuild build-for-testing`
- [ ] Run each test of `DeepseekV4IntegrationTests` one at a time, and stop at the first failure
- [ ] Do NOT run `longGenerationPastTwelveThousandTokensCompletes`; the user skipped it permanently
- [ ] Correct any failure

## Job 2 — measure the cache in an agentic shape

Model the new suite on `Qwen36UpstreamPromptCacheAssessmentTests.swift`.

- [ ] (a) Is round N+1's rendered prompt a true prefix extension of round N's, across a tool round?
- [ ] (b) Given a good prefix, does the cache skip the work? Count tokens fed against tokens skipped, and time prefill against a cold control
- [ ] Make the shape agentic: user turn, then a tool call in the DSML format, then a tool result, then more assistant text
- [ ] Measure chat mode and thinking mode
- [ ] Tell whether the DeepSeek-V4 cache is trimmable, which `RewindToCommonPrefixRule` needs
- [ ] Print each measurement with a `DSV4 CACHE:` prefix

Measure and report only. Do not build a correction. #deepseek-v4