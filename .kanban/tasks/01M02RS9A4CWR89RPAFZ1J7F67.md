---
assignees:
- claude-code
position_column: todo
position_ordinal: 9c80
title: Reuse the prompt cache across turns inside MLXLanguageModel.Executor
---
## What

Card `^z2996cp` restored the usage channel send and gave it a test. It could not
give `cachedTokenCount` a real value, because the executor reuses nothing. A
person split that feature into this card on 2026-08-15.

## The measurement that made this card

Taken on 2026-08-15, on branch `catch-up-upstream`:

| The question | The measurement |
| --- | --- |
| Does `MLXFoundationModels` hold cache-reuse machinery? | `grep -r "KVCache\|cachedTokens\|PromptCacheReuse\|ChatSession" Libraries/MLXFoundationModels/` gives **0** matches |
| Does the executor pass a cache? | Every generation entry point takes the default `cache: nil` |
| Does the executor reuse a prompt? | `respond(...)` renders the whole transcript again on each call |
| Does `GenerateCompletionInfo` hold the count? | It holds no reuse field |

Thus every turn pays a full prefill, and
`Libraries/MLXFoundationModels/MLXLanguageModel.swift`
`Executor.reusedPromptTokenCount` is 0 because 0 is true, not because a value is
missing.

## What the consumers see

`FoundationModelsRouter` holds two assertions that stay red until this card
lands, in `LanguageModelSessionBackendTests.swift`:

- `:563` `turn2Usage.input.cachedTokenCount > 0`
- `:574` `abs(turn2Usage.input.cachedTokenCount - turn1ProcessedTokenCount) <= tolerance`

A consumer that starts a second turn on the same session pays the whole prompt
again. On a long agentic transcript that cost is the largest part of a turn.

## Where the parts already are

`MLXLMCommon` holds the machinery, and `ChatSession` is the only consumer of it:

- `PromptCacheReusePolicy`, `ExtendCachedPrefixRule` and `RewindToCommonPrefixRule`
  decide what a new render may reuse.
- The reused count is computed inside that policy, and the value dies in its
  `switch`.
- `KVCache.isTrimmable` states whether a layer can rewind.

## What to watch out for

Card `^mscrreq` holds a measured reason a good prefix is still not sufficient:
`ExtendCachedPrefixRule` compares a new render against the LEDGER, which is the
earlier render PLUS the tokens that turn generated. An encoder that primes a
generation region breaks that prefix by one token. Read `^mscrreq` before you
design this.

Memory of this project also records: template priming breaks the prompt cache,
thus prove reuse with a real-weights two-round test and not with a unit test
alone.

## Acceptance Criteria

- [ ] `MLXLanguageModel.Executor` holds a key/value cache across the turns of one
      session, and invalidates it when the transcript is not an extension.
- [ ] `emitUsage` reports the true number of reused tokens at each of its five
      call sites, and `Executor.reusedPromptTokenCount` leaves.
- [ ] A test proves reuse through the CHANNEL, the way
      `Tests/MLXFoundationModelsTests/UsageChannelSendTests.swift` does.
- [ ] A real-weights two-round test shows the second turn feeds fewer tokens than
      the first, and that its answer is unchanged.
- [ ] `swift build` is clean, and
      `SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1 swift test` is green with
      no new warning.
- [ ] `xcodebuild build-for-testing -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS'`
      gives `TEST BUILD SUCCEEDED`.

#eventplan