# MLXFoundationModels

An MLX adapter conforming to Apple's `FoundationModels.LanguageModel`. It provides `MLXLanguageModel` (analogous to `SystemLanguageModel`), usable directly with `LanguageModelSession`, so existing FoundationModels code (guided `@Generable` output, tool calling, streaming) works unchanged. Requires the macOS/iOS/visionOS 27.0 SDK.

## Usage

Build an `MLXLanguageModel` with the `#huggingFaceLanguageModel` macro and pass it to a `LanguageModelSession`.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = #huggingFaceLanguageModel(
        configuration: LLMRegistry.qwen3_0_6b_4bit,
        capabilities: [.reasoning])
    let session = LanguageModelSession(model: model)

    let answer = try await session.respond(
        to: "I have three hours near the Loop in Chicago. Is the Art Institute or the Field Museum the better use of my time?")
    print(answer.content)
}
```

The macro synthesizes the `weightsLocation:` and `load:` parameters (Hugging Face download plus tokenizer loading) that you would otherwise pass to the `MLXLanguageModel` initializer by hand.

### Direct initializer

The macro above expands to the call below. Use the initializer directly to point `weightsLocation:` at your own on-disk directory or to swap `load:` for a different downloader or tokenizer.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = MLXLanguageModel(
        configuration: LLMRegistry.qwen3_0_6b_4bit,
        capabilities: [.reasoning],
        weightsLocation: { id in
            let cache = HubCache.default
            guard let repo = Repo.ID(rawValue: id) else { return cache.cacheDirectory }
            if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                return snapshot
            }
            return cache.repoDirectory(repo: repo, kind: .model)
        },
        load: { configuration, progress in
            try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progress: progress)
        })
    let session = LanguageModelSession(model: model)
}
```

## Capabilities

Declare what a model may do with the `capabilities:` list at construction. Declaration is explicit: the adapter does not infer capabilities from the model id; it defaults to `[.guidedGeneration]`. A request that exceeds the declared capabilities fails with a typed error.

| Capability | What it enables |
|---|---|
| `.guidedGeneration` | Grammar-constrained output. Pass a `GenerationSchema` to `respond(to:schema:)` or a `@Generable` type to `respond(to:generating:)`, and the result always matches the schema. See [`MLXGuidedGeneration`](../MLXGuidedGeneration/README.md). |
| `.toolCalling` | Expose Swift `Tool`s to the model. |
| `.reasoning` | Run "thinking" models that emit a reasoning trace. |
| `.vision` | Accept image inputs. |

### Required tool calls

`GenerationOptions(toolCallingMode: .required)` makes the turn call one of the enabled tools: a grammar constrains the output to one tool call, after an unconstrained reasoning phase when the model reasons. The call takes the frame of the model's tool-call format. A plain text syntax writes the `<tool_call>` JSON envelope. Inkling 2 (`ToolCallFormat.inkling`) writes its own tool frame, `<|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": ...}<|end_message|>`, and reasons first, at every effort including `none`, inside a thinking frame that the turn opens, whose text streams as reasoning. Harmony (`gpt_oss`) and ATEM (`atem`) write tool calls only in frames that have no required-mode grammar, so a required turn of such a model throws `LanguageModelError.unsupportedCapability` for `.toolCalling` before it generates; use `.allowed` for them.

### The schema in the prompt

During guided generation, `MLXLanguageModel` adds the response schema, including its `@Guide` descriptions, to the session's system prompt, so the model knows what each property is for. To leave the prompt unchanged, pass `includeSchemaInPrompt: false`. The grammar enforces the structure either way, but including the schema directly gives the model more information about what it's generating.

Consider `includeSchemaInPrompt: false` when:
- The schema is large relative to the model's context. Every guided request carries the full schema JSON, which also adds prefill time before the first token.
- Your instructions or prompt already describe the response.
- You want to control the wording or placement within the transcript, such as in cases where multiple different schemas are used within a single session.

### Where the grammar starts

The grammar of a schema pass starts at the end of the prompt, thus the prompt must end where the answer starts. Most chat templates write a generation prompt that ends there. The Inkling 2 template writes none: the model writes the header of its answer frame, `<|message_model|><|content_text|>`, after the prompt. Its reasoning protocol declares that header as its `answerOpening`, and a schema pass of such a model renders the prompt with thinking off and appends the header before the grammar starts. Without the header, the grammar forbids the tokens the model was trained to write first.

### The closing zones

Near the end of its token budget a guided pass closes the output: the soft zone favors the closing tokens (`"`, `}`, `]` and digits), and the hard zone allows nothing else. Each zone scales with the shortest valid output of the schema, and the budget caps it: the hard zone takes at most a quarter of the budget and the soft zone at most a half, unless the shortest valid output alone needs more. Before the caps, a schema whose shortest output took 32 tokens on Inkling 2 Mini set a hard zone of 256 tokens (32 × 8) over a budget of 256, and every string of the output closed at once or filled with `"}"`. At the default budget of 4096 tokens the caps change nothing for a schema whose shortest output takes fewer than 128 tokens.

## Multi-token prediction (MTP)

Pass an `MLXSpeculativeDecoding` to decode with the model's MTP head as a drafter. Each round the head drafts a few tokens and the model checks them in one forward pass, so a round can write several tokens for the cost of about one. The output is the output of plain decoding. The Qwen3.5-family targets (Qwen3.5, Qwen3.6 and Qwen3.8) have MTP heads.

```swift
import Foundation
import FoundationModels
import HuggingFace
import MLXFoundationModels
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

if #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) {
    let model = #huggingFaceLanguageModel(
        configuration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-4bit"),
        capabilities: [.guidedGeneration, .reasoning, .toolCalling],
        speculativeDecoding: try MLXSpeculativeDecoding(
            drafterConfiguration: ModelConfiguration(id: "mlx-community/Qwen3.8-27B-MTP-4bit"),
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            registration: .matchingTarget(.qwen35Text, .qwen35VLM)))
    let session = LanguageModelSession(model: model)
}
```

The direct `MLXLanguageModel` initializer takes the same `speculativeDecoding:` argument.

- **Pick the drafter family of the loaded model.** The macro loads the model with `loadModelContainer`, which tries `VLMModelFactory` first. With MLXVLM linked, Qwen3.8 therefore loads as a vision-language model, and its drafter is the `.qwen35VLM` family (MLXVLM). A model loaded by `LLMModelFactory` takes `.qwen35Text` (MLXLLM). `.matchingTarget(.qwen35Text, .qwen35VLM)` picks the right one for the model as it loaded. A drafter that cannot draft for the model leaves each turn to plain decode, and the `mtp-fallback` log states why, once for each drafter and model pair.
- **Block size and confidence floor.** `blockSize` (default 4) is the tokens of each round: up to `blockSize - 1` drafts and the model's own token. `minimumDraftProbability` (default 0.75) stops a round's chain of drafts at the first draft whose top-1 probability falls below it, and 0 drafts the whole block every round. The init throws `SpeculativeDecodingConfigurationError` for a block size below 2 or a floor outside 0...1. Use `init(drafterConfiguration:blockSize:minimumDraftProbability:drafterBytes:memoryPolicy:loadDrafter:)` to load the drafter with your own closure.
- **Memory policy.** Pass `memoryPolicy:` (a `SpeculativeDecodingMemoryPolicy`, such as `.recommendedWorkingSet`) and `drafterBytes:`, the estimated resident parameter bytes of the drafter, to admit the drafter beside the loaded model the way `ChatSession` admits its deferred drafter, with the same admission code. Each call weighs the parameters of the loaded model plus `drafterBytes` against the budget before the drafter loads. A drafter the policy does not admit never loads: under `.fallbackToDefault` the call decodes plainly and its response reports `"speculativeDecodingFallbackReason": "memoryBudgetExceeded"`, and under `.fail` the call throws `SpeculativeDecodingMemoryError`. Without `drafterBytes` the policy has nothing to weigh before the load, so the drafter loads and the policy weighs its parameters instead, with the same two outcomes. `#huggingFaceLanguageModel` forwards the `speculativeDecoding:` argument as written, so the policy reaches the model through the macro too.
- **Drafter loading and eviction.** The model loads the drafter one time, on the first generation or warm-up that needs it, and keeps it beside its weights in the shared model cache. Concurrent first uses share one load. `evict()` drops the drafter with the weights and the prompt cache entries of the model, and cancels a drafter load in flight; a custom `loadDrafter` closure must honor that cancellation. Two models that name the same drafter checkpoint each load their own copy.
- **Which turns speculate.** Text, reasoning, allowed-tool and think-then-call turns speculate at any temperature. Schema (`@Generable`) output and required-tool calls speculate too: each draft takes the grammar mask, a rejected draft rolls the grammar matcher back over it and its forced tokens, and the target still decides every token, so the output is the output of plain guided decoding, except where the two best logits of a decision lie close enough for a batched verify and a one-token forward to order them apart. They speculate on both load paths, the text path and the vision-language path of the router. A guided turn takes the prompt cache of its session as a text turn does, so it feeds the target and the drafter only the prompt after the cached prefix. The drafter resumes the drafter state a text turn left in the cache; after a guided turn the next turn starts its drafter cold where the cache ends. A guided turn that cannot speculate decodes plainly and reports why in `"speculativeDecodingFallbackReason"`.

### What each response reports

Each call of a model with `speculativeDecoding:` set writes its MTP outcome into the metadata of its response entry (`Transcript.Response.metadata`), the same metadata that carries `"incompleteOutput"`:

| Key | Type | Value |
| --- | --- | --- |
| `"speculativeDecoding"` | `Bool` | `true` when MTP drafted in at least one round of the call. |
| `"speculativeDecodingFallbackReason"` | `String` | The `SpeculativeDecodingFallbackReason` case name, such as `"incompatibleDrafter"`, when a pass of the call fell back to plain decode. Absent when no pass fell back. |
| `"draftedTokenCount"` | `Int` | The drafts the call proposed. |
| `"acceptedDraftTokenCount"` | `Int` | The drafts of the call that the model accepted. |

One call can run several passes: think-then-call runs a reasoning phase and a grammar phase, and an allowed-tool turn with a schema can run a schema pass after its tool pass. The two counts are the sums over every pass of the call, and the fallback reason is the first fallback a pass of the call reported, in the order the passes ran. Text, reasoning and allowed-tool passes report their counts in the speculative telemetry of their completion info; schema and required-tool grammar passes report them through the guided-generation diagnostic sink, which the call binds without recording any decoding detail. A streamed call and a call that returns its whole response report the same metadata. A call that is cancelled or throws after it opened its response entry still writes the four keys into that entry: the counts then cover the passes that reported before the call stopped, which are the passes that completed and the rounds a grammar pass ran before the throw. A generation-loop pass that stops before its completion info reports nothing of its own. A call of a model without `speculativeDecoding:` writes none of these keys.

### What the prompt cache keeps

The executor prompt cache keeps the MTP drafter state with each session's entry, so a follow-up turn resumes the drafter instead of feeding it the whole conversation again:

- **Reuse.** A turn whose prompt extends the cached tokens feeds the model and the drafter only the new suffix. A turn that speculates with another drafter than the one that left the state starts its drafter cold.
- **Checkpoints.** The checkpoint of a hybrid model (the Qwen3.5 family) keeps the drafter state at its position. A turn that goes back to the checkpoint resumes a copy of that state.
- **Spill and restore.** An entry that leaves memory under the budget of `configurePromptCache(memoryBudgetBytes:)` goes to a spill file with its drafter state. A spill file restores the drafter state for the drafter of the turn. A file written before drafter states were saved, a state of another drafter, a state of another cache kind or window, or a state that does not read back leaves the main caches restored and starts the drafter cold.
- **Budget and eviction.** The drafter state counts in the bytes of the entry, so the memory budget controls it too. `releasePromptCache(sessionID:)` and `evict()` drop it with the entry.

A cold drafter start can lower the acceptance; it does not change the output.

On an M5 Ultra, Qwen3.8-27B-4bit with its 4-bit MTP head decoded a text turn at 90.4 tokens per second against 58.5 without MTP, the second turn of a cached session at 80.9 against 57.5, and guided JSON at 68.2 against 50.8. See "Multi-token prediction (MTP)" in the MLXLMCommon `using` article for the full numbers.

### Copy the cache of a session to a fork

A fork is a new session with its own identifier. To reuse the cache of its parent, call `copyPromptCache(fromSession:toSession:)` before the first pass of the fork, then bind `.session(forkID)` for the passes of the fork. Do not bind the scope of the parent for the fork: that pass takes the cache of the parent and leaves the tokens of the fork in it.

```swift
switch await model.copyPromptCache(fromSession: parentID, toSession: forkID) {
case .copied(let tokenCount): print("the fork starts with \(tokenCount) cached tokens")
case .sourceNotFound, .sourceInUse, .refused: break  // The fork reads its prompt in full.
}
```

- **Whole copy.** The copy has every token of the parent, for each KV-cache configuration of the parent. The first pass of the fork rewinds it to the common prefix with its render. A hybrid model goes back to the longest checkpoint of the copy that the render keeps, or reads the prompt in full: the fork never resumes a recurrent state past the point where the two renders part. Each pass of a hybrid model keeps two checkpoints, one at the end of the instructions and one at the end of the last message of its prompt, so a fork whose first prompt keeps only the instructions of the parent still reads the instructions from the copy.
- **Prefix copy.** `copyPromptCache(fromSession:toSession:keepingTokens:)` keeps the first `tokenCount` tokens of the ledger of the parent, to make a template of the instructions alone. Caches that rewind that far keep exactly those tokens. Caches that cannot (a hybrid model, or a sliding window that wrapped) keep the longest checkpoint of the parent at or below `tokenCount`. A parent on disk is cloned only whole. A `tokenCount` beyond the ledger gives `.refused(.tokenCountBeyondLedger)`, and a parent with no prefix to keep gives `.refused(.noPrefixToKeep)`.
- **Independent sessions.** A pass of one session does not change the cache of the other. `releasePromptCache(sessionID:)` of one session leaves the other.
- **Copy-on-write.** Each cache of the copy is a new object whose arrays share their buffers with the parent until one session writes. A parent on disk gives a clone of its file.
- **Budget.** The copy counts the bytes of the parent in `promptCacheUsage` and the memory budget, also the bytes it shares with the parent: while both sessions live, the shared bytes count two times. It is the most recently used entry.
- **No wait.** When a pass of the parent runs, the call returns `.sourceInUse` at once. A destination that has a cache gives `.refused(.destinationHasCache)`.

## Memory

`MLXLanguageModel` does not set `Memory.cacheLimit`, so the MLX buffer pool follows MLX's default, as it does under `ChatSession`. Plain, MTP and guided decode (schema output and required-tool calls) clear the pool on the first token and every 256 emitted tokens after it, and the pool can still hold several GiB between clears on a long context: 16.0 GiB at most in a guided Qwen3.8-27B-4bit decode of 9,502 tokens at 110,647 rendered tokens of context on an M5 Ultra.

No clear runs when a decode ends. The pool keeps what the decode freed until the next generation clears it, whatever kind of decode that generation runs. A prompt, or a suffix past its cached tokens, long enough to prefill in chunks clears it after the first chunk, and a shorter one clears it on the first token. The pool held 9.7 GiB after a Qwen3.8-27B-4bit MTP decode at 110,737 rendered tokens of context on an M5 Ultra, with the KV cache still held by its prompt cache entry. A released KV cache joins the pool too: 17.86 GiB once that entry was dropped. A guided decode checks its KV cache into its prompt cache entry, as a plain decode does. Before guided decodes kept a prompt cache entry, their KV cache joined the pool when they ended: 9.3 to 18.4 GiB after decodes of 8,323 to 20,894 tokens at 110,647 to 110,648 rendered tokens of context. `evict()` and `evictAll()` release the pool without a generation: each one clears it after it drops the weights and prompt cache entries. The clear covers the whole process, so it also empties freed buffers another loaded model would have reused. Weights that an in-flight `respond()` still holds, and a prompt cache entry in a running or queued spill, reach the pool after that clear, and the next generation clears them. The clear can wait for a spill write, because a safetensors save holds the MLX eval lock for its run.

An iOS or visionOS host that runs under a jetsam limit may want to set `Memory.cacheLimit` itself before the first load; the adapter leaves that value in place.

## Availability

`MLXLanguageModel` exposes an `availability` property — `.available`, `.downloading`, `.unavailable(...)` — for gating on model and download state.

## SwiftPM trait

`FoundationModelsIntegration` is the single SwiftPM trait that turns on the adapter, and it is enabled by default. It is the integration point: with it on, `MLXFoundationModels` provides the `MLXLanguageModel` bridge to `FoundationModels.LanguageModel`.

The full capabilities also require the macOS/iOS/visionOS 27.0 SDK. The bridge is guarded by both the trait and the SDK, so anything short of "trait on, 27.0 SDK" compiles `MLXFoundationModels` down to an empty module.

| Trait | SDK | What you get |
|---|---|---|
| On (default) | 27.0 | The full `MLXLanguageModel` adapter bridging to `FoundationModels.LanguageModel`. |
| On (default) | Older | Nothing; the adapter (and its download-progress observable) is compiled out. |
| Off (`.disableDefaultTraits`) | Any | Nothing compiled in. Use this for iOS-17-era consumers that want `MLXLLM` / `MLXLMCommon` without the adapter. |

## See also

To learn more about the `LanguageModel` protocol this adapter conforms to, see [Bring an LLM provider to the Foundation Models framework](https://www.youtube.com/watch?v=u06ZVpSl0J4) from WWDC26.
