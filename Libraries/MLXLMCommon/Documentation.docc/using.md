# Using mlx-swift-lm

How to use mlx-swift-lm in your own tools and applications

## Overview

Using mlx-swift-lm to add LLM, VLM or text embedding capabilities to your own
software is straightforward:

- add a depdendency on `mlx-swift-lm`
- add a dependency on a _Downloader_ and _Tokenizer_
- adapt the API of the Downloader and Tokenizer to conform to the protocols

Then make use of the model:

```swift
import MLXLMCommon
    
let downloader: any Downloader = ...
let tokenizerLoader: any TokenizerLoader = ...

let model = try await loadModel(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-4B-4bit"
)
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

## Downloaders and Tokenizers

There are 3 general ways to select and use concrete Downloader and Tokenizer implementations:

- implementing protocols
- using an integration package
- using [MLXHuggingFace](MLXHuggingFace) macros

If you are <doc:upgrade> from mlx-swift-lm 2.x the macros will be the
simplest way, but consider <doc:#Integration-Packages> as there are alternate
implementations that may provide features and capabilities that you want.

### Implementing Protocols

The other two methods use exactly this technique to wrap concrete 
implementations in the mlx-swift-lm protocol.  You can do this yourself
if you have custom code or simply wish to see how it works.

mlx-swift-lm requires implementation of at least the two tokenizer protocols:

- ``Downloader`` -- required if you need to download weights.  Not needed if you have local weights.
- ``Tokenizer`` -- adapt the concrete tokenizers to the mlx-swift-lm protocol.
- ``TokenizerLoader`` -- factory for ``Tokenizer`` implementations.

You can look at <doc:#Integration-Packages> implementations for examples
of how to write these -- there are only a few properties and methods
and they typically have trivial mappings to the concrete implementation.

A tokenizer can override ``Tokenizer/makeStreamingDetokenizer()`` to provide its
own incremental decoder. The default returns ``NaiveStreamingDetokenizer``.
Each call must create independent state and preserve special tokens used by tool
and reasoning parsers. Emit only stable text from `next()`; return any remaining
text from ``StreamingDetokenizer/finish()``. Generation flushes the decoder before
finishing downstream parsers, and framed protocols start a fresh decoder for each
payload. A consumer that stops early can discard the remaining state.

This example shows adapting `HuggingFace.HubClient` to the `Downloader` protocol:

```swift
import HuggingFace
import MLXLMCommon

struct HubDownloader: MLXLMCommon.Downloader {
    private let upstream: HubClient

    init(_ upstream: HubClient) {
        self.upstream = upstream
    }
    
    init() {
        self.upstream = HubClient()
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {                        
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }                    
}

// now you can use it
let downloader = HubDownloader()
let tokenizerLoader: any TokenizerLoader = ...

let model = try await loadModel(
    from: downloader,
    using: tokenizerLoader,
    id: "mlx-community/Qwen3-4B-4bit"
)
```

### Integration Packages

Integration packages provide an adapter that encapsulates a concrete
implementation.  Adding a dependency on the adapter will transitively
add a dependency on the implementation.

So which adapter do you chose?

- `huggingface/swift-transformers`
    - this is the package that mlx-swift-lm originally integrated with

No additional integration packages are provided at this time, but feel free to
contribute one!

See <doc:#Xcode-projects> for information about how to hook it up.

### MLXHuggingFace Macros

To provide parity with mlx-swift-lm 2.x there is a built in integration with
the HuggingFace downloader and tokenizer implementations using macros.

Add these dependencies to your project (see <doc:#Xcode-projects>):

- [https://github.com/huggingface/swift-huggingface](https://github.com/huggingface/swift-huggingface)
- [https://github.com/huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)

and add `HuggingFace`, `Tokenizers`, `MLXLLM`, `MLXLMCommon` and `MLXHuggingFace` as libraries that your project links.

You can use the integration like this:

```swift
import MLXLLM
import MLXLMCommon
import MLXHuggingFace

import HuggingFace
import Tokenizers

let modelConfiguration = LLMRegistry.gemma31bQAT4bit

let model = try await #huggingFaceLoadModelContainer(
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

or if you prefer more explicit downloader and tokenizer loading for more
control:

```swift
import HuggingFace
import Tokenizers

import MLXLLM
import MLXLMCommon
import MLXHuggingFace

let modelConfiguration = LLMRegistry.gemma31bQAT4bit

let model = try await LLMModelFactory.shared.loadContainer(
    from: #hubDownloader(),
    using: #huggingFaceTokenizerLoader(),
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

`#hubDownloader()` provides an integration just like what is shown in <doc:#Implementing-Protocols> and `#huggingFaceTokenizerLoader()`
provides something similar to load the tokenizers.

See <doc:upgrade> for more information on upgrading from a 2.x release.

## Multi-token prediction (MTP)

Some checkpoints ship a small multi-token prediction head beside the model. A
``ChatSession`` can use that head as a drafter: each round the head proposes a
few tokens, and the model checks them all in one forward pass. The model keeps
the drafts it agrees with and adds one token of its own, so a round can write
several tokens for about the cost of one. The output is the output of plain
decoding (see <doc:#Measured-speed> for how the tokens are compared).

The Qwen3.5-family targets (Qwen3.5, Qwen3.6 and Qwen3.8, text and
vision-language) have MTP heads. See <doc:model-compatibility> for the list.

### Turn MTP on

Load the drafter and pass it to the session in a ``SpeculativeDecodingConfig``.
The drafter must be from the same family as the model: `.qwen35Text` (MLXLLM)
loads a drafter for a model that `LLMModelFactory` loaded.

```swift
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

let model = try await LLMModelFactory.shared.loadContainer(
    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
    configuration: LLMRegistry.qwen3_8_27b_4bit)

let drafter = try await MTPDrafterRegistration.qwen35Text.loadContainer(
    from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
    configuration: MTPDrafterRegistry.qwen3_8_27b_mtp_4bit)

let session = ChatSession(
    model, speculativeDecoding: try SpeculativeDecodingConfig(mtpDrafter: drafter))
print(try await session.respond(to: "What is 17 * 23?"))
```

A drafter that cannot draft for the model does not stop the session. Each turn
then decodes without speculation and reports
``SpeculativeDecodingFallbackReason/incompatibleDrafter``. Call
``MTPDrafterContainer/validateCompatibility(with:)`` to learn why, or to refuse
the pair before the first turn.

### Load the drafter only when it fits

The deferred form loads the drafter on the first turn that speculates, and only
when the memory policy admits it beside the model. The session and its forks
then share the loaded drafter.

```swift
let session = ChatSession(
    model,
    speculativeDecoding: try SpeculativeDecodingConfig(
        mtpDrafterConfiguration: MTPDrafterRegistry.qwen3_8_27b_mtp_4bit,
        mtpDrafterBytes: estimatedDrafterBytes,
        from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
        registration: .matchingTarget(.qwen35Text, .qwen35VLM),
        memoryPolicy: .recommendedWorkingSet))
```

- `registration` names the factory that loads the drafter and the drafter model
  types it registers first. `.qwen35Text` (MLXLLM) is for a model that
  `LLMModelFactory` loaded. `.qwen35VLM` (MLXVLM) is for a model that
  `VLMModelFactory` loaded. ``MTPDrafterRegistration/matchingTarget(_:_:)``
  picks the family that drafts for the model as it loaded. Use it when the model
  comes from `loadModelContainer`, which loads a Qwen3.5-family model as a
  vision-language model whenever MLXVLM is linked.
- `mtpDrafterBytes` is the estimate that the policy weighs before the drafter
  loads. After the load, each turn weighs the drafter's parameters and the
  drafter state that the session carries from the previous turn.
- ``SpeculativeDecodingMemoryPolicy`` with its default action
  (``SpeculativeDecodingMemoryAction/fallbackToDefault``) decodes without
  speculation when the drafter does not fit, and reports
  ``SpeculativeDecodingFallbackReason/memoryBudgetExceeded``. The
  ``SpeculativeDecodingMemoryAction/fail`` action throws
  ``SpeculativeDecodingMemoryError`` instead. Neither action loads the drafter.
- `SpeculativeDecodingConfig(mtpDrafterBytes:memoryPolicy:loadMTPDrafter:)`
  takes your own loader closure in place of a registration.

### Block size and confidence floor

Each round holds up to `blockSize` tokens: up to `blockSize - 1` drafts and the
model's own token. The default ``SpeculativeDecodingConfig/defaultMTPBlockSize``
is 5. The Qwen head chains its drafts, and
``SpeculativeDecodingConfig/defaultMTPMinimumDraftProbability`` (0.75) stops the
chain at the first draft whose top-1 probability falls below that floor. A round
whose first draft (the seed) falls below the floor drafts that one token alone.
Confident text such as math thus drafts deep, and uncertain prose drafts
shallow. `minimumDraftProbability: 0` drafts the whole block every round, and
llama.cpp users know the floor as `--spec-draft-p-min`.

```swift
let config = try SpeculativeDecodingConfig(
    mtpDrafter: drafter, blockSize: 3, minimumDraftProbability: 0)
```

A drafter caps the block at its ``MTPDrafterModel/maximumBlockSize``. The Qwen
drafters cap it at 6, which keeps each round on the fused verify path. A block
size below 2 or a floor outside 0...1 throws
``SpeculativeDecodingConfigurationError``.

### Temperature and sampling

The Qwen drafters speculate at any temperature. Note that
`GenerateParameters()` samples at temperature 0.6 by default: pass
`temperature: 0` for greedy output. At a temperature above zero,
``GenerateParameters/mtpSampledVerification`` chooses how a round checks its
drafts. Both modes emit tokens with the distribution of plain decoding.

- ``MTPSampledVerification/sampleAndMatch`` (the default): the head drafts its
  argmax tokens, the model samples each row with the session's sampler, and the
  round keeps the drafts the samples match. One GPU readback checks a round.
- ``MTPSampledVerification/rejectionSampling``: the head samples its drafts, and
  the model accepts each one with probability `min(1, p(x) / q(x))`, as in the
  speculative sampling papers.

```swift
var parameters = GenerateParameters(temperature: 0.7, topP: 0.95)
parameters.mtpSampledVerification = .rejectionSampling
let session = ChatSession(
    model, speculativeDecoding: try SpeculativeDecodingConfig(mtpDrafter: drafter),
    generateParameters: parameters)
```

On Qwen3.8-27B-4bit at temperature 0.7 and top-p 0.95, sample-and-match decoded
faster overall (see <doc:#Measured-speed>).

### Prompt cache reuse

MTP keeps the multi-turn prompt cache of a ``ChatSession`` working. The drafter
has its own small cache, and the session keeps it at the same position as the
model's cache:

- A follow-up turn reuses the cached prefix and feeds both the model and the
  drafter only the new suffix.
- The checkpoints of a hybrid model (such as the Qwen3.5 family) also keep the
  drafter state. A turn that goes back into the conversation, such as an edit
  or a regenerate in a ``ChatSession/fork(history:)``, returns to the longest
  checkpoint that the new prompt starts with, and resumes the drafter from the
  state kept there.
- ``ChatSession/saveCache(to:)`` writes the drafter state beside the KV cache.
  A session made with
  `ChatSession(_:promptCache:speculativeDecoding:)` from
  ``loadPromptCacheSnapshot(url:)`` resumes the drafter without a new feed over
  the cached prefix.
- The drafter cold-starts at the cached position when it cannot resume: for a
  file saved before drafter states were saved (such a file still loads), for a
  state that another drafter saved, for a drafter cache of another kind or
  window than the session's parameters give, and for damaged values. A cold
  start can lower the acceptance. It does not change the output.

To save and restore a snapshot without a session, see
`PromptCacheSnapshot.drafterContinuation` and
`SerializedMTPDrafterContinuation` in the kv-cache reference of the
`mlx-swift-lm` skill.

### KV cache settings

- With `maxKVSize` (or a typed capacity), the main attention caches rotate. The
  drafter cache then rotates with the same window and keeps a small rewind
  reserve, so it stays bounded and can drop rejected drafts after the ring
  wraps. MTP keeps speculating past the window, and its tokens are the tokens of
  plain decoding with the same `maxKVSize`.
- With `kvBits` (a quantized main cache), the drafter cache stays in full
  precision. It is one layer, so it is small. The tokens are the tokens of plain
  decoding with the same KV setting.
- A drafter state saved with one cache kind or window does not restore into
  another. The drafter cold-starts instead.

### When MTP falls back

A turn that cannot speculate decodes without speculation and does not throw. The
`.info` event of the turn says why:

```swift
for try await event in session.streamDetails(to: prompt) {
    if case .info(let info) = event {
        if let reason = info.speculativeDecodingFallbackReason {
            print("no MTP:", reason, info.passthroughReason ?? "")
        } else if let telemetry = info.speculativeDecodingTelemetry {
            print("acceptance", telemetry.acceptanceRate,
                  "tokens per verify", telemetry.meanEmittedTokensPerTargetCall)
        }
    }
}
```

``GenerateCompletionInfo/speculativeDecodingFallbackReason`` is one of the
``SpeculativeDecodingFallbackReason`` cases:

| Reason | When |
|---|---|
| ``SpeculativeDecodingFallbackReason/incompatibleDrafter`` | The drafter cannot draft for the model, for example a text drafter for a model that loaded as a VLM. |
| ``SpeculativeDecodingFallbackReason/memoryBudgetExceeded`` | The memory policy did not admit the drafter. |
| ``SpeculativeDecodingFallbackReason/unsupportedSampling`` | The drafter needs greedy decoding and the temperature is above zero. The Qwen drafters do not need it. |
| ``SpeculativeDecodingFallbackReason/unsupportedMedia`` | The prompt carries images, video or audio. |
| ``SpeculativeDecodingFallbackReason/unsupportedCache`` | The first MTP cache check refused the cache. |
| ``SpeculativeDecodingFallbackReason/unavailableContinuation`` | The drafter state is missing or cannot follow the main cache. |

``GenerateCompletionInfo/passthroughReason`` holds the reason in words. It is
also set when a stream stops speculating partway through a turn. An
incompatible drafter writes one warning to the `mtp-fallback` logger for each
drafter and model pair, so a drafter that every turn skips leaves one line in
the log. A request for log probabilities with MTP on throws
``ChatSession/ObservationError/unsupportedSpeculativeDecoding``
(see <doc:generation-observability>).

### Measured speed

Qwen3.8-27B-4bit with its 4-bit MTP head (`mlx-community/Qwen3.8-27B-MTP-4bit`)
on an M5 Ultra, Release build. Decode speed is in tokens per second. Acceptance
is the share of drafts the model kept, and tokens per verify is the tokens each
model forward pass wrote. Each one-turn prompt writes 192 tokens. The integration tests
`GenerationBenchmarkTests` and the FoundationModels MTP suites in
`IntegrationTesting` measure these numbers.

Greedy, one turn (`qwen38FourBitMTPGeneration`, load average about 4):

| Setting | Prompt | Plain | MTP | Speedup | Acceptance | Tokens per verify |
|---|---|---|---|---|---|---|
| Default (block 5, floor 0.75) | math | 59.5 | 100.4 | 1.69x | 86.8% | 3.31 |
| Default (block 5, floor 0.75) | code | 59.5 | 87.4 | 1.47x | 80.2% | 2.26 |
| Default (block 5, floor 0.75) | prose | 59.6 | 83.4 | 1.40x | 76.3% | 2.13 |
| Block 2, no floor | math | 59.7 | 92.9 | 1.56x | 93.9% | 1.96 |
| Block 2, no floor | code | 59.7 | 83.2 | 1.40x | 73.6% | 1.75 |
| Block 2, no floor | prose | 59.6 | 80.7 | 1.35x | 69.0% | 1.70 |

Sampled, one turn, temperature 0.7, top-p 0.95, seed 38
(`qwen38FourBitMTPSampledGeneration`, default block and floor):

| Mode | Prompt | Plain | MTP | Speedup | Acceptance | Tokens per verify |
|---|---|---|---|---|---|---|
| `sampleAndMatch` (default) | math | 58.4 | 92.5 | 1.58x | 83.3% | 2.95 |
| `sampleAndMatch` (default) | code | 58.4 | 77.1 | 1.32x | 65.3% | 2.02 |
| `sampleAndMatch` (default) | prose | 58.4 | 79.8 | 1.37x | 70.1% | 2.13 |
| `rejectionSampling` | math | 58.3 | 85.0 | 1.46x | 82.3% | 3.15 |
| `rejectionSampling` | code | 58.4 | 76.3 | 1.31x | 74.0% | 2.40 |
| `rejectionSampling` | prose | 58.4 | 72.8 | 1.25x | 72.3% | 2.11 |

Multi-turn ``ChatSession``, greedy, default block and floor
(`qwen38FourBitMTPSystemPromptCache`: three chats with a system prompt of about
2,200 tokens; follow-ups, follow-ups with thinking on, and an edit; load average
8.6 at the start and 7.0 at the end, from other work on the machine):

- Each follow-up turn reuses the cache and prefills 24 to 38 new tokens, and the
  drafter is fed the same suffix.
- The edit turn goes back to the end of the first reply in a fork. It reuses
  2,169 cached tokens and prefills 123, as the plain session does.
- Follow-up turns, the edit turn included, decode at 85.1 to 102.9 tokens per
  second with MTP against 55.7 to 56.1 without it (1.53x to 1.83x), with an
  acceptance of 77.6% to 92.7%. The cold first turns decode at 86.6 to 104.9.

FoundationModels `MLXLanguageModel`, default block and floor (load average 13.9
at the start and 8.3 at the end, from other work on the machine; each test runs
its plain and MTP sessions back to back, so the speedups are the comparable
figures):

| Case | Plain | MTP | Speedup |
|---|---|---|---|
| Text turn | 58.5 | 90.4 | 1.55x |
| Second turn of a cached session | 57.5 | 80.9 | 1.41x |
| Guided JSON (a generation schema) | 50.8 | 68.2 | 1.34x |

The second cached turn recomputes 26 of its 233 prompt tokens, as plain does,
and the drafter resumes from the state the first turn left.

Before this work, on the same machine at block 2, MTP decoded 80.1 / 71.2 /
70.2 tokens per second on math / code / prose, and 82.5 / 74.4 / 71.6 after the
port of upstream PR 724 (batched greedy verification). The first recorded
baseline, on an M5 Max at block 2 and temperature 0, was math 34.9 to 52.3,
code 33.2 to 43.4 and prose 25.0 to 26.3 tokens per second (plain to MTP).

A greedy MTP turn cannot always match plain decoding bit for bit: the verify
pass runs several tokens through the model in one forward pass, and the 4-bit
kernels round several rows differently from one. The benchmarks therefore
accept an MTP turn that parts from plain decoding only where plain's two best
logits lie within 0.5 of each other (`MTPChatTokenMatch` and its
`nearTieLogitTolerance` in BenchmarkHelpers). Plain decoding flips the same way
between a warm and a cold prompt cache.

## Xcode projects

You can read the [Xcode documentation](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app).

Click on your project (the top item in the Xcode navigator) and select the **Project** (top item).  Then select **Package Dependencies** and click `+` to add a new dependency.

For all integration methods you will need to add:

- [https://github.com/ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)

Beyond that, chose one of the 3 integration methods and add either the adapter packages OR the implementation packages if using macros/local implemenentation.  See <doc#Integration-Packages>.

## Package.swift / SwiftPM

In your Package.swift add a reference to mlx-swift-lm, chosing either the `main` branch or something that tracks versions:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),
```

Beyond that, chose one of the 3 integration methods and add either the adapter packages OR the implementation packages if using macros/local implemenentation.  See <doc#Integration-Packages>.

You can use the <doc:#MLXHuggingFace-Macros> like this:

```swift
.package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
```

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
        .product(name: "HuggingFace", package: "swift-huggingface"),
        .product(name: "Tokenizers", package: "swift-transformers"),
    ]),
```
