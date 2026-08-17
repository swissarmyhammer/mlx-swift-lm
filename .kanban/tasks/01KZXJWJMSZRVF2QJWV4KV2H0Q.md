---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m03pqtx39bwj177r79gtvskn
  text: |-
    ### One premise of this card is not correct — read before you decide

    Measured on 2026-08-15, by reading the source.

    **The card's option 3 says to "correct the doc comment of `wiredMemoryTicket` so
    that it does not promise what it cannot give". That comment promises nothing of
    the kind.** `Libraries/MLXLMCommon/Evaluate.swift:1774` reads:

    > `wiredMemoryTicket`: Optional wired memory ticket for policy-based
    > coordination across concurrent tasks. This is opt-in and only applied on GPU
    > devices that support wired memory control (macOS 15 / iOS 18 / tvOS 18 or
    > newer).

    It offers coordination across concurrent tasks. It does not offer a residency
    benefit for the weights, thus there is nothing dishonest to correct.

    **The ticket is sized for the KV cache, not for the weights.** Each of the four
    policy documentation blocks of `Libraries/MLXLMCommon/WiredMemoryPolicies.swift`
    gives the same example, at lines 26, 72, 96 and 129:

    ```swift
    let ticket = policy.ticket(size: kvBytes, kind: .active)
    ```

    `kvBytes`. `WiredMemoryTicket.withWiredLimit(ticket)` wraps the generation at
    `Evaluate.swift:2563`, which is the correct scope for a buffer that generation
    itself allocates. Thus the ticket is right for what it covers, and it was never
    the mechanism for the weights.

    ### The gap the card names IS real, and it is separate

    No load path raises the wired limit before it allocates the weights:

    - `Libraries/MLXLMCommon/Load.swift`, `loadWeights(...)`
    - `Libraries/MLXLLM/LLMModelFactory.swift`, the `load(from:using:...)` path

    A buffer joins the Metal residency set when it is MADE, thus a limit raised
    after the load changes nothing. The measurement of `^3gh7rb5` states the size
    of that: 2.10 s for one decode step against 0.068 s, which is 31 times.

    **A working helper already exists, and it lives in the wrong place.**
    `raiseWiredMemoryLimit` appears in 2 files of `IntegrationTesting/` and in **0**
    files of `Libraries/`. `DeepseekV4IntegrationTests.wiredMemoryLimitCoversTheWholeCheckpoint`
    proves it works: it asserts `wiredMemory.isFullyApplied` and that the request
    covers the whole 141 GiB checkpoint. Thus this repository already holds a proven
    implementation, and no shipping consumer can reach it.

    ### What the decision actually is

    The doc-correction half of option 3 falls away. What remains is one question:
    does the library raise the wired limit before the weight load, and does it do so
    by itself or only when a consumer asks?
  timestamp: 2026-08-15T21:55:08.323484+00:00
- actor: claude-code
  id: 01m03qsw4gmfdb7z0e9admwjkm
  text: |-
    ### Research — how the raise must reach the load path

    Read the MLX source of the wired-memory API, and every caller of `loadWeights`.

    **There is NO synchronous route to the wired limit.** `mlx_set_wired_limit` sits
    behind `Cmlx`, which the `mlx-swift` package does not export as a product, thus
    this repository cannot call it. The two public routes are:

    - `Memory.withWiredLimit(_:_ body:)`, synchronous form — DEPRECATED and a NO-OP.
      The source states it: "This synchronous overload is deprecated and is now a
      no-op."
    - `WiredMemoryTicket.start()` — `async`, through the `WiredMemoryManager` actor.
      This is the only form that applies a limit.

    Thus the raise must be awaited. `loadWeights(...)` is synchronous today, and it
    allocates the first weight buffer inside itself, so the raise cannot stand
    before that allocation without an `await` inside the function.

    **The signature change this forces**, and every caller of it:

    - `MLXLMCommon.loadWeights(modelDirectory:model:quantization:perLayerQuantization:)`
      becomes `async throws`.
    - `MLXLMCommon.convert(...)`, both overloads, become `async throws`, because
      `convert` calls `loadWeights` and is the only synchronous caller in the
      library.
    - Callers to update: `MLXLLM/LLMModelFactory`, `MLXVLM/VLMModelFactory`,
      `MLXEmbedders/ModelFactory`, `MLXLMCommon/MTPDrafterModelFactory`,
      `MLXLMCommon/ModelConversion`, `MLXLLM/ModelConversion`, plus
      `Tests/MLXLMTests/MixedPrecisionQuantLoadTests` and four files of
      `IntegrationTesting/`. Each outer entry point is already `async`.

    **How the manager composes limits.** `WiredMemoryManager.desiredLimit()` takes
    the MAXIMUM across policy groups, never the sum. A ticket held for the weights
    therefore does not add to a ticket taken later at generation time; the larger of
    the two wins. The weight ticket must thus carry its own headroom for the KV
    cache and the per-step working set.

    **How the limit is held and never lowered.** The manager restores the baseline
    when the last ticket ends, and a lower limit empties the residency set. The new
    `ModelWeightResidency` actor therefore starts the new ticket BEFORE it ends the
    one it replaces, so the applied limit never falls between the two.

    **Sizing without allocating.** `safetensorWeightURLs(in:)` gives the weight
    files before any array is made, and the file system gives their sizes. Thus the
    request is sized from the checkpoint bytes with no allocation at all.
  timestamp: 2026-08-15T22:13:43.696709+00:00
- actor: claude-code
  id: 01m03swssjqrkj3a7y8nx004cq
  text: |-
    ### The library now raises the wired limit by itself

    `MLXLMCommon.ModelWeightResidency` (new file
    `Libraries/MLXLMCommon/WeightResidency.swift`) is the one owner of the Metal
    wired limit for model weights. `MLXLMCommon.loadWeights` asks it to cover the
    weight files BEFORE it reads the first one:

    ```swift
    let weightURLs = try safetensorWeightURLs(in: modelDirectory)
    await ModelWeightResidency.shared.raise(
        toCoverWeightBytes: weightFileBytes(of: weightURLs))
    ```

    - The size comes from the FILE SIZES on disk, thus nothing is allocated to
      measure it. The request is the weight bytes plus a 25% share for the KV cache
      and the working set of one decode step, clamped to
      `GPU.maxRecommendedWorkingSetBytes()`.
    - The limit never falls. A request below the standing limit changes nothing,
      and a raise starts its new ticket BEFORE it ends the ticket it replaces.
    - Where no wired-memory control exists — an older system, or a device that is
      not a Metal GPU — every request is a no-op that returns `nil`.

    **The `LLMModelFactory.load(from:using:...)` path of the card is covered by that
    one call.** `LLMModelFactory._load` calls `try await loadWeights(...)`, thus it
    raises the limit before it allocates a weight. A second raise in the factory
    would be a copy of the first, thus the code holds one. The same holds for
    `MLXVLM`, `MLXEmbedders`, `MTPDrafterModelFactory` and `convert`.

    ### A BUG this work found: the size of a symbolic link, not the size of the weights

    `URLResourceValues` reads the LINK, and a `huggingface_hub` snapshot holds a
    symbolic link to a blob for each weight file. The first measurement showed it:

    ```
    MEASURE weight bytes: 76
    MEASURE outcome: WiredMemoryOutcome(requestedBytes: 95, appliedBytes: 95)
    ```

    95 bytes, against the 745,270,382 bytes of the checkpoint. The raise did
    nothing at all, and every build and every other assertion stayed green.
    `weightFileBytes(of:)` now resolves each path first, and
    `aSymbolicLinkCountsTheSizeOfItsTarget` guards it.

    ### The public API this forced

    There is no synchronous route to the wired limit (see the research comment
    above), thus these declarations are now `async throws`:

    - `MLXLMCommon.loadWeights(modelDirectory:model:quantization:perLayerQuantization:)`
    - `MLXLMCommon.convert(...)`, both overloads

    Every caller in this repository is updated. Each outer entry point was already
    `async`.
  timestamp: 2026-08-15T22:50:16.754668+00:00
- actor: claude-code
  id: 01m03sx7bgxt9qa9xqdjwhde9x
  text: |-
    ### Acceptance criterion 3 — the measurement on smaller models

    Measured on 2026-08-15, M3 Ultra (512 GiB, `Mac15,14`), DEBUG build, decode
    through `TokenIterator`, 64 steady steps for each run, the first step dropped.
    Each run is its OWN PROCESS, because the wired limit is process-global and
    never falls. "Without" replicates the load path this repository had before this
    change: the same weight read with no raise at all.

    | checkpoint | weight bytes | request applied | decode step WITHOUT | decode step WITH |
    |---|---|---|---|---|
    | `mlx-community/gemma-3-270m-it-4bit` | 150,939,130 | 188,673,912 | 14.4 / 17.7 / 17.8 ms | 12.8 / 15.9 / 18.2 ms |
    | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 695,283,921 | 869,104,901 | 6.88 / 7.07 / 7.13 / 7.16 ms | 6.69 / 6.71 / 6.85 / 7.26 ms |
    | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 4,077,480,215 | 5,096,850,268 | 13.7 / 13.8 / 13.9 / 14.4 / 14.7 ms | 13.5 / 13.7 / 13.9 ms |

    Each cell is the median step of one run, and the runs are listed in sorted
    order.

    **The effect on a small model is ZERO, inside the run-to-run noise.** The
    manager applied the whole request each time — `appliedBytes` equals
    `requestedBytes` in every row — thus the raise took effect and gave no
    measurable time back. The card measured 31 times faster at 141 GiB. The cost
    this change removes therefore does NOT fall linearly with the weight bytes: a
    4 GiB working set stays hot on a 512 GiB machine whatever the wired limit says,
    and a 141 GiB working set does not.

    The third model is above the two the card named. It was added because the two
    small models both measured zero, and 4 GiB shows that the answer is still zero
    one order of magnitude up. `mlx-community/DeepSeek-V4-Flash-4bit` was NOT
    loaded.

    The raise costs nothing where it gives nothing: the wired limit is a limit, not
    a reservation, and it is clamped to the recommended working-set size of the
    device.
  timestamp: 2026-08-15T22:50:30.640414+00:00
- actor: claude-code
  id: 01m03sy8f2dzd6gtb0xe6fhmck
  text: |-
    ### implement — changed

    - evidence:
      - **Files (17):**
        - new `Libraries/MLXLMCommon/WeightResidency.swift` — `WiredMemoryOutcome`,
          `ModelWeightResidency`
        - new `Tests/MLXLMTests/ModelWeightResidencyTests.swift` — 4 no-weights
          tests
        - `Libraries/MLXLMCommon/Load.swift` — `weightFileBytes(of:)`,
          `loadWeights` raises before the first read, now `async throws`
        - `Libraries/MLXLMCommon/WiredMemoryPolicies.swift` —
          `recommendedWorkingSetBytes()` shared in place of a second copy
        - `Libraries/MLXLMCommon/ModelConversion.swift`,
          `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift`,
          `Libraries/MLXLLM/LLMModelFactory.swift`,
          `Libraries/MLXLLM/ModelConversion.swift`,
          `Libraries/MLXVLM/VLMModelFactory.swift`,
          `Libraries/MLXEmbedders/ModelFactory.swift` — callers
        - `Tests/MLXLMTests/MixedPrecisionQuantLoadTests.swift` — callers
        - `IntegrationTesting/…/DeepseekV4SharedCheckpoint.swift` — the local
          wired-limit copy is deleted; it reads
          `ModelWeightResidency.shared.outcome` instead
        - `IntegrationTesting/…/DeepseekV4IntegrationTests.swift`,
          `…/Gemma4AssistantDraftModelIntegrationTests.swift`,
          `…/MTPIteratorEndToEndDiagnosticTests.swift`,
          `…/MTPRung4TokenParityTests.swift`
        - `docs/deepseek-v4-support.md` — "a consumer must do the same" is no
          longer true
      - **Decode-step numbers, `mlx-community/Llama-3.2-1B-Instruct-4bit`:**
        6.88 / 7.07 / 7.13 / 7.16 ms without the raise, 6.69 / 6.71 / 6.85 /
        7.26 ms with it. Zero effect, inside the noise. Two more checkpoints in the
        measurement comment.
      - **Red then green,** each proven with a filtered run:
        - `theRequestCoversTheWeightsAndAWorkingSetAboveThem` — red, the stub
          answered 8388608 against 10485760
        - `aSmallerRequestNeverLowersTheLimit` — red
        - `theLoadRaisesTheLimitBeforeItAllocatesAWeight` — red
        - `aSymbolicLinkCountsTheSizeOfItsTarget` — red with
          `resolvingSymlinksInPath()` taken out, green with it back
      - **Five bundles:** `MLXLMTests` 475 XCTest with 2 failures + 832 Swift
        Testing passed; `MLXGuidedGenerationTests` 70 passed;
        `MLXFoundationModelsTests` 141 passed; `CXGrammarTests` 7 passed;
        `MLXHuggingFaceMacrosTests` 5 XCTest passed. 480 XCTest and 1050 Swift
        Testing, which is 1530 against the baseline of 1526 plus the 4 new tests.
      - **The 2 failures are PRE-EXISTING and this change did not cause them.**
        `TurboQuantIntegrationTests` is flaky. Proof: the work was stashed, the
        package was rebuilt at HEAD, and that class was run 5 times — every one of
        the 5 runs failed, with cosine values from 0.8818 to 0.9573 across
        `testStandardAttentionRepeatFactors`, `testAffineKeyModeMatchesReference`
        and `testRawKeyModeBFloat16MatchesReference`. The same class passed 2 of 3
        runs of the same binary with the change in place. The values move run to
        run although the inputs are seeded. No card is opened for it, because this
        card forbids one.
      - **Builds:** `xcodebuild build-for-testing -scheme mlx-swift-lm-Package` →
        `TEST BUILD SUCCEEDED`, with no new warning.
        `xcodebuild build-for-testing -project IntegrationTesting/…` →
        `TEST BUILD SUCCEEDED`.
      - `swift-format format --in-place --configuration .swift-format` ran on every
        changed Swift file.
    - next: `/review`
  timestamp: 2026-08-15T22:51:04.546161+00:00
- actor: claude-code
  id: 01m03taphbq6808b1tybzxta09
  text: |
    ### review — clean

    - evidence: `review sha 9d574be~1..9d574be`. 0 findings, 0 confirmed, 0 refuted. 16 Swift files reviewed, 0 not reviewed; 9 validators attempted, 0 failed, 0 skipped. The 9 are every validator whose glob covers `*.swift`: swift, code-hygiene, code-security, completeness, complexity, duplication, missing-docs, reuse, test-integrity. The other 3 files of the 19 are the two kanban files and `docs/deepseek-v4-support.md`.
    - out-of-diff findings discarded: 0. The engine raised no candidate at all, thus none needed the standing rule.
    - order of the raise: confirmed. `loadWeights` awaits `ModelWeightResidency.shared.raise(toCoverWeightBytes:)` before the `for url in weightURLs` loop that calls `loadArraysAndMetadata`. Nothing is allocated to measure the size, because `weightFileBytes(of:)` reads the file system.
    - the limit never falls: confirmed. `raise` returns early unless `requestedBytes > highWaterMarkBytes`, and it starts the new ticket before it ends the ticket it replaces.
    - symlink resolution: confirmed. `weightFileBytes(of:)` calls `resolvingSymlinksInPath()` before it reads `.fileSizeKey`. `aSymbolicLinkCountsTheSizeOfItsTarget` makes a real symbolic link to an 8 MiB blob and expects 8 MiB, thus it goes red if the resolution is removed.
    - clamp and no-op path: confirmed. `min(recommended, limitBytes(forWeightBytes:))` clamps to `GPU.maxRecommendedWorkingSetBytes()`, and a `nil` from `recommendedWorkingSetBytes()` returns the outcome unchanged.
    - async propagation: no behavior change beyond `await`. The one other change is `recommendedWorkingSetBytes()` losing `private` so that `WeightResidency.swift` can call it.
    - test integrity: all 4 tests can fail. `safetensorWeightURLs` sorts the index values, thus `a-absent.safetensors` is read first and the ordering test truly allocates nothing before it throws.
    - next: none. Every acceptance item was already checked and no prior review section existed, thus the card moves to done.
  timestamp: 2026-08-15T22:57:52.171494+00:00
position_column: done
position_ordinal: ff8680
title: 'Wired memory: a ticket taken at generate time is too late, thus every large model pays a per-token residency cost'
---
## What

Task `^3gh7rb5` measured a 31-times decode speed-up on `mlx-community/DeepSeek-V4-Flash-4bit` (141 GB, 43 layers, M3 Ultra 512 GiB, release build):

- default Metal wired limit: **2.10 s** for one decode step
- wired limit raised BEFORE the weights are allocated: **0.068 s** for one decode step
- wired limit raised AFTER the load: **2.10 s**, that is no change at all

## Why

MLX gives its Metal residency set a capacity of zero unless the process raises the wired limit (`wired_limit_{0}` in `mlx/backend/metal/allocator.h`), and it takes only buffers of 1 MB or less from the one heap that set holds (`heap_size_ = 1 << 20`). Every model weight tensor is larger than 1 MB. Thus each weight buffer sits outside the residency set, and Metal makes the whole weight set resident again for each command buffer, which is once for each decode step.

A buffer joins the residency set when it is made, thus the limit must stand BEFORE the load. It must also never fall again, because a lower limit empties the set.

## The decision, made by the user on 2026-08-15

Raise the Metal wired limit inside the weight-load path of the library, by
itself, so that every consumer gets the speed with no code of its own. The
user's words: "this isn't a real decision -- how about you work on it to make
it usable fast."

The doc-correction half of the earlier option 3 falls away. The comment of
`wiredMemoryTicket` promises coordination across concurrent tasks, which is
what it gives, thus there is nothing dishonest to correct. That ticket is sized
for the KV cache and it stays as it is.

## Acceptance Criteria

- [x] The decision is recorded on this card, with the numbers above.
- [x] The chosen path is implemented: `MLXLMCommon.ModelWeightResidency` raises
      the limit inside `loadWeights`, before the first weight file is read.
- [x] A measurement on a second, smaller model shows the size of the effect
      there. Measured on three checkpoints from 151 MB to 4.1 GB: the effect is
      zero, inside the run-to-run noise. See the measurement comment.

#deepseek-v4 #performance