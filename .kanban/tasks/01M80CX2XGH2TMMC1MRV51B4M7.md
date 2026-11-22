---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m81k7tk0hjam3s1x88nqgtgw
  text: |-
    Resolved in 8cba0cfa. Each cause removed across the whole file:
    - [x] Inkling2Attention.swift:27 duplication — `l2Clip` and `l2ClipScaled` share `clippedStates(_:scale:threshold:)` (float32 clipped states); `l2ClipScaled` keeps only the tau scaling and rounding.
    - [x] Inkling2Attention.swift:45 no_unchecked_sendable — `QuantizedLinearBox` drops the Sendable conformance (nothing sends it; `FusedQuantizedLinearProjectionCache` beside it is not Sendable either).
    - [x] Inkling2Attention.swift:237 duplication — key and value convolution fusion share `prepareConvolutionProjection(_:linear:cache:key:)`.
    - [x] Inkling2Configuration.swift missing-docs — a documentation comment on all 54 flagged public declarations (plus the struct inits).
    - [x] Inkling2DecoderLayer.swift:548 no_unchecked_sendable — `MessageControllersBox` drops the Sendable conformance.
    - Same cause elsewhere: `Inkling2DecodeKernels` (Inkling2Kernels.swift) holds only immutable `MLXFastKernel`s (Sendable), so it is now a plain `Sendable`. `Inkling2ConstantBox` already carries the swiftlint disable comment with its invariant.

    Review `review sha HEAD~1..HEAD` of 8cba0cfa: 0 findings.
    Tests: swift build --build-tests clean; MLXLMTests 2561 passed (3 known issues), MLXGuidedGenerationTests passed, MLXFoundationModelsTests 570 passed, CXGrammarTests 7 passed, MLXHuggingFaceMacrosTests 11 passed.
  timestamp: 2026-11-22T12:21:03.712387+00:00
position_column: done
position_ordinal: ffffd180
title: 'Inkling2 split files: findings on moved code (from ^04yn8q4 review of f0ce3a2c)'
---
The review of f0ce3a2c (split of `Libraries/MLXVLM/Models/Inkling2.swift` into six files) ran the duplication validator over the split files for the first time. These findings land on code that the commit only moved; the same code was in `Inkling2.swift` at f0ce3a2c~1. They were filed here, out of the ^04yn8q4 scope, at the user's direction. Each item is the engine text word for word.

- [ ] `Libraries/MLXVLM/Models/Inkling2Attention.swift:27` `duplication/duplication` — `l2ClipScaled` repeats the body of `l2Clip` line for line: the float32 scale, the L2 norm, and the clip against `threshold / (norm + normEpsilon)`. It differs only by the trailing `tau` scaling. The two copies will drift when the clip math changes, and a fix to one will be missed in the other. Extract a shared plain helper, for example `clippedStates(_ x: MLXArray, scale: MLXArray, threshold: MLXArray) -> MLXArray` that returns the float32 clipped states before the dtype cast. Call it from `l2Clip` and from `l2ClipScaled`, and keep only the `tau` scaling and the rounding in `l2ClipScaled`.
- [ ] `Libraries/MLXVLM/Models/Inkling2Attention.swift:45` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.
- [ ] `Libraries/MLXVLM/Models/Inkling2Attention.swift:237` `duplication/duplication` — The value-convolution fusion block in `prepareFusedProjections` is a copy of the key-convolution block directly above it. Only the names differ (`dynamicKeyConvolution`/`fusedKey`/`keyProj`/`k_proj` against `dynamicValueConvolution`/`fusedValue`/`valueProj`/`v_proj`). The two copies must stay in lockstep for the prepare, view-install and cache-invalidation sequence to remain correct. Extract a private helper, for example `prepareConvolutionProjection(_ convolution: Inkling2DynamicShortConvolution?, linear: Linear, cache: FusedQuantizedLinearProjectionCache, key: String)`, that runs the prepare call and installs the `<key>` and `weight_proj` views. Call it once for the key and once for the value.
- [ ] `Libraries/MLXVLM/Models/Inkling2Configuration.swift` `code-hygiene/missing-docs-swift` — AllPublicDeclarationsHaveDocumentation: add a documentation comment for each public declaration. The engine flagged 54 of them: `Inkling2LoopConfiguration` and its `coreStart`, `coreEnd`, `passes`, `gateRank`, `core`; `Inkling2TextConfiguration` and its `hiddenSize`, `hiddenLayers`, `vocabularySize`, `unpaddedVocabularySize`, `attentionHeads`, `kvHeads`, `headDim`, `slidingAttentionHeads`, `slidingKVHeads`, `slidingHeadDim`, `slidingWindow`, `localLayerIds`, `relativeDim`, `relativeExtent`, `logScalingFloor`, `logScalingAlpha`, `rmsNormEps`, `qkClipThreshold`, `relativeClipThreshold`, `convolutionKernelSize`, `layersPerBlock`, `experts`, `expertsPerToken`, `intermediateSize`, `gateSoftcap`, `upSoftcap`, `routeScale`, `finalLogitSoftcap`, `loop`, `init`; `Inkling2VisionConfiguration` and its `hiddenSize`, `hiddenLayers`, `attentionHeads`, `patchSize`, `mergeSize`, `positionGridSize`, `projectionHiddenSize`, `projectorBias`, `mlpRatio`, `layerNormEps`, `ropeTheta`, `init`; `Inkling2Configuration` and its `modelType`, `textConfiguration`, `visionConfiguration`, `imageTokenId`, `init`.
- [ ] `Libraries/MLXVLM/Models/Inkling2DecoderLayer.swift:548` `code-hygiene/disallowed-constructs-swift` — no_unchecked_sendable: Instead of @unchecked Sendable, write a plain Sendable conformance or a @preconcurrency import. If the type really must be @unchecked Sendable, write // swiftlint:disable:next no_unchecked_sendable above it with the synchronization invariant that makes the type thread-safe.

A finding shows one example of a cause: remove each cause from the whole file. Build with `swift build --build-tests`, run all five bundles with `xcrun xctest` per CLAUDE.md, and run `swift-format` before the commit.

#inkling