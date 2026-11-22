// Copyright © 2026 Apple Inc.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

// The Inkling 2 DSpark drafter (`mtp.safetensors`) and the target side it needs: the feature
// taps of the looped stack and the in-place rewind of a verify pass. The tiny target is the
// model of `Inkling2ModelTests`, whose eight layer visits are the prelude, two passes of the
// loop core and the coda.

/// The tiny target of `Inkling2ModelTests` with an MTP configuration in its `config.json`, as
/// `mlx-community/Inkling-2-Mini-oQ4e-mtp` carries it: two draft layers, a window of six, so the
/// drafter attends the newest five context rows, and blocks of four.
private let tinyDraftJSON = """
    {
      "model_type": "inkling2_loop",
      "text_config": {
        "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 50,
        "unpadded_vocab_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "sliding_window_size": 8, "local_layer_ids": [0, 1, 2, 4],
        "d_rel": 8, "rel_extent": 16, "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
        "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 2, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0, "route_scale": 2.0, "final_logit_softcapping": 30.0,
        "loop_config": {
          "core_start": 2, "core_end": 4, "num_passes": 2, "gate_rank": 8,
          "is_varloop": false
        }
      },
      "mtp_config": {
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 96,
        "rms_norm_eps": 1e-06, "rope_theta": 1000000.0, "local_layer_ids": [0],
        "sliding_window_size": 6, "block_size": 4, "markov_rank": 8,
        "markov_head_type": "rnn_hidden", "q_bias": false, "o_bias": false,
        "target_layer_ids": [1, 3, 5, 8]
      }
    }
    """

/// The feature taps of the tiny drafter.
// swiftlint:disable:next no_magic_numbers the taps tinyDraftJSON names, which the tests compare with what the target and the drafter read
private let tinyTaps = [1, 3, 5, 8]

/// The random seed of the tiny drafter's weights.
private let tinyDrafterSeed: UInt64 = 11

/// The scale of the tiny drafter's random normal weights.
private let tinyDrafterWeightScale: Float = 0.3

/// The step between the tokens of a test prompt, and the first token: a prompt walks the tiny
/// vocabulary rather than repeating one token.
private let promptTokenStride = 7
private let promptTokenOffset = 3

/// The hidden size of the tiny target and drafter.
private let tinyHidden = 64

/// The tokens of the tiny vocabulary that are not padding.
private let tinyVocabulary = 48

/// The total verify block of the tiny drafter: the anchor and three drafts.
private let tinyBlock = 4

/// The convolutions of each layer visit of the target, thus the slots of its convolution cache.
private let convolutionSlots = 4

/// The largest logit difference after a float32 rewind and a replay of the kept tokens.
private let float32RewindTolerance: Float = 1e-3

/// The largest logit difference after a bfloat16 rewind and a replay of the kept tokens.
private let bfloat16RewindTolerance: Float = 0.1

private func tinyDraftConfiguration() throws -> Inkling2DraftConfiguration {
    try JSONDecoder().decode(Inkling2DraftConfiguration.self, from: Data(tinyDraftJSON.utf8))
}

/// The tiny target, with the MTP configuration that names its default taps.
private func tinyTarget() throws -> Inkling2 {
    try Inkling2ModelTests.makeModel(json: tinyDraftJSON)
}

/// A tiny drafter with random weights.
private func tinyDrafter() throws -> Inkling2DraftModel {
    let drafter = Inkling2DraftModel(try tinyDraftConfiguration())
    MLXRandom.seed(tinyDrafterSeed)
    let parameters = Dictionary(
        uniqueKeysWithValues: drafter.parameters().flattened().map { key, value in
            (key, MLXRandom.normal(value.shape) * tinyDrafterWeightScale)
        })
    try drafter.update(
        parameters: ModuleParameters.unflattened(parameters), verify: .noUnusedKeys)
    eval(drafter)
    return drafter
}

/// A prompt of `count` tokens of the tiny vocabulary.
private func prompt(_ count: Int) -> MLXArray {
    MLXArray(
        (0 ..< count).map { Int32(($0 * promptTokenStride + promptTokenOffset) % tinyVocabulary) }
    ).reshaped(1, -1)
}

/// A state that asks the target for the drafter taps.
private func emitState(taps: [Int]? = tinyTaps) -> LMOutput.State {
    var state = LMOutput.State()
    state[mtpEmitFlagKey] = true
    state[mtpTargetLayerIdsKey] = taps
    return state
}

// MARK: - Configuration and loading

@Suite("Inkling2 DSpark drafter loading")
struct Inkling2DraftLoadingTests {

    @Test("The drafter's parameters carry the names and shapes of mtp.safetensors")
    func parameterNames() throws {
        let drafter = Inkling2DraftModel(try tinyDraftConfiguration())
        let shapes = Dictionary(
            uniqueKeysWithValues: drafter.parameters().flattened().map { ($0.0, $0.1.shape) })
        let layer = [
            "dflare_norm.weight": [64], "input_layernorm.weight": [64],
            "post_attention_layernorm.weight": [64], "mlp.gate_proj.weight": [96, 64],
            "mlp.up_proj.weight": [96, 64], "mlp.down_proj.weight": [64, 96],
            "self_attn.q_proj.weight": [64, 64], "self_attn.k_proj.weight": [32, 64],
            "self_attn.v_proj.weight": [32, 64], "self_attn.o_proj.weight": [64, 64],
            "self_attn.k_proj_ctx.weight": [32, 64], "self_attn.v_proj_ctx.weight": [32, 64],
            "self_attn.q_norm.weight": [16], "self_attn.k_norm.weight": [16],
            "self_attn.k_norm_ctx.weight": [16],
        ]
        var expected: [String: [Int]] = [
            "fuse_weight": [3, 4], "mask_embed": [64], "norm.weight": [64],
            "markov_head.dflare_norm.weight": [64], "markov_head.hidden_out.weight": [64, 8],
            "markov_head.hidden_out.bias": [64], "markov_head.markov_w1.weight": [50, 8],
            "markov_head.rnn_proj.weight": [24, 144], "markov_head.rnn_proj.bias": [24],
        ]
        for index in 0 ..< 2 {
            for (key, shape) in layer {
                expected["layers.\(index).\(key)"] = shape
            }
        }
        #expect(shapes == expected)
    }

    @Test("The drafter reads its geometry from the MTP configuration of the target")
    func geometry() throws {
        let drafter = Inkling2DraftModel(try tinyDraftConfiguration())
        #expect(drafter.maximumBlockSize == tinyBlock)
        #expect(drafter.targetLayerIds == tinyTaps)
        #expect(drafter.hiddenSize == tinyTaps.count * tinyHidden)
        #expect(drafter.requiresPromptPrefill)
        #expect(!drafter.requiresSharedTargetKV)
        #expect(drafter.requiresGreedySampling)
        // A caller that sets no block size, as a router does, decodes in blocks of three.
        #expect(drafter.preferredBlockSize == 3)
        #expect(SpeculativeDecodingConfig.mtpBlockSize(nil, for: drafter) == 3)
    }

    @Test("The factory loads mtp.safetensors and skips the target weights beside it")
    func factoryLoadsTheHead() async throws {
        await Inkling2DraftRegistration.register()
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "inkling2-dspark-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(tinyDraftJSON.utf8).write(to: directory.appending(component: "config.json"))
        try save(
            arrays: ["model.norm.weight": MLXArray.zeros([tinyHidden])],
            url: directory.appending(component: "model.safetensors"))
        let source = try tinyDrafter()
        try save(
            arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
            url: directory.appending(component: "mtp.safetensors"))

        let context = try await MTPDrafterModelFactory.shared.load(
            from: directory, using: UnusedTokenizerLoader())

        let loaded = try #require(context.model as? Inkling2DraftModel)
        let loadedNorm = try #require(
            loaded.parameters().flattened().first { $0.0 == "norm.weight" }?.1)
        let sourceNorm = try #require(
            source.parameters().flattened().first { $0.0 == "norm.weight" }?.1)
        #expect(maxAbsoluteDifference(loadedNorm, sourceNorm) == 0)
    }

    @Test("The registry names the oQ4e checkpoint that ships the head")
    func registry() {
        #expect(MTPDrafterRegistry.shared.contains(id: "mlx-community/Inkling-2-Mini-oQ4e-mtp"))
    }

    @Test("The registry entry of the oQ4e checkpoint downloads the head and config.json alone")
    func registryDownloadsTheHeadAlone() {
        #expect(
            MTPDrafterRegistry.inkling2_mini_oq4e_mtp.downloadPatterns == [
                Inkling2DraftRegistration.weightFile, "config.json",
            ])
        let others = MTPDrafterRegistry.shared.models.filter {
            $0.name != MTPDrafterRegistry.inkling2_mini_oq4e_mtp.name
        }
        #expect(!others.isEmpty)
        #expect(others.allSatisfy { $0.downloadPatterns == nil })
    }

    @Test("A drafter-only load of the oQ4e checkpoint at a pinned revision fetches only the head")
    func pinnedDrafterLoadFetchesTheHeadAlone() async throws {
        await Inkling2DraftRegistration.register()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "inkling2-dspark-download-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let downloader = RecordingDownloader(directory: root)
        let id = MTPDrafterRegistry.inkling2_mini_oq4e_mtp.name
        let directory = downloader.directory(for: id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(tinyDraftJSON.utf8).write(to: directory.appending(component: "config.json"))
        let source = try tinyDrafter()
        try save(
            arrays: Dictionary(uniqueKeysWithValues: source.parameters().flattened()),
            url: directory.appending(component: Inkling2DraftRegistration.weightFile))
        let revision = "3eea042751cd296b7266628a8342c1749242e1fe"

        let container = try await MTPDrafterModelFactory.shared.loadContainer(
            from: downloader, using: UnusedTokenizerLoader(),
            configuration: ModelConfiguration(id: id, revision: revision))

        #expect(
            downloader.calls == [
                .init(
                    id: id, revision: revision,
                    patterns: [Inkling2DraftRegistration.weightFile, "config.json"])
            ])
        #expect(await container.perform { $0.model is Inkling2DraftModel })
    }

    @Test("A target the drafter cannot draft for throws the drafter, the expectation and the type")
    func incompatibleTargetThrows() throws {
        let drafter = Inkling2DraftModel(try tinyDraftConfiguration())
        let target = PrefillStepRecordingModel(ownDefaultStepSize: nil)
        #expect(
            throws: MTPDrafterCompatibilityError.incompatibleTarget(
                drafter: "Inkling2DraftModel", expected: "a test target",
                actual: "PrefillStepRecordingModel")
        ) {
            try drafter.requireCompatibleTarget(target, expected: "a test target")
        }
    }

    @Test("A registered drafter configuration builds the drafter its predicate accepts")
    func registeredConfigurationBuildsTheDrafter() async throws {
        let registry = ModelTypeRegistry<any MTPDrafterModel>()
        await registry.registerDrafter(
            modelTypes: Inkling2DraftRegistration.modelTypes,
            configuration: Inkling2DraftConfiguration.self,
            matches: { $0.draft.markovHeadType == Inkling2DraftConfiguration.markovHeadType },
            create: { Inkling2DraftModel($0) })
        let drafter = try await registry.createModel(
            configuration: Data(tinyDraftJSON.utf8), modelType: "inkling2_loop")
        #expect(drafter is Inkling2DraftModel)
    }

    @Test("A registered drafter configuration its predicate rejects builds no drafter")
    func rejectedConfigurationBuildsNoDrafter() async throws {
        let registry = ModelTypeRegistry<any MTPDrafterModel>()
        await registry.registerDrafter(
            modelTypes: Inkling2DraftRegistration.modelTypes,
            configuration: Inkling2DraftConfiguration.self, matches: { _ in false },
            create: { Inkling2DraftModel($0) })
        await #expect(throws: (any Error).self) {
            _ = try await registry.createModel(
                configuration: Data(tinyDraftJSON.utf8), modelType: "inkling2_loop")
        }
    }
}

// MARK: - Target taps and rewind

@Suite("Inkling2 DSpark target", .serialized)
struct Inkling2DraftTargetTests {

    @Test("The target emits one row of the four taps per position")
    func emitsTaps() throws {
        let target = try tinyTarget()
        let tokens = prompt(5)
        let output = target(LMInput.Text(tokens: tokens), cache: nil, state: emitState())
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        #expect(hidden.shape == [1, 5, tinyTaps.count * tinyHidden])
        #expect(output.logits.shape == [1, 5, tinyVocabulary])
    }

    @Test("Without named taps the target emits the taps of its own MTP configuration")
    func emitsConfiguredTaps() throws {
        let target = try tinyTarget()
        let output = target(
            LMInput.Text(tokens: prompt(3)), cache: nil, state: emitState(taps: nil))
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        #expect(hidden.dim(-1) == tinyTaps.count * tinyHidden)
    }

    @Test("A chunked prefill hands every prompt row to the drafter feed")
    func prefillFeedsEveryRow() throws {
        let target = try tinyTarget()
        let tokenCount = 30
        var fed = 0
        var state = emitState()
        state[mtpPrefillChunkHandlerKey] = { tokens, hidden, _ in
            #expect(hidden.dim(1) == tokens.dim(-1))
            fed += tokens.dim(-1)
        }
        let result = try target.prepare(
            LMInput(tokens: prompt(tokenCount)), cache: target.newCache(parameters: nil),
            state: state, prefill: .init(stepSize: 7))
        guard case .logits(let output) = result else {
            Issue.record("prepare did not return logits")
            return
        }
        let remaining = try #require(output.state?[mtpLastHiddenStatesKey])
        #expect(fed + remaining.dim(1) == tokenCount)
        #expect(output.logits.shape == [1, 1, tinyVocabulary])
    }

    @Test("The target advertises the in-place rewind of a verify pass")
    func verifiesInPlace() throws {
        let target = try tinyTarget()
        let cache = try target.newCache(parameters: nil)
        #expect(
            MTPSpeculativeTokenIterator.verifiesInPlace(
                mainModel: target, mainCache: cache, drafts: tinyBlock - 1))
    }

    @Test(
        "A verify pass through the decode kernels computes each token as a decode step does",
        arguments: [1, 3])
    func sequentialDecodeMatchesDecodeSteps(promptLength: Int) throws {
        let target = try Inkling2ModelTests.makeModel(bfloat16: true, json: tinyDraftJSON)
        let tokens = prompt(promptLength + tinyBlock)

        let difference = try verifyDifferenceFromDecodeSteps(
            target, prefix: [tokens[0..., ..<promptLength]], verify: tokens[0..., promptLength...])

        #expect(difference == 0)
    }

    @Test("A pass through the decode kernels leaves an ArraysCache states of their own")
    func sequentialDecodeCopiesPlainConvolutionStates() throws {
        let target = try Inkling2ModelTests.makeModel(bfloat16: true, json: tinyDraftJSON)
        let cache: [KVCache] = try target.newCache(parameters: nil).map { entry in
            let list = try #require(entry as? CacheList)
            return CacheList(list.children[0], ArraysCache(size: convolutionSlots))
        }
        _ = target(prompt(tinyBlock), cache: cache)
        eval(target(prompt(tinyBlock), cache: cache))

        let states = cache.compactMap { ($0 as? CacheList)?.children[1] as? ArraysCache }
            .flatMap { convolution in (0 ..< convolutionSlots).compactMap { convolution[$0] } }
        #expect(!states.isEmpty)
        // A view of a longer window steps over the window's rows from one batch row to the next.
        for state in states {
            let batchStride = state.asData(access: .noCopy).strides[0]
            #expect(batchStride <= state.dim(1) * state.dim(-1))
        }
    }

    @Test(
        "Rewinding a verify pass leaves the caches of the kept tokens",
        arguments: [6, 12], [(1, false), (3, false), (1, true), (3, true)])
    func rewindAfterPartialAccept(promptLength: Int, rejection: (Int, Bool)) throws {
        let (rejected, bfloat16) = rejection
        let target = try Inkling2ModelTests.makeModel(bfloat16: bfloat16, json: tinyDraftJSON)
        let tokens = prompt(promptLength + tinyBlock + 1)
        let verified = tokens[0..., promptLength ..< (promptLength + tinyBlock)]
        let kept = tinyBlock - rejected
        let next = tokens[0..., (promptLength + tinyBlock)...]

        let rewound = try target.newCache(parameters: nil)
        _ = target(tokens[0..., ..<promptLength], cache: rewound)
        eval(target(verified, cache: rewound))
        #expect(rewindSpeculativePromptCache(rewound, numTokens: rejected) == rejected)
        let rewoundLogits = target(next, cache: rewound)

        let replayed = try target.newCache(parameters: nil)
        _ = target(tokens[0..., ..<promptLength], cache: replayed)
        eval(target(verified[0..., ..<kept], cache: replayed))
        let replayedLogits = target(next, cache: replayed)

        // In bfloat16 the replay feeds the kept tokens as a pass of their own, so the next
        // token's logits round as another pass would.
        let tolerance = bfloat16 ? bfloat16RewindTolerance : float32RewindTolerance
        #expect(maxAbsoluteDifference(rewoundLogits, replayedLogits) < tolerance)
        for (a, b) in zip(rewound, replayed) {
            #expect(a.offset == b.offset)
        }
    }
}

// MARK: - Convolution cache

@Suite("Speculative convolution cache")
struct SpeculativeConvolutionCacheTests {

    /// The rows of a window: the three rows of the state before the pass, then one per token.
    private static func window(tokens: Int, channels: Int = 2) -> MLXArray {
        MLXArray(0 ..< Int32((3 + tokens) * channels)).reshaped(1, 3 + tokens, channels)
    }

    @Test("A rewind takes the state after the kept tokens from the kept windows")
    func rewindRecutsState() {
        let cache = SpeculativeConvolutionCache(size: 1, maximumVerifyPositions: 8)
        let window = Self.window(tokens: 5)
        cache[0] = window[0..., 5..., 0...]
        cache.advance(5)
        cache.keepVerifyWindows([window])

        #expect(cache.maxTrimCount == 5)
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 3)
        #expect(cache[0].map { maxAbsoluteDifference($0, window[0..., 3 ..< 6, 0...]) } == 0)
        #expect(cache.trim(3) == 3)
        #expect(cache[0].map { maxAbsoluteDifference($0, window[0..., 0 ..< 3, 0...]) } == 0)
    }

    @Test("The next forward pass drops the windows of a verify pass")
    func advanceDropsWindows() {
        let cache = SpeculativeConvolutionCache(size: 1, maximumVerifyPositions: 8)
        let window = Self.window(tokens: 4)
        cache[0] = window[0..., 4..., 0...]
        cache.advance(4)
        cache.keepVerifyWindows([window])
        cache.advance(1)
        #expect(cache.maxTrimCount == 0)
        #expect(cache.trim(1) == 0)
    }

    @Test("A copy holds the states but not the windows")
    func copyDropsWindows() throws {
        let cache = SpeculativeConvolutionCache(size: 1, maximumVerifyPositions: 8)
        let window = Self.window(tokens: 4)
        cache[0] = window[0..., 4..., 0...]
        cache.advance(4)
        cache.keepVerifyWindows([window])
        let copy = try #require(cache.copy() as? SpeculativeConvolutionCache)
        #expect(copy.offset == 4)
        #expect(copy.maxTrimCount == 0)
    }
}

// MARK: - Drafting

@Suite("Inkling2 DSpark drafting", .serialized)
struct Inkling2DraftingTests {

    /// Prefills the drafter over a prompt of `count` tokens and returns the prefill hidden rows.
    private func primed(
        _ drafter: Inkling2DraftModel, target: Inkling2, count: Int
    ) throws -> MTPDrafterState {
        let output = target(LMInput.Text(tokens: prompt(count)), cache: nil, state: emitState())
        let hidden = try #require(output.state?[mtpLastHiddenStatesKey])
        var state = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(
            target: target, promptTokens: prompt(count), targetHidden: hidden,
            firstBonus: MLXArray([Int32(5)]), positionDeltas: nil, state: &state,
            sampler: ArgMaxSampler())
        return state
    }

    @Test("A block proposes the requested drafts and leaves the drafter state", arguments: [2, 4])
    func blockShapes(blockSize: Int) throws {
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        var state = try primed(drafter, target: target, count: 9)
        #expect(state.nextPosition == 9)
        let drafts = drafter.draftBlock(
            target: target, lastToken: MLXArray([Int32(5)]), lastHidden: MLXArray.zeros([1, 1, 1]),
            sharedKV: [:], positionDeltas: nil, queryOffset: 9, blockSize: blockSize,
            state: &state, sampler: ArgMaxSampler())
        #expect(drafts.shape == [1, blockSize - 1])
        #expect(drafts.max().item(Int.self) < tinyVocabulary)
        #expect(state.nextPosition == 9)
        #expect(state.isAligned)
    }

    @Test("Finalizing after a partial accept rewinds the drafter rings past their window")
    func finalizeRewindsRings() throws {
        let target = try tinyTarget()
        let drafter = try tinyDrafter()
        var state = try primed(drafter, target: target, count: 9)
        let verify = target(
            LMInput.Text(tokens: prompt(tinyBlock)), cache: nil, state: emitState())
        let verifyHidden = try #require(verify.state?[mtpLastHiddenStatesKey])
        drafter.commitDrafterState(
            target: target, targetHidden: verifyHidden, draftTokens: prompt(tinyBlock - 1),
            acceptedCount: 2, finalToken: MLXArray([Int32(7)]), positionDeltas: nil,
            state: &state, sampler: ArgMaxSampler())
        #expect(state.nextPosition == 12)

        let finalized = drafter.finalizeDrafterState(
            target: target, targetBoundaryHidden: verifyHidden[0..., 1 ..< 2, 0...],
            targetProcessedTokenCount: 11, discardedTargetTokens: 1, positionDeltas: nil,
            state: &state)
        #expect(finalized)
        #expect(state.nextPosition == 10)
        #expect(state.isAligned)
    }

    @Test(
        "Greedy MTP decodes the tokens of plain greedy decoding",
        arguments: [5, 12], [false, true])
    func greedyMatchesPlainGreedy(promptLength: Int, bfloat16: Bool) throws {
        let target = try Inkling2ModelTests.makeModel(bfloat16: bfloat16, json: tinyDraftJSON)
        let drafter = try tinyDrafter()
        let input = LMInput(tokens: prompt(promptLength).reshaped(-1))
        let parameters = GenerateParameters(maxTokens: 20, temperature: 0)
        var baseline = try TokenIterator(input: input, model: target, parameters: parameters)
        var iterator = try MTPSpeculativeTokenIterator(
            input: input, mainModel: target, drafter: drafter, parameters: parameters,
            blockSize: tinyBlock)

        var expected: [Int] = []
        while let token = baseline.next() { expected.append(token) }
        var output: [Int] = []
        while let token = iterator.next() { output.append(token) }

        #expect(output == expected)
        #expect(iterator.passthroughReason == nil)
        #expect(iterator.proposedCount > 0)
    }
}
