// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import Testing

@testable import MLXFoundationModels

/// Tests that send the prefill caches of a tiny Qwen model through ``ExecutorPromptCacheFile``.
///
/// The executor reads a prompt cache file into the fresh caches that `newCache(parameters:)`
/// gives, and not into caches that it builds from the class names of the file. These tests use
/// that path with random weights, thus no download is necessary:
///
/// - Qwen3.5 is a hybrid model. Its `newCache` gives a `MambaCache` for each linear-attention
///   layer and a `KVCacheSimple` for each full-attention layer.
/// - Qwen3-VL gives a `KVCacheSimple` for each layer.
///
/// Each model keeps its M-RoPE anchor in the model state under its real key. A restored entry
/// must continue the prompt as the live caches do, and as a cold prefill of the whole prompt
/// does.
///
/// The cold comparison needs float32 arithmetic on all paths. The `MLXTestPrecision` target
/// sets `MLX_ENABLE_TF32=0` when this bundle loads, thus a split forward and a single forward
/// agree to approximately 1e-6 also on a GPU with neural accelerators (see
/// `Float32PrecisionTests`).
@Suite("A tiny Qwen entry goes through the executor prompt cache file into newCache templates")
struct ExecutorPromptCacheQwenFileTests: PromptCacheSpoolFixtures {

    // MARK: - Fixture values

    /// The model that the key of each entry names.
    private static let modelID = "test-org/tiny-qwen-prompt-cache-file"

    /// The session that the key of each entry names.
    private static let sessionID = "tiny-qwen-session"

    /// The generation of each file.
    private static let generation: UInt64 = 1

    /// The seed of the initializer weights of each tiny model.
    private static let weightSeed: UInt64 = 1

    /// The number of layers of each tiny model. The configurations below state the same value.
    private static let layerCount = 4

    /// The number of tokens of the prompt that the prefill puts in the caches.
    private static let promptLength = 40

    /// The number of tokens that each continuation adds after the prompt.
    private static let continuationLength = 8

    /// The seed of the prompt tokens.
    private static let promptSeed = 0

    /// The seed of the continuation tokens.
    private static let continuationSeed = 3

    /// The multiplier of the token pattern.
    private static let tokenStride = 13

    /// The offset of the token pattern.
    private static let tokenOffset = 7

    /// The number of plain text token IDs. The special token IDs of the tiny models (500 and
    /// up) are above this range.
    private static let textTokenRange = 480

    /// The largest difference between the logits of the restored caches and the logits of the
    /// live caches. The two continuations run the same arrays, thus they agree to the float
    /// noise.
    private static let warmTolerance: Float = 1e-6

    /// The largest difference between the logits of the restored caches and the logits of a
    /// cold prefill of the whole prompt. A split prefill adds rounding differences.
    private static let coldTolerance: Float = 1e-3

    /// The vision tower of each tiny model. The tests send text only, but the configuration
    /// needs a vision tower.
    private static let visionConfiguration = """
        "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32, "intermediate_size": 64,
        "out_hidden_size": 64, "num_heads": 2, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 64
        """

    /// The special token IDs and the vocabulary of each tiny model.
    private static let specialTokens = """
        "image_token_id": 500, "video_token_id": 501, "vision_start_token_id": 502,
        "vision_end_token_id": 503, "vocab_size": 512
        """

    /// The configuration of the tiny Qwen3.5 model: four layers, and a full-attention layer
    /// after each linear-attention layer.
    private static let qwen35Configuration = """
        {
            "model_type": "qwen3_5_vl", \(specialTokens),
            "text_config": {
                "model_type": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 4,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "vocab_size": 512, "full_attention_interval": 2,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "max_position_embeddings": 4096,
                "rope_parameters": {
                    "type": "default", "mrope_section": [8, 4, 4], "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            },
            "vision_config": { \(visionConfiguration) }
        }
        """

    /// The configuration of the tiny Qwen3-VL model: four attention layers.
    private static let qwen3VLConfiguration = """
        {
            "model_type": "qwen3_vl", \(specialTokens),
            "text_config": {
                "model_type": "qwen3_vl", "hidden_size": 64, "num_hidden_layers": 4,
                "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 16, "vocab_size": 512, "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-6, "rope_theta": 100000.0,
                "rope_scaling": { "type": "default", "mrope_section": [4, 2, 2] }
            },
            "vision_config": { \(visionConfiguration), "deepstack_visual_indexes": [0, 1] }
        }
        """

    // MARK: - The tiny models

    /// One tiny Qwen model of this suite.
    enum TinyQwen: CaseIterable, CustomTestStringConvertible {

        /// The hybrid Qwen3.5 model.
        case qwen35

        /// The Qwen3-VL model.
        case qwen3VL

        /// The model name that `ContinuationStateError` names.
        var modelName: String {
            switch self {
            case .qwen35: "Qwen35"
            case .qwen3VL: "Qwen3VL"
            }
        }

        /// The real key of the M-RoPE anchor in the model state.
        var ropeDeltasKeyName: String {
            switch self {
            case .qwen35: "qwen35.ropeDeltas"
            case .qwen3VL: "qwen35vl.ropeDeltas"
            }
        }

        /// The class names of the caches that `newCache(parameters:)` gives, layer by layer.
        var cacheClassNames: [String] {
            switch self {
            case .qwen35: ["MambaCache", "KVCacheSimple", "MambaCache", "KVCacheSimple"]
            case .qwen3VL:
                Array(
                    repeating: "KVCacheSimple", count: ExecutorPromptCacheQwenFileTests.layerCount)
            }
        }

        /// The typed key of the M-RoPE anchor.
        var ropeDeltasKey: LMOutput.Key<MLXArray> {
            LMOutput.Key(ropeDeltasKeyName)
        }

        /// The name of the case in the test report.
        var testDescription: String { modelName }

        /// Makes the tiny model with fixed initializer weights.
        ///
        /// - Returns: The model.
        /// - Throws: The error of the configuration decoder.
        func makeModel() throws -> any LanguageModel {
            let random = MLXRandom.RandomState(seed: ExecutorPromptCacheQwenFileTests.weightSeed)
            switch self {
            case .qwen35:
                let configuration = try JSONDecoder().decode(
                    Qwen35Configuration.self,
                    from: Data(ExecutorPromptCacheQwenFileTests.qwen35Configuration.utf8))
                return withRandomState(random) { Qwen35(configuration) }
            case .qwen3VL:
                let configuration = try JSONDecoder().decode(
                    Qwen3VLConfiguration.self,
                    from: Data(ExecutorPromptCacheQwenFileTests.qwen3VLConfiguration.utf8))
                return withRandomState(random) { Qwen3VL(configuration) }
            }
        }
    }

    /// One prefill of a tiny model, written to a file.
    struct WrittenPrefill {

        /// The model.
        let model: any LanguageModel

        /// The caches that the prefill filled. The file holds a copy of their state.
        let liveCaches: [KVCache]

        /// The entry that the file holds.
        let entry: ExecutorPromptCacheEntry

        /// The model state of the prefill.
        let state: LMOutput.State

        /// The prompt of the prefill, with shape `(1, promptLength)`.
        let prompt: MLXArray

        /// The URL of the file.
        let url: URL
    }

    // MARK: - Fixture builders

    /// The key of each entry of this suite.
    private static var key: ExecutorPromptCacheKey {
        ExecutorPromptCacheKey(modelID: modelID, sessionID: sessionID)
    }

    /// Makes plain text tokens in a fixed pattern.
    ///
    /// - Parameters:
    ///   - count: The number of tokens.
    ///   - seed: The shift of the pattern.
    /// - Returns: An array of shape `(1, count)`.
    private static func textTokens(_ count: Int, seed: Int) -> MLXArray {
        let values = (0 ..< count).map {
            Int32(($0 * tokenStride + tokenOffset + seed) % textTokenRange)
        }
        return MLXArray(values).expandedDimensions(axis: 0)
    }

    /// The token IDs of a token array, as a ledger.
    ///
    /// - Parameter tokens: An array of shape `(1, count)`.
    /// - Returns: The token IDs.
    private static func ledger(_ tokens: MLXArray) -> [Int] {
        tokens.asArray(Int32.self).map(Int.init)
    }

    /// The model output of a prepare that ran the whole input, or nil for a prepare that left
    /// tokens for the first step.
    ///
    /// - Parameter result: The result of `prepare`.
    /// - Returns: The model output, or nil.
    private static func output(of result: PrepareResult) -> LMOutput? {
        if case .logits(let output) = result {
            return output
        }
        return nil
    }

    /// Runs `tokens` into `caches` and gives back the output.
    ///
    /// - Parameters:
    ///   - model: The model.
    ///   - tokens: The tokens, with shape `(1, count)`.
    ///   - caches: The caches to fill.
    ///   - state: The model state of the tokens before `tokens`, or nil.
    /// - Returns: The model output.
    /// - Throws: The error of the prepare, or an issue when the prepare gives no logits.
    private static func run(
        _ model: any LanguageModel, _ tokens: MLXArray, caches: [KVCache], state: LMOutput.State?
    ) throws -> LMOutput {
        let result = try model.prepare(
            LMInput(text: .init(tokens: tokens)), cache: caches, state: state, prefill: .init())
        return try #require(output(of: result))
    }

    /// The logits of the last position of an output.
    ///
    /// - Parameter output: The model output.
    /// - Returns: The logits, with shape `(1, vocabulary)`.
    private static func lastLogits(_ output: LMOutput) -> MLXArray {
        output.logits[0..., -1, 0...]
    }

    /// The largest absolute difference between two arrays.
    ///
    /// - Parameters:
    ///   - lhs: The first array.
    ///   - rhs: The second array.
    /// - Returns: The difference.
    private static func maxAbsDifference(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        abs(lhs - rhs).max().item(Float.self)
    }

    /// Prefills the prompt into the fresh caches of `tiny` and writes the entry to a file in
    /// `directory`.
    ///
    /// The ledger is the prompt. The render ledger is the prompt and the continuation, as the
    /// render of a later pass is.
    ///
    /// - Parameters:
    ///   - tiny: The tiny model.
    ///   - directory: The folder of the file. The function makes it.
    /// - Returns: The prefill and the URL of its file.
    /// - Throws: The error of the prefill or of the write.
    private static func writePrefill(of tiny: TinyQwen, in directory: URL) throws
        -> WrittenPrefill
    {
        let model = try tiny.makeModel()
        let prompt = textTokens(promptLength, seed: promptSeed)
        let liveCaches = try model.newCache(parameters: nil)
        let state = try #require(run(model, prompt, caches: liveCaches, state: nil).state)
        let tokens = ledger(prompt)
        let continuation = ledger(textTokens(continuationLength, seed: continuationSeed))
        let entry = ExecutorPromptCacheEntry(
            caches: liveCaches, tokens: tokens, renderTokens: tokens + continuation, state: state)

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(
            ExecutorPromptCacheFile.fileName(for: key, generation: generation))
        try ExecutorPromptCacheFile.write(ExecutorPromptCacheFile.prepare(entry, key: key), to: url)
        return WrittenPrefill(
            model: model, liveCaches: liveCaches, entry: entry, state: state, prompt: prompt,
            url: url)
    }

    /// The class name of each cache.
    ///
    /// - Parameter caches: The caches.
    /// - Returns: One class name for each cache.
    private static func classNames(_ caches: [KVCache]) -> [String] {
        caches.map { String(describing: type(of: $0)) }
    }

    /// The object identity of each cache.
    ///
    /// - Parameter caches: The caches.
    /// - Returns: One identity for each cache.
    private static func identities(_ caches: [KVCache]) -> [ObjectIdentifier] {
        caches.map { ObjectIdentifier($0 as AnyObject) }
    }

    // MARK: - The round trip

    @Test(
        "a restored entry fills the newCache templates and continues as the live caches do",
        arguments: TinyQwen.allCases)
    func aRestoredEntryContinuesAsTheLiveCaches(_ tiny: TinyQwen) throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let written = try Self.writePrefill(of: tiny, in: directory)
        let model = written.model
        let templates = try model.newCache(parameters: nil)

        let restored = try ExecutorPromptCacheFile.read(
            from: written.url, key: Self.key, templates: templates)

        #expect(Self.classNames(templates) == tiny.cacheClassNames)
        #expect(Self.classNames(restored.caches) == tiny.cacheClassNames)
        #expect(Self.identities(restored.caches) == Self.identities(templates))
        let ledgerLength = written.entry.tokens.count
        #expect(
            restored.caches.map(\.offset) == Array(repeating: ledgerLength, count: templates.count))
        #expect(restored.tokens == written.entry.tokens)
        #expect(restored.renderTokens == written.entry.renderTokens)
        let savedDeltas = try #require(written.state[tiny.ropeDeltasKey])
        let restoredDeltas = try #require(restored.state?[tiny.ropeDeltasKey])
        #expect(restoredDeltas.shape == savedDeltas.shape)
        #expect(restoredDeltas.dtype == savedDeltas.dtype)
        #expect(arrayEqual(restoredDeltas, savedDeltas).item(Bool.self))

        let continuation = Self.textTokens(Self.continuationLength, seed: Self.continuationSeed)
        let liveLogits = Self.lastLogits(
            try Self.run(model, continuation, caches: written.liveCaches, state: written.state))
        let restoredLogits = Self.lastLogits(
            try Self.run(model, continuation, caches: restored.caches, state: restored.state))
        let coldLogits = Self.lastLogits(
            try Self.run(
                model, concatenated([written.prompt, continuation], axis: 1),
                caches: model.newCache(parameters: nil), state: nil))

        #expect(Self.maxAbsDifference(restoredLogits, liveLogits) <= Self.warmTolerance)
        #expect(Self.maxAbsDifference(restoredLogits, coldLogits) <= Self.coldTolerance)
    }

    // MARK: - The negative control

    @Test(
        "a restored entry without its model state cannot continue",
        arguments: TinyQwen.allCases)
    func aRestoredEntryWithoutStateThrows(_ tiny: TinyQwen) throws {
        let directory = Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let written = try Self.writePrefill(of: tiny, in: directory)
        let model = written.model
        let restored = try ExecutorPromptCacheFile.read(
            from: written.url, key: Self.key, templates: model.newCache(parameters: nil))
        let continuation = Self.textTokens(Self.continuationLength, seed: Self.continuationSeed)

        #expect(restored.state?[tiny.ropeDeltasKey] != nil)
        #expect(
            throws: ContinuationStateError.missingState(
                model: tiny.modelName, key: tiny.ropeDeltasKeyName)
        ) {
            try model.prepare(
                LMInput(text: .init(tokens: continuation)), cache: restored.caches, state: nil,
                prefill: .init())
        }
    }
}

#endif
