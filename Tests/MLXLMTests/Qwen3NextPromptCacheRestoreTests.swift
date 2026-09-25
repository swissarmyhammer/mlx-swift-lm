// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Tests that save the prefill caches of a tiny Qwen3-Next model to a prompt cache file, restore
/// the file into the fresh caches of `newCache(parameters:)`, and decode on the restored caches.
///
/// Qwen3-Next is a hybrid model. Its `newCache` gives a `MambaCache` for each linear-attention
/// layer and a `KVCacheSimple` for each full-attention layer. The saved values of a `MambaCache`
/// do not hold its offset. Only the offset record of the file brings the offset back, thus the
/// test checks the offset of each layer.
///
/// The model computes in float32. The `MLXTestPrecision` target sets `MLX_ENABLE_TF32=0` when
/// this bundle loads, thus a split forward and a single forward agree to approximately 1e-6 also
/// on a GPU with neural accelerators.
struct Qwen3NextPromptCacheRestoreTests {

    // MARK: - Fixture values

    /// The seed of the initializer weights of the tiny model.
    private static let weightSeed: UInt64 = 43

    /// The prompt that the prefill puts in the caches. Each token is in the vocabulary of the
    /// tiny model.
    private static let prompt: [Int32] = [1, 7, 3, 9, 2, 11, 5, 13, 4, 8]

    /// The number of greedy tokens that each decode adds after the prompt.
    private static let decodeStepCount = 4

    /// The class names of the caches that `newCache(parameters:)` gives, layer by layer.
    private static let cacheClassNames = [
        "MambaCache", "KVCacheSimple", "MambaCache", "KVCacheSimple",
    ]

    /// The largest difference between the logits of the restored caches and the logits of the
    /// live caches. The two decodes run the same arrays, thus they agree to the float noise.
    private static let warmTolerance: Float = 1e-6

    /// The largest difference between the logits of the restored caches and the logits of a
    /// cold prefill of the prompt and the first token. A split forward adds rounding differences.
    private static let coldTolerance: Float = 1e-3

    // MARK: - Helpers

    /// Makes the tiny model with fixed initializer weights.
    ///
    /// - Returns: The model.
    /// - Throws: The error of the configuration decoder.
    private static func makeModel() throws -> Qwen3NextModel {
        let configuration = try Qwen3NextCompiledDecodeTests.configuration()
        return withRandomState(MLXRandom.RandomState(seed: weightSeed)) {
            Qwen3NextModel(configuration)
        }
    }

    /// Runs tokens into caches and gives back the logits of the last position.
    ///
    /// - Parameters:
    ///   - model: The model.
    ///   - tokens: The tokens.
    ///   - caches: The caches to fill.
    /// - Returns: The logits, with shape `(1, vocabulary)`.
    private static func lastLogits(
        _ model: Qwen3NextModel, _ tokens: [Int32], caches: [KVCache]
    ) -> MLXArray {
        let logits = model(MLXArray(tokens).expandedDimensions(axis: 0), cache: caches)
        let last = logits[0..., -1, 0...]
        eval(last)
        return last
    }

    /// The token with the largest logit.
    ///
    /// - Parameter logits: The logits, with shape `(1, vocabulary)`.
    /// - Returns: The token.
    private static func greedyToken(_ logits: MLXArray) -> Int32 {
        argMax(logits, axis: -1).item(Int32.self)
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

    // MARK: - The round trip

    @Test("A restored Qwen3-Next prefill fills the newCache templates with each offset")
    func restoredPrefillFillsTheTemplates() throws {
        let model = try Self.makeModel()
        let liveCaches = try model.newCache(parameters: nil)
        _ = Self.lastLogits(model, Self.prompt, caches: liveCaches)
        let url = PromptCacheTemplateRestoreTests.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: liveCaches)
        let templates = try model.newCache(parameters: nil)

        let restored = try loadPromptCacheSnapshot(url: url, into: templates).cache

        #expect(templates.map { String(describing: type(of: $0)) } == Self.cacheClassNames)
        #expect(restored.count == templates.count)
        #expect(
            liveCaches.map(\.offset)
                == Array(repeating: Self.prompt.count, count: Self.cacheClassNames.count))
        for (layer, (cache, template)) in zip(restored, templates).enumerated() {
            #expect(
                PromptCacheTemplateRestoreTests.isSameInstance(cache, template),
                "layer \(layer): restores into its template")
            #expect(cache.offset == Self.prompt.count, "layer \(layer): offset")
            PromptCacheTemplateRestoreTests.expectSameContents(
                cache, liveCaches[layer], "layer \(layer)")
        }
    }

    @Test("A restored Qwen3-Next prefill decodes as the live caches and as a cold prefill")
    func restoredPrefillDecodesAsTheLiveCaches() throws {
        let model = try Self.makeModel()
        let liveCaches = try model.newCache(parameters: nil)
        var token = Self.greedyToken(Self.lastLogits(model, Self.prompt, caches: liveCaches))
        let url = PromptCacheTemplateRestoreTests.temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try savePromptCache(url: url, cache: liveCaches)
        let restoredCaches = try loadPromptCacheSnapshot(
            url: url, into: model.newCache(parameters: nil)
        ).cache
        let coldLogits = Self.lastLogits(
            model, Self.prompt + [token], caches: try model.newCache(parameters: nil))

        for step in 0 ..< Self.decodeStepCount {
            let liveLogits = Self.lastLogits(model, [token], caches: liveCaches)
            let restoredLogits = Self.lastLogits(model, [token], caches: restoredCaches)
            #expect(
                Self.maxAbsDifference(restoredLogits, liveLogits) <= Self.warmTolerance,
                "step \(step): restored and live logits")
            if step == 0 {
                #expect(
                    Self.maxAbsDifference(restoredLogits, coldLogits) <= Self.coldTolerance,
                    "step 0: restored and cold logits")
            }
            token = Self.greedyToken(liveLogits)
        }
        let decodedLength = Self.prompt.count + Self.decodeStepCount
        #expect(restoredCaches.map(\.offset) == liveCaches.map(\.offset))
        #expect(
            restoredCaches.map(\.offset)
                == Array(repeating: decodedLength, count: Self.cacheClassNames.count))
    }
}
