// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXGuidedGeneration
@testable import MLXLLM

/// Guided generation feeds the target every token it emits: a decision whose commit forces
/// tokens feeds the decided token, then the forced tokens. The target's cache thus holds the
/// prompt and every emitted token, and each decision reads logits conditioned on all of them.
///
/// Each generation runs on the plain path and on the MTP path. On the MTP path the drafts and
/// their forced tokens reach the target through verify forwards, and the loop rewinds the rows
/// of each verify the target rejected. The test thus compares the tokens the target's cache
/// keeps, as ``CacheRecordingTarget`` tracks them, and not the raw inputs of the forwards.
///
/// The fixture's target decodes through ``CacheRecordingTarget``, and its grammar writes JSON
/// over ``ToolCallAlphabetTokenizer``, whose keys the grammar forces after the decided quote
/// that opens them.
@Suite(.serialized)
struct GuidedGenerationModelFeedTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [7, 21, 30, 12, 4]

    /// The emitted tokens the cache does not keep: the token of the decision that terminates
    /// the grammar, at most.
    private static let unfedFinalTokens = 1

    /// The block sizes of the generations: `nil` decodes plainly, 2 speculates the shipped
    /// single draft, and 4 speculates chains of three drafts.
    private static let blockSizes: [Int?] = [nil, 2, 4]

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Self.blockSizes)
    func theTargetIsFedEveryEmittedToken(schema: GuidedFixtureSchema, blockSize: Int?) throws {
        let fixture = try Qwen35MTPFixture()
        let recorder = try CacheRecordingTarget(copying: fixture.target)

        let run = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: blockSize,
            target: recorder)

        let emitted = ToolCallAlphabetTokenizer().encode(text: run.text, addSpecialTokens: false)
        let cached = try #require(recorder.cachedTokens)
        let generated = Array(cached.dropFirst(Self.prompt.count))
        #expect(run.sink.speculationFallbackReason == nil)
        #expect((run.sink.draftedTokenCount > 0) == (blockSize != nil))
        #expect(!run.sink.fastForwardTokenIDs.isEmpty)
        #expect(Array(cached.prefix(Self.prompt.count)) == Self.prompt)
        #expect(emitted.starts(with: generated))
        // A decision whose commit terminates the grammar emits its token and stops, so the
        // cache never takes that last token. A generation that stops any other way fed them all.
        #expect(emitted.count - generated.count <= Self.unfedFinalTokens)
    }
}

/// The fixture's target, which tracks the tokens its cache keeps.
///
/// Each forward appends the tokens it feeds. Before it does, the record drops the tokens the
/// cache no longer holds: the rows of the last verify the loop rewound. The record thus follows
/// the offset of the target's full-attention cache, which each rewind trims in place.
///
/// A subclass of the target, and not a wrapper, because the MTP drafter drafts for a Qwen 3.5
/// target only, and reads that target's embedding and LM head. It prepares the prompt as the
/// target does, and a prompt shorter than one prefill chunk feeds through
/// ``callAsFunction(_:cache:state:)`` like every other forward of guided generation.
private final class CacheRecordingTarget: Qwen35TextModel {

    /// The tokens the cache held when the last forward began, then the tokens it fed.
    private var fedTokens: [Int] = []

    /// The cache of the last forward, whose offset ends the record after the generation.
    private var lastCache: [KVCache]?

    /// The tokens the cache holds, in order, or `nil` when no forward ran on a cache with a
    /// full-attention layer.
    var cachedTokens: [Int]? {
        lastCache.flatMap(Self.attentionOffset).map { Array(fedTokens.prefix($0)) }
    }

    /// Makes a target with the configuration and the weights of `target`.
    ///
    /// - Parameter target: The target whose configuration and weights the copy takes.
    /// - Throws: The error of the weight update.
    init(copying target: Qwen35TextModel) throws {
        super.init(target.configuration)
        try update(parameters: target.parameters(), verify: .noUnusedKeys)
        eval(self)
    }

    /// Records the tokens of `input`, then runs the target's forward.
    ///
    /// - Parameters:
    ///   - input: The tokens of the forward.
    ///   - cache: The cache of each layer.
    ///   - state: The model state of the forward.
    /// - Returns: The output of the target's forward.
    override func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        record(input.tokens, cache: cache)
        return super.callAsFunction(input, cache: cache, state: state)
    }

    /// Drops the tokens `cache` no longer holds from ``fedTokens``, then appends `tokens`.
    ///
    /// - Parameters:
    ///   - tokens: The tokens of the forward.
    ///   - cache: The cache of the forward.
    private func record(_ tokens: MLXArray, cache: [KVCache]?) {
        if let held = cache.flatMap(Self.attentionOffset) {
            fedTokens = Array(fedTokens.prefix(held))
        }
        lastCache = cache
        fedTokens += tokens.reshaped(-1).asArray(Int32.self).map(Int.init)
    }

    /// The tokens the full-attention layer of `cache` holds, or `nil` when it has none.
    ///
    /// - Parameter cache: The cache of each layer.
    /// - Returns: The offset of the first layer that is not recurrent.
    private static func attentionOffset(of cache: [KVCache]) -> Int? {
        cache.first { !($0 is MambaCache) }?.offset
    }
}

#endif
