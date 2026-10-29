// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// What the prefill of a guided generation gives: the logits of the last prompt position and,
/// when the prefill fed an MTP drafter, the prompt segment the final forward evaluated with one
/// target hidden row for each of its tokens.
struct GuidedPrefill {

    /// The logits of the final prefill forward, shape `[B, positions, vocabulary]`.
    let logits: MLXArray

    /// The prompt tokens the final forward evaluated, or `nil` when `prepare` evaluated the
    /// whole prompt itself.
    let segmentTokens: MLXArray?

    /// One target hidden row for each token of ``segmentTokens``, or `nil` when the forward
    /// emitted none.
    let segmentHidden: MLXArray?

    /// The target position deltas of the final forward, or `nil` for a target without them.
    let positionDeltas: MLXArray?
}

/// The target model of one guided generation, with its key/value cache and the model state its
/// forwards carry from one to the next.
///
/// Plain decoding feeds the target one token per forward. An MTP round feeds it a verify of
/// several tokens that asks for its drafter state, and rewinds the rejected tail in place.
struct GuidedTarget {

    /// The target model.
    let model: any LanguageModel

    /// The cache of the target and its timeline.
    let cacheStorage: KVCacheStorage

    /// The model state the next forward carries, without the MTP exchange values.
    private(set) var state: LMOutput.State?

    /// The cache of each layer.
    var cache: [KVCache] { cacheStorage.cache }

    /// Makes a new cache for `model`.
    ///
    /// - Parameters:
    ///   - model: The target model.
    ///   - parameters: The parameters of the cache.
    /// - Throws: The error of the cache plan.
    init(model: any LanguageModel, parameters: GenerateParameters) throws {
        let plan = try parameters.kvCachePlan()
        self.model = model
        self.cacheStorage = try plan.validated(
            KVCacheStorage(try model.newCache(parameters: parameters), plan: plan))
    }

    /// Prefills the prompt.
    ///
    /// With a drafter feed, `prepare` hands the feed the hidden rows of each prompt chunk, and
    /// the final forward emits the hidden rows of the segment after the chunks.
    ///
    /// - Parameters:
    ///   - input: The prompt.
    ///   - parameters: The prefill parameters.
    ///   - drafterFeed: The handler of each prefill chunk's hidden rows, or `nil`.
    /// - Returns: The logits of the prompt and the segment the final forward evaluated.
    /// - Throws: The error of the prefill or of the cache plan.
    mutating func prefill(
        _ input: LMInput, parameters: PrefillParameters, drafterFeed: MTPPrefillChunkHandler?
    ) throws -> GuidedPrefill {
        let inputLength = input.text.cacheSequenceLength
        let output: LMOutput
        let segment: MLXArray?
        switch try model.prepare(
            input, cache: cache, state: drafterFeed.map { LMOutput.State().feeding($0) },
            prefill: parameters)
        {
        case .tokens(let tokens):
            let remainingLength = tokens.cacheSequenceLength
            precondition(
                remainingLength <= inputLength,
                "LanguageModel.prepare returned more tokens than it received")
            cacheStorage.commitProcessedTokens(inputLength - remainingLength)
            output = model(
                tokens[text: .newAxis], cache: cache,
                state: drafterFeed.map { _ in Self.emittingState(from: nil) })
            cacheStorage.commitProcessedTokens(remainingLength)
            segment = tokens.tokens
        case .logits(let result):
            cacheStorage.commitProcessedTokens(inputLength)
            output = result
            segment = nil
        }
        try cacheStorage.plan.applyAndValidate(to: cacheStorage)
        state = drafterFeed == nil ? output.state : output.state?.withoutMTPExchange
        return GuidedPrefill(
            logits: output.logits, segmentTokens: segment,
            segmentHidden: output.state?[mtpLastHiddenStatesKey],
            positionDeltas: output.state?[mtpPositionDeltasKey])
    }

    /// Feeds `tokens` one forward at a time, as plain guided decoding feeds them.
    ///
    /// One-token forwards take the optimized single-query attention kernel. A forward of several
    /// tokens on a populated cache is proven against them on the Qwen 3.5 targets only, which an
    /// MTP round verifies in place.
    ///
    /// - Parameter tokens: The tokens, at least one.
    /// - Returns: The logits of the last token, unevaluated.
    mutating func feedOneAtATime(_ tokens: [Int32]) -> MLXArray {
        var logits: MLXArray?
        for token in tokens {
            let input = LMInput.Text(tokens: MLXArray([token]))
            let result = model(
                input[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
            cacheStorage.commitProcessedTokens(input.cacheSequenceLength)
            state = result.state
            logits = result.logits
        }
        cacheStorage.plan.apply(to: cacheStorage)
        guard let logits else {
            preconditionFailure("A guided feed holds at least one token")
        }
        return logits
    }

    /// One forward over `tokens` that asks the target for its drafter state. A forward of
    /// several tokens also keeps the snapshot slots that ``keepVerified(_:of:)`` rewinds into.
    ///
    /// - Parameter tokens: The tokens of the verify.
    /// - Returns: The logits and the drafter state of each token.
    mutating func verify(_ tokens: [Int32]) -> LMOutput {
        var verifyState = Self.emittingState(from: state)
        verifyState[mtpVerifyKeepsSnapshotsKey] = tokens.count > 1 ? true : nil
        let output = model(
            LMInput.Text(tokens: MLXArray(tokens))[text: .newAxis], cache: cache,
            state: verifyState)
        state = output.state?.withoutMTPExchange
        return output
    }

    /// Keeps the first `kept` of the `verified` tokens of the last verify, and rewinds the rest
    /// of them in place.
    ///
    /// - Parameters:
    ///   - kept: The tokens to keep, at least one.
    ///   - verified: The tokens of the last verify.
    func keepVerified(_ kept: Int, of verified: Int) {
        let rejected = verified - kept
        if rejected > 0 {
            let rewound = rewindSpeculativePromptCache(cache, numTokens: rejected)
            precondition(
                rewound == rejected,
                "The target verifies in place, but rewound \(rewound) of \(rejected) tokens")
        }
        cacheStorage.commitProcessedTokens(kept)
        cacheStorage.plan.apply(to: cacheStorage)
    }

    /// The most tokens one forward of the target's verify holds: its in-place rewind depth and
    /// the token at its head.
    var mostVerifiedTokens: Int {
        ((model as? any SpeculativeCacheRewindModel)?.maximumNativeTargetCacheRewind ?? 0) + 1
    }

    /// Whether a verify of `count` tokens runs in place: each query attends as a one-token
    /// forward at its position would, and the target can rewind all but the first token.
    ///
    /// - Parameter count: The tokens of the verify.
    /// - Returns: `true` when the verify runs in place.
    func verifiesInPlace(tokens count: Int) -> Bool {
        let rewound = count - 1
        guard rewound > 0 else { return true }
        return MTPSpeculativeTokenIterator.verifiesInPlace(
            mainModel: model, mainCache: cache, drafts: rewound)
    }

    /// `state`, asking the target's forward for its drafter state.
    private static func emittingState(from state: LMOutput.State?) -> LMOutput.State {
        var emitting = state ?? LMOutput.State()
        emitting[mtpEmitFlagKey] = true
        return emitting
    }
}
