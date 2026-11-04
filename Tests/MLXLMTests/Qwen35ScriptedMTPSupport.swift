import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension Qwen35MTPFixture {

    /// The bound of a cache that MTP decode built against a cache that one prefill built: both
    /// add in float32, each in its own order.
    static let rebuiltCacheTolerance = AttentionKernelTolerance.float32

    /// Expects `cache` of the target to hold what one prefill of `tokens` builds: the same
    /// offsets, and states within ``rebuiltCacheTolerance``.
    ///
    /// - Parameters:
    ///   - cache: The cache of the target after an MTP decode.
    ///   - tokens: The tokens the cache should hold.
    ///   - comment: What the expectations report when they fail.
    ///   - sourceLocation: The caller of the expectations.
    /// - Throws: The error of the target's cache factory.
    func expectCache(
        _ cache: [KVCache], holdsThePrefillOf tokens: [Int], comment: Comment = "",
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let rebuilt = try target.newCache(parameters: nil)
        eval(
            target(
                LMInput.Text(tokens: MLXArray(tokens.map(Int32.init))[.newAxis]),
                cache: rebuilt, state: nil
            ).logits)
        #expect(
            cache.map(\.offset) == rebuilt.map(\.offset), comment, sourceLocation: sourceLocation)
        for (got, expected) in zip(cache.flatMap(\.state), rebuilt.flatMap(\.state)) {
            let gap = AttentionKernelTolerance.worstGap(got, expected)
            #expect(
                gap <= Self.rebuiltCacheTolerance, "\(comment) worst gap \(gap)",
                sourceLocation: sourceLocation)
        }
    }

    /// An MTP iterator of the fixture's drafter over `input`.
    ///
    /// - Parameters:
    ///   - input: The prompt tokens.
    ///   - blockSize: The tokens of each round: the drafts and the bonus token.
    ///   - parameters: The generation parameters.
    ///   - minimumDraftProbability: The drafter top-1 probability below which a chain stops.
    ///   - cache: The main cache, or `nil` for a new one.
    ///   - mtpContinuation: The drafter continuation the main cache was left with, or `nil`.
    /// - Returns: The iterator.
    /// - Throws: The error of the iterator's validation or prefill.
    func mtpIterator(
        over input: [Int], blockSize: Int, parameters: GenerateParameters,
        minimumDraftProbability: Float = 0, cache: [KVCache]? = nil,
        mtpContinuation: MTPDrafterContinuation? = nil
    ) throws -> MTPSpeculativeTokenIterator {
        try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray(input.map(Int32.init))), mainModel: target,
            drafter: drafter, mainCache: cache, mtpContinuation: mtpContinuation,
            parameters: parameters, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability)
    }

    /// A greedy MTP iterator of the fixture's drafter over `input`.
    ///
    /// - Parameters:
    ///   - input: The prompt tokens.
    ///   - blockSize: The tokens of each round: the drafts and the bonus token.
    ///   - maxTokens: The tokens the iterator generates at most.
    ///   - minimumDraftProbability: The drafter top-1 probability below which a chain stops.
    ///   - cache: The main cache, or `nil` for a new one.
    ///   - mtpContinuation: The drafter continuation the main cache was left with, or `nil`.
    /// - Returns: The iterator.
    /// - Throws: The error of the iterator's validation or prefill.
    func greedyMTPIterator(
        over input: [Int], blockSize: Int, maxTokens: Int, minimumDraftProbability: Float = 0,
        cache: [KVCache]? = nil, mtpContinuation: MTPDrafterContinuation? = nil
    ) throws -> MTPSpeculativeTokenIterator {
        try mtpIterator(
            over: input, blockSize: blockSize,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            minimumDraftProbability: minimumDraftProbability, cache: cache,
            mtpContinuation: mtpContinuation)
    }

    /// The drafter continuation a greedy prefill of `tokens` holds at their end: the reference
    /// a decode that leaves the main cache holding `tokens` must match.
    ///
    /// - Parameters:
    ///   - tokens: The tokens of the prefill.
    ///   - blockSize: The tokens of each round of the prefilled iterator.
    /// - Returns: The continuation, or `nil` when the prefill leaves none.
    /// - Throws: The error of the iterator's validation or prefill.
    func prefillContinuation(of tokens: [Int], blockSize: Int) throws -> MTPDrafterContinuation? {
        try greedyMTPIterator(over: tokens, blockSize: blockSize, maxTokens: 1)
            .promptBoundaryContinuation
    }
}

/// The tokens `cache` holds.
///
/// - Parameter cache: The layers of a main cache.
/// - Returns: The processed token count of the cache.
func heldTokenCount(of cache: [KVCache]) -> Int {
    KVCacheStorage(cache, plan: .disabled).processedTokenCount
}

/// Reads the tokens of `iterator` until it ends or `limit` tokens are read.
///
/// - Parameters:
///   - iterator: A token iterator: a plain or a speculative one.
///   - limit: The tokens to read at most; every token when omitted.
/// - Returns: The tokens read, in order.
func drain<Tokens: IteratorProtocol>(
    _ iterator: inout Tokens, limit: Int = .max
) -> [Int] where Tokens.Element == Int {
    var tokens: [Int] = []
    while tokens.count < limit, let token = iterator.next() {
        tokens.append(token)
    }
    return tokens
}

/// The text and the completion information of one chat turn.
///
/// - Parameter stream: The events of the turn.
/// - Returns: The text the turn wrote and its completion information.
/// - Throws: The error of the turn, or a missing completion information.
func collectChatTurn(
    _ stream: AsyncThrowingStream<Generation, Error>
) async throws -> (text: String, info: GenerateCompletionInfo) {
    var text = ""
    var info: GenerateCompletionInfo?
    for try await event in stream {
        text += event.chunk ?? ""
        info = event.info ?? info
    }
    return (text, try #require(info))
}

/// The drafts of a script: the prompt and the plain greedy decode of the target after it. Each
/// round spoils its drafts at one depth, so that the target rejects every draft from that depth
/// on. The depths of the rounds cycle through each count of accepted drafts, the last of which
/// spoils nothing.
///
/// A round is known by its position, the tokens the target cache holds before it, thus a
/// drafter of the plan keeps no round state of its own. The positions follow from the depths:
/// a round that accepts `d` drafts moves the position by `d + 1`.
struct ScriptedDraftPlan {

    /// The prompt and the plain greedy tokens after it.
    private let script: [Int]

    /// The count of accepted drafts of the round at each position of the plan.
    private let acceptedDraftsAtPosition: [Int: Int]

    /// The tokens of the vocabulary of the target.
    private let vocabularySize: Int

    /// - Parameters:
    ///   - script: The prompt and the plain greedy tokens after it.
    ///   - promptLength: The tokens of the prompt: the position of the first round.
    ///   - blockSize: The tokens of each round: the drafts and the bonus token.
    ///   - vocabularySize: The tokens of the vocabulary of the target.
    init(script: [Int], promptLength: Int, blockSize: Int, vocabularySize: Int) {
        self.script = script
        self.vocabularySize = vocabularySize
        var plan: [Int: Int] = [:]
        var position = promptLength
        var round = 0
        while position < script.count {
            let accepted = round % blockSize
            plan[position] = accepted
            position += accepted + 1
            round += 1
        }
        self.acceptedDraftsAtPosition = plan
    }

    /// The drafts of the round at `queryOffset`.
    ///
    /// - Parameters:
    ///   - queryOffset: The position of the round: the tokens the target cache holds before it.
    ///   - blockSize: The tokens of the round: the drafts and the bonus token.
    ///   - batchSize: The sequences of the round.
    /// - Returns: The drafts, shape `[batchSize, blockSize - 1]`.
    func drafts(queryOffset: Int, blockSize: Int, batchSize: Int) -> MLXArray {
        let draftCount = blockSize - 1
        let spoiledDepth = acceptedDraftsAtPosition[queryOffset] ?? draftCount
        let drafts = (0 ..< draftCount).map { depth in
            let position = queryOffset + 1 + depth
            let token = position < script.count ? script[position] : 0
            return Int32(depth == spoiledDepth ? (token + 1) % vocabularySize : token)
        }
        return MLXArray(drafts, [batchSize, draftCount])
    }
}

/// A stateless drafter that proposes the drafts of a ``ScriptedDraftPlan``.
final class ScriptedRejectionDrafter: Module, MTPDrafterModel {

    /// The drafts the drafter proposes.
    private let plan: ScriptedDraftPlan

    var maximumBlockSize: Int? { nil }
    var requiresSharedTargetKV: Bool { false }
    var requiresGreedySampling: Bool { true }

    /// - Parameter plan: The drafts the drafter proposes.
    init(plan: ScriptedDraftPlan) {
        self.plan = plan
    }

    func validateCompatibility(with target: any LanguageModel) throws {}

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        plan.drafts(queryOffset: queryOffset, blockSize: blockSize, batchSize: lastToken.dim(0))
    }
}

/// A resumable drafter that proposes the drafts of a ``ScriptedDraftPlan`` and keeps the
/// drafter state of a Qwen MTP drafter: its prefill, its commit of each round, and its
/// continuation at the end of a generation are those of `stateModel`, at any block size.
///
/// A chat test sets a new ``plan`` between turns, since the drafts of a round depend on the
/// tokens of the turn that runs it.
final class ScriptedResumableDrafter: Module, ResumableMTPDrafterModel {

    /// The drafts the drafter proposes.
    var plan: ScriptedDraftPlan

    /// The Qwen MTP drafter whose state this drafter keeps.
    private let stateModel: Qwen35MTPDraftModel

    var maximumBlockSize: Int? { nil }
    var requiresSharedTargetKV: Bool { false }
    var requiresPromptPrefill: Bool { true }
    var requiresGreedySampling: Bool { true }

    /// - Parameters:
    ///   - plan: The drafts the drafter proposes.
    ///   - stateModel: The Qwen MTP drafter whose state this drafter keeps.
    init(plan: ScriptedDraftPlan, stateModel: Qwen35MTPDraftModel) {
        self.plan = plan
        self.stateModel = stateModel
    }

    func validateCompatibility(with target: any LanguageModel) throws {
        try stateModel.validateCompatibility(with: target)
    }

    func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        stateModel.makeState(parameters: parameters)
    }

    func appendDrafterState(
        target: any LanguageModel, tokens: MLXArray, targetHidden: MLXArray,
        positionDeltas: MLXArray?, state: inout MTPDrafterState, sampler: any LogitSampler
    ) -> Bool {
        stateModel.appendDrafterState(
            target: target, tokens: tokens, targetHidden: targetHidden,
            positionDeltas: positionDeltas, state: &state, sampler: sampler)
    }

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        plan.drafts(queryOffset: queryOffset, blockSize: blockSize, batchSize: lastToken.dim(0))
    }

    func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)], positionDeltas: MLXArray?, queryOffset: Int,
        blockSize: Int, state: inout MTPDrafterState, sampler: any LogitSampler
    ) -> MLXArray {
        draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            sampler: sampler)
    }
}
