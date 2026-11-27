// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private let clusterMTPLogger = MLXLogger(label: "cluster-mtp-iterator")

/// Generator of tokens with MTP speculative decoding on a ``ClusterLanguageModel``, whose rank 0
/// holds the drafter.
///
/// Each round is one ``RemoteModelOp/speculate(_:)``, one round trip to the worker: every rank
/// rewinds the drafts the round before rejected, rank 0 drafts after the token the round starts
/// from, and every rank verifies the token and the drafts with one forward. The iterator
/// verifies the drafts in this process, from the logits of every verified position, with the
/// processor and the sampler of the stream: it accepts the longest prefix of drafts that the
/// target tokens match, as ``MTPSpeculativeTokenIterator`` verifies the argmax drafts of a
/// stateful drafter, so a greedy stream decodes the tokens of plain greedy decoding.
///
/// The drafts a round rejects stay in the caches of the worker until the next round rewinds
/// them. When the generation ends, ``finalizeGeneration()`` trims them with the verified tokens
/// the consumer did not take, so the cache then holds the prompt and each emitted token a
/// forward fed, as the token ledger of a prompt cache expects.
package struct ClusterMTPTokenIterator: TokenIteratorProcessing {

    /// The model whose worker runs the rounds.
    private let model: ClusterLanguageModel

    /// The one remote cache of the generation.
    private let remote: RemoteKVCache

    /// The cache and its timeline.
    private let cacheStorage: KVCacheStorage

    /// The logit processor of the stream.
    var processor: LogitProcessor?

    /// The sampler of the stream.
    private let sampler: LogitSampler

    /// The maximum number of tokens to generate, or `nil` for no limit.
    package let maxTokens: Int?

    /// The most drafts of each round: the block size less the round's token.
    private let draftLimit: Int

    /// The drafter top-1 probability below which a chaining drafter stops a round's chain.
    private let minimumDraftProbability: Float

    /// Wall-clock time spent priming the cache with the prompt before the first generated token.
    package var promptPrefillTime: TimeInterval = 0

    /// The token the next round starts from: the last token sampled, which no forward fed.
    private var roundToken: Int32 = 0

    /// The tokens of the last round that the consumer has yet to take, and the next to take.
    private var pendingTokens: [Int] = []
    private var pendingIndex = 0

    /// The pending tokens the cache holds: the drafts the last round accepted.
    private var committedPendingTokenCount = 0

    /// The drafts of the last round that the target rejected, which the cache still holds.
    private var rejectedCount = 0

    /// The drafts the rounds proposed, and those the target accepted.
    private var proposedCount = 0
    private var acceptedCount = 0

    /// Whether a round failed, which ends the generation.
    private var failed = false

    /// Round and token telemetry.
    private var telemetry = SpeculativeDecodingTelemetry()

    /// Whether the cache holds the token the most recent ``next()`` call returned: an accepted
    /// draft, which the verify of its round fed. The last token of a round is returned before
    /// any forward feeds it.
    package private(set) var hasCachedLastToken = false

    /// The cadence that returns the freed buffers of the MLX pool to the system.
    private var bufferPoolClearing = GenerationBufferPoolClearing()

    /// Prefills `input` on the worker and samples the first token.
    ///
    /// - Parameters:
    ///   - input: The prompt, or the suffix of the prompt that the cache does not hold.
    ///   - model: The cluster model.
    ///   - drafter: The drafter of rank 0 of `model`: ``ClusterLanguageModel/mtpDrafter``.
    ///   - cacheStorage: The cache of `model` and its timeline.
    ///   - parameters: The generation parameters.
    ///   - blockSize: The tokens of each round, or `nil` for the drafter's preferred block.
    ///   - minimumDraftProbability: The confidence floor of a chaining drafter, from 0 to 1.
    ///   - components: Optional behavioral components, e.g. a custom ``LogitProcessor``.
    /// - Throws: An ``MTPDrafterCompatibilityError`` when `drafter` is not the drafter of
    ///   `model`, a ``SpeculativeDecodingConfigurationError`` for a block of no draft or a floor
    ///   outside 0 to 1, ``ClusterModelError/foreignCache`` for a cache `model` did not make,
    ///   or the error of the prefill.
    package init(
        input: LMInput, model: ClusterLanguageModel, drafter: RemoteMTPDrafter,
        cacheStorage: KVCacheStorage, parameters: GenerateParameters, blockSize: Int?,
        minimumDraftProbability: Float, components: GenerationComponents = .init()
    ) throws {
        try drafter.validateCompatibility(with: model)
        try SpeculativeDecodingConfig.validate(minimumDraftProbability: minimumDraftProbability)
        let requestedBlockSize = SpeculativeDecodingConfig.mtpBlockSize(blockSize, for: drafter)
        try SpeculativeDecodingConfig.validate(mtpBlockSize: requestedBlockSize)
        let cacheStorage = try cacheStorage.plan.validated(cacheStorage)
        self.model = model
        self.remote = try model.remoteCache(of: cacheStorage.cache)
        self.cacheStorage = cacheStorage
        try components.validate(parameters: parameters)
        self.processor = components.logitProcessor(parameters: parameters, model: model)
        self.sampler = parameters.makeSampler()
        self.maxTokens = parameters.maxTokens
        self.draftLimit =
            Swift.min(requestedBlockSize, drafter.maximumBlockSize ?? requestedBlockSize) - 1
        self.minimumDraftProbability = minimumDraftProbability

        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prepare(input, prefill: parameters.prefill)
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    /// Prefills the prompt, and samples the token the first round starts from.
    private mutating func prepare(_ input: LMInput, prefill: PrefillParameters) throws {
        processor?.prompt(input.text.tokens)
        prefill.prefix?(cacheStorage.processedTokenCount)
        let prepared = try model.prepare(
            input, cache: cacheStorage.cache, state: nil, prefill: prefill)
        cacheStorage.commitPrefill(
            of: input, result: prepared, role: "ClusterLanguageModel.prepare")
        let logits: MLXArray
        switch prepared {
        case .tokens(let rest):
            logits = model.nextTokenLogits(rest, cache: cacheStorage.cache, state: nil).logits
            cacheStorage.commitProcessedTokens(rest.cacheSequenceLength)
            let total = input.text.tokens.size
            prefill.progress?(total, total)
        case .logits(let output):
            logits = output.logits
        }
        try model.checkFailure()
        roundToken = draw(logits[0..., -1, 0...])
        pendingTokens = [Int(roundToken)]
    }

    /// The token count the telemetry recorded.
    package var tokenCount: Int { telemetry.emittedTokenCount }

    /// The round telemetry, or `nil` before the first round that drafted.
    package var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }

    /// The most positions a layer of the cache evicted.
    package var evictedTokenCount: Int {
        KVCacheTree.maxEvictedTokenCount(in: cacheStorage.cache)
    }

    /// Un-counts the most recently emitted token in the telemetry.
    package mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    /// The next token: a token the last round left, or the first of a new round. `nil` at the
    /// token limit, and after a round failed.
    ///
    /// - Returns: The token, or `nil`.
    package mutating func next() -> Int? {
        guard !isAtTokenLimit, !failed else { return nil }
        if pendingIndex == pendingTokens.count {
            autoreleasepool { runRound() }
        }
        guard pendingIndex < pendingTokens.count else { return nil }
        let token = pendingTokens[pendingIndex]
        hasCachedLastToken = pendingIndex < committedPendingTokenCount
        pendingIndex += 1
        bufferPoolClearing.clearPoolIfDue(tokenCount: tokenCount)
        telemetry.recordGeneratedToken()
        return token
    }

    // MARK: - Rounds

    /// The most drafts the next round may propose: as many as the token limit leaves room for
    /// beside the token the round ends with.
    private var roundDraftLimit: Int {
        guard let maxTokens else { return draftLimit }
        return Swift.max(0, Swift.min(draftLimit, maxTokens - tokenCount - 1))
    }

    /// Runs one round on the worker, and fills the pending buffer with the tokens it emits.
    private mutating func runRound() {
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        committedPendingTokenCount = 0
        guard
            let round = model.speculate(
                on: remote, from: roundToken, rejectedCount: rejectedCount,
                draftLimit: roundDraftLimit, minimumDraftProbability: minimumDraftProbability)
        else {
            failed = true
            return
        }
        cacheStorage.commitDroppedTokens(rejectedCount)
        cacheStorage.commitProcessedTokens(round.drafts.count + 1)
        let (tokens, accepted) = verify(drafts: round.drafts, logits: round.logits[0])
        pendingTokens = tokens
        committedPendingTokenCount = accepted
        rejectedCount = round.drafts.count - accepted
        roundToken = Int32(tokens[tokens.count - 1])
        guard !round.drafts.isEmpty else { return }
        proposedCount += round.drafts.count
        acceptedCount += accepted
        telemetry.recordRound(
            drafted: round.drafts.count, accepted: accepted,
            targetVerified: round.drafts.count + 1, draftModelCalls: 1)
    }

    /// Verifies `drafts` against the target logits of each verified position.
    ///
    /// One batched readback serves a processor that keeps no state between rows, with argmax
    /// rows at temperature 0 and sampled rows above it; any other stream verifies row by row,
    /// so its processor learns of the emitted tokens alone.
    ///
    /// - Parameters:
    ///   - drafts: The drafts of the round.
    ///   - logits: The target logits of each verified position, shape `[drafts + 1, vocabulary]`.
    /// - Returns: The emitted tokens, the last of them the token the next round starts from, and
    ///   the drafts the target accepted.
    private mutating func verify(drafts: [Int32], logits: MLXArray) -> (
        tokens: [Int], accepted: Int
    ) {
        let draftTokens = MLXArray(drafts)
        let processorIsStateless = processor == nil || processor is SuppressTokensProcessor
        if processorIsStateless, sampler is ArgMaxSampler || sampler is any DistributionLogitSampler
        {
            let rows = processor?.process(logits: logits) ?? logits
            let result =
                sampler is ArgMaxSampler
                ? batchedGreedyMTPVerification(logits: rows, draftTokens: draftTokens)
                : BatchedMTPVerification.sampled(
                    logits: rows, draftTokens: draftTokens, sampler: sampler)
            return (result.tokens, result.accepted)
        }
        let walk = BatchedMTPVerification.walk(draftTokens: drafts.map(Int.init)) { row in
            MLXArray([draw(logits[row ..< (row + 1)])])
        }
        return (concatenated(walk.emitted).asArray(Int.self), walk.accepted)
    }

    /// Draws a token from the logits of one position: the processor shapes them, the sampler
    /// draws, and the processor learns of the token.
    ///
    /// - Parameter logits: The logits, shape `[1, vocabulary]`.
    /// - Returns: The token.
    private mutating func draw(_ logits: MLXArray) -> Int32 {
        let processed = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: processed)
        processor?.didSample(token: token)
        return token.item(Int32.self)
    }
}

extension ClusterMTPTokenIterator: GenerationFinalizingTokenIterator {

    /// Trims the drafts the last round rejected, and the accepted drafts the consumer did not
    /// take, from the caches of every rank.
    mutating func finalizeGeneration() {
        defer { processor?.finalizeGeneration() }
        let consumed = Swift.min(pendingIndex, committedPendingTokenCount)
        let discarded = rejectedCount + committedPendingTokenCount - consumed
        rejectedCount = 0
        committedPendingTokenCount = consumed
        guard discarded > 0, !failed else { return }
        let trimmed = remote.trim(discarded)
        cacheStorage.commitDroppedTokens(trimmed)
        if trimmed != discarded {
            assertionFailure("The worker trimmed \(trimmed) of \(discarded) speculated tokens")
            clusterMTPLogger.error(
                "The worker trimmed \(trimmed) of \(discarded) speculated tokens; the cache "
                    + "holds positions no token ledger names, and the prompt cache drops it")
        }
    }
}

extension ClusterMTPTokenIterator: GenerationCacheProviding {

    /// The cache of the generation.
    package var generationCache: [KVCache] { cacheStorage.cache }
}

extension ClusterMTPTokenIterator: MTPStatsCollecting {

    /// The drafts the rounds proposed.
    package var proposedDraftTokens: Int { proposedCount }

    /// The drafts the target accepted.
    package var acceptedDraftTokens: Int { acceptedCount }

    /// Nil: every round of the stream speculates, and rank 0 keeps the drafter state beside the
    /// cache, so no continuation is discarded.
    package var passthroughReason: String? { nil }
}

extension MTPSpeculation {

    /// The iterator of an MTP generation of `drafter` on `mainModel`: a
    /// ``ClusterMTPTokenIterator`` for the drafter rank 0 of a cluster model holds, else an
    /// ``MTPSpeculativeTokenIterator``.
    ///
    /// A stream of the cluster drafter that asks for log probabilities decodes plainly, as the
    /// MTP iterator stands down for it.
    ///
    /// - Parameters: As
    ///   ``MTPSpeculativeTokenIterator/init(input:mainModel:drafter:mainCache:mainState:mtpContinuation:parameters:blockSize:minimumDraftProbability:components:)``
    ///   takes them.
    /// - Returns: The iterator, after its prefill.
    /// - Throws: The error of the iterator.
    package static func makeMTPIterator(
        input: LMInput, mainModel: any LanguageModel, drafter: any MTPDrafterModel,
        mainCache: [KVCache]?, mainState: LMOutput.State?,
        mtpContinuation: MTPDrafterContinuation?, parameters: GenerateParameters,
        blockSize: Int?, minimumDraftProbability: Float, components: GenerationComponents
    ) throws -> any TokenIteratorProtocol {
        guard let remoteDrafter = drafter as? RemoteMTPDrafter,
            let cluster = mainModel as? ClusterLanguageModel
        else {
            return try MTPSpeculativeTokenIterator(
                input: input, mainModel: mainModel, drafter: drafter, mainCache: mainCache,
                mainState: mainState, mtpContinuation: mtpContinuation, parameters: parameters,
                blockSize: blockSize, minimumDraftProbability: minimumDraftProbability,
                components: components)
        }
        let storage = KVCacheStorage(
            try mainCache ?? cluster.newCache(parameters: parameters),
            plan: try parameters.kvCachePlan())
        return try makeClusterIterator(
            input: input, model: cluster, drafter: remoteDrafter, cacheStorage: storage,
            parameters: parameters, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability, components: components)
    }

    /// The iterator of an MTP generation of `drafter`, which rank 0 of `model` holds: a
    /// ``ClusterMTPTokenIterator``, or a ``TokenIterator`` for a stream that asks for log
    /// probabilities, whose tokens report the log probabilities of plain decode.
    ///
    /// - Parameters: As ``ClusterMTPTokenIterator/init(input:model:drafter:cacheStorage:parameters:blockSize:minimumDraftProbability:components:)``
    ///   takes them.
    /// - Returns: The iterator, after its prefill.
    /// - Throws: The error of the iterator.
    package static func makeClusterIterator(
        input: LMInput, model: ClusterLanguageModel, drafter: RemoteMTPDrafter,
        cacheStorage: KVCacheStorage, parameters: GenerateParameters, blockSize: Int?,
        minimumDraftProbability: Float, components: GenerationComponents
    ) throws -> any TokenIteratorProtocol {
        guard parameters.logProbabilities == nil else {
            var iterator = try TokenIterator(
                input: input, model: model, cacheStorage: cacheStorage, parameters: parameters,
                components: components)
            iterator.speculationFallback = MTPDrafterFallback(
                reason: .unsupportedSampling,
                statedReason: MTPSpeculativeTokenIterator.logProbabilitiesReason)
            return iterator
        }
        return try ClusterMTPTokenIterator(
            input: input, model: model, drafter: drafter, cacheStorage: cacheStorage,
            parameters: parameters, blockSize: blockSize,
            minimumDraftProbability: minimumDraftProbability, components: components)
    }
}
