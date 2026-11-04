// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// The logger of the shifted MTP drafters.
private let qwenMTPLogger = MLXLogger(label: "qwen-mtp")

package func normalizedMTPTokenBatch(_ tokens: MLXArray) -> MLXArray {
    switch tokens.ndim {
    case 1:
        return tokens[.newAxis, 0...]
    default:
        return tokens
    }
}

package func normalizedMTPColumn(_ tokens: MLXArray) -> MLXArray {
    let tokens = normalizedMTPTokenBatch(tokens)
    return tokens.dim(-1) == 1 ? tokens : tokens[0..., (-1)...]
}

extension MTPDraftBlockOutput {

    /// The smallest round a chain of MTP proposals serves: one proposal and the bonus.
    package static let minimumChainedMTPBlockSize = 2

    /// The vocabulary logits of the last position of an MTP predictor's hidden rows, through the
    /// target's LM head, or its tied embedding when it has none.
    private static func mtpLastLogits(
        hidden: MLXArray, targetEmbedTokens: Embedding, lmHead: Linear?
    ) -> MLXArray {
        let logits = lmHead.map { $0(hidden) } ?? targetEmbedTokens.asLinear(hidden)
        return logits[0..., -1, 0...]
    }

    /// The proposal a shifted MTP predictor seeds from the hidden row of its last committed
    /// entry, with the processed logits it was drawn from.
    ///
    /// - Parameters:
    ///   - seedHidden: The predictor's hidden row of its last committed entry, shape `[B, 1, hidden]`.
    ///   - targetEmbedTokens: The target's token embedding, the LM head when `lmHead` is `nil`.
    ///   - lmHead: The target's LM head, or `nil` for tied embeddings.
    ///   - processor: The processor that shapes the logits, or `nil`.
    ///   - sampler: The sampler that draws the proposal.
    /// - Returns: The proposal, shape `[B, 1]`, and its processed logits, shape `[B, 1, vocabulary]`.
    package static func seededMTPDraft(
        seedHidden: MLXArray, targetEmbedTokens: Embedding, lmHead: Linear?,
        processor: (any LogitProcessor)?, sampler: any LogitSampler
    ) -> MTPDraftBlockOutput {
        var logits = mtpLastLogits(
            hidden: seedHidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead)
        logits = processor?.process(logits: logits) ?? logits
        return MTPDraftBlockOutput(
            tokens: normalizedMTPColumn(sampler.sample(logits: logits)),
            processedLogits: logits.expandedDimensions(axis: 1))
    }

    /// The probability the softmax of `logits` gives its most likely token, the lowest over the
    /// batch, as a lazy scalar: `exp(max - logSumExp)`, without the softmax over the vocabulary.
    ///
    /// Reading it waits for the logits, so only a chain with a confidence floor reads it.
    fileprivate static func lowestTopProbability(of logits: MLXArray) -> MLXArray {
        MLX.exp(MLX.min(logits.max(axis: -1) - MLX.logSumExp(logits, axis: -1)))
    }

    /// Whether `sampler` draws without random state, so a draw made and then dropped leaves
    /// every later draw as it was: an argmax, alone or behind a suppressed-token mask.
    private static func drawsWithoutRandomState(_ sampler: any LogitSampler) -> Bool {
        if let suppressing = sampler as? SuppressTokensSampler {
            return drawsWithoutRandomState(suppressing.base)
        }
        return sampler is ArgMaxSampler
    }

    /// Chains up to `blockSize - 1` proposals of a shifted MTP predictor from `lastToken`: each
    /// step feeds the token the step before drew and appends one tentative entry to `cache`.
    ///
    /// Each proposal is drawn with `sampler` from the logits `processor` shaped. The processor
    /// learns of each proposal it shaped the next step for; pass a copy the round discards.
    ///
    /// A step whose processed logits give their most likely token a probability below
    /// `minimumDraftProbability` ends the chain without a proposal, as llama.cpp's
    /// `--spec-draft-p-min` does; the entry that step appended stays tentative with the others.
    ///
    /// - Parameters:
    ///   - targetEmbedTokens: The target's token embedding, the LM head when `lmHead` is `nil`.
    ///   - lmHead: The target's LM head, or `nil` for tied embeddings.
    ///   - inputEmbedding: The embedding of the tokens the predictor reads.
    ///   - lastToken: The token the chain starts from, shape `[B]` or `[B, 1]`.
    ///   - lastHidden: The target hidden row of the position before `lastToken`.
    ///   - queryOffset: The absolute position of `lastToken`.
    ///   - blockSize: The tokens of the round: up to `blockSize - 1` proposals and the bonus.
    ///   - minimumDraftProbability: The top-1 probability below which the chain stops; zero
    ///     drafts every step.
    ///   - processor: The processor that shapes each step's logits, or `nil`.
    ///   - sampler: The sampler that draws each proposal.
    ///   - cache: The predictor's cache.
    ///   - forward: One predictor forward over the embedded input and the previous hidden rows.
    /// - Returns: The proposals, shape `[B, n]`, and their processed logits, shape
    ///   `[B, n, vocabulary]`, with `n` up to `blockSize - 1`, or `nil` when the floor stopped
    ///   the chain before its first proposal.
    package static func chainedMTPDrafts(
        targetEmbedTokens: Embedding,
        lmHead: Linear?,
        inputEmbedding: Embedding,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        queryOffset: Int,
        blockSize: Int,
        minimumDraftProbability: Float,
        processor: (any LogitProcessor)?,
        sampler: any LogitSampler,
        cache: [KVCache],
        forward: ShiftedMTPDrafter.Forward
    ) -> MTPDraftBlockOutput? {
        precondition(
            blockSize >= minimumChainedMTPBlockSize,
            "blockSize must be >= \(minimumChainedMTPBlockSize)")
        precondition(!cache.isEmpty, "Qwen MTP drafter cache must not be empty")

        var token = lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
        var hidden = lastHidden
        var processor = processor
        var tokens: [MLXArray] = []
        var processedLogits: [MLXArray] = []
        tokens.reserveCapacity(blockSize - 1)
        processedLogits.reserveCapacity(blockSize - 1)
        // A chain whose dropped steps leave nothing behind reads each step's confidence one step
        // late: the GPU runs the next step while the host waits for the read, and a step below
        // the floor costs one drafter step more.
        let readsAhead =
            minimumDraftProbability > 0 && processor == nil && drawsWithoutRandomState(sampler)
        var unreadProbability: MLXArray?

        for stepIndex in 0 ..< (blockSize - 1) {
            hidden = forward(inputEmbedding(token), hidden, cache, queryOffset + stepIndex)
            var logits = mtpLastLogits(
                hidden: hidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead)
            logits = processor?.process(logits: logits) ?? logits
            if minimumDraftProbability > 0, !readsAhead,
                lowestTopProbability(of: logits).item(Float.self) < minimumDraftProbability
            {
                break
            }
            let next = sampler.sample(logits: logits)
            if readsAhead {
                let probability = lowestTopProbability(of: logits)
                asyncEval(next, probability)
                if let unread = unreadProbability,
                    unread.item(Float.self) < minimumDraftProbability
                {
                    // The step before fell below the floor: drop its draft, and the entry this
                    // step appended, as a chain that read the floor at that step would stop.
                    tokens.removeLast()
                    processedLogits.removeLast()
                    trimPromptCache(cache, numTokens: 1)
                    unreadProbability = nil
                    break
                }
                unreadProbability = probability
            }
            processor?.didSample(token: next)
            token = next.ndim == 1 ? next.reshaped([next.dim(0), 1]) : next
            tokens.append(token)
            processedLogits.append(logits.expandedDimensions(axis: 1))
        }
        if let unreadProbability, unreadProbability.item(Float.self) < minimumDraftProbability {
            tokens.removeLast()
            processedLogits.removeLast()
        }

        guard !tokens.isEmpty else { return nil }
        return MTPDraftBlockOutput(
            tokens: concatenated(tokens, axis: 1),
            processedLogits: concatenated(processedLogits, axis: 1))
    }
}

/// The parts of a shifted MTP predictor (Qwen NextN) that one round of drafts reads: the
/// target's embedding and LM head, the embedding of the tokens the predictor reads, and one
/// predictor forward.
///
/// A round after a commit starts from the seed proposal the commit left, and chains the
/// predictor on its own hidden rows for the drafts after the seed, as llama.cpp chains its MTP
/// drafts. A round with no seed chains every draft from the round's bonus token.
package struct ShiftedMTPDrafter {

    /// One predictor forward over the embedded input and the previous hidden rows, which appends
    /// one entry to the cache for each input token.
    package typealias Forward = (
        _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
        _ positionOffset: Int
    ) -> MLXArray

    /// The target's token embedding, the LM head when ``lmHead`` is `nil`.
    private let targetEmbedTokens: Embedding

    /// The target's LM head, or `nil` for tied embeddings.
    private let lmHead: Linear?

    /// The embedding of the tokens the predictor reads.
    private let inputEmbedding: Embedding

    /// One predictor forward.
    private let forward: Forward

    /// The longest rewind of a drafter state, in rounds of the most tokens: a finalize removes
    /// the tentative entries of a proposal (fewer than one round), the verified lookahead the
    /// main cache discards (fewer than one round), and the boundary entry.
    private static let longestRewindInRounds = 2

    /// The rows a drafter cache ring keeps beyond its attention window, so that each rewind of
    /// a drafter state stays exact after the ring wraps.
    ///
    /// - Parameter maximumBlockSize: The most tokens of a round of the drafter.
    /// - Returns: The rewind reserve of each drafter cache ring.
    package static func rewindReserve(maximumBlockSize: Int) -> Int {
        longestRewindInRounds * maximumBlockSize
    }

    /// The cache of each predictor layer in a stream of `parameters`.
    ///
    /// The drafter holds the entries of the main cache one position earlier, and attends to the
    /// window ``GenerateParameters/mtpDrafterAttentionWindow`` names:
    ///
    /// - When the main caches rotate (`maxKVSize`, or a typed capacity) or the drafter has a
    ///   window, each drafter layer is a ``RotatingKVCache`` of the narrower of the two windows,
    ///   with its kept prefix (the main one when the windows are equal), and a rewind reserve of
    ///   ``rewindReserve(maximumBlockSize:)`` rows (or the stream's own reserve when larger). The
    ///   drafter then stays bounded, and each rewind of its state stays exact after its ring
    ///   wraps. Its entries keep counting logical positions, as the main cache does.
    /// - Otherwise each layer is a ``KVCacheSimple``. A quantized main cache (`kvBits`, a
    ///   quantizing ``KVCachePlan``) leaves the drafter in full precision: the plan applies to
    ///   the main cache only, and the drafter cache is small next to it.
    ///
    /// - Parameters:
    ///   - layerCount: The predictor layers.
    ///   - parameters: The generation parameters of the stream, or `nil` for the default
    ///     parameters.
    ///   - maximumBlockSize: The most tokens of a round of the drafter.
    /// - Returns: One cache for each predictor layer.
    package static func makeCache(
        layerCount: Int, parameters: GenerateParameters?, maximumBlockSize: Int
    ) -> [KVCache] {
        let parameters = parameters ?? GenerateParameters()
        guard let capacity = drafterCacheCapacity(parameters: parameters) else {
            return (0 ..< layerCount).map { _ in KVCacheSimple() }
        }
        let reserve = Swift.max(
            rewindReserve(maximumBlockSize: maximumBlockSize),
            parameters.kvCache?.rewind?.maxTokens ?? 0)
        return (0 ..< layerCount).map { _ in capacity.makeRotatingCache(rewindCapacity: reserve) }
    }

    /// The window of the drafter cache of a stream of `parameters`: the narrower of the
    /// capacity of the main full-attention caches and the drafter's own attention window, the
    /// main one when the two are equal, or `nil` when neither bounds the drafter.
    ///
    /// - Parameter parameters: The generation parameters of the stream.
    /// - Returns: The capacity, or `nil`.
    private static func drafterCacheCapacity(
        parameters: GenerateParameters
    ) -> KVCacheConfiguration.Capacity? {
        [mainCacheCapacity(parameters: parameters), parameters.mtpDrafterAttentionWindow]
            .compactMap { $0 }
            .min { $0.maxTokens < $1.maxTokens }
    }

    /// The capacity of the main full-attention caches of a stream of `parameters`, or `nil`
    /// when they do not rotate.
    ///
    /// The main caches of the stream are built from the same parameters first, so a request
    /// they refuse never reaches the drafter. Should one reach it, the drafter keeps an unbounded
    /// cache: it then decodes exactly, with more memory.
    ///
    /// - Parameter parameters: The generation parameters of the stream.
    /// - Returns: The capacity, or `nil`.
    private static func mainCacheCapacity(
        parameters: GenerateParameters
    ) -> KVCacheConfiguration.Capacity? {
        do {
            return try parameters.effectiveKVCacheCapacity()
        } catch {
            assertionFailure("MTP drafter cache of invalid KV cache parameters: \(error)")
            qwenMTPLogger.error(
                "The KV cache parameters are invalid (\(error)); the MTP drafter cache is unbounded"
            )
            return nil
        }
    }

    /// - Parameters:
    ///   - targetEmbedTokens: The target's token embedding.
    ///   - lmHead: The target's LM head, or `nil` for tied embeddings.
    ///   - inputEmbedding: The embedding of the tokens the predictor reads.
    ///   - forward: One predictor forward.
    package init(
        targetEmbedTokens: Embedding, lmHead: Linear?, inputEmbedding: Embedding,
        forward: @escaping Forward
    ) {
        self.targetEmbedTokens = targetEmbedTokens
        self.lmHead = lmHead
        self.inputEmbedding = inputEmbedding
        self.forward = forward
    }

    /// The drafts of one round as tokens: the seed the last commit left, then the drafts chained
    /// after it until the state's ``MTPDrafterState/minimumDraftProbability`` stops the chain,
    /// or a chain from `lastToken` when there is no seed.
    ///
    /// - Parameters:
    ///   - lastToken: The bonus token of the round, shape `[B]` or `[B, 1]`.
    ///   - lastHidden: The target hidden row that predicted `lastToken`.
    ///   - queryOffset: The absolute position of `lastToken`.
    ///   - blockSize: The tokens of the round: up to `blockSize - 1` drafts and the bonus.
    ///   - sampler: The sampler of each chained draft.
    ///   - state: The drafter state; it holds one tentative entry for each chained step.
    /// - Returns: The drafts, shape `[B, n]` with `n` from 1 to `blockSize - 1`.
    package func draftTokens(
        lastToken: MLXArray, lastHidden: MLXArray, queryOffset: Int, blockSize: Int,
        sampler: any LogitSampler, state: inout MTPDrafterState
    ) -> MLXArray {
        guard let seed = state.takeSeed() else {
            return unseededChain(
                from: lastToken, hidden: lastHidden, queryOffset: queryOffset,
                blockSize: blockSize, processor: nil, sampler: sampler, state: &state
            ).tokens
        }
        guard
            let chained = chainAfterSeed(
                seed.token, seed: seed, blockSize: blockSize, processor: nil, sampler: sampler,
                state: &state)
        else { return seed.token }
        return concatenated([seed.token, chained.tokens], axis: 1)
    }

    /// The drafts of one round with the processed logits each was drawn from: the seed drawn
    /// again from its hidden row, then the drafts chained after it until the state's
    /// ``MTPDrafterState/minimumDraftProbability`` stops the chain, or a chain from `lastToken`
    /// when there is no seed.
    ///
    /// - Parameters:
    ///   - lastToken: The bonus token of the round, shape `[B]` or `[B, 1]`.
    ///   - lastHidden: The target hidden row that predicted `lastToken`.
    ///   - queryOffset: The absolute position of `lastToken`.
    ///   - blockSize: The tokens of the round: up to `blockSize - 1` drafts and the bonus.
    ///   - processor: The processor that shapes each draft's logits, or `nil`. It learns of each
    ///     draft it shaped the next draft for; pass a copy the round discards.
    ///   - sampler: The sampler of each draft.
    ///   - state: The drafter state; it holds one tentative entry for each chained step.
    /// - Returns: The drafts, shape `[B, n]`, and their processed logits, shape
    ///   `[B, n, vocabulary]`, with `n` from 1 to `blockSize - 1`.
    package func draftsWithLogits(
        lastToken: MLXArray, lastHidden: MLXArray, queryOffset: Int, blockSize: Int,
        processor: (any LogitProcessor)?, sampler: any LogitSampler, state: inout MTPDrafterState
    ) -> MTPDraftBlockOutput {
        guard let seed = state.takeSeed() else {
            return unseededChain(
                from: lastToken, hidden: lastHidden, queryOffset: queryOffset,
                blockSize: blockSize, processor: processor, sampler: sampler, state: &state)
        }
        let first = MTPDraftBlockOutput.seededMTPDraft(
            seedHidden: Self.hidden(of: seed), targetEmbedTokens: targetEmbedTokens,
            lmHead: lmHead, processor: processor, sampler: sampler)
        var processor = processor
        processor?.didSample(token: first.tokens[0..., 0])
        guard
            let chained = chainAfterSeed(
                first.tokens, seed: seed, blockSize: blockSize, processor: processor,
                sampler: sampler, state: &state)
        else { return first }
        return MTPDraftBlockOutput(
            tokens: concatenated([first.tokens, chained.tokens], axis: 1),
            processedLogits: concatenated(
                [first.processedLogits, chained.processedLogits], axis: 1))
    }

    /// The predictor's hidden row a seed proposal was drawn from.
    private static func hidden(of seed: MTPDrafterState.SeedProposal) -> MLXArray {
        guard let hidden = seed.hidden else {
            preconditionFailure("A Qwen MTP seed proposal carries the hidden row it came from")
        }
        return hidden
    }

    /// The drafts chained after the seed of a round of `blockSize` tokens, or `nil` when the
    /// round holds the seed only, the seed falls below the state's floor, or the floor stopped
    /// the chain at its first step.
    ///
    /// The chain is a round of its own that starts from the seed: its first step feeds the seed
    /// with the hidden row the seed was drawn from, at the position of the seed's entry. A seed
    /// below the floor runs no chained step, so its round costs a round of one draft.
    private func chainAfterSeed(
        _ seedToken: MLXArray, seed: MTPDrafterState.SeedProposal, blockSize: Int,
        processor: (any LogitProcessor)?, sampler: any LogitSampler,
        state: inout MTPDrafterState
    ) -> MTPDraftBlockOutput? {
        let chainBlockSize = blockSize - 1
        guard chainBlockSize >= MTPDraftBlockOutput.minimumChainedMTPBlockSize,
            state.chainsAfterSeed(of: seed.topProbability)
        else { return nil }
        return chain(
            from: seedToken, hidden: Self.hidden(of: seed), queryOffset: state.nextPosition,
            blockSize: chainBlockSize, minimumDraftProbability: state.minimumDraftProbability,
            processor: processor, sampler: sampler, state: &state)
    }

    /// Chains every draft of a round from its bonus token, for a state without a seed. The
    /// floor does not apply: the round has no other draft.
    private func unseededChain(
        from token: MLXArray, hidden: MLXArray, queryOffset: Int, blockSize: Int,
        processor: (any LogitProcessor)?, sampler: any LogitSampler,
        state: inout MTPDrafterState
    ) -> MTPDraftBlockOutput {
        guard
            let drafts = chain(
                from: token, hidden: hidden, queryOffset: queryOffset, blockSize: blockSize,
                minimumDraftProbability: 0, processor: processor, sampler: sampler,
                state: &state)
        else {
            preconditionFailure("A chain without a confidence floor drafts every step")
        }
        return drafts
    }

    /// Chains up to `blockSize - 1` drafts from `token` and records the tentative entries its
    /// steps appended.
    private func chain(
        from token: MLXArray, hidden: MLXArray, queryOffset: Int, blockSize: Int,
        minimumDraftProbability: Float, processor: (any LogitProcessor)?,
        sampler: any LogitSampler, state: inout MTPDrafterState
    ) -> MTPDraftBlockOutput? {
        state.appendingTentativeProposal { cache in
            MTPDraftBlockOutput.chainedMTPDrafts(
                targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
                inputEmbedding: inputEmbedding, lastToken: token, lastHidden: hidden,
                queryOffset: queryOffset, blockSize: blockSize,
                minimumDraftProbability: minimumDraftProbability, processor: processor,
                sampler: sampler, cache: cache, forward: forward)
        }
    }
}

extension MTPDrafterState {

    /// A seed proposal the last commit computed: its token, shape `[B, 1]`, the predictor's
    /// hidden row it was drawn from, and the top-1 probability of its logits.
    package typealias SeedProposal = (
        token: MLXArray, hidden: MLXArray?, topProbability: MLXArray?
    )

    /// Takes the seed proposal the last commit computed, or `nil` when there is none.
    ///
    /// The seed holds no tentative entry, so the state holds none after it is taken.
    ///
    /// - Returns: The seed token, shape `[B, 1]`, the predictor's hidden row it was drawn
    ///   from, and its ``seedTopProbability``.
    package mutating func takeSeed() -> SeedProposal? {
        guard let seedToken else { return nil }
        let seed = (token: seedToken, hidden: seedHidden, topProbability: seedTopProbability)
        dropSeed()
        proposalAppended = 0
        return seed
    }

    /// Draws the seed proposal of the next round from the predictor's hidden row of the last
    /// committed entry, and keeps the row and the top-1 probability of the seed's logits.
    ///
    /// The probability stays lazy: only a chained round with a confidence floor reads it.
    ///
    /// - Parameters:
    ///   - hidden: The predictor's hidden row of the last committed entry, shape `[B, 1, hidden]`.
    ///   - targetEmbedTokens: The target's token embedding, the LM head when `lmHead` is `nil`.
    ///   - lmHead: The target's LM head, or `nil` for tied embeddings.
    ///   - sampler: The sampler that draws the seed.
    package mutating func drawSeed(
        hidden: MLXArray, targetEmbedTokens: Embedding, lmHead: Linear?,
        sampler: any LogitSampler
    ) {
        let seed = MTPDraftBlockOutput.seededMTPDraft(
            seedHidden: hidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
            processor: nil, sampler: sampler)
        seedHidden = hidden
        seedToken = seed.tokens
        seedTopProbability = MTPDraftBlockOutput.lowestTopProbability(
            of: seed.processedLogits[0..., 0, 0...])
    }

    /// Whether a chain may draft after a seed of `topProbability` under
    /// ``minimumDraftProbability``: always without a floor or without the seed's probability,
    /// and else when the seed reaches the floor. Reading the probability waits for the seed.
    ///
    /// - Parameter topProbability: The top-1 probability of the seed's logits, or `nil`.
    /// - Returns: `true` when the chain drafts.
    package func chainsAfterSeed(of topProbability: MLXArray?) -> Bool {
        guard minimumDraftProbability > 0, let topProbability else { return true }
        return topProbability.item(Float.self) >= minimumDraftProbability
    }

    /// Runs `propose`, which appends tentative entries to ``cache``, and records the entries it
    /// appended as the entries of the current proposal.
    ///
    /// A chain that a confidence floor stops appends fewer entries than its round has tokens,
    /// so the count is read from the cache rather than from the round.
    ///
    /// - Parameter propose: The chained proposal over ``cache``, which must not be empty.
    /// - Returns: The value of `propose`.
    package mutating func appendingTentativeProposal<Proposal>(
        _ propose: ([KVCache]) -> Proposal
    ) -> Proposal {
        let entriesBefore = cache.first?.offset ?? 0
        let proposal = propose(cache)
        proposalAppended = (cache.first?.offset ?? 0) - entriesBefore
        nextPosition += proposalAppended
        return proposal
    }
}
