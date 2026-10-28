// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

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

package func sampleMTPSeed(
    hidden: MLXArray,
    targetEmbedTokens: Embedding,
    lmHead: Linear?,
    sampler: any LogitSampler
) -> MLXArray {
    MTPDraftBlockOutput.seededMTPDraft(
        seedHidden: hidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead,
        processor: nil, sampler: sampler
    ).tokens
}

extension MTPDraftBlockOutput {

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

    /// Chains `blockSize - 1` proposals of a shifted MTP predictor from `lastToken`: each step
    /// feeds the token the step before drew and appends one tentative entry to `cache`.
    ///
    /// Each proposal is drawn with `sampler` from the logits `processor` shaped. The processor
    /// learns of each proposal it shaped the next step for; pass a copy the round discards.
    ///
    /// - Parameters:
    ///   - targetEmbedTokens: The target's token embedding, the LM head when `lmHead` is `nil`.
    ///   - lmHead: The target's LM head, or `nil` for tied embeddings.
    ///   - inputEmbedding: The embedding of the tokens the predictor reads.
    ///   - lastToken: The token the chain starts from, shape `[B]` or `[B, 1]`.
    ///   - lastHidden: The target hidden row of the position before `lastToken`.
    ///   - queryOffset: The absolute position of `lastToken`.
    ///   - blockSize: The tokens of the round: `blockSize - 1` proposals and the bonus.
    ///   - processor: The processor that shapes each step's logits, or `nil`.
    ///   - sampler: The sampler that draws each proposal.
    ///   - cache: The predictor's cache.
    ///   - forward: One predictor forward over the embedded input and the previous hidden rows.
    /// - Returns: The proposals, shape `[B, blockSize - 1]`, and their processed logits, shape
    ///   `[B, blockSize - 1, vocabulary]`.
    package static func chainedMTPDrafts(
        targetEmbedTokens: Embedding,
        lmHead: Linear?,
        inputEmbedding: Embedding,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        queryOffset: Int,
        blockSize: Int,
        processor: (any LogitProcessor)?,
        sampler: any LogitSampler,
        cache: [KVCache],
        forward: (
            _ inputsEmbeds: MLXArray, _ hiddenStates: MLXArray, _ cache: [KVCache],
            _ positionOffset: Int
        ) -> MLXArray
    ) -> MTPDraftBlockOutput {
        precondition(blockSize >= 2, "blockSize must be >= 2")
        precondition(!cache.isEmpty, "Qwen MTP drafter cache must not be empty")

        var token = lastToken.ndim == 1 ? lastToken.reshaped([lastToken.dim(0), 1]) : lastToken
        var hidden = lastHidden
        var processor = processor
        var tokens: [MLXArray] = []
        var processedLogits: [MLXArray] = []
        tokens.reserveCapacity(blockSize - 1)
        processedLogits.reserveCapacity(blockSize - 1)

        for stepIndex in 0 ..< (blockSize - 1) {
            hidden = forward(inputEmbedding(token), hidden, cache, queryOffset + stepIndex)
            var logits = mtpLastLogits(
                hidden: hidden, targetEmbedTokens: targetEmbedTokens, lmHead: lmHead)
            logits = processor?.process(logits: logits) ?? logits
            let next = sampler.sample(logits: logits)
            processor?.didSample(token: next)
            token = next.ndim == 1 ? next.reshaped([next.dim(0), 1]) : next
            tokens.append(token)
            processedLogits.append(logits.expandedDimensions(axis: 1))
        }

        return MTPDraftBlockOutput(
            tokens: concatenated(tokens, axis: 1),
            processedLogits: concatenated(processedLogits, axis: 1))
    }
}

extension MTPDrafterState {

    /// Takes the seed proposal the last commit computed, or `nil` when there is none.
    ///
    /// The seed holds no tentative entry, so the state holds none after it is taken.
    ///
    /// - Returns: The seed token, shape `[B, 1]`, and the predictor's hidden row it was drawn
    ///   from.
    package mutating func takeSeed() -> (token: MLXArray, hidden: MLXArray?)? {
        guard let seedToken else { return nil }
        let seed = (token: seedToken, hidden: seedHidden)
        self.seedToken = nil
        seedHidden = nil
        proposalAppended = 0
        return seed
    }

    /// Runs `propose`, which appends `blockSize - 1` tentative entries to ``cache``, and
    /// records them as the entries of the current proposal.
    ///
    /// - Parameters:
    ///   - blockSize: The tokens of the round: `blockSize - 1` proposals and the bonus.
    ///   - propose: The chained proposal over ``cache``.
    /// - Returns: The value of `propose`.
    package mutating func appendingTentativeProposal<Proposal>(
        blockSize: Int, _ propose: ([KVCache]) -> Proposal
    ) -> Proposal {
        proposalAppended = blockSize - 1
        let proposal = propose(cache)
        nextPosition += proposalAppended
        return proposal
    }
}
