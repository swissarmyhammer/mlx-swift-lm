// Copyright © 2026 Apple Inc.

import MLX

/// The grammar state one guided decision reads: the matcher's mask there, its additive array, the
/// tokens the generation counted before it, the run of whitespace tokens before it, and whether
/// the output before it stands inside an open JSON string.
///
/// Plain decoding computes one position after each decision. An MTP round computes the positions
/// of its drafts ahead of the target, and its verify reads the same positions.
struct GuidedPosition {

    /// The mask of the matcher at this position.
    let mask: MaskResult

    /// The additive array of ``mask``, or `nil` when the grammar forces every token.
    let maskArray: MLXArray?

    /// The tokens the generation counted before this position, sampled and fast-forwarded.
    let tokenCount: Int

    /// The run of whitespace tokens the generation sampled before this position.
    let whitespace: WhitespaceRunTracker

    /// The string state of the output before this position, its sampled and fast-forwarded
    /// tokens alike.
    let stringState: JSONStringState

    /// Whether `mask` is the mask of this position.
    ///
    /// - Parameter mask: A mask of the matcher.
    /// - Returns: `true` when both masks allow the same tokens and end the same way.
    func hasMask(_ mask: MaskResult) -> Bool {
        self.mask.mask == mask.mask && self.mask.needsApply == mask.needsApply
            && self.mask.isTerminated == mask.isTerminated
    }
}

/// The fixed rules of one guided generation: its grammar, its token budget with the bias of each
/// zone of that budget, and the tokens that stop it.
///
/// The rules decide the logits of every position. Plain decoding and the drafts and the verify of
/// an MTP round all read them, so each reaches the same decision at the same position.
struct GuidedDecisionRules {

    /// The bias that takes a token out of the running: the hard zone writes it on every
    /// non-closing token, and the stop-token penalty on every stop token.
    private static let suppressingBias: Float32 = -10000

    /// The bias the hard zone adds to the string-closing tokens
    /// (``JSONStringTokens/stringClosingTokenIDs``) while a string is open
    /// (``JSONStringState/closesWithQuote``). Inside a string the grammar allows `}`, `]` and the
    /// digits as content, and a model that prefers them there writes `666…` or `}266}266…` to the
    /// end of the budget without closing the string. Outside a string a `"` opens one, thus it
    /// gets no preference there: after `[` or `{`, `]` and `}` keep their tier. Measured on Gemma
    /// with `HardReserveStressTests` before the floor of the hard reserve rose to three times the
    /// shortest valid output: without the preference, budgets of 4 and 8 times that output ended
    /// in incomplete output at floors of 1 and 2 times it; with it, every pass at 8 and 16 times
    /// completed.
    static let hardZoneStringClosingBias: Float32 = 100

    /// The grammar constraint, whose matcher every decision advances.
    let constraint: GrammarConstraint

    /// The tokens the generation may count.
    let maxTokens: Int

    /// The tokens of the grammar's vocabulary: the valid bits of each mask.
    let vocabSize: Int

    /// The width of the model's logits.
    let logitDim: Int

    /// The tokens that stop the generation where the grammar allows them.
    let stopTokenIDs: Set<Int>

    /// The tokenizer's unknown token, which stops the generation too.
    let unknownTokenID: Int?

    /// The tokens before the budget at which the closing bias nudges the generation to close.
    let completionReserve: Int

    /// The tokens before the budget at which only closing tokens remain; 0 turns the zone off.
    let hardReserve: Int

    /// The bias that favors closing tokens, or `nil` for no budget zones.
    let closingBias: MLXArray?

    /// The bias of the hard zone: closing tokens only, stop tokens penalized; nil without a hard
    /// zone or a closing bias.
    private let hardZoneBias: MLXArray?

    /// The bias of the hard zone inside an open string: ``hardZoneBias`` with the string-closing
    /// tokens preferred.
    private let hardZoneInStringBias: MLXArray?

    /// The tokens that move the string state, and the tokens that close a string.
    let stringTokens: JSONStringTokens

    /// The bias against whitespace tokens once a whitespace run grows, or `nil` for none.
    let whitespaceBias: MLXArray?

    /// The tokens the whitespace run counts.
    let whitespaceTokenIDs: Set<Int>

    /// Creates the rules of one generation.
    ///
    /// - Parameters:
    ///   - constraint: The grammar constraint.
    ///   - maxTokens: The tokens the generation may count.
    ///   - vocabSize: The tokens of the grammar's vocabulary.
    ///   - logitDim: The width of the model's logits.
    ///   - stopTokenIDs: The tokens that stop the generation.
    ///   - unknownTokenID: The tokenizer's unknown token.
    ///   - completionReserve: The start of the soft closing zone, in tokens before the budget.
    ///   - hardReserve: The start of the hard closing zone, in tokens before the budget.
    ///   - closingBias: The bias that favors closing tokens, or `nil`.
    ///   - whitespaceBias: The bias against whitespace tokens, or `nil`.
    ///   - whitespaceTokenIDs: The tokens the whitespace run counts.
    ///   - stringTokens: The tokens that move the string state, and those that close a string.
    init(
        constraint: GrammarConstraint, maxTokens: Int, vocabSize: Int, logitDim: Int,
        stopTokenIDs: Set<Int>, unknownTokenID: Int?, completionReserve: Int, hardReserve: Int,
        closingBias: MLXArray?, whitespaceBias: MLXArray?, whitespaceTokenIDs: Set<Int>,
        stringTokens: JSONStringTokens
    ) {
        self.constraint = constraint
        self.maxTokens = maxTokens
        self.vocabSize = vocabSize
        self.logitDim = logitDim
        self.stopTokenIDs = stopTokenIDs
        self.unknownTokenID = unknownTokenID
        self.completionReserve = completionReserve
        self.hardReserve = hardReserve
        self.closingBias = closingBias
        // The hard-zone biases serve a generation with a hard zone only.
        let hardZoneBias =
            hardReserve > 0
            ? closingBias.map {
                Self.hardZoneBias(closingBias: $0, stopTokenIDs: stopTokenIDs)
            } : nil
        self.hardZoneBias = hardZoneBias
        self.hardZoneInStringBias = hardZoneBias.map {
            $0
                + Self.stringClosingPreference(
                    length: $0.shape[0], stringClosingTokenIDs: stringTokens.stringClosingTokenIDs)
        }
        self.stringTokens = stringTokens
        self.whitespaceBias = whitespaceBias
        self.whitespaceTokenIDs = whitespaceTokenIDs
    }

    /// The hard-zone bias of `closingBias`: every non-closing token suppressed, and every stop
    /// token penalized, because the grammar may allow a stop at an intermediate valid state
    /// before every required field is present.
    private static func hardZoneBias(closingBias: MLXArray, stopTokenIDs: Set<Int>) -> MLXArray {
        let biasLength = closingBias.shape[0]
        var penalty = [Float32](repeating: 0, count: biasLength)
        for stopToken in stopTokenIDs where stopToken >= 0 && stopToken < biasLength {
            penalty[stopToken] = suppressingBias
        }
        return which(closingBias .> 0, Float32(0), suppressingBias) + MLXArray(penalty)
    }

    /// The preference the hard zone adds inside an open string: ``hardZoneStringClosingBias`` on
    /// each string-closing token, 0 elsewhere.
    ///
    /// - Parameters:
    ///   - length: The length of the bias.
    ///   - stringClosingTokenIDs: The tokens that close a string.
    /// - Returns: The preference.
    private static func stringClosingPreference(length: Int, stringClosingTokenIDs: Set<Int>)
        -> MLXArray
    {
        var preference = [Float32](repeating: 0, count: length)
        for token in stringClosingTokenIDs where token >= 0 && token < length {
            preference[token] = hardZoneStringClosingBias
        }
        return MLXArray(preference)
    }

    /// The position of the first decision: the matcher's mask before any token.
    ///
    /// - Returns: The position.
    /// - Throws: The error of the matcher.
    func startPosition() throws -> GuidedPosition {
        try position(
            tokenCount: 0, whitespace: WhitespaceRunTracker(whitespaceTokenIDs: whitespaceTokenIDs),
            stringState: JSONStringState())
    }

    /// The position the matcher stands at now.
    ///
    /// - Parameters:
    ///   - tokenCount: The tokens the generation counted before the position.
    ///   - whitespace: The whitespace run before the position.
    ///   - stringState: The string state of the output before the position.
    /// - Returns: The position.
    /// - Throws: The error of the matcher.
    func position(
        tokenCount: Int, whitespace: WhitespaceRunTracker, stringState: JSONStringState
    ) throws -> GuidedPosition {
        let mask = try constraint.computeMask()
        return GuidedPosition(
            mask: mask,
            maskArray: GuidedGenerationLoop.buildMaskArray(
                for: mask, vocabSize: vocabSize, logitDim: logitDim),
            tokenCount: tokenCount, whitespace: whitespace, stringState: stringState)
    }

    /// The bias of a decision at `position`, by the zone of the budget it falls in.
    ///
    /// The normal zone has no bias: the grammar mask already gates a stop on structural
    /// validity. The soft zone (``completionReserve`` tokens left) favors closing tokens, with no
    /// stop penalty, so the generation may stop wherever the grammar allows it. The hard zone
    /// (``hardReserve`` tokens left) suppresses every non-closing token and every stop token, and
    /// inside an open string prefers the tokens that close it (``hardZoneStringClosingBias``).
    /// The whitespace bias joins once the whitespace run grows.
    ///
    /// No bias applies where the grammar forces every token (`needsApply == false`): a bias
    /// without a mask could pick a stop token before the grammar accepts the output.
    ///
    /// - Parameter position: The position of the decision.
    /// - Returns: The bias, or `nil` for none.
    func bias(at position: GuidedPosition) -> MLXArray? {
        guard position.mask.needsApply else { return nil }
        var active: MLXArray?
        if let closingBias {
            if let hardZoneBias, position.tokenCount >= maxTokens - hardReserve {
                active =
                    position.stringState.closesWithQuote ? hardZoneInStringBias : hardZoneBias
            } else if position.tokenCount >= maxTokens - completionReserve {
                active = closingBias
            }
        }
        if let whitespaceBias, position.whitespace.isActive {
            active = active.map { $0 + whitespaceBias } ?? whitespaceBias
        }
        return active
    }

    /// `logits` with the mask and the bias of `position` added.
    ///
    /// - Parameters:
    ///   - logits: Logits of one or more rows, shape `[..., logitDim]`.
    ///   - position: The position of the decision of each row.
    /// - Returns: The constrained logits.
    func constrainedLogits(_ logits: MLXArray, at position: GuidedPosition) -> MLXArray {
        GuidedGenerationLoop.constrainedLogits(
            logits, maskArray: position.maskArray, bias: bias(at: position))
    }

    /// Whether `token`, decided at `position`, stops the generation: a stop token or the unknown
    /// token where the grammar exposed a mask.
    ///
    /// Where the grammar forces every token the decided value is irrelevant: the commit surfaces
    /// the forced tokens, and a stop there would end output the grammar has not accepted.
    ///
    /// - Parameters:
    ///   - token: The decided token.
    ///   - position: The position of the decision.
    /// - Returns: `true` when the generation stops at `token`.
    func endsGeneration(_ token: Int, at position: GuidedPosition) -> Bool {
        position.mask.needsApply && (token == unknownTokenID || stopTokenIDs.contains(token))
    }

    /// The whitespace run and the string state after a committed `token` and the tokens the
    /// grammar fast-forwarded after it.
    ///
    /// The whitespace run counts the sampled `token` only: the grammar forced the rest. The
    /// string state reads every committed token, forced or not.
    ///
    /// - Parameters:
    ///   - token: The committed token.
    ///   - forced: The tokens the grammar fast-forwarded after it.
    ///   - position: The position `token` was decided at.
    /// - Returns: The whitespace run and the string state after the forced tokens.
    func after(token: Int, forced: [Int32], from position: GuidedPosition) -> (
        whitespace: WhitespaceRunTracker, stringState: JSONStringState
    ) {
        var whitespace = position.whitespace
        if whitespaceBias != nil {
            _ = whitespace.record(tokenID: token)
        }
        return (
            whitespace,
            stringTokens.state(after: token, forced: forced, from: position.stringState)
        )
    }
}
