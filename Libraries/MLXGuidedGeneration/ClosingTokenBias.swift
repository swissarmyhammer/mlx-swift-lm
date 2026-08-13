// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Utility that identifies JSON-closing tokens in a tokenizer's vocabulary
/// and produces a logit bias array.
public enum ClosingTokenBias {

    // MARK: - Constants

    private static let tier1Bias: Float = 200.0
    private static let tier2Bias: Float = 100.0

    private static let tier2Characters: Set<String> = [
        "\"", "}", "]",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    // MARK: - Public API

    /// Returns an MLXArray of shape [vocabSize]. Closing tokens get a large
    /// positive value (tiered by priority), all others get 0.0.
    ///
    /// Tier 1 (+200): EOS token
    /// Tier 2 (+100): `"`, `}`, `]`, single digits `0`-`9`
    public static func compute(tokenizer: any Tokenizer, eosTokenId: Int?) -> MLXArray {
        // Discover vocab size by scanning token IDs
        var vocabSize = 0
        while tokenizer.convertIdToToken(vocabSize) != nil {
            vocabSize += 1
            if vocabSize > 500_000 { break }
        }

        var biases = [Float](repeating: 0.0, count: vocabSize)

        for id in 0 ..< vocabSize {
            if let token = tokenizer.convertIdToToken(id),
                tier2Characters.contains(token)
            {
                biases[id] = tier2Bias
            }
        }

        // Tier 1 applied last so it overrides tier 2 if EOS overlaps
        if let eos = eosTokenId, eos >= 0, eos < vocabSize {
            biases[eos] = tier1Bias
        }

        return MLXArray(biases)
    }

    /// Returns an MLXArray of shape [count] with the tier-1 (+200) boost at
    /// each stop-token position and 0.0 everywhere else.
    ///
    /// This is the *soft-zone* bias: near the token budget the model should
    /// be nudged to stop as soon as the grammar makes stopping legal, but
    /// its content must never be distorted -- a boost on a stop token that
    /// the grammar mask holds at `-inf` is a no-op, so mid-string/mid-value
    /// content decodes exactly as in the normal zone. Boosting content
    /// characters instead (the old soft-zone behavior of applying the full
    /// ``compute(tokenizer:eosTokenId:)`` array) is what corrupted long
    /// tool-call arguments into `}7}7…`/digit runaways (kanban y4s0w2j).
    ///
    /// - Parameters:
    ///   - stopTokenIDs: Token ids that terminate generation (see
    ///     `GuidedGenerationLoop.buildStopTokenIDs`). Out-of-range ids are
    ///     ignored.
    ///   - count: The returned array's length (the closing-bias length, so
    ///     the two zone arrays stay interchangeable downstream).
    /// - Returns: The additive soft-zone bias array.
    public static func eosOnlyBoost(stopTokenIDs: Set<Int>, count: Int) -> MLXArray {
        var boost = [Float](repeating: 0.0, count: count)
        for id in stopTokenIDs where id >= 0 && id < count {
            boost[id] = tier1Bias
        }
        return MLXArray(boost)
    }
}
