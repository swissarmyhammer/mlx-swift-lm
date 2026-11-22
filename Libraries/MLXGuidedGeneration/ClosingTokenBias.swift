// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon

/// Utility that identifies JSON-closing tokens in a tokenizer's vocabulary
/// and produces a logit bias array.
public enum ClosingTokenBias {

    // MARK: - Constants

    private static let tier1Bias: Float = 200.0
    private static let tier2Bias: Float = 100.0

    /// The tier-2 character that ends a string.
    static let stringClosingCharacter = "\""

    /// The bias of ``stringClosingCharacter``: one above the rest of tier 2, which marks it in
    /// the bias array. The hard zone of `GuidedDecisionRules` reads that mark and prefers the
    /// quote wherever the grammar allows it: inside a string the grammar allows `}`, `]` and the
    /// digits as content too, and a model that keeps writing them never closes the string.
    static let stringClosingBias: Float = tier2Bias + 1

    private static let tier2Characters: Set<String> = [
        "\"", "}", "]",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]

    // MARK: - Public API

    /// Returns an MLXArray of shape [vocabSize]. Closing tokens get a large
    /// positive value (tiered by priority), all others get 0.0.
    ///
    /// Tier 1 (+200): EOS token
    /// Tier 2 (+100): `}`, `]`, single digits `0`-`9`, and `"` one above (+101,
    /// ``stringClosingBias``)
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
                biases[id] = token == stringClosingCharacter ? stringClosingBias : tier2Bias
            }
        }

        // Tier 1 applied last so it overrides tier 2 if EOS overlaps
        if let eos = eosTokenId, eos >= 0, eos < vocabSize {
            biases[eos] = tier1Bias
        }

        return MLXArray(biases)
    }
}
