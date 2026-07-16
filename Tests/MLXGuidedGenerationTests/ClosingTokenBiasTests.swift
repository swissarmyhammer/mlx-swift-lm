// Copyright © 2025 Apple Inc.

import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

// MARK: - Stub Tokenizer

/// Tokenizer with a fixed vocabulary list. Token at index `i` has ID `i`.
private struct ListTokenizer: MLXLMCommon.Tokenizer {
    let tokens: [String]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        self.tokens.firstIndex(of: token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < self.tokens.count else { return nil }
        return self.tokens[id]
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - Tests

@Suite
struct ClosingTokenBiasTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    @Test("Tier-2 closing characters get +100 bias")
    func tier2CharactersGetHundredBias() {
        let tok = ListTokenizer(tokens: [
            "\"",  // 0
            "}",  // 1
            "]",  // 2
            "0",  // 3
            "5",  // 4
            "9",  // 5
            "abc",  // 6 (not closing)
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: nil)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 100.0)  // "
        #expect(values[1] == 100.0)  // }
        #expect(values[2] == 100.0)  // ]
        // Digits must NOT be closing-tier: boosting them sustains digit
        // runaways inside string/number content (kanban y4s0w2j -- the
        // `tripCities202506041234...` / `10101010...` live failures), and
        // they never need a boost to complete a numeric value: when digits
        // are the only grammar-legal class, the hard zone's uniform
        // suppression preserves their relative order anyway.
        #expect(values[3] == 0.0)  // 0
        #expect(values[4] == 0.0)  // 5
        #expect(values[5] == 0.0)  // 9
        #expect(values[6] == 0.0)  // abc
    }

    @Test("eosOnlyBoost boosts exactly the stop-token positions")
    func eosOnlyBoostBoostsStopTokensOnly() {
        let boost = ClosingTokenBias.eosOnlyBoost(stopTokenIDs: [1, 3], count: 5)
        let values = boost.asArray(Float.self)

        #expect(values == [0.0, 200.0, 0.0, 200.0, 0.0])
    }

    @Test("eosOnlyBoost ignores out-of-range stop ids and handles an empty set")
    func eosOnlyBoostIgnoresOutOfRangeAndEmpty() {
        let outOfRange = ClosingTokenBias.eosOnlyBoost(stopTokenIDs: [-1, 7], count: 3)
        #expect(outOfRange.asArray(Float.self) == [0.0, 0.0, 0.0])

        let empty = ClosingTokenBias.eosOnlyBoost(stopTokenIDs: [], count: 2)
        #expect(empty.asArray(Float.self) == [0.0, 0.0])
    }

    @Test("EOS token gets +200 bias overriding any tier-2 setting")
    func eosTokenGetsTwoHundredBiasOverridingTier2() {
        let tok = ListTokenizer(tokens: [
            "}",  // 0 - tier 2
            "<EOS>",  // 1 - EOS
            "abc",  // 2 - none
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: 1)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 100.0)  // tier 2 only
        #expect(values[1] == 200.0)  // EOS
        #expect(values[2] == 0.0)
    }

    @Test("EOS that overlaps with a tier-2 character takes the +200 bias")
    func eosOverlapsTier2() {
        let tok = ListTokenizer(tokens: [
            "\"",  // 0 - tier 2 AND EOS
            "abc",  // 1
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: 0)
        let values = bias.asArray(Float.self)

        // EOS bias overrides tier-2
        #expect(values[0] == 200.0)
        #expect(values[1] == 0.0)
    }

    @Test("Unknown / non-closing tokens receive 0.0 bias")
    func unknownTokensGetZeroBias() {
        let tok = ListTokenizer(tokens: [
            "hello",
            "world",
            "abc",
            "{",  // opening - not in tier 2
            "[",  // opening - not in tier 2
        ])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: nil)
        let values = bias.asArray(Float.self)

        #expect(values == [0.0, 0.0, 0.0, 0.0, 0.0])
    }

    @Test("Vocab size discovery scans until convertIdToToken returns nil")
    func vocabSizeDiscoveryWorks() {
        let tok = ListTokenizer(tokens: ["a", "b", "}", "]", "\""])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: nil)

        // Discovered vocab size should be 5
        #expect(bias.shape == [5])
    }

    @Test("Out-of-range EOS id is ignored")
    func outOfRangeEOSIgnored() {
        let tok = ListTokenizer(tokens: ["a", "}"])
        let bias = ClosingTokenBias.compute(tokenizer: tok, eosTokenID: 999)
        let values = bias.asArray(Float.self)

        #expect(values[0] == 0.0)
        #expect(values[1] == 100.0)  // tier 2 still applies
    }
}
