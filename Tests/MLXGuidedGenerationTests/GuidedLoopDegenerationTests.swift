// Copyright © 2026 Apple Inc.
//
// Regression tests for kanban task `y4s0w2j`: constrained tool-calling
// decode on real hardware degenerated into repeated-token runaways
// (`}7}7}7…`, `10101010…`, digit floods) for GLM-4.7-Flash and the
// Devstral family. Two loop-policy defects combined to cause it:
//
// 1. The soft zone applied the full closing-token bias (+100 on `"`, `}`,
//    `]`, and formerly digits) to every sampled token, and +100 dwarfs any
//    real logit gap — so once the (tiny, `structuralReserve * 3`-sized)
//    normal zone was spent, string *content* was forced into the closing
//    set, corrupting long tool-call arguments instead of closing them.
//    The soft zone must bias EOS only: a +200 boost on a stop token the
//    grammar masks to `-inf` is a no-op mid-string, so content decodes
//    exactly as in the normal zone, while trivial schemas (kanban
//    t3nynaj's Int-overflow case) still flip to EOS the moment stopping
//    becomes legal.
//
// 2. Greedy argmax has no way out of a short repeating cycle when every
//    cycling token stays grammar-legal (inside a JSON string, the boosted
//    `}` never closes anything). `RepetitionCycleTracker` detects the
//    cycle and suppresses its ids so decode must pick a different legal
//    token and can make structural progress again.
//
// Uses the same byte-fallback vocab + probe-model pattern as
// `FastForwardSampledTokenKVCacheTests`, with scripted logits so each
// degenerate preference is deterministic without real model weights.

import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon
import MLXNN
import Testing

/// Minimal byte-level tokenizer: token id == byte value, plus an EOS token
/// at id 255. Matches `GrammarTokenizer`'s byte-fallback vocab below.
private struct DegenerationByteTokenizer: MLXLMCommon.Tokenizer {
    static let eosTokenID = 255
    static let eosTokenString = "</s>"

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map(Int.init)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(
            decoding: tokenIds.filter { $0 != Self.eosTokenID }.map(UInt8.init),
            as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token == Self.eosTokenString { return Self.eosTokenID }
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        if id == Self.eosTokenID { return Self.eosTokenString }
        guard id >= 0, id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
    }

    var bosToken: String? { nil }
    var eosToken: String? { Self.eosTokenString }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

/// `UserInputProcessor` stand-in; `GuidedGenerationLoop.run` never touches
/// `context.processor`.
private struct DegenerationUnusedProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        fatalError("GuidedGenerationLoop.run must not use context.processor")
    }
}

/// Language model whose logits are scripted per forward pass: call `n`
/// (0 = the prompt prefill) returns the preference array
/// `preferences(n)`, expanded to `[1, inputSize, vocabSize]`.
private final class ScriptedPreferenceModel: Module, MLXLMCommon.LanguageModel,
    KVCacheDimensionProvider
{
    static let vocabSize = 256

    var kvHeads: [Int] { [1] }
    private var callCount = 0
    private let preferences: (Int) -> [Float]

    init(preferences: @escaping (Int) -> [Float]) {
        self.preferences = preferences
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let floats = preferences(callCount)
        let logits = MLXArray(floats)
        callCount += 1
        // Same last-position logits at every sequence position; the loop
        // only reads the last one. The logit dimension follows the
        // preference array so tests can model vocabs larger than 256
        // (e.g. an appended turn-ender token).
        return broadcast(logits[.newAxis, .newAxis, 0...], to: [1, inputs.size, floats.count])
    }
}

@Suite
struct GuidedLoopDegenerationTests {

    init() {
        _ = MetalLibraryTestBootstrap.ensureColocatedMetallib
    }

    private func makeGrammarTokenizer() throws -> GrammarTokenizer {
        let vocab: [String] = (0 ..< ScriptedPreferenceModel.vocabSize).map { byte in
            String(format: "<0x%02X>", byte)
        }
        return try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            eosTokenId: Int32(DegenerationByteTokenizer.eosTokenID)
        )
    }

    private func makeContext(model: ScriptedPreferenceModel) -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(id: "guided-loop-degeneration-test"),
            model: model,
            processor: DegenerationUnusedProcessor(),
            tokenizer: DegenerationByteTokenizer()
        )
    }

    /// - Throws: Rethrows setup/generation errors as test failures.
    @Test(
        """
        The soft zone never corrupts string content: closing-tier characters are \
        not boosted over the model's own content preference
        """
    )
    func softZoneDoesNotCorruptStringContent() throws {
        let quote = Int(Character("\"").asciiValue!)
        let closeBrace = Int(Character("}").asciiValue!)
        // Diverse letters so the content itself never looks like a
        // degenerate repetition cycle.
        let letters = Array("abcdefghijklmnop".utf8).map(Int.init)

        // Content phase (first 20 generated tokens): the model prefers a
        // fresh letter each step (its genuine content), with '}' as a
        // runner-up — the exact shape of a code-writing model mid-string.
        // Wrap-up phase: it prefers the closing quote.
        let model = ScriptedPreferenceModel { call in
            var floats = [Float](repeating: 0.0, count: ScriptedPreferenceModel.vocabSize)
            if call <= 20 {
                floats[letters[call % letters.count]] = 10.0
                floats[closeBrace] = 5.0
                floats[quote] = 1.0
            } else {
                floats[quote] = 50.0
                floats[closeBrace] = 5.0
            }
            return floats
        }

        let hostTokenizer = DegenerationByteTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: try makeGrammarTokenizer(),
            jsonSchema: #"{"type": "string"}"#,
            fastForward: true,
            hostTokenizer: hostTokenizer
        )
        let closingBias = ClosingTokenBias.compute(
            tokenizer: hostTokenizer, eosTokenID: DegenerationByteTokenizer.eosTokenID)

        var emittedText = ""
        let maxTokens = 64
        // completionReserve == maxTokens puts EVERY sampled token in the
        // soft zone — the post-ad4c4cf live situation for tool-call
        // envelopes, whose `structuralReserve * 3` normal zone is ~60 of
        // 4096 tokens.
        try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            context: makeContext(model: model),
            constraint: constraint,
            maxTokens: maxTokens,
            vocabSize: ScriptedPreferenceModel.vocabSize,
            completionReserve: maxTokens,
            closingBias: closingBias
        ) { text in
            emittedText += text
            return true
        }

        #expect(
            !emittedText.dropFirst().dropLast().contains("}"),
            """
            soft-zone bias must not override string content with closing \
            characters, got \(emittedText)
            """
        )
        // Call 0's logits are consumed by the grammar-forced opening
        // quote, so the visible content starts at the second letter.
        #expect(
            emittedText.contains("bcdefgh"),
            "the model's own content must decode, got \(emittedText)")
    }

    /// - Throws: Rethrows setup/generation errors as test failures.
    @Test(
        """
        A hard-zone repetition cycle is broken by suppression instead of \
        exhausting the budget as incomplete output
        """
    )
    func hardZoneCycleIsBrokenBySuppression() throws {
        let quote = Int(Character("\"").asciiValue!)
        let closeBrace = Int(Character("}").asciiValue!)
        let letterA = Int(Character("a").asciiValue!)

        // The model always prefers '}' — inside a JSON string that is
        // grammar-legal *content*, so the hard zone's closing-set boost
        // sustains `}}}}}…` forever (the `}7}7…` live signature).
        let model = ScriptedPreferenceModel { _ in
            var floats = [Float](repeating: 0.0, count: ScriptedPreferenceModel.vocabSize)
            floats[closeBrace] = 10.0
            floats[quote] = 5.0
            floats[letterA] = 20.0  // non-closing: suppressed in the hard zone
            return floats
        }

        let hostTokenizer = DegenerationByteTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: try makeGrammarTokenizer(),
            jsonSchema: #"{"type": "string"}"#,
            fastForward: true,
            hostTokenizer: hostTokenizer
        )
        let closingBias = ClosingTokenBias.compute(
            tokenizer: hostTokenizer, eosTokenID: DegenerationByteTokenizer.eosTokenID)

        var emittedText = ""
        let maxTokens = 64
        // hardReserve == maxTokens puts every token in the hard zone.
        let result = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            context: makeContext(model: model),
            constraint: constraint,
            maxTokens: maxTokens,
            vocabSize: ScriptedPreferenceModel.vocabSize,
            completionReserve: maxTokens,
            hardReserve: maxTokens,
            closingBias: closingBias
        ) { text in
            emittedText += text
            return true
        }

        #expect(
            result.tokenCount < maxTokens,
            "the cycle must be broken well before the budget, got \(result.tokenCount)"
        )
        #expect(
            emittedText.hasSuffix("\""),
            "the string must actually close once the cycle is suppressed, got \(emittedText)"
        )
    }

    /// - Throws: Rethrows setup/generation errors as test failures.
    @Test(
        """
        Legitimate short repetition survives: a constant JSON array decodes intact \
        instead of tripping the cycle breaker
        """
    )
    func legitimateConstantArraySurvives() throws {
        let openBracket = Int(Character("[").asciiValue!)
        let closeBracket = Int(Character("]").asciiValue!)
        let comma = Int(Character(",").asciiValue!)
        let digitSeven = Int(Character("7").asciiValue!)

        // A correct, constant 8-element array — `[7,7,7,7,7,7,7,7]` — is a
        // period-2 sampled run of 15 tokens. The cycle breaker must not
        // mistake it for degeneration and corrupt the tail (kanban
        // y4s0w2j adversarial review, finding 1).
        let model = ScriptedPreferenceModel { call in
            var floats = [Float](repeating: 0.0, count: ScriptedPreferenceModel.vocabSize)
            switch call {
            case 0:
                floats[openBracket] = 5.0
            case 1 ... 15:
                floats[call % 2 == 1 ? digitSeven : comma] = 10.0
            default:
                floats[closeBracket] = 10.0
            }
            return floats
        }

        let hostTokenizer = DegenerationByteTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: try makeGrammarTokenizer(),
            jsonSchema: #"{"type": "array", "items": {"type": "integer"}}"#,
            fastForward: true,
            hostTokenizer: hostTokenizer
        )
        let closingBias = ClosingTokenBias.compute(
            tokenizer: hostTokenizer, eosTokenID: DegenerationByteTokenizer.eosTokenID)

        var emittedText = ""
        let maxTokens = 64
        try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            context: makeContext(model: model),
            constraint: constraint,
            maxTokens: maxTokens,
            vocabSize: ScriptedPreferenceModel.vocabSize,
            completionReserve: maxTokens,
            closingBias: closingBias
        ) { text in
            emittedText += text
            return true
        }

        #expect(
            emittedText == "[7,7,7,7,7,7,7,7]",
            "a legitimate constant array must decode intact, got \(emittedText)")
    }

    /// - Throws: Rethrows setup/generation errors as test failures.
    @Test(
        """
        Every registered stop token is grammar-gated: a turn-ender whose bytes are \
        valid string content is never sampled mid-string, so the envelope completes
        """
    )
    func registeredStopTokensAreNeverSampledAsStringContent() throws {
        let quote = Int(Character("\"").asciiValue!)
        let letterA = Int(Character("a").asciiValue!)
        // A GLM-style secondary stop token (`<|user|>`/`<|endoftext|>`):
        // its literal bytes are perfectly valid JSON string *content*, so
        // only registering it as an xgrammar stop token can keep it out of
        // mid-string sampling. This is the live GLM-4.7-Flash failure:
        // an unregistered stop id was sampled mid-`response`, the loop
        // stopped, and the truncated envelope leaked into the reply.
        let turnEnder = 256

        // The model strongly prefers ending its turn mid-string — the
        // constrained format is foreign to it — then falls back to content.
        let model = ScriptedPreferenceModel { call in
            var floats = [Float](repeating: 0.0, count: 257)
            if call <= 5 {
                floats[turnEnder] = 30.0
                floats[letterA] = 10.0
                floats[quote] = 1.0
            } else {
                floats[quote] = 50.0
                floats[letterA] = 10.0
            }
            return floats
        }

        let hostTokenizer = DegenerationByteTokenizer()
        var vocab: [String] = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
        vocab.append("ENDTURN")
        let grammarTokenizer = try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            stopTokenIds: [Int32(DegenerationByteTokenizer.eosTokenID), Int32(turnEnder)]
        )
        let constraint = try GrammarConstraint(
            tokenizer: grammarTokenizer,
            jsonSchema: #"{"type": "string"}"#,
            fastForward: true,
            hostTokenizer: hostTokenizer
        )
        let closingBias = ClosingTokenBias.compute(
            tokenizer: hostTokenizer, eosTokenID: DegenerationByteTokenizer.eosTokenID)

        var emittedText = ""
        let maxTokens = 64
        let context = ModelContext(
            // The turn-ender is a stop token on the host side too — the
            // live case's `configuration.eosTokenIds`.
            configuration: ModelConfiguration(
                id: "guided-loop-degeneration-test", eosTokenIds: [turnEnder]),
            model: model,
            processor: DegenerationUnusedProcessor(),
            tokenizer: hostTokenizer
        )
        // Soft zone from token 0 — the boost must be inert for a
        // grammar-gated stop token outside accept states.
        let result = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            context: context,
            constraint: constraint,
            maxTokens: maxTokens,
            vocabSize: vocab.count,
            completionReserve: maxTokens,
            closingBias: closingBias
        ) { text in
            emittedText += text
            return true
        }

        #expect(
            !emittedText.contains("ENDTURN"),
            "a stop token must never decode as string content, got \(emittedText)")
        #expect(
            emittedText.contains("aaaaa"),
            """
            decode must continue with real content instead of stopping at the \
            unmasked turn-ender, got \(emittedText)
            """
        )
        #expect(
            emittedText.hasSuffix("\""),
            "the string must complete instead of truncating at a stop token, got \(emittedText)"
        )
        #expect(result.tokenCount < maxTokens)
    }

    /// - Throws: Rethrows setup/generation errors as test failures.
    @Test(
        """
        A trivial integer schema still stops via the soft zone's EOS boost \
        the moment stopping is grammar-legal (the t3nynaj guarantee)
        """
    )
    func integerSchemaStillStopsViaEOSBoost() throws {
        let digitSeven = Int(Character("7").asciiValue!)

        // Digit-degenerate model: always prefers '7' (the t3nynaj
        // Int-overflow signature).
        let model = ScriptedPreferenceModel { _ in
            var floats = [Float](repeating: 0.0, count: ScriptedPreferenceModel.vocabSize)
            floats[digitSeven] = 10.0
            return floats
        }

        let hostTokenizer = DegenerationByteTokenizer()
        let constraint = try GrammarConstraint(
            tokenizer: try makeGrammarTokenizer(),
            jsonSchema: #"{"type": "integer"}"#,
            fastForward: true,
            hostTokenizer: hostTokenizer
        )
        let closingBias = ClosingTokenBias.compute(
            tokenizer: hostTokenizer, eosTokenID: DegenerationByteTokenizer.eosTokenID)

        var emittedText = ""
        let maxTokens = 64
        let result = try GuidedGenerationLoop.run(
            input: LMInput(tokens: MLXArray([Int32(0)])),
            context: makeContext(model: model),
            constraint: constraint,
            maxTokens: maxTokens,
            vocabSize: ScriptedPreferenceModel.vocabSize,
            completionReserve: maxTokens,
            closingBias: closingBias
        ) { text in
            emittedText += text
            return true
        }

        #expect(
            result.tokenCount <= 3,
            "the EOS boost must stop a digit-degenerate integer immediately, got \(result.tokenCount) tokens: \(emittedText)"
        )
    }
}
