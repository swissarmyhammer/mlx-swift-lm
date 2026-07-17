// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// Byte-per-token stub host tokenizer for `CompletionReserve.estimate`,
/// mirroring `CompletionReserveTests.StubTokenizer` (that suite is
/// `MLXGuidedGeneration`-only and file-scoped, so it isn't reachable here).
private struct ToolEnvelopeReserveStubTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        Array(text.utf8).map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(bytes: tokenIds.map { UInt8($0 & 0xFF) }, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        guard let byte = token.utf8.first, token.utf8.count == 1 else { return nil }
        return Int(byte)
    }

    func convertIdToToken(_ id: Int) -> String? {
        guard id >= 0, id < 256 else { return nil }
        return String(UnicodeScalar(UInt8(id)))
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

/// Unit-level regression coverage for `ConstraintSetup.reserves(forMaxTokens:)`
/// against the tool-calling envelope schema (kanban 7f091xq).
///
/// `PromptCacheToolReasoningReuseTests.toolCallingRoundReusesPromptCacheOnRoundTwo`
/// (real-model, `IntegrationTesting`) is the end-to-end reproduction; this
/// suite pins the same failure at the pure-formula level so it can't
/// silently regress again without a real model or an xcodebuild run.
@Suite
struct ToolEnvelopeReserveZoneTests {

    /// The exact shape `SchemaConverter.encodeToolCallingEnvelopeJSON`
    /// produces for a real tool (`get_weather`, closed-enum `location`) paired
    /// with the synthetic `mlx_final_answer` tool (open-ended string
    /// `response`): a `oneOf` of `{name: {const}, arguments: <params>}`
    /// objects. Every alternative names its tool via `const`, never
    /// `enum`/`type`.
    private static let toolEnvelopeJSON = """
        {"oneOf":[{"type":"object","required":["name","arguments"],"additionalProperties":false,"properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","required":["location"],"properties":{"location":{"type":"string","enum":["Tokyo","Paris","New York"]}}}}},{"type":"object","required":["name","arguments"],"additionalProperties":false,"properties":{"name":{"const":"mlx_final_answer"},"arguments":{"type":"object","required":["response"],"properties":{"response":{"type":"string"}}}}}]}
        """

    /// A cheap real `GrammarConstraint` (compiled from a trivial, unrelated
    /// schema -- `reserves(forMaxTokens:)` never touches `constraint` or
    /// `xgTokenizer`) so `ConstraintSetup` can be constructed directly,
    /// mirroring `ToolCallingSchemaTests.makeByteTokenizer()`.
    private func makeByteConstraint() throws -> (GrammarTokenizer, GrammarConstraint) {
        let vocabSize = 256
        let vocab: [String] = (0 ..< vocabSize).map { byte in
            String(format: "<0x%02X>", byte)
        }
        let tokenizer = try GrammarTokenizer(
            vocab: vocab, vocabType: .byteFallback, eosTokenID: Int32(vocabSize - 1))
        let constraint = try GrammarConstraint(
            tokenizer: tokenizer, jsonSchema: #"{"type":"integer"}"#, fastForward: false)
        return (tokenizer, constraint)
    }

    @Test(
        "The tool-envelope schema's hard-closing zone never consumes the entire token budget"
    )
    func toolEnvelopeHardZoneLeavesRealRunway() throws {
        let (tokenizer, constraint) = try makeByteConstraint()
        let structuralReserve = CompletionReserve.estimate(
            schemaJSON: Self.toolEnvelopeJSON, tokenizer: ToolEnvelopeReserveStubTokenizer())
        let setup = ConstraintSetup(
            xgTokenizer: tokenizer, constraint: constraint, maxTokens: 256,
            closingBias: MLXArray([Float(0)]), structuralReserve: structuralReserve,
            whitespaceBias: MLXArray([Float(0)]), whitespaceTokenIDs: []
        )

        let (completionReserve, hardReserve) = setup.reserves(forMaxTokens: 256)

        // Regression pin (kanban 7f091xq): before the fix, `const` wasn't
        // handled by `CompletionReserve`'s minimal-JSON synthesis, so
        // `structuralReserve` silently fell back to the 64-token default for
        // ANY tool-calling envelope (every `oneOf` alternative names its tool
        // via `const`). That made `hardReserve = structuralReserve * 8 = 512`
        // exceed `maxTokens = 256` outright.
        // `GuidedGenerationLoop.applyBiasAndSample`'s hard-zone check
        // (`tokenCount >= maxTokens - hardReserve`) was then true from token
        // 0, forcing the ENTIRE 256-token budget through the hard-closing
        // bias -- crushing legitimate content for the whole generation. (At
        // the time `ClosingTokenBias` also treated `0`-`9` as closing-tier,
        // so a rambling digit string inside an open-ended field
        // (`mlx_final_answer`'s `response`) could run to the full budget
        // instead of ever selecting the closing quote; kanban y4s0w2j
        // removed digits from the tier, but an all-hard-zone budget remains
        // wrong on its own.)
        #expect(
            hardReserve < 256,
            "hard zone must leave real runway, not consume the entire budget")
        #expect(
            completionReserve < 256,
            "soft zone must leave real runway, not consume the entire budget")
        #expect(hardReserve <= 256 / 4, "hard zone must not exceed a quarter of the budget")

        // Not just `completionReserve >= hardReserve` (guaranteed by the
        // `Swift.max` in the formula regardless): the SOFT zone -- the
        // gentle "closing bias, no EOS penalty" phase strictly between
        // `hardReserve` and `completionReserve` -- must have nonzero width.
        // Capping both `hardReserve` and `normalZoneLength` at the SAME
        // `maxTokens / 2` ceiling (an earlier version of this fix) let them
        // collide exactly at that ceiling for schemas like this one, making
        // `completionReserve == hardReserve` and collapsing the soft zone
        // to zero width -- generation would jump straight from the
        // unbiased normal zone into the hard-forced-closing zone with no
        // transition (caught by adversarial review of kanban 7f091xq).
        #expect(
            completionReserve > hardReserve,
            "the soft zone must have nonzero width, not collapse onto the hard zone")

        // `structuralReserve` itself must reflect the schema's real
        // minimal-JSON size, not the 64-token fallback -- the deeper
        // root-cause pin (also covered directly in
        // `MLXGuidedGenerationTests.CompletionReserveTests`).
        let expectedMinimalJSON =
            #"{"name":"get_weather","arguments":{"location":"Tokyo"}}"#.utf8.count
        #expect(structuralReserve == expectedMinimalJSON)
    }

    @Test(
        "A schema whose structural reserve is still inaccurately large cannot swallow the whole budget"
    )
    func hardReserveCapHoldsEvenForAnInflatedStructuralReserve() throws {
        let (tokenizer, constraint) = try makeByteConstraint()
        // Simulates the failure mode directly: an inflated/fallback
        // `structuralReserve` (as `CompletionReserve.estimate` used to
        // produce for ANY schema containing an unrecognized construct)
        // large enough that the old `structuralReserve * 8` formula alone
        // would exceed `maxTokens`.
        let setup = ConstraintSetup(
            xgTokenizer: tokenizer, constraint: constraint, maxTokens: 256,
            closingBias: MLXArray([Float(0)]), structuralReserve: 64,
            whitespaceBias: MLXArray([Float(0)]), whitespaceTokenIDs: []
        )

        let (completionReserve, hardReserve) = setup.reserves(forMaxTokens: 256)

        #expect(hardReserve == 64, "hardReserve must cap at maxTokens / 4, not 64 * 8 = 512")
        #expect(completionReserve >= hardReserve, "completionReserve must never fall below hardReserve")
        #expect(
            completionReserve > hardReserve,
            "the soft zone must have nonzero width, not collapse onto the hard zone")
        #expect(completionReserve < 256)
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
