// Copyright © 2025 Apple Inc.

import MLXGuidedGeneration
import MLXLMCommon
import Testing

// MARK: - Stub Tokenizer

/// Minimal tokenizer stub: each input character maps to one token.
/// Token count therefore equals string length.
private struct StubTokenizer: MLXLMCommon.Tokenizer {
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

// MARK: - Tests

@Suite
struct CompletionReserveTests {

    private let tokenizer = StubTokenizer()

    @Test("Empty object schema returns token count of '{}'")
    func emptyObjectSchemaTokenCount() {
        let schema = #"{"type":"object"}"#
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        // Minimal JSON for {} object with no required fields => "{}" (2 chars => 2 tokens)
        #expect(reserve == 2)
    }

    @Test("Malformed JSON returns the default reserve")
    func malformedJSONReturnsDefault() {
        let reserve = CompletionReserve.estimate(
            schemaJSON: "not a schema",
            tokenizer: tokenizer,
            defaultReserve: 99
        )
        #expect(reserve == 99)
    }

    @Test("Object with required string property returns expected count")
    func objectWithRequiredStringProperty() {
        let schema =
            #"{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}"#
        // Minimal JSON: {"name":""} (11 chars => 11 tokens)
        let expected = #"{"name":""}"#.utf8.count
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        #expect(reserve == expected)
    }

    @Test("Default reserve falls back to 64 when not provided")
    func defaultReserveDefault() {
        let reserve = CompletionReserve.estimate(schemaJSON: "garbage", tokenizer: tokenizer)
        #expect(reserve == 64)
    }

    @Test("Object with a required const property returns expected count instead of falling back")
    func objectWithRequiredConstProperty() {
        // Mirrors the tool-calling envelope's `{"name": {"const": "..."}}`
        // shape (`SchemaConverter.encodeToolCallingEnvelopeJSON`) -- every
        // `oneOf` alternative names its tool via `const`, not `enum`/`type`.
        let schema = #"{"type":"object","required":["name"],"properties":{"name":{"const":"get_weather"}}}"#
        // Minimal JSON: {"name":"get_weather"} (25 chars => 25 tokens)
        let expected = #"{"name":"get_weather"}"#.utf8.count
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        #expect(reserve == expected)
    }

    @Test("Tool-calling envelope schema (oneOf of const-named alternatives) synthesizes its real minimal JSON instead of falling back to the default reserve")
    func toolCallingEnvelopeSchemaSynthesizesMinimalJSON() {
        // The exact shape `SchemaConverter.encodeToolCallingEnvelopeJSON`
        // produces for a real tool + the synthetic `mlx_final_answer` tool:
        // a `oneOf` of `{name: {const}, arguments: <params>}` objects.
        // Regression (kanban 7f091xq): before the fix, `const` wasn't
        // handled, so `synthesizeMinimalJSON` returned `nil` for every
        // alternative's `name` property, and the WHOLE envelope fell back to
        // the 64-token default reserve regardless of its actual (much
        // smaller) minimal-JSON size.
        let schema = """
            {"oneOf":[{"type":"object","required":["name","arguments"],"additionalProperties":false,"properties":{"name":{"const":"get_weather"},"arguments":{"type":"object","required":["location"],"properties":{"location":{"type":"string","enum":["Tokyo","Paris","New York"]}}}}},{"type":"object","required":["name","arguments"],"additionalProperties":false,"properties":{"name":{"const":"mlx_final_answer"},"arguments":{"type":"object","required":["response"],"properties":{"response":{"type":"string"}}}}}]}
            """
        // `oneOf` synthesizes from the first alternative only.
        let expected = #"{"name":"get_weather","arguments":{"location":"Tokyo"}}"#.utf8.count
        let reserve = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer)
        #expect(reserve == expected)
    }
}
