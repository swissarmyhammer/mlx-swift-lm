// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Testing
import Foundation
import MLXGuidedGeneration
import MLXLMCommon
import FoundationModels
@testable import MLXFoundationModels

/// Schemas for fake developer-defined tools used across these tests.

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct WeatherArgs {
    @Guide(description: "City and state, e.g. 'San Francisco, CA'.")
    var location: String
}

@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
@Generable
private struct AddArgs {
    @Guide(description: "First addend.")
    var a: Int
    @Guide(description: "Second addend.")
    var b: Int
}

/// Unit tests for the tool-calling schema and grammar builders.
///
/// Covers both:
/// - `SchemaConverter.encodeToolCallingEnvelopeJSON(tools:)` - the inner
///   `{oneOf: [{name, arguments}, ...]}` JSON envelope, which must compile
///   cleanly with xgrammar's JSON-schema constructor and is also fed to
///   `CompletionReserve` as the structural-reserve seed.
/// - `SchemaConverter.encodeToolCallingGrammar(tools:)` - the xgrammar
///   structural-tag JSON envelope of the form
///   `{type: "structural_tag", format: {type: "or", elements: [tag(...,
///   json_schema), json_schema]}}`. The wrapped arm dispatches Qwen-style
///   `<tool_call>...</tool_call>` delimiters; the bare arm accepts the
///   raw envelope. Shape-only assertions here; real-tokenizer compilation
///   is exercised by the integration suite (the byte-tokenizer used in
///   these unit tests doesn't define Qwen's `<tool_call>` special tokens).
@Suite
struct ToolCallingSchemaTests {

    // MARK: - Envelope Structure

    @Test
    func emptyToolListThrows() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(throws: SchemaConverter.SchemaConversionError.noTools) {
            _ = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [])
        }
    }

    @Test
    func singleToolProducesOneOfWithSingleEntry() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: [weather])
        let parsed = try parseAsDictionary(json)

        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        #expect(oneOf.count == 1)

        let entry = oneOf[0]
        #expect(entry["type"] as? String == "object")
        #expect(entry["additionalProperties"] as? Bool == false)

        let properties = try #require(entry["properties"] as? [String: Any])
        let nameSchema = try #require(properties["name"] as? [String: Any])
        #expect(nameSchema["const"] as? String == "get_weather")
        #expect(properties["arguments"] != nil, "arguments schema must be nested verbatim")
    }

    @Test
    func multipleToolsProduceOneEntryPerTool() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let tools = [
            Transcript.ToolDefinition(
                name: "get_weather",
                description: "Get weather",
                parameters: WeatherArgs.generationSchema
            ),
            Transcript.ToolDefinition(
                name: "add",
                description: "Add two numbers",
                parameters: AddArgs.generationSchema
            ),
        ]

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: tools)
        let parsed = try parseAsDictionary(json)
        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        #expect(oneOf.count == 2)

        // Names preserved and in order supplied.
        let names: [String] = oneOf.compactMap { entry in
            (entry["properties"] as? [String: Any])
                .flatMap { $0["name"] as? [String: Any] }
                .flatMap { $0["const"] as? String }
        }
        #expect(names == ["get_weather", "add"])
    }

    @Test
    func finalAnswerToolFitsInEnvelope() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let finalAnswer = FinalAnswerTool.makeToolDefinition(responseSchema: nil)
        let tools = [
            Transcript.ToolDefinition(
                name: "get_weather",
                description: "Get weather",
                parameters: WeatherArgs.generationSchema
            ),
            finalAnswer,
        ]

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(tools: tools)
        let parsed = try parseAsDictionary(json)
        let oneOf = try #require(parsed["oneOf"] as? [[String: Any]])
        #expect(oneOf.count == 2)

        let names: [String] = oneOf.compactMap { entry in
            (entry["properties"] as? [String: Any])
                .flatMap { $0["name"] as? [String: Any] }
                .flatMap { $0["const"] as? String }
        }
        #expect(names.contains(FinalAnswerTool.toolName))
    }

    // MARK: - Grammar Compilation

    @Test
    func envelopeCompilesWithXGrammar() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let finalAnswer = FinalAnswerTool.makeToolDefinition(responseSchema: nil)
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get weather",
            parameters: WeatherArgs.generationSchema
        )

        let json = try SchemaConverter.encodeToolCallingEnvelopeJSON(
            tools: [weather, finalAnswer]
        )

        // Build a minimal byte-fallback tokenizer and attempt to compile the
        // envelope as a grammar.
        let tokenizer = try makeByteTokenizer()
        _ = try GrammarConstraint(tokenizer: tokenizer, jsonSchema: json, fastForward: false)
    }

    // MARK: - Grammar Builder

    @Test
    func grammarBuilderRejectsEmptyToolList() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        #expect(throws: SchemaConverter.SchemaConversionError.noTools) {
            _ = try SchemaConverter.encodeToolCallingGrammar(tools: [])
        }
    }

    @Test
    func grammarExposesWrappedAndBareAlternatives() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )

        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [weather])
        let parsed = try parseAsDictionary(grammar)

        #expect(parsed["type"] as? String == "structural_tag")

        let format = try #require(parsed["format"] as? [String: Any])
        #expect(format["type"] as? String == "or")

        let elements = try #require(format["elements"] as? [[String: Any]])
        #expect(elements.count == 2)

        // Wrapped arm: tag(<tool_call>\n ... \n</tool_call>) embedding the envelope.
        let wrapped = elements[0]
        #expect(wrapped["type"] as? String == "tag")
        #expect(wrapped["begin"] as? String == "<tool_call>\n")
        #expect(wrapped["end"] as? [String] == ["\n</tool_call>"])

        let wrappedContent = try #require(wrapped["content"] as? [String: Any])
        #expect(wrappedContent["type"] as? String == "json_schema")
        #expect(
            wrappedContent["json_schema"] != nil, "wrapped arm must embed an envelope schema")

        // Bare arm: json_schema embedding the same envelope.
        let bare = elements[1]
        #expect(bare["type"] as? String == "json_schema")
        #expect(bare["json_schema"] != nil, "bare arm must embed an envelope schema")
    }

    @Test
    func grammarEmbedsValidEnvelopeJSON() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [weather])
        let parsed = try parseAsDictionary(grammar)

        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        try #require(elements.count == 2)

        // Wrapped arm: drill content.json_schema and assert envelope shape.
        let wrappedSchema = try #require(
            (elements[0]["content"] as? [String: Any])?["json_schema"] as? [String: Any]
        )
        let wrappedOneOf = try #require(wrappedSchema["oneOf"] as? [[String: Any]])
        #expect(wrappedOneOf.count == 1, "single tool produces a single envelope entry")

        // Bare arm: drill json_schema and assert the same envelope shape.
        let bareSchema = try #require(elements[1]["json_schema"] as? [String: Any])
        let bareOneOf = try #require(bareSchema["oneOf"] as? [[String: Any]])
        #expect(bareOneOf.count == 1, "single tool produces a single envelope entry")
    }

    // MARK: - Format-Driven Structural Tag

    /// The seam: each `ToolCallFormat` maps to its own wrapper spec, and
    /// every format without a bespoke wrapper falls back to Qwen's so
    /// nothing regresses.
    @Test
    func structuralTagSeamSelectsWrapperPerFormat() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // No inference and the explicit Qwen/Llama JSON format both take the
        // default wrapper.
        #expect(SchemaConverter.ToolCallStructuralTag.forFormat(nil) == .qwen)
        #expect(SchemaConverter.ToolCallStructuralTag.forFormat(.json) == .qwen)
        // Unmapped families fall back to the Qwen wrapper.
        #expect(SchemaConverter.ToolCallStructuralTag.forFormat(.lfm2) == .qwen)
        #expect(SchemaConverter.ToolCallStructuralTag.forFormat(.gemma4) == .qwen)
        // GLM-4.5/4.6/4.7 (glm4_moe) gets its own bespoke wrapper.
        #expect(SchemaConverter.ToolCallStructuralTag.forFormat(.glm4) == .glm4)
        // The two specs are genuinely distinct.
        #expect(SchemaConverter.ToolCallStructuralTag.glm4 != .qwen)
    }

    /// Regression guard: a `nil` format (no inference) must produce exactly
    /// the historical Qwen `<tool_call>\n … \n</tool_call>` wrapper.
    @Test
    func defaultFormatProducesQwenWrapper() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let grammar = try SchemaConverter.encodeToolCallingGrammar(tools: [weather])
        let parsed = try parseAsDictionary(grammar)
        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        let wrapped = try #require(elements.first)
        #expect(wrapped["begin"] as? String == "<tool_call>\n")
        #expect(wrapped["end"] as? [String] == ["\n</tool_call>"])
    }

    /// Regression guard (AC2): the explicit `.json` (Qwen/Llama) format
    /// selects the same wrapper as the `nil`-format default — the Qwen
    /// `<tool_call>\n … \n</tool_call>` delimiters, unchanged. (The full
    /// serialized JSON is *not* compared for byte equality: xgrammar reads
    /// it as an unordered object and `JSONSerialization` does not emit a
    /// stable key order, so the wrapper is the meaningful invariant.)
    @Test
    func qwenJSONFormatSelectsSameWrapperAsDefault() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let byDefault = try wrappedArm(
            of: SchemaConverter.encodeToolCallingGrammar(tools: [weather]))
        let byJSON = try wrappedArm(
            of: SchemaConverter.encodeToolCallingGrammar(tools: [weather], format: .json))
        #expect(byDefault.begin == "<tool_call>\n")
        #expect(byDefault.end == ["\n</tool_call>"])
        #expect(byJSON.begin == byDefault.begin)
        #expect(byJSON.end == byDefault.end)
    }

    /// AC3: the GLM4 (`glm4_moe`) format wires its native wrapper —
    /// `<tool_call>`/`</tool_call>` with no inner newlines (see
    /// `GLM4ToolCallParser` fixtures) — distinct from Qwen's newline-padded
    /// wrapper.
    @Test
    func glm4FormatProducesGLM4Wrapper() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let weather = Transcript.ToolDefinition(
            name: "get_weather",
            description: "Get current weather",
            parameters: WeatherArgs.generationSchema
        )
        let grammar = try SchemaConverter.encodeToolCallingGrammar(
            tools: [weather], format: .glm4)
        let parsed = try parseAsDictionary(grammar)

        #expect(parsed["type"] as? String == "structural_tag")
        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        #expect(elements.count == 2)

        // Wrapped arm carries GLM's newline-free wrapper.
        let wrapped = elements[0]
        #expect(wrapped["type"] as? String == "tag")
        #expect(wrapped["begin"] as? String == "<tool_call>")
        #expect(wrapped["end"] as? [String] == ["</tool_call>"])

        // The embedded envelope is unchanged from the default path.
        let wrappedContent = try #require(wrapped["content"] as? [String: Any])
        #expect(wrappedContent["type"] as? String == "json_schema")

        // Bare arm still present.
        let bare = elements[1]
        #expect(bare["type"] as? String == "json_schema")
    }

    // MARK: - Helpers

    private func parseAsDictionary(_ json: String) throws -> [String: Any] {
        let data = Data(json.utf8)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Envelope JSON did not parse as an object: \(json)")
            return [:]
        }
        return obj
    }

    /// Parses a structural-tag grammar string and returns the wrapped arm's
    /// `begin`/`end` delimiters — the wrapper invariant the format seam
    /// selects — without depending on `JSONSerialization`'s (unstable) key
    /// ordering.
    private func wrappedArm(of grammar: String) throws -> (begin: String, end: [String]) {
        let parsed = try parseAsDictionary(grammar)
        let format = try #require(parsed["format"] as? [String: Any])
        let elements = try #require(format["elements"] as? [[String: Any]])
        let wrapped = try #require(elements.first)
        let begin = try #require(wrapped["begin"] as? String)
        let end = try #require(wrapped["end"] as? [String])
        return (begin, end)
    }

    private func makeByteTokenizer() throws -> GrammarTokenizer {
        let vocabSize = 256
        let vocab: [String] = (0 ..< vocabSize).map { byte in
            String(format: "<0x%02X>", byte)
        }
        return try GrammarTokenizer(
            vocab: vocab,
            vocabType: .byteFallback,
            eosTokenId: Int32(vocabSize - 1)
        )
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
