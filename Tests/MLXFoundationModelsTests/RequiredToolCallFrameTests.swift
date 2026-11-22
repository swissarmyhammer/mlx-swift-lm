// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The frame of a required tool turn follows the tool-call format of the model, and each frame
/// reads back the call its grammar constrains.
@Suite
struct RequiredToolCallFrameTests {

    /// A tool call of `get_weather` for Rome in Inkling 2's tool frame, as the grammar writes it.
    private static let inklingCall =
        #"<|message_model|>get_weather<|content_invoke_tool_json|>{"name": "get_weather", "args": {"city": "Rome"}}<|end_message|>"#

    @Test("Inkling 2 constrains to its own tool frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingUsesItsToolFrame() throws {
        #expect(try RequiredToolCallFrame(format: .inkling) == .inklingFrame)
    }

    @Test(
        "A plain tool-call syntax constrains to the tool-call envelope",
        arguments: [nil, ToolCallFormat.json, .qwen35, .gemma4, .lfm2])
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func plainSyntaxUsesTheEnvelope(format: ToolCallFormat?) throws {
        #expect(try RequiredToolCallFrame(format: format) == .toolCallEnvelope)
    }

    @Test(
        "Harmony and ATEM have no required frame and name their protocol",
        arguments: [ToolCallFormat.gptOSS, .atem])
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func frameProtocolsWithoutARequiredFrameThrow(format: ToolCallFormat) {
        #expect(throws: RequiredToolCallFrame.UnsupportedFormat(format: format)) {
            try RequiredToolCallFrame(format: format)
        }
        #expect(
            RequiredToolCallFrame.UnsupportedFormat(format: format).description
                .contains(format.rawValue))
    }

    @Test("Only the Inkling 2 frame opens the thinking for the model")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func onlyInklingOpensTheThinking() {
        #expect(
            RequiredToolCallFrame.inklingFrame.reasoningOpenerMarkers
                == ["<|message_model|>", "<|content_thinking|>"])
        #expect(RequiredToolCallFrame.toolCallEnvelope.reasoningOpenerMarkers.isEmpty)
    }

    /// The tools of the grammar tests.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func grammarTools() -> [Transcript.ToolDefinition] {
        ["get_weather", "get_time"].map {
            Transcript.ToolDefinition(
                name: $0, description: "A tool.", parameters: String.generationSchema)
        }
    }

    /// The structural tag grammar of the Inkling 2 frame for `tools`, as JSON.
    ///
    /// - Parameter tools: the tools of the turn.
    /// - Returns: the grammar text, its `type`, and its tags, one per tool.
    /// - Throws: the error of the grammar, or of a grammar that does not decode as a structural
    ///   tag.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private static func inklingGrammar(tools: [Transcript.ToolDefinition]) throws -> (
        text: String, type: String?, tags: [[String: Any]]
    ) {
        let grammar = try RequiredToolCallFrame.inklingFrame.grammar(tools: tools)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(grammar.utf8)) as? [String: Any])
        let format = try #require(object["format"] as? [String: Any])
        let tags = try #require(format["elements"] as? [[String: Any]])
        return (grammar, object["type"] as? String, tags)
    }

    @Test("The Inkling 2 grammar opens each tool's frame and closes it at the end of the message")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingGrammarWritesTheToolFrame() throws {
        let tools = Self.grammarTools()
        let (grammar, type, tags) = try Self.inklingGrammar(tools: tools)

        #expect(type == "structural_tag")
        #expect(
            tags.map { $0["begin"] as? String }
                == tools.map {
                    "<|message_model|>\($0.name)<|content_invoke_tool_json|>"
                        + #"{"name": "\#($0.name)", "args": "#
                })
        #expect(tags.allSatisfy { $0["end"] as? [String] == ["}<|end_message|>"] })
        #expect(!grammar.contains("<tool_call>"))
    }

    @Test("The Inkling 2 frame reads back each call that its grammar's tags enclose")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingFrameReadsBackTheCallsOfItsGrammar() throws {
        let tools = Self.grammarTools()
        let tags = try Self.inklingGrammar(tools: tools).tags
        try #require(tags.count == tools.count)

        for (tool, tag) in zip(tools, tags) {
            let begin = try #require(tag["begin"] as? String)
            let end = try #require((tag["end"] as? [String])?.first)
            let call = try #require(
                RequiredToolCallFrame.inklingFrame.call(in: begin + #"{"city": "Rome"}"# + end))
            #expect(call.name == tool.name)
            #expect((call.arguments as? [String: String]) == ["city": "Rome"])
        }
    }

    @Test("The Inkling 2 grammar needs a tool")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingGrammarNeedsATool() {
        #expect(throws: SchemaConverter.SchemaConversionError.noTools) {
            try RequiredToolCallFrame.inklingFrame.grammar(tools: [])
        }
    }

    @Test("The Inkling 2 frame reads the name and the arguments of its call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingFrameReadsItsCall() throws {
        let call = try #require(RequiredToolCallFrame.inklingFrame.call(in: Self.inklingCall))
        #expect(call.name == "get_weather")
        #expect((call.arguments as? [String: String]) == ["city": "Rome"])
    }

    @Test(
        "The Inkling 2 frame reads no call from output outside its frame",
        arguments: [
            #"{"name": "get_weather", "args": {}}"#,
            #"<|message_model|>get_weather{"name": "get_weather", "args": {}}<|end_message|>"#,
            #"<|message_model|>get_time<|content_invoke_tool_json|>{"name": "get_weather", "args": {}}<|end_message|>"#,
            "<|message_model|>get_weather<|content_invoke_tool_json|>no JSON<|end_message|>",
        ])
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func inklingFrameRejectsOtherOutput(output: String) {
        #expect(RequiredToolCallFrame.inklingFrame.call(in: output) == nil)
    }

    @Test("A call without arguments keeps its name, for the caller to drop")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func callWithoutArgumentsKeepsItsName() throws {
        let call = try #require(
            RequiredToolCallFrame.inklingFrame.call(
                in: #"<|message_model|>f<|content_invoke_tool_json|>{"name": "f"}<|end_message|>"#))
        #expect(call.name == "f")
        #expect(call.arguments == nil)
    }

    @Test("The envelope frame reads the wrapped and the bare envelope, and not the Inkling frame")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func envelopeFrameReadsTheEnvelope() throws {
        let frame = RequiredToolCallFrame.toolCallEnvelope
        let wrapped = try #require(
            frame.call(
                in: "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>"))
        #expect(wrapped.name == "f")
        #expect((wrapped.arguments as? [String: Int]) == ["a": 1])
        #expect(frame.call(in: #"{"name": "f", "arguments": {}}"#)?.name == "f")
        #expect(frame.call(in: Self.inklingCall) == nil)
    }
}

#endif
