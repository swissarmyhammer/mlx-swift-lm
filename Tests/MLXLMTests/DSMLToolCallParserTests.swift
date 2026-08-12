// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4 emits tool calls in DSML. Every DSML tag is built around the
// `｜DSML｜` marker, whose delimiter is FULLWIDTH VERTICAL LINE U+FF5C — not
// the ASCII `|` it looks like. The fixtures below carry that exact character,
// matching the write side (`DeepSeekV4ChatEncoder`) byte for byte.

import Testing

@testable import MLXLMCommon

@Suite
struct DSMLToolCallParserTests {

    /// One complete tool-call block, exactly as the write side renders it.
    private static let weatherCall = """
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
        <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
        <｜DSML｜parameter name="detailed" string="false">true</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """

    /// One block that carries two invokes — a parallel call round.
    private static let parallelCalls = """
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_weather">
        <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
        </｜DSML｜invoke>
        <｜DSML｜invoke name="get_time">
        <｜DSML｜parameter name="timezone" string="true">CET</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """

    // MARK: - Format inference

    @Test("model_type deepseek_v4 resolves the DSML format")
    func inferDeepSeekV4ResolvesDSML() {
        #expect(ToolCallFormat.infer(from: "deepseek_v4") == .dsml)
        #expect(ToolCallFormat.dsml.rawValue == "dsml")
    }

    @Test("deepseek_v3 keeps its present path: no inferred format")
    func inferDeepSeekV3ResolvesNoFormat() {
        #expect(ToolCallFormat.infer(from: "deepseek_v3") == nil)
    }

    // MARK: - Parsing

    @Test("parse extracts the name and the typed arguments")
    func parseExtractsNameAndArguments() throws {
        let parser = ToolCallFormat.dsml.makeParser()
        let call = try #require(parser.parse(content: Self.weatherCall, tools: nil))

        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["city"] == .string("Paris"))
        #expect(call.function.arguments["days"] == .int(3))
        #expect(call.function.arguments["detailed"] == .bool(true))
    }

    @Test("a string parameter keeps text that looks like JSON")
    func parseKeepsJSONLookingTextWhenMarkedString() throws {
        let content = """
            <｜DSML｜invoke name="echo">
            <｜DSML｜parameter name="text" string="true">[1, 2]</｜DSML｜parameter>
            </｜DSML｜invoke>
            """
        let parser = ToolCallFormat.dsml.makeParser()
        let call = try #require(parser.parse(content: content, tools: nil))
        #expect(call.function.arguments["text"] == .string("[1, 2]"))
    }

    @Test("a non-string parameter decodes JSON arrays and objects")
    func parseDecodesJSONForNonStringParameters() throws {
        let content = """
            <｜DSML｜invoke name="plan">
            <｜DSML｜parameter name="steps" string="false">["walk", "run"]</｜DSML｜parameter>
            </｜DSML｜invoke>
            """
        let parser = ToolCallFormat.dsml.makeParser()
        let call = try #require(parser.parse(content: content, tools: nil))
        #expect(call.function.arguments["steps"] == .array([.string("walk"), .string("run")]))
    }

    @Test("parse returns nothing for text that is not a tool call")
    func parseReturnsNilForNonToolText() {
        let parser = ToolCallFormat.dsml.makeParser()
        #expect(parser.parse(content: "The weather in Paris is sunny.", tools: nil) == nil)
        #expect(parser.parse(content: "", tools: nil) == nil)
    }

    @Test("parseEOS recovers every invoke of a parallel round")
    func parseEOSRecoversParallelInvokes() {
        let parser = ToolCallFormat.dsml.makeParser()
        let calls = parser.parseEOS(Self.parallelCalls, tools: nil)

        #expect(calls.count == 2)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.last?.function.name == "get_time")
        #expect(calls.last?.function.arguments["timezone"] == .string("CET"))
    }

    // MARK: - Streaming

    @Test("the streaming processor extracts the call and keeps the prose")
    func processorExtractsCallFromStream() {
        let processor = ToolCallProcessor(format: .dsml)

        var text = ""
        for chunk in ["Let me check.\n", Self.weatherCall] {
            if let passthrough = processor.processChunk(chunk) {
                text += passthrough
            }
        }
        processor.processEOS()

        #expect(text == "Let me check.\n")
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }
}
