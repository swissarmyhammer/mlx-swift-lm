// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4 emits tool calls in DSML. Every DSML tag is built around the
// `｜DSML｜` marker, whose delimiter is FULLWIDTH VERTICAL LINE U+FF5C — not
// the ASCII `|` it looks like. The fixtures below carry that exact character,
// matching the write side (`DeepSeekV4ChatEncoder`) byte for byte.

import Foundation
import Testing

@testable import MLXLLM
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

    // MARK: - Format declaration
    //
    // `ToolCallFormat.infer(from:)` and its model_type table are gone (upstream
    // #502): a model now declares its own format through
    // `ChatConventionsProviding` instead of a central lookup. These tests build
    // each model from a small synthetic checkpoint (no real weights needed) and
    // read the declaration straight off the model instance.

    @Test("the DeepSeek-V4 model declares the DSML format")
    func deepSeekV4ModelDeclaresDSML() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(DeepSeekV4SyntheticCheckpoint.configJSON.utf8),
            modelType: "deepseek_v4")
        #expect(model.toolCallFormat == .dsml)
        #expect(ToolCallFormat.dsml.rawValue == "dsml")
    }

    @Test("the DeepSeek-V3 model declares no tool-call format")
    func deepSeekV3ModelDeclaresNoFormat() async throws {
        // deepseek_v3 is an architecture shared by DeepSeek-V3 and DeepSeek-R1;
        // see the "Chat conventions" note on `DeepseekV3Model` for why it
        // deliberately declares nothing.
        let json = """
            {
                "model_type": "deepseek_v3",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "moe_intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "n_routed_experts": 4,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "n_group": 2,
                "topk_group": 1,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "first_k_dense_replace": 1,
                "moe_layer_freq": 1,
                "q_lora_rank": 4,
                "kv_lora_rank": 4,
                "qk_rope_head_dim": 2,
                "qk_nope_head_dim": 2,
                "v_head_dim": 4,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "max_position_embeddings": 128,
                "attention_bias": false
            }
            """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8),
            modelType: "deepseek_v3")
        #expect(model.toolCallFormat == nil)
    }

    // MARK: - Parsing

    @Test("parse extracts the name and the typed arguments")
    func parseExtractsNameAndArguments() throws {
        let parser = ToolCallFormat.dsml.createParser()
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
        let parser = ToolCallFormat.dsml.createParser()
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
        let parser = ToolCallFormat.dsml.createParser()
        let call = try #require(parser.parse(content: content, tools: nil))
        #expect(call.function.arguments["steps"] == .array([.string("walk"), .string("run")]))
    }

    @Test("parse returns nothing for text that is not a tool call")
    func parseReturnsNilForNonToolText() {
        let parser = ToolCallFormat.dsml.createParser()
        #expect(parser.parse(content: "The weather in Paris is sunny.", tools: nil) == nil)
        #expect(parser.parse(content: "", tools: nil) == nil)
    }

    @Test("parseEOS recovers every invoke of a parallel round")
    func parseEOSRecoversParallelInvokes() {
        let parser = ToolCallFormat.dsml.createParser()
        let calls = parser.parseEOS(Self.parallelCalls, tools: nil)

        #expect(calls.count == 2)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.last?.function.name == "get_time")
        #expect(calls.last?.function.arguments["timezone"] == .string("CET"))
    }

    // MARK: - The round trip

    /// One call whose parameter names are NOT in alphabetical order.
    ///
    /// A render that sorts the names writes `cursor`, `loc`, `path`. Thus a
    /// byte comparison against this fixture fails as soon as the order of the
    /// model is lost.
    private static let unsortedCall = """
        <｜DSML｜tool_calls>
        <｜DSML｜invoke name="view_file">
        <｜DSML｜parameter name="path" string="true">/tmp/notes.txt</｜DSML｜parameter>
        <｜DSML｜parameter name="cursor" string="false">3</｜DSML｜parameter>
        <｜DSML｜parameter name="loc" string="false">[1, 2]</｜DSML｜parameter>
        </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """

    /// The tag that opens a DSML block of calls.
    private static let toolCallsOpenTag = "<｜DSML｜tool_calls>"

    /// The tag that closes a DSML block of calls.
    private static let toolCallsCloseTag = "</｜DSML｜tool_calls>"

    /// The DSML block of calls that one prompt holds.
    ///
    /// - Parameter prompt: the prompt to read.
    /// - Returns: the block, or `nil` when the prompt holds none.
    private static func toolCallsBlock(of prompt: String) -> String? {
        guard let open = prompt.range(of: toolCallsOpenTag),
            let close = prompt.range(of: toolCallsCloseTag)
        else { return nil }
        return String(prompt[open.lowerBound ..< close.upperBound])
    }

    /// Renders one assistant turn of calls through the DeepSeek-V4 prompt path.
    ///
    /// The path is the one a second round of a conversation takes: the message
    /// generator writes the raw dictionary of the turn, and the encoder renders
    /// that dictionary again.
    ///
    /// - Parameter calls: the calls of the turn.
    /// - Returns: the prompt.
    private static func rendered(_ calls: [ToolCall]) -> String {
        let raw = DefaultMessageGenerator().generate(messages: [.assistant("", toolCalls: calls)])
        return DeepSeekV4ChatEncoder().encode(
            messages: DeepSeekV4ChatEncoder.Message.messages(from: raw), thinkingMode: .chat)
    }

    @Test("a parsed call renders again with the parameter order of the model")
    func roundTripKeepsTheParameterOrderOfTheModel() throws {
        let parser = ToolCallFormat.dsml.createParser()
        let calls = parser.parseEOS(Self.unsortedCall, tools: nil)

        let block = try #require(Self.toolCallsBlock(of: Self.rendered(calls)))
        #expect(block == Self.unsortedCall)
    }

    @Test("the parser records the arguments as JSON text in the order of the model")
    func parseRecordsArgumentsJSONInTheOrderOfTheModel() throws {
        let parser = ToolCallFormat.dsml.createParser()
        let call = try #require(parser.parse(content: Self.unsortedCall, tools: nil))

        #expect(
            call.function.argumentsJSON
                == #"{"path": "/tmp/notes.txt", "cursor": 3, "loc": [1, 2]}"#)
    }

    @Test("a call that records no order still renders every argument")
    func callWithoutRecordedOrderStillRendersEveryArgument() throws {
        let call = ToolCall(
            function: .init(name: "view_file", arguments: ["path": "/tmp/notes.txt", "cursor": 3]))

        let block = try #require(Self.toolCallsBlock(of: Self.rendered([call])))
        #expect(call.function.argumentsJSON == nil)
        #expect(
            block == """
                <｜DSML｜tool_calls>
                <｜DSML｜invoke name="view_file">
                <｜DSML｜parameter name="cursor" string="false">3</｜DSML｜parameter>
                <｜DSML｜parameter name="path" string="true">/tmp/notes.txt</｜DSML｜parameter>
                </｜DSML｜invoke>
                </｜DSML｜tool_calls>
                """)
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
