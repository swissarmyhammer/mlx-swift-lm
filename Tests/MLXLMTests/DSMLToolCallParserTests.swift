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
