// Copyright © 2026 Apple Inc.
//
// DeepSeek-V4 ships no chat template, thus `DeepSeekV4ChatEncoder` is the
// only correct prompt builder for the family (card ^f0ymw6b). These tests
// pin the wiring of card ^mjrzkgm: a model that the type registry identifies
// as `deepseek_v4` builds its prompt with the encoder, and every other model
// keeps its present path.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.serialized)
struct DeepSeekV4EncoderWiringTests {

    // MARK: - Fixtures

    /// A DeepSeek-V4-like tokenizer for prompt assertions: `encode` maps each
    /// UTF-8 byte of the text to one token id so a test can read the prompt
    /// back, and `applyChatTemplate` answers as
    /// ``chatTemplateIdentifiers`` states.
    private struct ByteTokenizer: MLXLMCommon.Tokenizer {

        /// The identifiers `applyChatTemplate` answers with, or `nil` when
        /// this tokenizer carries no chat template and must throw.
        ///
        /// The default is `nil`, which is the DeepSeek-V4 checkpoint as the
        /// family's port assumes it: no template, thus the encoder builds the
        /// prompt. A test that stands for a checkpoint WITH a template gives
        /// this property a value.
        var chatTemplateIdentifiers: [Int]?

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            Array(text.utf8).map { Int($0) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            String(decoding: tokenIds.map { UInt8(clamping: $0) }, as: UTF8.self)
        }

        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            guard let chatTemplateIdentifiers else {
                throw TokenizerError.missingChatTemplate
            }
            return chatTemplateIdentifiers
        }
    }

    /// Builds the prompt processor of the DeepSeek-V4 path: the byte
    /// tokenizer wrapped by the model's own `promptTokenizer(wrapping:)`.
    private func makeDeepSeekV4Processor() throws -> LLMUserInputProcessor {
        let model = DeepSeekV4Model(try DeepSeekV4SyntheticCheckpoint.configuration())
        return LLMUserInputProcessor(
            tokenizer: model.promptTokenizer(wrapping: ByteTokenizer()),
            configuration: ModelConfiguration(id: "test/deepseek-v4-synthetic"),
            messageGenerator: DefaultMessageGenerator(),
            missingChatTemplateRefusal: model.missingChatTemplateRefusal)
    }

    /// Reads the rendered prompt text back out of a prepared input.
    private func renderedText(of input: LMInput) -> String {
        ByteTokenizer().decode(
            tokenIds: input.text.tokens.asArray(Int.self), skipSpecialTokens: false)
    }

    // MARK: - Detection

    @Test("DeepSeekV4Model wraps the loaded tokenizer with the encoder")
    func deepSeekV4ModelWrapsThePromptTokenizer() throws {
        let model = DeepSeekV4Model(try DeepSeekV4SyntheticCheckpoint.configuration())
        let tokenizer = model.promptTokenizer(wrapping: ByteTokenizer())
        #expect(tokenizer is DeepSeekV4EncodingTokenizer)
    }

    @Test("a model that is not DeepSeek-V4 keeps its loaded tokenizer")
    func otherModelsKeepTheirTokenizer() {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 16, hiddenLayers: 1, intermediateSize: 16, attentionHeads: 2,
            headDim: 8,
            rmsNormEps: 0.00001, vocabularySize: 16, kvHeads: 1,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 8,
            slidingWindow: 8, slidingWindowPattern: 1,
            maxPositionEmbeddings: 32
        )
        let model = Gemma3TextModel(config)
        let tokenizer = model.promptTokenizer(wrapping: ByteTokenizer())
        #expect(!(tokenizer is DeepSeekV4EncodingTokenizer))
        #expect(tokenizer is ByteTokenizer)
    }

    // MARK: - The rendered prompt

    @Test("the prompt path renders exactly what the encoder renders")
    func processorRendersWithTheEncoder() throws {
        let processor = try makeDeepSeekV4Processor()

        let input = try processor.prepare(
            input: UserInput(chat: [.system("Be brief."), .user("Hello")]))

        let expected = DeepSeekV4ChatEncoder().encode(
            messages: [.system(content: "Be brief."), .user(content: "Hello")],
            thinkingMode: .thinking)
        #expect(renderedText(of: input) == expected)
    }

    @Test("the thinking flag of the reasoning row selects the chat mode")
    func thinkingFlagSelectsChatMode() throws {
        let processor = try makeDeepSeekV4Processor()

        let input = try processor.prepare(
            input: UserInput(chat: [.user("Hello")], additionalContext: ["thinking": false]))

        let expected = DeepSeekV4ChatEncoder().encode(
            messages: [.user(content: "Hello")], thinkingMode: .chat)
        #expect(renderedText(of: input) == expected)
        #expect(renderedText(of: input).hasSuffix(DeepSeekV4ChatEncoder.SpecialToken.thinkEnd))
    }

    @Test("tool specifications render into the system turn")
    func toolsRenderIntoTheSystemTurn() throws {
        let processor = try makeDeepSeekV4Processor()
        let weatherTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string"] as [String: any Sendable]
                    ] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let input = try processor.prepare(
            input: UserInput(chat: [.system("Be brief."), .user("Hello")], tools: [weatherTool]))

        let text = renderedText(of: input)
        #expect(text.contains("## Tools"))
        #expect(text.contains("get_weather"))
    }

    @Test("a tool round trip replays the calls and the results in DSML")
    func toolRoundTripReplaysCallsAndResults() throws {
        let processor = try makeDeepSeekV4Processor()
        let call = MLXLMCommon.ToolCall(
            function: .init(
                name: "get_weather", arguments: ["city": "Paris"] as [String: any Sendable]),
            id: "call_1")

        let input = try processor.prepare(
            input: UserInput(
                chat: [
                    .user("Weather in Paris?"),
                    .assistant("", toolCalls: [call]),
                    .tool(#"{"forecast": "sunny"}"#, id: "call_1"),
                ]))

        let text = renderedText(of: input)
        #expect(
            text.contains("<\(DeepSeekV4ChatEncoder.SpecialToken.dsml)invoke name=\"get_weather\">")
        )
        #expect(text.contains(#"<tool_result>{"forecast": "sunny"}</tool_result>"#))
    }

    // MARK: - The order of the members of a JSON object

    // A `ToolSpec` and a ``ToolCall`` both carry a Swift `Dictionary`, which
    // keeps no order. The published reference writes each object in one fixed
    // order, and the model saw that order in training, thus the mapping has to
    // impose it. The two expectations below are transcriptions of the
    // published golden file `encoding/tests/test_output_1.txt` of
    // `deepseek-ai/DeepSeek-V4-Flash`.

    /// The `get_weather` schema of the published input `test_input_1.json`,
    /// as the Swift dictionary a caller writes.
    private static let weatherToolSpec: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "parameters": [
                "type": "object",
                "properties": [
                    "location": [
                        "type": "string",
                        "description": "The city name",
                    ] as [String: any Sendable],
                    "unit": [
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
                "required": ["location"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    /// The line that the published golden output writes for
    /// ``weatherToolSpec`` under `### Available Tool Schemas`.
    private static let publishedWeatherSchemaLine: String =
        #"{"name": "get_weather", "description": "Get the weather for a specific location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, "required": ["location"]}}"#

    /// The DSML block that the published golden output writes for the
    /// `get_weather` call of `test_input_1.json`.
    private static let publishedWeatherCallBlock: String = {
        let marker = DeepSeekV4ChatEncoder.SpecialToken.dsml
        return """
            <\(marker)invoke name="get_weather">
            <\(marker)parameter name="location" string="true">Beijing</\(marker)parameter>
            <\(marker)parameter name="unit" string="true">celsius</\(marker)parameter>
            </\(marker)invoke>
            """
    }()

    @Test("a tool schema renders in the member order the published reference writes")
    func toolSchemaRendersInThePublishedMemberOrder() throws {
        let processor = try makeDeepSeekV4Processor()

        let input = try processor.prepare(
            input: UserInput(
                chat: [.system("Be brief."), .user("Hello")], tools: [Self.weatherToolSpec]))

        #expect(renderedText(of: input).contains(Self.publishedWeatherSchemaLine))
    }

    @Test("the arguments of a replayed call render in one stable order")
    func replayedCallArgumentsRenderInOneStableOrder() throws {
        let processor = try makeDeepSeekV4Processor()
        let call = MLXLMCommon.ToolCall(
            function: .init(
                name: "get_weather",
                arguments: ["unit": "celsius", "location": "Beijing"] as [String: any Sendable]),
            id: "call_001")

        let input = try processor.prepare(
            input: UserInput(
                chat: [
                    .user("What's the weather in Beijing?"),
                    .assistant("", toolCalls: [call]),
                ], tools: [Self.weatherToolSpec]))

        #expect(renderedText(of: input).contains(Self.publishedWeatherCallBlock))
    }

    @Test("the plain-text path of every other model does not use the encoder")
    func otherModelPathDoesNotUseTheEncoder() throws {
        let processor = LLMUserInputProcessor(
            tokenizer: ByteTokenizer(),
            configuration: ModelConfiguration(id: "test/no-template"),
            messageGenerator: DefaultMessageGenerator(),
            missingChatTemplateRefusal: nil)

        let input = try processor.prepare(input: UserInput(prompt: "Hello"))

        let text = renderedText(of: input)
        #expect(text.contains("Hello"))
        #expect(!text.contains(DeepSeekV4ChatEncoder.SpecialToken.user))
    }

    // MARK: - ChatSession end to end

    /// Records every text that reaches `encode(text:addSpecialTokens:)`, and
    /// keeps every id inside the 12-entry vocabulary of the synthetic
    /// checkpoint. Mutable state under a lock, hence a class.
    private final class RecordingByteTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private let lock = NSLock()
        private var prompts: [String] = []

        /// Every text the session asked this tokenizer to encode.
        var encodedPrompts: [String] {
            lock.lock()
            defer { lock.unlock() }
            return prompts
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            lock.lock()
            prompts.append(text)
            lock.unlock()
            return Array(text.utf8.prefix(8)).map { Int($0) % 12 }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "ok" }
        func convertTokenToId(_ token: String) -> Int? { nil }
        func convertIdToToken(_ id: Int) -> String? { nil }
        var bosToken: String? = nil
        var eosToken: String? = nil
        var unknownToken: String? = nil

        func applyChatTemplate(
            messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            throw TokenizerError.missingChatTemplate
        }
    }

    @Test("a ChatSession over a DeepSeek-V4 model prompts through the encoder")
    func chatSessionUsesTheEncoder() async throws {
        let model = DeepSeekV4Model(try DeepSeekV4SyntheticCheckpoint.configuration())
        eval(model)
        let recorder = RecordingByteTokenizer()
        let tokenizer = model.promptTokenizer(wrapping: recorder)
        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test/deepseek-v4-synthetic"),
            messageGenerator: DefaultMessageGenerator(),
            missingChatTemplateRefusal: model.missingChatTemplateRefusal)
        let context = ModelContext(
            configuration: processor.configuration, model: model, processor: processor,
            tokenizer: tokenizer)

        let session = ChatSession(
            context, generateParameters: GenerateParameters(maxTokens: 2))
        _ = try await session.respond(to: "Hello")

        let prompt = try #require(recorder.encodedPrompts.first)
        #expect(prompt.hasPrefix(DeepSeekV4ChatEncoder.SpecialToken.beginOfSentence))
        #expect(prompt.contains(DeepSeekV4ChatEncoder.SpecialToken.user + "Hello"))
        #expect(prompt.contains(DeepSeekV4ChatEncoder.SpecialToken.assistant))
    }

    // MARK: - The factory wiring

    // A model declares its prompt path with `promptTokenizer(wrapping:)`, and
    // the declaration only reaches the model when the LOAD installs it. These
    // tests read the production load, thus they fail when the hook stays
    // unused, which no test above can see: each one installs the wrapper by
    // hand.

    /// The one identifier the chat template of ``templateTokenizer`` writes
    /// for a conversation.
    ///
    /// The value sits outside the 0...255 range a byte tokenizer writes, thus
    /// a prompt that holds it can only come from the chat template of the
    /// wrapped tokenizer.
    private static let chatTemplateSentinel = 4242

    /// A tokenizer that renders every conversation with a chat template of its
    /// own, as the published checkpoint does.
    ///
    /// The `mlx-community/DeepSeek-V4-Flash-4bit` snapshot ships an 883-byte
    /// `chat_template.jinja` that reads the `system`, `user` and `assistant`
    /// roles and holds no `tools` variable at all. Such a tokenizer never
    /// throws ``TokenizerError/missingChatTemplate``, thus the refusal in
    /// ``DeepSeekV4Model/missingChatTemplateRefusal`` never fires, and a
    /// prompt that drops every tool looks correct and gives no error.
    private static var templateTokenizer: ByteTokenizer {
        ByteTokenizer(chatTemplateIdentifiers: [chatTemplateSentinel])
    }

    /// A loader that gives one ``templateTokenizer`` and reads no file.
    private struct TemplateTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            DeepSeekV4EncoderWiringTests.templateTokenizer
        }
    }

    /// Writes a synthetic `deepseek_v4` checkpoint that a factory load reads.
    ///
    /// The directory holds the `config.json` of
    /// ``DeepSeekV4SyntheticCheckpoint`` and one safetensors file that carries
    /// the model's own initial parameters. The load filter of
    /// ``DeepSeekV4Model`` passes a checkpoint that already carries module
    /// paths through unchanged, thus those tensors satisfy the whole-model
    /// verification that the weight update runs.
    ///
    /// - Returns: the checkpoint directory. The caller removes it.
    private func writeSyntheticCheckpoint() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "deepseek-v4-factory-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(DeepSeekV4SyntheticCheckpoint.configJSON.utf8).write(
            to: directory.appendingPathComponent("config.json"))

        let model = DeepSeekV4Model(try DeepSeekV4SyntheticCheckpoint.configuration())
        eval(model)
        var weights: [String: MLXArray] = [:]
        for (path, array) in model.parameters().flattened() {
            weights[path] = array
        }
        try MLX.save(
            arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    /// Loads the synthetic checkpoint through the production factory.
    ///
    /// - Parameter directory: the checkpoint directory.
    /// - Returns: the loaded context.
    private func loadSyntheticCheckpoint(at directory: URL) async throws -> ModelContext {
        try await LLMModelFactory.shared._load(
            configuration: ResolvedModelConfiguration(directory: directory),
            tokenizerLoader: TemplateTokenizerLoader())
    }

    @Test("a factory load installs the DeepSeek-V4 prompt tokenizer")
    func factoryLoadInstallsThePromptTokenizer() async throws {
        let directory = try writeSyntheticCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let context = try await loadSyntheticCheckpoint(at: directory)

        #expect(
            context.tokenizer is DeepSeekV4EncodingTokenizer,
            "the load must install the prompt tokenizer that the model asks for")
    }

    @Test("a factory load prompts through the encoder, not the checkpoint template")
    func factoryLoadPromptsThroughTheEncoder() async throws {
        let directory = try writeSyntheticCheckpoint()
        defer { try? FileManager.default.removeItem(at: directory) }

        let context = try await loadSyntheticCheckpoint(at: directory)
        let input = try await context.processor.prepare(
            input: UserInput(
                chat: [.system("Be brief."), .user("Hello")], tools: [Self.weatherToolSpec]))

        let identifiers = input.text.tokens.asArray(Int.self)
        #expect(
            !identifiers.contains(Self.chatTemplateSentinel),
            "the load must not build the prompt with the chat template of the checkpoint")
        let text = ByteTokenizer().decode(tokenIds: identifiers, skipSpecialTokens: false)
        #expect(text.contains("## Tools"), "the prompt must carry the tools section")
        #expect(
            text.contains(Self.publishedWeatherSchemaLine),
            "the prompt must carry the schema of the offered tool")
    }
}
