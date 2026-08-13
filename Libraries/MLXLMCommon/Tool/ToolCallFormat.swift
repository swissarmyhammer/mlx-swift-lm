// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ToolCallParser Protocol

/// Protocol for parsing tool call content from model output.
///
/// Different models use different formats for tool calls. This protocol provides
/// a common interface for parsing tool calls from model output text.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public protocol ToolCallParser: Sendable {
    /// The start tag that indicates a tool call is beginning.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var startTag: String? { get }

    /// The end tag that indicates a tool call has ended.
    /// Returns `nil` for inline formats that don't use wrapper tags.
    var endTag: String? { get }

    /// Whether this format has no literal marker preceding its content and
    /// must therefore have the *entire* response buffered before parsing.
    ///
    /// Tagged formats (`startTag != nil`) and inline JSON-envelope formats
    /// (`startTag == nil` but the function name is embedded inside the JSON,
    /// e.g. `Llama3ToolCallParser`) can both be detected incrementally as
    /// text streams in. Formats where non-JSON content -- such as a bare
    /// function name -- may precede the parseable payload with no marker at
    /// all cannot be distinguished from ordinary prose mid-stream, so
    /// ``ToolCallProcessor`` defers all parsing for them to end-of-sequence.
    /// Defaults to `false`.
    var buffersEntireResponse: Bool { get }

    /// Parse the content into a `ToolCall`.
    /// - Parameters:
    ///   - content: The text content to parse (may include tags)
    ///   - tools: Optional tool schemas for type-aware parsing
    /// - Returns: A `ToolCall` if parsing succeeds, `nil` otherwise
    func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall?

    /// Parse remaining buffered content at end-of-sequence.
    ///
    /// Called when generation ends to extract any tool calls still in the buffer.
    /// The default implementation splits on `startTag` (if present) and parses
    /// each segment individually.
    ///
    /// - Parameters:
    ///   - toolCallBuffer: The text remaining in the tool-call buffer when
    ///     generation ended.
    ///   - tools: Optional tool schemas for type-aware parsing.
    /// - Returns: The tool calls recovered from `toolCallBuffer`, in the
    ///   order they appear.
    func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]
}

extension ToolCallParser {
    /// Default implementation: `false`, because most formats carry a literal
    /// marker (a start tag or a JSON envelope) that lets tool calls be
    /// detected incrementally as text streams in. Only markerless formats
    /// (e.g. ``GLM4BareToolCallParser``) override this to `true`.
    public var buffersEntireResponse: Bool { false }

    /// Default implementation: for tagged formats, splits the buffer on
    /// `startTag` and parses each non-empty segment individually (recovering
    /// multiple calls left in the buffer); for untagged formats, parses the
    /// whole buffer as a single call.
    ///
    /// - Parameters:
    ///   - toolCallBuffer: The text remaining in the tool-call buffer when
    ///     generation ended.
    ///   - tools: Optional tool schemas for type-aware parsing.
    /// - Returns: The tool calls recovered from `toolCallBuffer`, in the
    ///   order they appear.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        if let startTag {
            return
                toolCallBuffer
                .components(separatedBy: startTag)
                .filter { !$0.isEmpty }
                .compactMap { parse(content: $0, tools: tools) }
        } else {
            guard let toolCall = parse(content: toolCallBuffer, tools: tools) else {
                return []
            }
            return [toolCall]
        }
    }
}

// MARK: - ToolCallFormat Enum

/// Supported tool call formats for different language models.
///
/// This enum defines the various tool call formats used by different LLM families.
/// Each format has its own syntax for encoding function names and arguments.
///
/// The raw string values can be used for JSON serialization or CLI parameters.
///
/// Reference: https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/tool_parsers
public enum ToolCallFormat: String, Sendable, Codable, CaseIterable {
    /// Default JSON format used by Llama, Qwen, and most models.
    /// Example: `<tool_call>{"name": "func", "arguments": {...}}</tool_call>`
    case json

    /// LFM2/LFM2.5 Pythonic format with model-specific tags.
    /// Example: `<|tool_call_start|>[func(arg='value')]<|tool_call_end|>`
    case lfm2

    /// XML function format used by Nemotron, Qwen3 Coder, Qwen3 Next, and similar models.
    /// Example: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    case xmlFunction = "xml_function"

    /// Qwen 3.5's XML function format with a framed Hermes-JSON compatibility dialect.
    ///
    /// Qwen 3.5 is prompted to emit `xmlFunction`, but can sporadically emit the
    /// Qwen/Hermes JSON dialect used by earlier Qwen models instead. Both dialects
    /// use the same `<tool_call>` frame; this format accepts either payload without
    /// enabling bare JSON recovery.
    case qwen35 = "qwen3_5"

    /// GLM4 format with arg_key/arg_value tags.
    /// Example: `func<arg_key>k</arg_key><arg_value>v</arg_value>`
    case glm4

    /// GLM4 bare format used by the original (pre-4.7, non-MoE) GLM-4
    /// checkpoints, such as GLM-4-9B-0414. No wrapper tags or JSON envelope:
    /// the function name appears alone, followed by a bare JSON object of
    /// just the arguments.
    /// Example: `get_weather\n{"location": "Paris", "unit": "celsius"}`
    case glm4Bare = "glm4_bare"

    /// Gemma function call format.
    /// Example: `<start_function_call>call:name{key:value,k:<escape>str<escape>}<end_function_call>`
    case gemma

    /// Gemma4 function call format.
    /// Example: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`
    case gemma4

    /// Kimi K2 format with functions prefix.
    /// Example: `functions.name:0<|tool_call_argument_begin|>{"key": "value"}`
    case kimiK2 = "kimi_k2"

    /// MiniMax M2 format with invoke/parameter tags.
    /// Example: `<invoke name="f"><parameter name="k">v</parameter></invoke>`
    case minimaxM2 = "minimax_m2"

    /// Muse Glimmer's Onyx ATEM invoke/parameter format.
    /// Example: `<atem:function_calls><atem:invoke name="f">...</atem:invoke></atem:function_calls>`
    case atem

    /// Mistral V11+ format with [TOOL_CALLS] and [ARGS] delimiters.
    /// Example: `[TOOL_CALLS]get_weather [ARGS]{"location": "Tokyo"}`
    case mistral

    /// Llama 3 inline JSON format.
    /// Example: `<|python_tag|>{ "name": "func", "parameters": {...} }`
    case llama3

    /// GPT-OSS full Harmony response protocol.
    ///
    /// Not a tool-call JSON dialect: selects the Harmony frame parser + router
    /// (channels, recipients, terminators). Tool calls appear only as
    /// `commentary to=functions.<name>` frames within that protocol.
    /// Example: `<|channel|>commentary to=functions.get_weather<|message|>{"location": "Tokyo"}<|call|>`
    case gptOSS = "gpt_oss"

    // MARK: - Factory Methods

    /// The opening wrapper tag shared by the ``json`` and ``xmlFunction``
    /// tool-call envelopes.
    private static let toolCallStartTag = "<tool_call>"

    /// The closing wrapper tag shared by the ``json`` and ``xmlFunction``
    /// tool-call envelopes.
    private static let toolCallEndTag = "</tool_call>"

    /// Create the appropriate parser for this format.
    /// - Returns: A parser instance configured for this format
    public func makeParser() -> any ToolCallParser {
        switch self {
        case .json:
            return JSONToolCallParser(
                startTag: Self.toolCallStartTag, endTag: Self.toolCallEndTag)
        case .lfm2:
            return PythonicToolCallParser(
                startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        case .xmlFunction:
            return XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .qwen35:
            return Qwen35ToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        case .glm4:
            return GLM4ToolCallParser()
        case .glm4Bare:
            return GLM4BareToolCallParser()
        case .gemma:
            return GemmaFunctionParser(
                startTag: "<start_function_call>", endTag: "<end_function_call>",
                escapeMarker: "<escape>")
        case .gemma4:
            return GemmaFunctionParser(
                startTag: "<|tool_call>", endTag: "<tool_call|>", escapeMarker: "<|\"|>")
        case .kimiK2:
            return KimiK2ToolCallParser()
        case .minimaxM2:
            return MiniMaxM2ToolCallParser()
        case .atem:
            return ATEMToolCallParser()
        case .mistral:
            return MistralToolCallParser()
        case .llama3:
            return Llama3ToolCallParser()
        case .gptOSS:
            return JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")
        }
    }

    /// Builds the response-protocol decoder used by streaming text generation.
    ///
    /// Ordinary formats share the detokenized tool-call decoder. Protocols
    /// with token-level framing provide their own implementation here, keeping
    /// concrete model behavior out of the generic evaluation loop.
    package func makeTokenStreamDecoder(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>
    ) -> any TokenStreamDecoder {
        if let decoder = makeProtocolTokenStreamDecoder(
            tokenizer: tokenizer, tools: tools, stopStrings: stopStrings)
        {
            return decoder
        }
        return StandardTokenStreamDecoder(
            tokenizer: tokenizer, format: self, tools: tools, stopStrings: stopStrings)
    }

    /// Builds a decoder only for formats which own a framed token protocol.
    /// Generic callers use this factory and never depend on an Onyx/Harmony
    /// concrete type. Plain text tool syntaxes return `nil` and stay on their
    /// standard detokenized routing path.
    package func makeProtocolTokenStreamDecoder(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>
    ) -> (any TokenStreamDecoder)? {
        switch self {
        case .gptOSS:
            return HarmonyStreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings)

        case .atem:
            return OnyxStreamAdapter(
                tokenizer: tokenizer, tools: tools, stopStrings: stopStrings)

        case .json, .lfm2, .xmlFunction, .qwen35, .glm4, .gemma, .gemma4, .kimiK2, .minimaxM2,
            .mistral, .llama3:
            return nil
        }
    }

    /// Cache-reuse rules required by this protocol's on-device token stream.
    ///
    /// Most formats are plain text dialects: what the model generates is what a
    /// chat-template re-render produces, so the standard prefix rules suffice
    /// and this returns no extra rules. A protocol that keeps state in the KV
    /// cache which the template cannot reproduce contributes a rule here.
    ///
    /// - Parameter tokenizer: used to resolve protocol control tokens; a
    ///   tokenizer lacking them yields no rules, leaving the model on the
    ///   standard path.
    func promptCacheReuseRules(tokenizer: any Tokenizer) -> [any PromptCacheReuseRule] {
        switch self {
        case .gptOSS:
            return HarmonyToolRestartRule(tokenizer: tokenizer).map { [$0] } ?? []
        case .atem:
            return OnyxToolRestartRule(tokenizer: tokenizer).map { [$0] } ?? []
        case .json, .lfm2, .xmlFunction, .qwen35, .glm4, .gemma, .gemma4, .kimiK2, .minimaxM2,
            .mistral,
            .llama3:
            return []
        }
    }

    /// Number of structured call commits a protocol's chat-template render
    /// contributes to the prompt. Harmony renders at most one call per
    /// assistant message; Onyx renders every call as its own assistant frame.
    func promptCacheStructuredToolCallCount(in messages: [Chat.Message]) -> Int {
        switch self {
        case .gptOSS:
            messages.count {
                $0.role == .assistant && $0.tool?.calls?.isEmpty == false
            }
        case .atem:
            messages.reduce(into: 0) { count, message in
                if message.role == .assistant {
                    count += message.tool?.calls?.count ?? 0
                }
            }
        case .json, .lfm2, .xmlFunction, .qwen35, .glm4, .gemma, .gemma4, .kimiK2, .minimaxM2,
            .mistral, .llama3:
            0
        }
    }

    /// Generate an ID compatible with this tool-call syntax.
    /// - Returns: A fresh unique ID (9 characters for ``mistral``, OpenAI-style
    ///   `call_`-prefixed for all other formats)
    public func generateToolCallID() -> String {
        let uuid = UUID().uuidString.replacingOccurrences(of: "-", with: "")

        switch self {
        case .mistral:
            return String(uuid.prefix(9))
        default:
            return "call_" + uuid.lowercased()
        }
    }
}
