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

    /// XML function format used by Nemotron, Qwen3 Coder, Qwen3.5, and similar models.
    /// Example: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`
    case xmlFunction = "xml_function"

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

    /// MiniMax M3 namespaced XML format: parameters are arbitrary
    /// `<key>value</key>` children rather than M2's `<parameter name="k">v</parameter>`
    /// attribute style. Every tag is prefixed with M3's literal namespace token.
    ///
    /// See ``MiniMaxM3ToolCallParser`` for the full format and parsing details.
    /// Example: `]<]minimax[>[<invoke name="f">]<]minimax[>[<k>v]<]minimax[>[</k>]<]minimax[>[</invoke>`
    case minimaxM3 = "minimax_m3"

    /// Mistral V11+ format with [TOOL_CALLS] and [ARGS] delimiters.
    /// Example: `[TOOL_CALLS]get_weather [ARGS]{"location": "Tokyo"}`
    case mistral

    /// Llama 3 inline JSON format.
    /// Example: `<|python_tag|>{ "name": "func", "parameters": {...} }`
    case llama3

    // MARK: - Factory Methods

    /// The opening wrapper tag shared by the ``json`` and ``xmlFunction``
    /// tool-call envelopes.
    private static let toolCallStartTag = "<tool_call>"

    /// The closing wrapper tag shared by the ``json`` and ``xmlFunction``
    /// tool-call envelopes.
    private static let toolCallEndTag = "</tool_call>"

    /// Create the appropriate parser for this format.
    /// - Returns: A parser instance configured for this format
    public func createParser() -> any ToolCallParser {
        switch self {
        case .json:
            return JSONToolCallParser(
                startTag: Self.toolCallStartTag, endTag: Self.toolCallEndTag)
        case .lfm2:
            return PythonicToolCallParser(
                startTag: "<|tool_call_start|>", endTag: "<|tool_call_end|>")
        case .xmlFunction:
            return XMLFunctionParser(
                startTag: Self.toolCallStartTag, endTag: Self.toolCallEndTag)
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
        case .minimaxM3:
            return MiniMaxM3ToolCallParser()
        case .mistral:
            return MistralToolCallParser()
        case .llama3:
            return Llama3ToolCallParser()
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

    /// How an ``inferenceTable`` entry's `value` is compared against the
    /// lowercased `model_type`.
    private enum ModelTypeMatch {
        case exact
        case prefix

        /// Whether `type` satisfies this match kind for `value`.
        func matches(_ type: String, against value: String) -> Bool {
            switch self {
            case .exact:
                return type == value
            case .prefix:
                return type.hasPrefix(value)
            }
        }
    }

    /// First-match `model_type` → format lookup consumed by
    /// ``infer(from:configData:)``.
    ///
    /// Order matters: the first matching entry wins, so an exact dense-family
    /// match must precede the prefix match that would otherwise swallow it
    /// (e.g. "glm4" exact → `.glm4Bare` before "glm4" prefix → `.glm4`).
    private static let inferenceTable:
        [(match: ModelTypeMatch, value: String, format: ToolCallFormat)] = [
            // LFM2 family (lfm2, lfm2_moe, lfm2_5, lfm25, etc.)
            (.prefix, "lfm2", .lfm2),

            // GLM4 dense/non-MoE model (exact match): the original, pre-4.7
            // architecture (e.g. GLM-4-9B-0414). Its chat template never taught
            // the GLM-4.7 `<tool_call>`/`<arg_key>` envelope -- it emits a bare
            // function-name line followed by a bare JSON object of arguments.
            (.exact, "glm4", .glm4Bare),

            // GLM4 MoE family (glm4_moe, glm4_moe_lite, etc.): GLM-4.5/4.6/4.7
            // descend from this architecture and DO use the envelope above.
            (.prefix, "glm4", .glm4),

            // Gemma4
            (.prefix, "gemma4", .gemma4),

            // Gemma
            (.exact, "gemma", .gemma),

            // Nemotron family (nemotron_h, etc.)
            (.prefix, "nemotron", .xmlFunction),

            // Qwen3.5 family (qwen3_5, qwen3_5_moe, etc.)
            (.prefix, "qwen3_5", .xmlFunction),

            // Qwen3-Next family (qwen3_next, etc.)
            (.prefix, "qwen3_next", .xmlFunction),

            // Mistral3 family (mistral3, mistral3_text, etc.) and Ministral3
            // (ministral3, e.g. Devstral-2-123B) — "ministral3" does not share
            // the "mistral3" prefix, so it needs its own entry.
            (.prefix, "mistral3", .mistral),
            (.prefix, "ministral3", .mistral),

            // MiniMax-M2 (model_type "minimax", arch MiniMaxM2ForCausalLM, e.g.
            // mlx-community/MiniMax-M2-4bit). Exact match, mirroring the dense
            // GLM4 style above: the only registered minimax architecture is M2,
            // and earlier MiniMax families (minimax_text_01, minimax_m1) neither
            // load here nor share M2's `<minimax:tool_call>` invoke format.
            (.exact, "minimax", .minimaxM2),

            // MiniMax-M3 family (minimax_m3, minimax_m3_vl, e.g.
            // mlx-community/MiniMax-M3-4bit). Prefix match so both the flat
            // text and VL model types resolve to the same row. Distinct from
            // M2: namespaced XML with arbitrary `<key>value</key>` parameter
            // children rather than `<parameter name="k">v</parameter>` -- see
            // ``MiniMaxM3ToolCallParser``.
            (.prefix, "minimax_m3", .minimaxM3),
        ]

    /// Infer the tool call format based on model type from config.json.
    ///
    /// This method maps known model types to their corresponding tool call formats,
    /// enabling automatic format detection when loading models.
    ///
    /// - Parameters:
    ///   - modelType: The `model_type` value from config.json
    ///   - configData: The raw config.json data for inspecting secondary signals (e.g. `rope_scaling` for Llama 3)
    /// - Returns: The appropriate `ToolCallFormat`, or `nil` to use the default format
    public static func infer(from modelType: String, configData: Data? = nil) -> ToolCallFormat? {
        let type = modelType.lowercased()

        // Llama family: the one case the table cannot express — "llama"
        // alone is ambiguous between Llama 1/2 (no inferable format) and
        // Llama 3, so secondary config.json signals decide.
        if type == "llama" {
            return inferLlamaFormat(configData: configData)
        }

        return inferenceTable.first { $0.match.matches(type, against: $0.value) }?.format
    }

    /// Resolves the Llama-family ambiguity via secondary config.json signals.
    ///
    /// `model_type == "llama"` covers both Llama 1/2 (no inferable tool
    /// format) and Llama 3 (`.llama3` inline JSON), so the format is decided
    /// by signals only Llama 3 carries.
    ///
    /// - Parameter configData: The raw config.json data.
    /// - Returns: `.llama3` when a Llama 3 signal is present, otherwise `nil`.
    private static func inferLlamaFormat(configData: Data?) -> ToolCallFormat? {
        guard let data = configData,
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        // Secondary signal 1: vocab_size >= 128000 (Llama 3 uses 128256, Llama 2 uses 32000)
        if let vocabSize = json["vocab_size"] as? Int, vocabSize >= 128000 {
            return .llama3
        }

        // Secondary signal 2: rope_scaling with rope_type == "llama3"
        if let ropeScaling = json["rope_scaling"] as? [String: Any],
            let ropeType = ropeScaling["rope_type"] as? String,
            ropeType == "llama3"
        {
            return .llama3
        }

        return nil
    }
}
