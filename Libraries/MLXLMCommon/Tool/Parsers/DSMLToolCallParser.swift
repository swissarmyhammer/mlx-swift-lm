// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for DeepSeek-V4's DSML tool-call format.
///
/// One `<｜DSML｜tool_calls>` block wraps the calls of a round. Each call is
/// one `<｜DSML｜invoke name="...">` element, and each argument is one
/// `<｜DSML｜parameter name="..." string="true|false">value</｜DSML｜parameter>`
/// child. The delimiter inside the `｜DSML｜` marker is FULLWIDTH VERTICAL
/// LINE U+FF5C — not the ASCII `|` it looks like — and the marker is one
/// token (id 128825) of the published DeepSeek-V4 tokenizer.
///
/// `string="true"` keeps the value as literal text. `string="false"` decodes
/// the value as JSON (numbers, booleans, arrays, objects). This typing rides
/// in the format itself, thus the tool schemas go unused here.
///
/// The write side of the format is ``DeepSeekV4ChatEncoder``.
/// References: https://github.com/ml-explore/mlx-lm/pull/1337
/// (`mlx_lm/tool_parsers/deepseek_dsml.py`) and `parse_tool_calls` in
/// `encoding/encoding_dsv4.py` of `deepseek-ai/DeepSeek-V4-Flash`.
public struct DSMLToolCallParser: ToolCallParser, Sendable {

    /// The `｜DSML｜` marker: `DSML` between two FULLWIDTH VERTICAL LINE
    /// (U+FF5C) delimiters. It carries no angle brackets of its own — each
    /// tag literal below puts them around it.
    private static let marker = "\u{FF5C}DSML\u{FF5C}"

    /// The opening of one invoke element, up to its `name=` attribute.
    private static let invokeOpen = "<\(marker)invoke name="

    /// The closing tag of one invoke element.
    private static let invokeClose = "</\(marker)invoke>"

    /// The opening of one parameter element, up to its `name=` attribute.
    private static let parameterOpen = "<\(marker)parameter name="

    /// The closing tag of one parameter element.
    private static let parameterClose = "</\(marker)parameter>"

    /// The attribute that separates the parameter name from its type flag.
    private static let stringAttribute = " string="

    /// The tag that opens a DSML tool-call block.
    public let startTag: String? = "<\(marker)tool_calls>"

    /// The tag that closes a DSML tool-call block.
    public let endTag: String? = "</\(marker)tool_calls>"

    /// Creates a new DSML tool call parser.
    public init() {}

    /// Parses `content` into a single ``ToolCall`` — the first invoke found.
    ///
    /// - Parameters:
    ///   - content: The text content to parse (may include the block tags).
    ///   - tools: Unused — the `string="true|false"` attribute types each
    ///     value, thus no schema lookup is necessary.
    /// - Returns: The first parsed ``ToolCall``, or `nil` when `content`
    ///   contains no invoke element.
    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        Self.extractToolCalls(from: content).first
    }

    /// Overrides the default EOS split (which only recognizes `startTag`
    /// boundaries) because DSML wraps every invocation of a round in ONE
    /// `<｜DSML｜tool_calls>` block: recovering a parallel round from
    /// leftover buffered content means walking each invoke inside that one
    /// wrapper, not splitting on repeated wrapper tags.
    ///
    /// - Parameters:
    ///   - toolCallBuffer: The text remaining in the tool-call buffer when
    ///     generation ended.
    ///   - tools: Unused — see ``parse(content:tools:)``.
    /// - Returns: The tool calls recovered from `toolCallBuffer`, in the
    ///   order they appear.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        Self.extractToolCalls(from: toolCallBuffer)
    }

    /// Walks every `<｜DSML｜invoke name="...">...</｜DSML｜invoke>` element
    /// of `content` and parses each one into a call.
    private static func extractToolCalls(from content: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var searchRange = content.startIndex ..< content.endIndex
        while let open = content.range(of: invokeOpen, range: searchRange) {
            guard
                let headerEnd = content.range(of: ">", range: open.upperBound ..< content.endIndex),
                let close = content.range(
                    of: invokeClose, range: headerEnd.upperBound ..< content.endIndex)
            else { break }

            let name = extractName(String(content[open.upperBound ..< headerEnd.lowerBound]))
            let body = String(content[headerEnd.upperBound ..< close.lowerBound])
            searchRange = close.upperBound ..< content.endIndex

            guard !name.isEmpty else { continue }
            let elements = parameters(in: body)
            calls.append(
                ToolCall(
                    function: .init(
                        name: name, arguments: arguments(of: elements),
                        argumentsJSON: argumentsJSON(of: elements))))
        }
        return calls
    }

    /// One `<｜DSML｜parameter>` element of an invoke.
    private struct Parameter {
        /// The name of the parameter.
        let name: String
        /// The value, in the Swift form that ``ToolCall`` carries.
        let value: any Sendable
        /// The value, in the JSON form that keeps the text of the model.
        let json: PythonStyleJSON
    }

    /// Reads the parameter elements of one invoke body.
    ///
    /// - Parameter body: the text between the two invoke tags.
    /// - Returns: the parameters, in the order the model wrote them.
    private static func parameters(in body: String) -> [Parameter] {
        var parameters: [Parameter] = []
        var searchRange = body.startIndex ..< body.endIndex
        while let open = body.range(of: parameterOpen, range: searchRange) {
            guard
                let headerEnd = body.range(of: ">", range: open.upperBound ..< body.endIndex),
                let close = body.range(
                    of: parameterClose, range: headerEnd.upperBound ..< body.endIndex)
            else { break }

            let header = String(body[open.upperBound ..< headerEnd.lowerBound])
            let value = String(body[headerEnd.upperBound ..< close.lowerBound])
            searchRange = close.upperBound ..< body.endIndex

            let element = parameter(header: header, value: value)
            guard !element.name.isEmpty else { continue }
            parameters.append(element)
        }
        return parameters
    }

    /// The arguments of one call, by name.
    ///
    /// A name that the model wrote twice keeps the value that comes last, which
    /// is what a Swift `Dictionary` does.
    ///
    /// - Parameter parameters: the parameters of the call.
    /// - Returns: the arguments.
    private static func arguments(of parameters: [Parameter]) -> [String: any Sendable] {
        var arguments: [String: any Sendable] = [:]
        for parameter in parameters {
            arguments[parameter.name] = parameter.value
        }
        return arguments
    }

    /// The arguments of one call, as JSON text in the order of the model.
    ///
    /// ``DeepSeekV4ChatEncoder`` writes one DSML parameter for each member of
    /// this text, in the order of the text, thus a replayed call reproduces the
    /// parameter order the model wrote.
    ///
    /// - Parameter parameters: the parameters of the call.
    /// - Returns: the JSON text.
    private static func argumentsJSON(of parameters: [Parameter]) -> String {
        PythonStyleJSON.object(
            parameters.map { PythonStyleJSON.Member(key: $0.name, value: $0.json) }
        ).pythonStyleText
    }

    /// Splits one parameter header into its name and its converted value.
    ///
    /// `string="true"` keeps the value as literal text, and `string="false"`
    /// decodes it as JSON. A header without the attribute — which the
    /// reference never writes — falls back to the same JSON decode the
    /// `false` flag asks for, so a bare number still comes through typed.
    ///
    /// - Parameters:
    ///   - header: the text between the parameter tag name and its `>`.
    ///   - value: the text between the two parameter tags.
    /// - Returns: the parameter.
    private static func parameter(header: String, value: String) -> Parameter {
        guard let attribute = header.range(of: stringAttribute) else {
            return decoded(name: extractName(header), value: value)
        }
        let name = extractName(String(header[..<attribute.lowerBound]))
        let isString = extractName(String(header[attribute.upperBound...])) == "true"
        return isString
            ? Parameter(name: name, value: value, json: .string(value))
            : decoded(name: name, value: value)
    }

    /// Decodes one `string="false"` value as JSON.
    ///
    /// `.fragmentsAllowed` accepts the bare scalars the format carries
    /// (`3`, `true`) beside objects and arrays. Text that is not JSON comes
    /// back as it is, the same fallback the upstream parser uses.
    ///
    /// ``PythonStyleJSON/parse(_:)`` keeps the digits of each number as the
    /// model wrote them, thus it reads the JSON form first. Text that only
    /// `JSONSerialization` reads takes the JSON form of the decoded value, thus
    /// the two forms of the parameter always carry the same value.
    ///
    /// - Parameters:
    ///   - name: the name of the parameter.
    ///   - value: the text between the two parameter tags.
    /// - Returns: the parameter.
    private static func decoded(name: String, value: String) -> Parameter {
        guard let data = value.data(using: .utf8),
            let decoded = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
        else { return Parameter(name: name, value: value, json: .string(value)) }
        let argument = asSendable(decoded)
        return Parameter(
            name: name, value: argument,
            json: PythonStyleJSON.parse(value) ?? PythonStyleJSON(sendable: argument))
    }
}
