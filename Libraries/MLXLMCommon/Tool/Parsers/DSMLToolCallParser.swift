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
///
/// ## Tolerance of the short invoke closing tag — card `^z5xrzg6`
///
/// This parser accepts ONE tag that the DSML syntax does not state:
/// `</｜DSML｜inv>`. That is a DELIBERATE divergence from the published
/// `parse_tool_calls`, which accepts the one syntax and raises on every other.
/// Do NOT "correct" it back before you read card `^z5xrzg6`.
///
/// The reason is measured, not supposed. With the real weights of
/// `mlx-community/DeepSeek-V4-Flash-4bit` and greedy decoding, the model
/// deterministically drops identifier 5406 (`oke`) from the closing tag of the
/// invoke element and writes `</｜DSML｜inv>`. At the losing step the winner
/// `>\n` (1018) stands at logit 28.25 and `oke` at 21.25 — a gap of 7.0 logits,
/// which repeats identifier for identifier across runs AND across processes. No
/// change of this port corrects it, thus a tool round can never complete while
/// the parser holds the one literal.
///
/// The rule this tolerance follows: **accept a closing tag whose name is a
/// prefix of the published name, cut at a token boundary of the published
/// tokenization of that name.** The published `tokenizer.json` writes each name
/// as:
///
/// | name | pieces | prefixes at a token boundary |
/// | --- | --- | --- |
/// | `invoke` | `inv` (40148), `oke` (5406) | `inv` |
/// | `tool_calls` | `tool` (72461), `_c` (4941), `alls` (12548) | `tool`, `tool_c` |
/// | `parameter` | `parameter` (41523) | none |
///
/// Thus the rule gives exactly one extra literal for `invoke`, and that literal
/// is exactly the string the weights write. The rule stops at the token
/// boundary on purpose: the vocabulary also holds `i` (75) and `in` (261), thus
/// `</｜DSML｜in>` is a legal output of a sampler, but it is a DIFFERENT word
/// rather than a lost piece of the published tokenization. This parser refuses
/// it, and it refuses every other text.
///
/// The rule gives nothing for `parameter`, whose name is one token. It gives two
/// literals for `tool_calls`, and neither is necessary here: ``extractToolCalls``
/// reads the closing tag of the BLOCK never — it walks the invoke elements alone
/// — and the only reader of ``endTag`` is `ToolCallProcessor`, which uses it to
/// flush its streaming buffer. A short block closing tag thus holds the buffer
/// to the end of generation, where ``parseEOS(_:tools:)`` recovers every call.
public struct DSMLToolCallParser: ToolCallParser, Sendable {

    /// The `｜DSML｜` marker: `DSML` between two FULLWIDTH VERTICAL LINE
    /// (U+FF5C) delimiters. It carries no angle brackets of its own — each
    /// tag literal below puts them around it.
    private static let marker = "\u{FF5C}DSML\u{FF5C}"

    /// The opening of one invoke element, up to its `name=` attribute.
    private static let invokeOpen = "<\(marker)invoke name="

    /// The closing tag of one invoke element.
    private static let invokeClose = "</\(marker)invoke>"

    /// The closing tag of one invoke element without the second piece of its
    /// name.
    ///
    /// The published tokenizer writes `invoke` as `inv` (40148) and `oke`
    /// (5406), thus this is the ONE prefix of that name at a token boundary,
    /// and it is the tag the real weights write. See the "Tolerance" section of
    /// the type documentation and card `^z5xrzg6`.
    private static let shortInvokeClose = "</\(marker)inv>"

    /// Every closing tag of an invoke element this parser accepts.
    private static let invokeCloseTags = [invokeClose, shortInvokeClose]

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

    /// The first closing tag of an invoke element that `content` holds at or
    /// after `start`.
    ///
    /// The search reads every tag of ``invokeCloseTags`` and keeps the one that
    /// comes first. No accepted tag is a substring of another, thus well-formed
    /// text always answers with ``invokeClose``.
    ///
    /// - Parameters:
    ///   - content: the text to search.
    ///   - start: where the search starts.
    /// - Returns: the range of the tag, or `nil` when the text holds none.
    private static func invokeCloseRange(
        in content: String, from start: String.Index
    ) -> Range<String.Index>? {
        invokeCloseTags
            .compactMap { content.range(of: $0, range: start ..< content.endIndex) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    /// Walks every `<｜DSML｜invoke name="...">...</｜DSML｜invoke>` element
    /// of `content` and parses each one into a call.
    private static func extractToolCalls(from content: String) -> [ToolCall] {
        var calls: [ToolCall] = []
        var searchRange = content.startIndex ..< content.endIndex
        while let open = content.range(of: invokeOpen, range: searchRange) {
            guard
                let headerEnd = content.range(of: ">", range: open.upperBound ..< content.endIndex),
                let close = invokeCloseRange(in: content, from: headerEnd.upperBound)
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
