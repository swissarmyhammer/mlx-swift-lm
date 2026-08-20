// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for MiniMax M3's namespaced XML tool-call format.
///
/// Verified against `mlx-community/MiniMax-M3-4bit`'s `chat_template.jinja`
/// and upstream mlx-lm PR #1416 (`mlx_lm/tool_parsers/minimax_m3.py`). Unlike
/// M2's `<parameter name="k">v</parameter>` attribute-style parameters, M3
/// renders every invocation and its arguments through the template's
/// `to_xml` macro, which emits arbitrary `<key>value</key>` children --
/// mappings become nested elements, iterables become `<item>` children, and
/// every synthesized tag is prefixed with the literal namespace token:
///
/// ```
/// ]<]minimax[>[<tool_call>
/// ]<]minimax[>[<invoke name="get_weather">
/// ]<]minimax[>[<location>Paris]<]minimax[>[</location>
/// ]<]minimax[>[</invoke>
/// ]<]minimax[>[</tool_call>
/// ```
///
/// M3 carries no tool schema through this format (unlike M2's
/// `<parameter name="k">`, whose type comes from the tool definition), so
/// every leaf value is JSON-sniffed the same way the reference parser does
/// (see ``coerceScalar(_:)``), and malformed argument XML surfaces an
/// explicit `__parse_error__` argument rather than silently degrading to an
/// empty `{}` -- a phantom-success failure mode the reference parser calls
/// out explicitly.
///
/// Reference: https://github.com/ml-explore/mlx-lm/pull/1416
public struct MiniMaxM3ToolCallParser: ToolCallParser, Sendable {

    /// The literal token M3's chat template prepends to every synthesized
    /// tag start/end (`to_xml` macro and the `<invoke>`/`<tool_call>`
    /// wrapper alike).
    private static let namespaceToken = "]<]minimax[>["

    /// The argument key a malformed or non-object argument payload surfaces
    /// under, mirroring the reference parser's `__parse_error__` sentinel.
    private static let parseErrorKey = "__parse_error__"

    /// The unprefixed XML wrapper tag name for a tool-call block, shared by
    /// ``startTag``, ``endTag``, and the wrapper-narrowing search in
    /// ``extractToolCalls(from:)`` so the literal never drifts between them.
    private static let toolCallTagName = "tool_call"

    /// The XML tag that marks the start of an M3 tool-call block, prefixed
    /// with the namespace token every synthesized tag carries.
    public let startTag: String? = namespaceToken + "<\(toolCallTagName)>"

    /// The XML tag that marks the end of an M3 tool-call block, prefixed
    /// with the namespace token every synthesized tag carries.
    public let endTag: String? = namespaceToken + "</\(toolCallTagName)>"

    /// Creates a new MiniMax M3 tool call parser.
    public init() {}

    /// Parses `content` into a single ``ToolCall`` by extracting the first
    /// `<invoke>` block found, after stripping M3's namespace tokens and any
    /// conversational prefix outside the `<tool_call>` wrapper.
    ///
    /// - Parameters:
    ///   - content: The text content to parse (may include tags).
    ///   - tools: Unused -- M3 carries no tool schema through this format,
    ///     so every leaf value is JSON-sniffed rather than typed from a
    ///     schema (see ``coerceScalar(_:)``).
    /// - Returns: The first parsed ``ToolCall``, or `nil` if `content`
    ///   contains no `<invoke>` block.
    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        Self.extractToolCalls(from: content).first
    }

    /// Overrides the default EOS split (which only recognizes `startTag`
    /// boundaries) because M3 wraps every invocation of a round in a
    /// SINGLE `<tool_call>` block: recovering more than one call from
    /// leftover buffered content means walking each `<invoke>` inside that
    /// one wrapper, not splitting on repeated wrapper tags.
    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        Self.extractToolCalls(from: toolCallBuffer)
    }

    /// Strips M3's namespace tokens, narrows to the `<tool_call>...</tool_call>`
    /// payload when present (dropping any conversational prefix the model
    /// emitted before it), and parses every `<invoke name="...">...</invoke>`
    /// block the payload contains.
    private static func extractToolCalls(from content: String) -> [ToolCall] {
        var text = content.replacingOccurrences(of: namespaceToken, with: "")

        if let start = text.range(of: "<\(toolCallTagName)>"),
            let end = text.range(of: "</\(toolCallTagName)>")
        {
            text = String(text[start.upperBound ..< end.lowerBound])
        }

        var calls: [ToolCall] = []
        var searchRange = text.startIndex ..< text.endIndex
        while let invokeStart = text.range(of: "<invoke name=", range: searchRange) {
            guard
                let nameEnd = text.range(
                    of: ">", range: invokeStart.upperBound ..< text.endIndex),
                let invokeEnd = text.range(
                    of: "</invoke>", range: nameEnd.upperBound ..< text.endIndex)
            else { break }

            let name = extractName(String(text[invokeStart.upperBound ..< nameEnd.lowerBound]))
            let body = String(text[nameEnd.upperBound ..< invokeEnd.lowerBound])
            searchRange = invokeEnd.upperBound ..< text.endIndex

            guard !name.isEmpty else { continue }
            calls.append(
                ToolCall(
                    function: .init(
                        name: name, arguments: parseArguments(body, functionName: name))))
        }
        return calls
    }

    /// Parses an `<invoke>` body into its arguments object.
    ///
    /// - Returns: The parsed `<key>value</key>` children as a dictionary; an
    ///   empty dictionary for a no-argument call (`<invoke name="x"></invoke>`);
    ///   or a single `__parse_error__` entry when the body is malformed XML or
    ///   parses to something other than an object (e.g. a bare `<item>` list).
    private static func parseArguments(_ body: String, functionName: String) -> [String:
        any Sendable]
    {
        guard let value = elementValue(from: body) else {
            return [
                parseErrorKey:
                    "MiniMax-M3 emitted malformed argument XML for tool '\(functionName)'. "
                    + "Re-issue this tool call with each argument as a simple <key>value</key> tag. "
                    + "Raw snippet: \(body.prefix(200))"
            ]
        }
        if let dict = value as? [String: any Sendable] {
            return dict
        }
        if let scalar = value as? String, scalar.isEmpty {
            // Empty body parses to an empty scalar -- a legitimate
            // no-argument call, not a malformed one.
            return [:]
        }
        return [
            parseErrorKey:
                "Arguments for tool '\(functionName)' did not parse to an object. "
                + "Re-issue the call with named <key>value</key> argument tags."
        ]
    }

    /// Recursively converts an XML fragment into a Python-`_element_to_python`-
    /// equivalent value: a leaf scalar when it has no child tags, a list when
    /// every child is `<item>`, or a dictionary keyed by child tag name.
    ///
    /// - Returns: The converted value, or `nil` when `fragment` contains
    ///   unbalanced or unclosed tags (surfaced by the caller as
    ///   `__parse_error__` rather than silently treated as a leaf/empty value).
    private static func elementValue(from fragment: String) -> (any Sendable)? {
        guard let elements = parseElements(in: fragment) else { return nil }

        if elements.isEmpty {
            return coerceScalar(fragment)
        }

        if elements.allSatisfy({ $0.tag == "item" }) {
            var items: [any Sendable] = []
            for element in elements {
                guard let value = elementValue(from: element.body) else { return nil }
                items.append(value)
            }
            return items
        }

        var dict: [String: any Sendable] = [:]
        for element in elements {
            guard let value = elementValue(from: element.body) else { return nil }
            dict[element.tag] = value
        }
        return dict
    }

    /// Scans `fragment` for its top-level sibling elements (`<tag>body</tag>`),
    /// tracking nesting depth by tag name so a child sharing its parent's tag
    /// name (unusual, but not disallowed) does not terminate the match early.
    ///
    /// - Returns: The sibling `(tag, body)` pairs in document order, an empty
    ///   array when `fragment` has no tags at all (a leaf, including ordinary
    ///   scalar text that merely contains a stray `<`/`>`, e.g. `"3 < 5"`),
    ///   or `nil` when a plausible tag is opened but never properly closed
    ///   (malformed XML).
    private static func parseElements(in fragment: String) -> [(tag: String, body: String)]? {
        var results: [(tag: String, body: String)] = []
        var remainder = Substring(fragment)

        while let openStart = remainder.range(of: "<") {
            guard
                let openEnd = remainder.range(
                    of: ">", range: openStart.upperBound ..< remainder.endIndex)
            else {
                // No closing '>' for this '<' at all (e.g. a comparison like
                // "3 < 5" with nothing resembling a tag after it). When no
                // sibling element has parsed successfully yet, this is
                // ordinary scalar text with a stray '<', not a broken tag --
                // fall back to a leaf rather than a false-positive parse
                // error. A '<' appearing after at least one real element has
                // already closed is more plausibly a genuine break in tag
                // structure, so that case still surfaces as malformed.
                return results.isEmpty ? [] : nil
            }

            let tagName = String(remainder[openStart.upperBound ..< openEnd.lowerBound])
            guard isPlausibleTagName(tagName) else {
                return results.isEmpty ? [] : nil
            }

            let openTag = "<\(tagName)>"
            let closeTag = "</\(tagName)>"

            guard
                let closeRange = findClosingTag(
                    in: remainder, openTag: openTag, closeTag: closeTag,
                    searchStart: openEnd.upperBound)
            else { return nil }

            let body = String(remainder[openEnd.upperBound ..< closeRange.lowerBound])
            results.append((tagName, body))
            remainder = remainder[closeRange.upperBound...]
        }

        return results
    }

    /// Finds the range of the closing tag matching an already-consumed
    /// opening `<tagName>`, tracking nesting depth so a child element
    /// sharing its parent's tag name does not terminate the match early.
    ///
    /// Extracted from ``parseElements(in:)`` to keep that function's nesting
    /// shallow: this depth-tracking search is a self-contained sub-problem
    /// (find the matching close for one already-opened tag), not part of
    /// the sibling-scanning loop that calls it.
    ///
    /// - Parameters:
    ///   - remainder: The substring being scanned by ``parseElements(in:)``.
    ///   - openTag: The literal opening tag, e.g. `<location>`.
    ///   - closeTag: The literal closing tag, e.g. `</location>`.
    ///   - searchStart: The index immediately after the already-consumed
    ///     opening tag.
    /// - Returns: The range of the matching closing tag, or `nil` if
    ///   `remainder` never closes the tag (malformed XML).
    private static func findClosingTag(
        in remainder: Substring, openTag: String, closeTag: String,
        searchStart: Substring.Index
    ) -> Range<Substring.Index>? {
        var depth = 1
        var searchStart = searchStart
        while depth > 0 {
            guard
                let nextClose = remainder.range(
                    of: closeTag, range: searchStart ..< remainder.endIndex)
            else { return nil }

            if let nextOpen = remainder.range(
                of: openTag, range: searchStart ..< remainder.endIndex),
                nextOpen.lowerBound < nextClose.lowerBound
            {
                depth += 1
                searchStart = nextOpen.upperBound
            } else {
                depth -= 1
                searchStart = nextClose.upperBound
                if depth == 0 { return nextClose }
            }
        }
        return nil
    }

    /// Whether `name` looks like a tag name the `to_xml` macro would actually
    /// generate (an argument key or `"item"`): non-empty, starting with a
    /// letter or underscore, and otherwise alphanumeric, `_`, or `-`.
    ///
    /// Distinguishes a real tag boundary from a stray `<...>`-shaped
    /// substring inside ordinary scalar text -- e.g. `"3 < 5 > 2"` extracts
    /// the implausible candidate `" 5 "` between its `<` and `>`, which this
    /// rejects so the caller falls back to treating the text as a leaf
    /// rather than misreading it as a broken tag.
    private static func isPlausibleTagName(_ name: String) -> Bool {
        guard let first = name.first, first.isLetter || first == "_" else { return false }
        return name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" }
    }

    /// Mirrors the reference parser's `_coerce_scalar`: JSON-decodes a leaf
    /// value's text when possible, so bare numbers/booleans/null round-trip
    /// as their native types, falling back to the trimmed raw string.
    ///
    /// M3's format carries no tool schema to disambiguate types (unlike M2's
    /// `convertValueWithTypes`), so every leaf is JSON-sniffed unconditionally,
    /// with `.fragmentsAllowed` so a bare scalar (not just an object/array)
    /// is accepted.
    private static func coerceScalar(_ text: String) -> any Sendable {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if let data = trimmed.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        {
            return asSendable(json)
        }
        return trimmed
    }
}
