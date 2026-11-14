// Copyright © 2025 Apple Inc.

import Foundation

/// Parser for the GLM4 tool-call dialects.
///
/// GLM-4.5 and later frame a call as
/// `<tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
/// GLM-4-0414 writes the function name on its own line followed by a JSON
/// arguments object, with no delimiters at all. `ToolCallProcessor` anchors
/// that markerless dialect on the declared tools before it reaches this parser.
///
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
public struct GLM4ToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<tool_call>"
    public let endTag: String? = "</tool_call>"
    public var supportsMarkerlessNamedJSON: Bool { true }

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        // Strip wrapper tags only at the boundaries so literal tag strings in
        // JSON arguments remain unchanged.
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if let start = startTag, text.hasPrefix(start) {
            text = String(text.dropFirst(start.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let end = endTag, text.hasSuffix(end) {
            text = String(text.dropLast(end.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Extract function name (everything before first <arg_key>)
        guard let argKeyStart = text.range(of: "<arg_key>") else {
            return parseNamedJSON(text, tools: tools)
        }
        let funcName = String(text[..<argKeyStart.lowerBound]).trimmingCharacters(
            in: .whitespacesAndNewlines)

        guard !funcName.isEmpty else { return nil }

        var arguments: [String: any Sendable] = [:]

        // Find all arg_key/arg_value pairs
        var searchRange = text.startIndex ..< text.endIndex
        while let keyStart = text.range(of: "<arg_key>", range: searchRange) {
            // Find </arg_key>
            guard
                let keyEnd = text.range(
                    of: "</arg_key>", range: keyStart.upperBound ..< text.endIndex)
            else { break }

            let key = String(text[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // Find <arg_value> after </arg_key>
            guard
                let valueStart = text.range(
                    of: "<arg_value>", range: keyEnd.upperBound ..< text.endIndex)
            else { break }

            // Find </arg_value>
            guard
                let valueEnd = text.range(
                    of: "</arg_value>", range: valueStart.upperBound ..< text.endIndex)
            else { break }

            let value = String(text[valueStart.upperBound ..< valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            // GLM4: deserialize if NOT a string type in schema
            func typed(_ name: String, _ value: String) -> any Sendable {
                isStringType(funcName: funcName, argName: name, tools: tools)
                    ? value : tryParseJSON(value) ?? value
            }

            if let nested = Self.nestedPairs(in: value) {
                // The value wraps the real arguments of the call.
                for (name, nestedValue) in nested {
                    arguments[name] = typed(name, nestedValue)
                }
            } else {
                arguments[key] = typed(key, value)
            }

            searchRange = valueEnd.upperBound ..< text.endIndex
        }

        return ToolCall(function: .init(name: funcName, arguments: arguments))
    }

    /// The argument pairs nested inside `value`, or `nil` when it nests none.
    ///
    /// Flint can open a JSON object inside one value and then write the
    /// arguments with GLM tags, as in
    /// `{"op</arg_key><arg_value>use skill","id</arg_key><arg_value>kanban"}`,
    /// or nest whole tagged pairs. Those pairs are the arguments of the call.
    /// The recovery applies only when every key it finds is an identifier, so
    /// free text that mentions the tags stays as it is.
    ///
    /// - Parameter value: an argument value.
    /// - Returns: the nested pairs in order, or `nil`.
    static func nestedPairs(in value: String) -> [(String, String)]? {
        guard value.contains("</arg_key>") else { return nil }
        // The separator of a key and its value inside a value.
        let nestedSeparator = /<\/arg_key>\s*<arg_value>/
        // What ends a nested value and starts the next key: GLM tags, or the
        // `","` of a JSON object.
        let nestedBoundary = /<\/arg_value>\s*<arg_key>|"\s*,\s*"/
        // A key that a nested pair may have.
        let nestedKey = /[A-Za-z_][A-Za-z0-9_.\-]{0,63}/

        let parts = value.split(separator: nestedSeparator, omittingEmptySubsequences: false)
        guard parts.count >= 2 else { return nil }

        let jsonOpened = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{")
        var keys = [String(parts[0])]
        var values: [String] = []
        for part in parts.dropFirst().dropLast() {
            guard let boundary = part.matches(of: nestedBoundary).last else { return nil }
            values.append(String(part[..<boundary.range.lowerBound]))
            keys.append(String(part[boundary.range.upperBound...]))
        }
        values.append(String(parts[parts.count - 1]))

        func cleanKey(_ raw: String) -> String? {
            var key = Substring(raw.trimmingCharacters(in: .whitespacesAndNewlines))
            for prefix in ["{", "<arg_key>", "\""] where key.hasPrefix(prefix) {
                key = key.dropFirst(prefix.count)
                key = Substring(key.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            if key.hasSuffix("\"") { key = key.dropLast() }
            return key.wholeMatch(of: nestedKey) == nil ? nil : String(key)
        }

        var lastValue = values[values.count - 1].trimmingCharacters(in: .whitespacesAndNewlines)
        if lastValue.hasSuffix("</arg_value>") {
            lastValue = String(lastValue.dropLast("</arg_value>".count))
        }
        if jsonOpened, lastValue.hasSuffix("}") {
            lastValue = String(lastValue.dropLast()).trimmingCharacters(in: .whitespaces)
            if lastValue.hasSuffix("\"") { lastValue = String(lastValue.dropLast()) }
        }
        values[values.count - 1] = lastValue

        var pairs: [(String, String)] = []
        for (rawKey, rawValue) in zip(keys, values) {
            guard let key = cleanKey(rawKey) else { return nil }
            pairs.append((key, rawValue.trimmingCharacters(in: .whitespacesAndNewlines)))
        }
        return pairs
    }

    /// Parses `name\n{json}`. The payload must be exactly one call and, when a
    /// schema is supplied, name one of its tools.
    private func parseNamedJSON(_ text: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard let call = NamedJSONCallScanner.split(text[...]),
            isDeclaredTool(String(call.name), tools: tools),
            let arguments = tryParseJSON(String(call.arguments)) as? [String: any Sendable]
        else { return nil }
        return ToolCall(function: .init(name: String(call.name), arguments: arguments))
    }
}
