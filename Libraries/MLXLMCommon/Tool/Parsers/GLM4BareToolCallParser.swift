// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for the original (pre-4.7, non-MoE) GLM-4 tool call format: a bare
/// function-name line followed by a bare JSON object of just the arguments,
/// with no wrapper tags and no `name`/`arguments` JSON envelope.
///
/// Example: `get_weather\n{"location": "Paris", "unit": "celsius"}`
///
/// Checkpoints such as `mlx-community/GLM-4-9B-0414-4bit` predate GLM-4.7 and
/// were never taught the `<tool_call>`/`<arg_key>` envelope that
/// ``GLM4ToolCallParser`` implements (see that type's reference to
/// `mlx_lm/tool_parsers/glm47.py`) -- their own chat template just prints the
/// function name and full JSON schema, then instructs the model (in Chinese)
/// to represent the call's arguments in JSON.
///
/// Because there is no literal marker preceding the function name, this
/// format cannot be reliably detected mid-stream the way tagged formats are:
/// ``buffersEntireResponse`` is `true`, so ``ToolCallProcessor`` defers all
/// parsing to end-of-sequence rather than trying to split live token chunks.
public struct GLM4BareToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = nil
    public let endTag: String? = nil
    public let buffersEntireResponse = true

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        let text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, let braceIndex = text.firstIndex(of: "{") else { return nil }

        let funcName = String(text[..<braceIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
        let argsText = String(text[braceIndex...]).trimmingCharacters(in: .whitespacesAndNewlines)

        guard !funcName.isEmpty,
            let argsDict = tryParseJSON(argsText) as? [String: any Sendable]
        else { return nil }

        return ToolCall(function: .init(name: funcName, arguments: argsDict))
    }
}
