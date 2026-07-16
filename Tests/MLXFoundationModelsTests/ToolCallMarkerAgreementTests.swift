// Copyright © 2026 Apple Inc.
//
// Regression tests for kanban task `y4s0w2j`, tool-call marker handling:
//
// 1. Constrain/parse agreement (acceptance criterion): whatever wrapper
//    `SchemaConverter.ToolCallStructuralTag.forFormat(_:)` constrains
//    generation to must be the same one `unwrapToolCallMarkers` extracts,
//    for EVERY inferred `ToolCallFormat` (and `nil`). The live failures
//    happened exactly when these two sides disagreed at runtime.
//
// 2. The malformed-output fallback must never leak `<tool_call>` marker
//    text into a reply: on real hardware, budget-truncated runaway output
//    failed the JSON parse and the raw buffer — Qwen wrapper included —
//    was streamed as the final reply text ("<tool_call>{\"name\":
//    \"runCode\", …" verbatim in composeChain replies).

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

@Suite("Tool-call marker constrain/parse agreement")
struct ToolCallMarkerAgreementTests {

    @Test("unwrapToolCallMarkers extracts the exact wrapper every format constrains to")
    func unwrapExtractsEveryFormatsConstrainedWrapper() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let envelope = #"{"name": "get_weather", "arguments": {"location": "Austin"}}"#

        // Every inferred format, plus nil (no inference), must round-trip:
        // wrap with the tag the grammar uses, then unwrap to the bare
        // envelope the parse side decodes.
        let formats: [ToolCallFormat?] = ToolCallFormat.allCases + [nil]
        for format in formats {
            let tag = SchemaConverter.ToolCallStructuralTag.forFormat(format)
            let wrapped = tag.begin + envelope + tag.end
            let unwrapped = MLXLanguageModel.Executor.unwrapToolCallMarkers(wrapped)
            #expect(
                unwrapped == envelope,
                "constrain and parse sides disagree for format \(String(describing: format))"
            )
        }
    }

    @Test("unwrapToolCallMarkers leaves the bare-JSON grammar arm untouched")
    func unwrapLeavesBareJSONUntouched() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        let envelope = #"{"name": "get_weather", "arguments": {"location": "Austin"}}"#
        #expect(MLXLanguageModel.Executor.unwrapToolCallMarkers(envelope) == envelope)
    }

    @Test("the malformed-output fallback never leaks tool-call marker text into a reply")
    func malformedFallbackNeverLeaksMarkers() {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
        // The live signature: a budget-truncated runaway — wrapper opened,
        // JSON garbage, no closing marker, unparseable.
        let runaway = "<tool_call>\n{\"name\": \"runCode\", \"arguments\": {\"code\": \"const result = findAPI1}7}7}7}7"
        let fallback = MLXLanguageModel.Executor.malformedToolCallFallbackText(runaway)

        #expect(!fallback.contains("<tool_call>"))
        #expect(!fallback.contains("</tool_call>"))
        #expect(
            fallback.contains("runCode"),
            "the fallback must still surface the failure content, got \(fallback)")
    }
}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
