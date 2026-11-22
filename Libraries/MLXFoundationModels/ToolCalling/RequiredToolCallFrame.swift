// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon

/// The frame a `.required` tool turn constrains its one tool call to, chosen by the tool-call
/// format of the model.
///
/// A model with a plain text tool-call syntax writes the `<tool_call>` JSON envelope (or the
/// bare envelope) that ``SchemaConverter/encodeToolCallingGrammar(tools:)`` constrains to.
/// A framed token protocol writes its tool calls only inside its own frames, thus the envelope
/// would put the call outside the protocol of the model:
///
/// - Inkling 2 (``ToolCallFormat/inkling``) gets a grammar of its own tool frame, and a
///   think-then-call phase, at every effort, inside a thinking frame that the turn opens for
///   it.
/// - Harmony (``ToolCallFormat/gptOSS``) and Onyx ATEM (``ToolCallFormat/atem``) have no
///   required-mode frame: a required turn of such a model throws
///   ``UnsupportedFormat`` before it generates.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum RequiredToolCallFrame: Equatable {
    /// The `<tool_call>` JSON envelope, `{"name": ..., "arguments": ...}`.
    case toolCallEnvelope

    /// Inkling 2's tool frame,
    /// `<|message_model|>NAME<|content_invoke_tool_json|>{"name": ..., "args": ...}<|end_message|>`.
    case inklingFrame

    /// A framed token protocol that has no required-mode frame.
    struct UnsupportedFormat: Error, Equatable, CustomStringConvertible {
        /// The tool-call format of the model.
        let format: ToolCallFormat

        var description: String {
            "Required tool calling is not supported for the \(format.rawValue) response "
                + "protocol, which writes tool calls only in its own frames. Use the .allowed "
                + "tool-calling mode."
        }
    }

    /// The control markers of an Inkling 2 frame.
    private enum InklingMarker {
        static let messageModel = "<|message_model|>"
        static let thinking = "<|content_thinking|>"
        static let toolJSON = "<|content_invoke_tool_json|>"
        static let endMessage = "<|end_message|>"
    }

    /// The frame of a model whose tool calls use `format`.
    ///
    /// - Parameter format: The tool-call format of the model, or `nil` for the default JSON
    ///   format.
    /// - Throws: ``UnsupportedFormat`` for a framed token protocol without a required-mode
    ///   frame.
    init(format: ToolCallFormat?) throws {
        guard let format else {
            self = .toolCallEnvelope
            return
        }
        switch format {
        case .inkling:
            self = .inklingFrame
        case .gptOSS, .atem:
            throw UnsupportedFormat(format: format)
        case .json, .lfm2, .xmlFunction, .qwen35, .glm4, .gemma, .gemma4, .kimiK2,
            .minimaxM2, .minimaxM3, .miniCPM5, .mistral, .llama3, .dsml:
            self = .toolCallEnvelope
        }
    }

    /// The structural-tag grammar of one call of one of `tools` in this frame.
    ///
    /// - Parameter tools: The tools the call may name; not empty.
    /// - Returns: The structural-tag JSON.
    /// - Throws: The error of the schema conversion.
    func grammar(tools: [Transcript.ToolDefinition]) throws -> String {
        switch self {
        case .toolCallEnvelope:
            try SchemaConverter.encodeToolCallingGrammar(tools: tools)
        case .inklingFrame:
            try SchemaConverter.encodeInklingToolCallingGrammar(tools: tools)
        }
    }

    /// The control markers the turn appends to the prompt, in order, to open the thinking of
    /// the think-then-call phase; empty when the model opens its thinking itself.
    ///
    /// An Inkling 2 turn opens with a frame marker the model chooses: thinking, text or a tool
    /// call. A required turn that reasons first opens the thinking frame for the model, thus
    /// the phase reasons inside the frame and closes it at `<|end_message|>`. Each marker is one
    /// special token, thus each encodes on its own.
    ///
    /// A frame with opener markers thinks before every call, at every effort: Inkling 2 still
    /// writes a thinking frame at the effort `none` before it picks a tool, and a grammar that
    /// opens the tool frame at once takes that step from it (measured on Inkling-2-Mini-4bit:
    /// with the tool frame forced first, the effort `none` called `get_balance` with the example
    /// id of its parameter description before `find_customer`).
    var reasoningOpenerMarkers: [String] {
        switch self {
        case .toolCallEnvelope: []
        case .inklingFrame: [InklingMarker.messageModel, InklingMarker.thinking]
        }
    }

    /// The name and the arguments of the call in `output`, the text the grammar of this frame
    /// constrained, or `nil` when `output` is no call of this frame.
    ///
    /// - Parameter output: The output of the guided generation.
    /// - Returns: The name of the tool, and the arguments of the call, which are `nil` when the
    ///   call names a tool but carries no arguments.
    func call(in output: String) -> (name: String, arguments: Any?)? {
        switch self {
        case .toolCallEnvelope:
            return Self.call(
                inJSON: MLXLanguageModel.Executor.unwrapToolCallMarkers(output),
                argumentsKey: "arguments")
        case .inklingFrame:
            var payload = output.trimmingCharacters(in: .whitespacesAndNewlines)
            guard payload.hasPrefix(InklingMarker.messageModel),
                let open = payload.range(of: InklingMarker.toolJSON)
            else { return nil }
            let header = payload[
                payload.index(
                    payload.startIndex, offsetBy: InklingMarker.messageModel.count)
                    ..< open.lowerBound]
            payload = String(payload[open.upperBound...])
            if payload.hasSuffix(InklingMarker.endMessage) {
                payload.removeLast(InklingMarker.endMessage.count)
            }
            guard let call = Self.call(inJSON: payload, argumentsKey: "args"),
                header.isEmpty || header == call.name
            else { return nil }
            return call
        }
    }

    /// The name and the arguments of a JSON call object, `{"name": ..., <argumentsKey>: ...}`.
    private static func call(inJSON text: String, argumentsKey: String)
        -> (name: String, arguments: Any?)?
    {
        guard
            let object = try? JSONSerialization.jsonObject(with: Data(text.utf8))
                as? [String: Any],
            let name = object["name"] as? String
        else { return nil }
        return (name, object[argumentsKey])
    }
}

#endif
#endif
