// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A tokenizer that writes the message frames of Inkling 2 with a vocabulary of a few dozen
/// tokens, for weight-free tests of a tool loop.
///
/// Each control marker of the Inkling protocol is one token, and each text of the test is one
/// token of its own: the test names its texts as `pieces`. The chat template writes the frames
/// the published template writes:
///
/// - the tool declaration, one system frame, when the turn declares tools;
/// - the thinking effort, one system frame before the first message that is not a system
///   message, whose level comes from the `reasoning_effort` template variable;
/// - each message as `<|message_role|>header<|content_kind|>payload<|end_message|>`, the
///   reasoning of an assistant message in a `<|content_thinking|>` frame before its text, each
///   tool call in a `<|content_invoke_tool_json|>` frame, and `<|content_model_end_sampling|>`
///   at the end of each assistant message.
///
/// A tool call renders as the published template writes it: compact JSON with sorted keys. A
/// model writes the same call with spaces, thus the render of a past tool call differs from the
/// tokens the model wrote, as it does on the real checkpoint.
///
/// The `ChatSession` tests of `MLXLMTests` and the executor tests of
/// `MLXFoundationModelsTests` share it.
public struct InklingFrameTokenizer: Tokenizer {

    /// The control markers of the Inkling protocol, one token each.
    public enum Marker: String, CaseIterable, Sendable {
        /// Opens a user message.
        case messageUser = "<|message_user|>"
        /// Opens a frame of the model.
        case messageModel = "<|message_model|>"
        /// Opens a system message.
        case messageSystem = "<|message_system|>"
        /// Opens a tool result.
        case messageTool = "<|message_tool|>"
        /// Opens a text payload.
        case contentText = "<|content_text|>"
        /// Opens a thinking payload.
        case contentThinking = "<|content_thinking|>"
        /// Opens a JSON tool-call payload.
        case contentToolJSON = "<|content_invoke_tool_json|>"
        /// Ends every frame.
        case endMessage = "<|end_message|>"
        /// Ends the turn of the model.
        case endSampling = "<|content_model_end_sampling|>"
        /// Stands for the whole tool declaration, which a test never reads.
        case toolDeclaration = "<|tool_declaration|>"
    }

    /// The text of the thinking-effort frame before its level.
    public static let effortPrefix = "Thinking effort level: "

    /// The level the published template writes when no `reasoning_effort` is set.
    public static let defaultEffort = "0.9"

    /// The texts of the vocabulary after the markers, in the order of their tokens.
    private let pieces: [String]

    /// The token of each text of ``pieces``.
    private let tokenOfPiece: [String: Int]

    /// Makes a tokenizer whose vocabulary holds the markers, the effort levels
    /// ``defaultEffort`` and ``InklingReasoningProtocol/thinkingOffEffort``, and `pieces`.
    ///
    /// - Parameter pieces: the texts of the test, one token each.
    public init(pieces: [String]) {
        let effortPieces = [Self.defaultEffort, InklingReasoningProtocol.thinkingOffEffort].map {
            Self.effortPrefix + $0
        }
        var unique: [String] = []
        for piece in effortPieces + pieces where !unique.contains(piece) {
            unique.append(piece)
        }
        self.pieces = unique
        self.tokenOfPiece = Dictionary(
            uniqueKeysWithValues: unique.enumerated().map { ($1, Marker.allCases.count + $0) })
    }

    /// The token that stands for a text outside the vocabulary.
    public var unknownTokenID: Int { Marker.allCases.count + pieces.count }

    /// The number of tokens: the markers, the pieces and the unknown token.
    public var vocabularySize: Int { unknownTokenID + 1 }

    /// The token of `marker`.
    ///
    /// - Parameter marker: a control marker.
    /// - Returns: its token.
    public func token(_ marker: Marker) -> Int {
        Marker.allCases.firstIndex(of: marker) ?? unknownTokenID
    }

    /// The token of `piece`, or ``unknownTokenID`` when the vocabulary does not hold it.
    ///
    /// - Parameter piece: a text of the vocabulary.
    /// - Returns: its token.
    public func token(_ piece: String) -> Int {
        tokenOfPiece[piece] ?? unknownTokenID
    }

    /// The text of the tool call that the published template writes: compact JSON with sorted
    /// keys.
    ///
    /// - Parameters:
    ///   - name: the name of the tool.
    ///   - arguments: the arguments of the call.
    /// - Returns: the JSON text of the payload.
    public static func renderedCall(name: String, arguments: [String: any Sendable]) -> String {
        let data =
            (try? JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys]))
            ?? Data("{}".utf8)
        return #"{"name":"\#(name)","args":"# + String(decoding: data, as: UTF8.self) + "}"
    }

    public var bosToken: String? { nil }
    public var eosToken: String? { Marker.endSampling.rawValue }
    public var unknownToken: String? { nil }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let marker = Marker(rawValue: text) {
            return [token(marker)]
        }
        return [token(text)]
    }

    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { convertIdToToken($0) ?? "" }.joined()
    }

    public func convertTokenToId(_ token: String) -> Int? {
        if let marker = Marker(rawValue: token) {
            return self.token(marker)
        }
        return tokenOfPiece[token]
    }

    public func convertIdToToken(_ id: Int) -> String? {
        if Marker.allCases.indices.contains(id) {
            return Marker.allCases[id].rawValue
        }
        let pieceIndex = id - Marker.allCases.count
        return pieces.indices.contains(pieceIndex) ? pieces[pieceIndex] : nil
    }

    /// The tool declaration, then each message, with the thinking effort before the first
    /// message that is not a system message.
    public func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let level =
            additionalContext?[InklingReasoningProtocol.effortKey] as? String ?? Self.defaultEffort
        let effort = frame(.messageSystem, .contentText, payload: Self.effortPrefix + level)
        var tokens: [Int] = []
        if tools?.isEmpty == false {
            tokens += [token(.messageSystem), token(.toolDeclaration), token(.endMessage)]
        }
        var wroteEffort = false
        for message in messages {
            let role = message["role"] as? String
            if !wroteEffort, role != "system" {
                tokens += effort
                wroteEffort = true
            }
            tokens += render(message, role: role, in: messages)
        }
        return wroteEffort ? tokens : tokens + effort
    }

    /// The frames of one message.
    private func render(
        _ message: [String: any Sendable], role: String?, in messages: [[String: any Sendable]]
    ) -> [Int] {
        let content = Self.text(of: message["content"])
        switch role {
        case "system":
            return frame(.messageSystem, .contentText, payload: content)
        case "tool":
            let name = message["name"] as? String ?? toolName(of: message, in: messages)
            return frame(.messageTool, header: name, .contentText, payload: content)
        case "assistant":
            return assistantFrames(message, content: content)
        default:
            return frame(.messageUser, .contentText, payload: content)
        }
    }

    /// The frames of an assistant message: its reasoning, its text, its tool calls, then the end
    /// of its turn.
    private func assistantFrames(_ message: [String: any Sendable], content: String) -> [Int] {
        let reasoning = message["reasoning_content"] as? String ?? ""
        let calls = message["tool_calls"] as? [[String: any Sendable]] ?? []
        var tokens: [Int] = []
        if !reasoning.isEmpty {
            tokens += frame(.messageModel, .contentThinking, payload: reasoning)
        }
        if !content.isEmpty || (calls.isEmpty && reasoning.isEmpty) {
            tokens += frame(.messageModel, .contentText, payload: content)
        }
        for call in calls {
            let function = call["function"] as? [String: any Sendable] ?? [:]
            let name = function["name"] as? String ?? ""
            let arguments = function["arguments"] as? [String: any Sendable] ?? [:]
            tokens += frame(
                .messageModel, header: name, .contentToolJSON,
                payload: Self.renderedCall(name: name, arguments: arguments))
        }
        return tokens + [token(.endSampling)]
    }

    /// The text of a message content: the content itself, or the joined text of its parts when
    /// a message generator writes the content as a list of parts.
    private static func text(of content: (any Sendable)?) -> String {
        if let parts = content as? [[String: String]] {
            return parts.compactMap { $0["text"] }.joined()
        }
        return content as? String ?? ""
    }

    /// The name of the tool whose call `message` answers, found by its call identifier.
    private func toolName(
        of message: [String: any Sendable], in messages: [[String: any Sendable]]
    ) -> String {
        let id = message["tool_call_id"] as? String
        let calls = messages.flatMap { $0["tool_calls"] as? [[String: any Sendable]] ?? [] }
        let call = calls.first { $0["id"] as? String == id }
        let function = call?["function"] as? [String: any Sendable]
        return function?["name"] as? String ?? ""
    }

    /// One frame: the role marker, the header, the kind marker, the payload, then the end.
    private func frame(
        _ role: Marker, header: String = "", _ kind: Marker, payload: String
    ) -> [Int] {
        let headerTokens = header.isEmpty ? [] : [token(header)]
        return [token(role)] + headerTokens + [token(kind), token(payload), token(.endMessage)]
    }
}
