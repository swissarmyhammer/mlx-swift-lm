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

    /// The error of a render whose tool call has arguments that are not a JSON object.
    public struct ArgumentsAreNotJSON: Error, Equatable {
        /// The name of the tool of the call.
        public let toolName: String

        /// Makes the error of a call of `toolName`.
        ///
        /// - Parameter toolName: the name of the tool of the call.
        public init(toolName: String) {
            self.toolName = toolName
        }
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

    /// The token of the thinking-effort text at `level`, ``effortPrefix`` followed by `level`.
    ///
    /// - Parameter level: an effort level, such as ``defaultEffort`` or
    ///   ``InklingReasoningProtocol/thinkingOffEffort``.
    /// - Returns: its token, or ``unknownTokenID`` when the vocabulary does not hold it.
    public func effortToken(_ level: String) -> Int {
        token(Self.effortPrefix + level)
    }

    /// The text of the tool call that the published template writes: compact JSON with sorted
    /// keys.
    ///
    /// - Parameters:
    ///   - name: the name of the tool.
    ///   - arguments: the arguments of the call.
    /// - Returns: the JSON text of the payload.
    /// - Throws: ``ArgumentsAreNotJSON`` when `arguments` is not a JSON object, or the error of
    ///   `JSONSerialization`.
    private static func renderedCall(name: String, arguments: [String: any Sendable]) throws
        -> String
    {
        guard JSONSerialization.isValidJSONObject(arguments) else {
            throw ArgumentsAreNotJSON(toolName: name)
        }
        let data = try JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys])
        return #"{"name":"\#(name)","args":"# + String(decoding: data, as: UTF8.self) + "}"
    }

    /// The vocabulary has no beginning-of-sequence token.
    public var bosToken: String? { nil }

    /// The end of the turn of the model, `<|content_model_end_sampling|>`.
    public var eosToken: String? { Marker.endSampling.rawValue }

    /// The vocabulary names no unknown token as text; ``unknownTokenID`` stands for it.
    public var unknownToken: String? { nil }

    /// The tokens of `text`: one token for each marker, as a real tokenizer encodes each special
    /// token, and one token for each text between two markers, its piece or ``unknownTokenID``.
    ///
    /// - Parameters:
    ///   - text: markers and pieces of the vocabulary.
    ///   - addSpecialTokens: ignored, the vocabulary has no special tokens to add.
    /// - Returns: the tokens of `text`.
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            if let marker = Marker.allCases.first(where: { rest.hasPrefix($0.rawValue) }) {
                tokens.append(token(marker))
                rest = rest.dropFirst(marker.rawValue.count)
                continue
            }
            let pieceEnd =
                Marker.allCases.compactMap { rest.range(of: $0.rawValue)?.lowerBound }.min()
                ?? rest.endIndex
            tokens.append(token(String(rest[..<pieceEnd])))
            rest = rest[pieceEnd...]
        }
        return tokens
    }

    /// The joined text of `tokenIds`, with an empty text for a token outside the vocabulary.
    ///
    /// - Parameters:
    ///   - tokenIds: the tokens to decode.
    ///   - skipSpecialTokens: ignored, every marker decodes to its text.
    /// - Returns: the text of the tokens.
    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { convertIdToToken($0) ?? "" }.joined()
    }

    /// The token of the marker or piece `token`, or nil outside the vocabulary.
    ///
    /// - Parameter token: a marker or a piece.
    /// - Returns: its token, or nil.
    public func convertTokenToId(_ token: String) -> Int? {
        if let marker = Marker(rawValue: token) {
            return self.token(marker)
        }
        return tokenOfPiece[token]
    }

    /// The marker or piece of `id`, or nil for the unknown token and outside the vocabulary.
    ///
    /// - Parameter id: a token.
    /// - Returns: its text, or nil.
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
            tokens += try render(message, role: role, in: messages)
        }
        return wroteEffort ? tokens : tokens + effort
    }

    /// The frames of one message.
    private func render(
        _ message: [String: any Sendable], role: String?, in messages: [[String: any Sendable]]
    ) throws -> [Int] {
        let content = Self.text(of: message["content"])
        switch role {
        case "system":
            return frame(.messageSystem, .contentText, payload: content)
        case "tool":
            let name = message["name"] as? String ?? toolName(of: message, in: messages)
            return frame(.messageTool, header: name, .contentText, payload: content)
        case "assistant":
            return try assistantFrames(message, content: content)
        default:
            return frame(.messageUser, .contentText, payload: content)
        }
    }

    /// The frames of an assistant message: its reasoning, its text, its tool calls, then the end
    /// of its turn.
    private func assistantFrames(_ message: [String: any Sendable], content: String) throws
        -> [Int]
    {
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
                payload: try Self.renderedCall(name: name, arguments: arguments))
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
