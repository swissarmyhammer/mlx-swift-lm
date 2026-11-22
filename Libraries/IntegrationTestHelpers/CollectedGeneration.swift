// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// The text, the reasoning, the tool calls and the completion info that the generation stream
/// of a chat turn carried.
///
/// ``collect(_:)`` keeps the whole turn. ``collectPasses(_:)`` keeps one value for each
/// generation pass of the turn, so a tool loop reads what each pass wrote.
public struct CollectedGeneration {
    /// The answer text the stream carried.
    public var text = ""
    /// The reasoning the stream carried, apart from ``text``.
    public var reasoning = ""
    /// The tool calls the stream carried, in order.
    public var toolCalls: [ToolCall] = []
    /// The tool calls the session rejected, in order. They are not in ``toolCalls``.
    public var rejectedToolCalls: [RejectedToolCall] = []
    /// The last completion info the stream carried, or `nil` when it carried none.
    public var info: GenerateCompletionInfo?

    /// Drains `stream` into one value for the whole turn.
    ///
    /// - Parameter stream: The generation stream of one turn.
    /// - Returns: The text, the reasoning and the tool calls of every pass, and the last
    ///   completion info.
    /// - Throws: The error of `stream`.
    public static func collect(_ stream: AsyncThrowingStream<Generation, Error>) async throws
        -> CollectedGeneration
    {
        var turn = CollectedGeneration()
        for try await event in stream { turn.record(event) }
        return turn
    }

    /// Drains `stream` into one value for each completion info it carries. Each completion
    /// info ends a pass; events after the last one belong to no pass.
    ///
    /// - Parameter stream: The generation stream of one turn.
    /// - Returns: The passes of the turn, in order.
    /// - Throws: The error of `stream`.
    public static func collectPasses(_ stream: AsyncThrowingStream<Generation, Error>)
        async throws -> [CollectedGeneration]
    {
        var passes: [CollectedGeneration] = []
        var pass = CollectedGeneration()
        for try await event in stream {
            pass.record(event)
            guard event.info != nil else { continue }
            passes.append(pass)
            pass = CollectedGeneration()
        }
        return passes
    }

    /// One line that names the token counts and the rates of the completion info, and the
    /// number of rejected tool calls when there are any.
    public var summary: String {
        let rejected =
            rejectedToolCalls.isEmpty ? "" : ", \(rejectedToolCalls.count) rejected tool call(s)"
        guard let info else { return "no completion info" + rejected }
        return
            "prompt \(info.promptTokenCount) tok (cached \(info.cachedPromptTokenCount)), "
            + "prefill \(String(format: "%.1f", info.promptTokensPerSecond)) tok/s, "
            + "generated \(info.generationTokenCount) tok, "
            + "decode \(String(format: "%.1f", info.tokensPerSecond)) tok/s" + rejected
    }

    /// Adds the text, the reasoning, the tool call, the rejected tool call or the completion
    /// info of `event`.
    private mutating func record(_ event: Generation) {
        switch event {
        case .chunk(let chunk): text += chunk
        case .reasoning(let chunk): reasoning += chunk
        case .toolCall(let call): toolCalls.append(call)
        case .rejectedToolCall(let rejection): rejectedToolCalls.append(rejection)
        case .info(let completion): info = completion
        case .steering: break
        }
    }
}
