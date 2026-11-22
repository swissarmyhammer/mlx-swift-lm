// Copyright © 2026 Apple Inc.
//
// One turn of an Inkling 2 chat, collected from its generation stream, shared by the Inkling 2
// checkpoint and MTP suites.

import Foundation
import MLXLMCommon

/// The text, the reasoning, the tool calls and the completion info of one turn.
struct Inkling2Turn {
    var text = ""
    var reasoning = ""
    var toolCalls: [ToolCall] = []
    var info: GenerateCompletionInfo?

    /// Drains `stream` into a turn.
    static func collect(_ stream: AsyncThrowingStream<Generation, Error>) async throws
        -> Inkling2Turn
    {
        var turn = Inkling2Turn()
        for try await event in stream {
            switch event {
            case .chunk(let chunk): turn.text += chunk
            case .reasoning(let chunk): turn.reasoning += chunk
            case .toolCall(let call): turn.toolCalls.append(call)
            case .info(let info): turn.info = info
            default: break
            }
        }
        return turn
    }

    /// One line that names the token counts and rates of the turn.
    var summary: String {
        guard let info else { return "no completion info" }
        return
            "prompt \(info.promptTokenCount) tok (cached \(info.cachedPromptTokenCount)), "
            + "prefill \(String(format: "%.1f", info.promptTokensPerSecond)) tok/s, "
            + "generated \(info.generationTokenCount) tok, "
            + "decode \(String(format: "%.1f", info.tokensPerSecond)) tok/s"
    }
}
