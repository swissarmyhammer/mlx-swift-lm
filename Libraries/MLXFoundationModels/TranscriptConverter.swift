// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXLMCommon
import os.log

/// Converts FoundationModels transcript entries to MLX chat message format.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
struct TranscriptConverter {

    private static let logger = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "TranscriptConverter")

    /// The MLX `Chat.Message` array for a collection of transcript entries.
    ///
    /// - Parameters:
    ///   - entries: Transcript entries from FoundationModels
    ///   - replayReasoning: whether the text of a `.reasoning` entry rides on
    ///     the assistant message of its turn, as ``Chat/Message/reasoning``.
    ///     The framework appends a turn's reasoning entry AFTER the response
    ///     entry of that turn -- the response entry opens with the turn, the
    ///     reasoning entry opens when the first reasoning fragment arrives --
    ///     and a think-then-call turn puts it BEFORE the tool-calls entry.
    ///     Thus a reasoning entry attaches to the assistant message that
    ///     follows it before any other entry, or, when none follows, to the
    ///     assistant message right before it. Consecutive reasoning entries
    ///     join with newlines, the way the segments of one entry do; a
    ///     reasoning entry with no assistant message on either side is
    ///     dropped. `false` (the default) drops every reasoning entry: the
    ///     answer carries forward, the chain-of-thought does not.
    ///     `ReasoningConfig.replaysReasoningIntoHistory` decides which model
    ///     family replays.
    /// - Returns: Array of MLX Chat.Message objects
    static func mlxMessages(
        for entries: some Collection<Transcript.Entry>, replayReasoning: Bool = false
    ) -> [Chat.Message] {
        var messages: [Chat.Message] = []
        var replay = ReasoningReplay()
        for entry in entries {
            if replayReasoning, case .reasoning(let reasoning) = entry {
                replay.hold(extractText(from: reasoning.segments))
                continue
            }
            guard var message = convert(entry: entry) else { continue }
            replay.settle(before: &message, in: &messages)
            messages.append(message)
            if message.role == .assistant {
                replay.anchor = messages.count - 1
            }
        }
        replay.settle(in: &messages)
        return messages
    }

    /// The reasoning text between two assistant messages, and the assistant
    /// message it falls back to.
    private struct ReasoningReplay {
        /// Reasoning text held until an assistant message takes it.
        var pending: String?

        /// The index in the messages of the assistant message that stands
        /// right before the pending text, or nil when another entry stands
        /// between.
        var anchor: Int?

        /// Holds `text` for the next assistant message.
        mutating func hold(_ text: String?) {
            guard let text else { return }
            pending = pending.map { $0 + "\n" + text } ?? text
        }

        /// Hands the pending text to `message` when it is an assistant
        /// message, and otherwise to the anchored one. Either way, `message`
        /// then stands between the anchor and any later reasoning.
        mutating func settle(before message: inout Chat.Message, in messages: inout [Chat.Message])
        {
            if message.role == .assistant {
                message.reasoning =
                    pending.map { pending in
                        message.reasoning.map { $0 + "\n" + pending } ?? pending
                    } ?? message.reasoning
                pending = nil
            } else {
                settle(in: &messages)
            }
            anchor = nil
        }

        /// Hands the pending text to the anchored assistant message, and drops
        /// it when no assistant message stands right before it.
        mutating func settle(in messages: inout [Chat.Message]) {
            if let anchor, let pending {
                messages[anchor].reasoning =
                    messages[anchor].reasoning.map { $0 + "\n" + pending } ?? pending
            }
            pending = nil
        }
    }

    /// Converts a single transcript entry to its chat message — the per-entry
    /// body of ``mlxMessages(for:replayReasoning:)``, which owns the
    /// cross-entry reasoning state around it.
    ///
    /// - Parameter entry: the entry to convert.
    /// - Returns: the message, or `nil` for an entry that produces none.
    private static func convert(entry: Transcript.Entry) -> Chat.Message? {
        switch entry {
        case .instructions(let instructions):
            // System message for model instructions. Labeled image
            // attachments ride along as message images, mirroring the
            // prompt path, so the `.vision` gate sees them and they are
            // not silently dropped.
            let text = extractText(from: instructions.segments)
            let images = extractImages(from: instructions.segments)
            guard text != nil || !images.isEmpty else {
                logger.warning(
                    "Skipping instructions entry with no text or image content")
                return nil
            }
            return Chat.Message.system(text ?? "", images: images)

        case .prompt(let prompt):
            // User message for prompts. Labeled image attachments
            // (public `.attachment` segments) ride along as message
            // images; text is still concatenated as before.
            let text = extractText(from: prompt.segments)
            let images = extractImages(from: prompt.segments)
            guard text != nil || !images.isEmpty else {
                logger.warning("Skipping prompt entry with no text or image content")
                return nil
            }
            return Chat.Message.user(text ?? "", images: images)

        case .response(let response):
            // Assistant message for previous responses
            guard let text = extractText(from: response.segments) else {
                logger.warning("Skipping response entry with no text content")
                return nil
            }
            return Chat.Message.assistant(text)

        case .reasoning:
            // Prior-turn reasoning is NOT replayed into the model's chat
            // history (per SKILL.md) unless the caller asks for replay, in
            // which case `mlxMessages` consumes the entry BEFORE dispatching
            // here. Dropped explicitly so a future SDK change is reviewed here
            // rather than silently absorbed by the catch-all below.
            logger.debug("Skipping reasoning entry (not replayed into chat history)")
            return nil

        case .toolCalls(let toolCalls):
            // Replay prior tool calls as an assistant message carrying the
            // structured calls. The model's tool-aware chat template renders
            // these into its native tool-call channel; DefaultMessageGenerator
            // serializes each id/name/arguments (see ToolCallIdTests). Without
            // this, a continuation round would re-issue the same call.
            let calls = toolCalls.map { call -> MLXLMCommon.ToolCall in
                let argumentsData = Data(call.arguments.jsonString.utf8)
                let arguments: [String: JSONValue]
                if let decoded = try? JSONDecoder().decode(
                    [String: JSONValue].self, from: argumentsData)
                {
                    arguments = decoded
                } else {
                    logger.warning(
                        "Failed to decode arguments for tool: \(call.toolName, privacy: .public)"
                    )
                    arguments = [:]
                }
                return MLXLMCommon.ToolCall(
                    function: .init(name: call.toolName, arguments: arguments),
                    id: call.id)
            }
            guard !calls.isEmpty else {
                logger.warning("Skipping toolCalls entry with no calls")
                return nil
            }
            return Chat.Message.assistant("", toolCalls: calls)

        case .toolOutput(let output):
            // Replay the tool result as a `tool` message correlated to its
            // originating call by id. Text remains verbatim; structured
            // GeneratedContent is serialized as JSON so the native chat
            // template can expose it to the continuation model turn.
            let content = extractToolOutputContent(from: output.segments)
            return Chat.Message.tool(content, id: output.id)

        default:
            // Skip unsupported entry types. Explicit `return nil` is a
            // tripwire: a newly added SDK entry type surfaces here for review
            // rather than being silently coerced into the wrong role.
            logger.debug("Skipping unsupported entry type")
            return nil
        }
    }

    /// Extracts supported tool-output content in transcript segment order.
    ///
    /// Foundation Models lowers `String` outputs to `.text` and
    /// `GeneratedContent`/`@Generable` outputs to `.structure`. MLX chat
    /// templates accept tool results as strings, so structured values retain
    /// their JSON representation. Attachments and custom segments are deferred
    /// until their media and prompt-representation contracts are implemented.
    private static func extractToolOutputContent(
        from segments: [Transcript.Segment]
    ) -> String {
        segments.compactMap { segment -> String? in
            switch segment {
            case .text(let textSegment):
                return textSegment.content
            case .structure(let structuredSegment):
                return structuredSegment.content.jsonString
            default:
                logger.debug("Skipping unsupported tool-output segment")
                return nil
            }
        }.joined(separator: "\n")
    }

    /// Extracts text content from transcript segments.
    ///
    /// Concatenates all text segments with newlines.
    /// Skips images, structured content, and other non-text segments.
    ///
    /// - Parameter segments: Array of transcript segments
    /// - Returns: Concatenated text, or nil if no text content found
    private static func extractText(from segments: [Transcript.Segment]) -> String? {
        let texts = segments.compactMap { segment -> String? in
            switch segment {
            case .text(let textSegment):
                return textSegment.content

            default:
                // Skip images, structured content, and local attention segment types
                logger.debug("Skipping non-text segment in extractText")
                return nil
            }
        }

        let combined = texts.joined(separator: "\n")
        return combined.isEmpty ? nil : combined
    }

    /// Extracts image inputs from image attachment segments.
    ///
    /// Each image attachment is handed over as its already-decoded
    /// `CIImage`. Segments that carry no image produce no input.
    ///
    /// - Parameter segments: Array of transcript segments
    /// - Returns: The image inputs found, in segment order
    private static func extractImages(from segments: [Transcript.Segment])
        -> [UserInput.Image]
    {
        segments.compactMap { segment -> UserInput.Image? in
            guard case .attachment(let attachment) = segment,
                case .image(let imageAttachment) = attachment.content
            else {
                return nil
            }
            return .ciImage(imageAttachment.ciImage)
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
