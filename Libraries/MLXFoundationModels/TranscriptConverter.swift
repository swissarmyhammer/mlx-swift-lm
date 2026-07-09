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
        subsystem: mlxFoundationModelsLoggingSubsystem, category: "TranscriptConverter")

    /// The MLX `Chat.Message` array for a collection of transcript entries.
    ///
    /// A single transcript entry can expand to more than one chat message
    /// (a `.toolCalls` entry carries one call per tool the model invoked in
    /// that round), so entries are flat-mapped rather than compact-mapped.
    ///
    /// - Parameter entries: Transcript entries from FoundationModels
    /// - Returns: Array of MLX Chat.Message objects
    static func mlxMessages(for entries: some Collection<Transcript.Entry>) -> [Chat
        .Message]
    {
        entries.flatMap { entry -> [Chat.Message] in
            switch entry {
            case .instructions(let instructions):
                // System message for model instructions. Labeled image
                // attachments ride along as message images, mirroring the
                // prompt path, so the `.vision` gate sees them and they are
                // not silently dropped.
                return makeTextImageMessage(
                    from: instructions.segments,
                    make: { Chat.Message.system($0, images: $1) },
                    emptyWarning: "Skipping instructions entry with no text or image content")

            case .prompt(let prompt):
                // User message for prompts. Labeled image attachments
                // (public `.attachment` segments) ride along as message
                // images; text is still concatenated as before.
                return makeTextImageMessage(
                    from: prompt.segments,
                    make: { Chat.Message.user($0, images: $1) },
                    emptyWarning: "Skipping prompt entry with no text or image content")

            case .response(let response):
                // Assistant message for previous responses. Includes
                // `.structure` segments (like `.toolOutput` below): a prior
                // turn's response may have been a guided/structured
                // generation (a `Generable` result carried as `.structure`,
                // not `.text`), and that content must survive replay into
                // the next turn's prompt rather than being silently dropped.
                guard let text = extractText(from: response.segments, includeStructure: true)
                else {
                    logger.warning("Skipping response entry with no text or structure content")
                    return []
                }
                return [Chat.Message.assistant(text)]

            case .toolCalls(let toolCalls):
                // One assistant message per tool call, each carrying the
                // exact `{"name": ..., "arguments": ...}` envelope text the
                // Executor's own tool-calling grammar generates (see
                // `unwrapToolCallMarkers` in MLXLanguageModel.swift) --
                // replayed verbatim rather than through Chat.Message's
                // structured `toolCalls:` parameter, so a continuation round
                // sees byte-identical history to what it actually produced,
                // not a template's own `tool_calls` re-rendering of it.
                return toolCalls.map { call in
                    Chat.Message.assistant(
                        "{\"\(ToolCallEnvelopeKey.name)\": \(jsonStringLiteral(call.toolName)), "
                            + "\"\(ToolCallEnvelopeKey.arguments)\": \(call.arguments.jsonString)}")
                }

            case .toolOutput(let toolOutput):
                // Tool-role message carrying the executed tool's result,
                // correlated to its call via `id`. Always emitted (even when
                // the tool produced no text) so a continuation round's prompt
                // always differs from the round that made the call --
                // dropping empty outputs would make the two rounds' rendered
                // prompts identical and risk the model repeating the same
                // call forever. Image attachment segments (e.g. a tool that
                // returns a photo) ride along as message images, mirroring
                // the instructions/prompt path, so they are not silently
                // dropped.
                return [
                    Chat.Message.tool(
                        extractToolOutputText(from: toolOutput.segments),
                        images: extractImages(from: toolOutput.segments),
                        id: toolOutput.id)
                ]

            case .reasoning:
                // Prior-turn reasoning is intentionally NOT replayed into the
                // model's chat history (per SKILL.md): the answer carries
                // forward, the chain-of-thought does not. Dropped explicitly so
                // a future SDK change is reviewed here rather than silently
                // absorbed by the catch-all below.
                logger.debug("Skipping reasoning entry (not replayed into chat history)")
                return []

            @unknown default:
                // Skip unrecognized future entry types.
                logger.debug("Skipping unsupported entry type")
                return []
            }
        }
    }

    /// Builds a system/user chat message from a text- and image-bearing
    /// entry (instructions or prompt), sharing the extract-guard-construct
    /// logic the two cases would otherwise duplicate.
    ///
    /// - Parameters:
    ///   - segments: The entry's transcript segments.
    ///   - make: Constructs the role-specific message from the extracted
    ///     text (empty string if none) and images.
    ///   - emptyWarning: Logged when the entry has neither text nor images.
    /// - Returns: A single-element array with the constructed message, or
    ///   an empty array when the entry has no content to carry.
    private static func makeTextImageMessage(
        from segments: [Transcript.Segment],
        make: (String, [UserInput.Image]) -> Chat.Message,
        emptyWarning: String
    ) -> [Chat.Message] {
        let text = extractText(from: segments)
        let images = extractImages(from: segments)
        guard text != nil || !images.isEmpty else {
            logger.warning("\(emptyWarning, privacy: .public)")
            return []
        }
        return [make(text ?? "", images)]
    }

    /// Encodes `value` as a JSON string literal (quotes included), escaping
    /// quotes, backslashes, and control characters per RFC 8259.
    ///
    /// - Parameter value: The string to encode.
    /// - Returns: The quoted, escaped JSON string literal.
    private static func jsonStringLiteral(_ value: String) -> String {
        if let data = try? JSONSerialization.data(withJSONObject: value, options: .fragmentsAllowed),
            let json = String(data: data, encoding: .utf8)
        {
            return json
        }
        // JSONSerialization cannot fail for a plain String in practice, but
        // escape manually rather than risk an unescaped quote or control
        // character reaching the constrained grammar's parser.
        var escaped = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": escaped += "\\\""
            case "\\": escaped += "\\\\"
            case "\n": escaped += "\\n"
            case "\r": escaped += "\\r"
            case "\t": escaped += "\\t"
            default:
                if scalar.value < 0x20 {
                    escaped += String(format: "\\u%04x", scalar.value)
                } else {
                    escaped.unicodeScalars.append(scalar)
                }
            }
        }
        escaped += "\""
        return escaped
    }

    /// Extracts and concatenates transcript segment text, with newlines
    /// between segments. Shared by `extractText` and `extractToolOutputText`,
    /// which differ only in whether `.structure` segments are included and
    /// how an empty result should read to their caller.
    ///
    /// - Parameters:
    ///   - segments: Array of transcript segments.
    ///   - includeStructure: When true, `.structure` segments are included,
    ///     rendered as their JSON; when false they're skipped like images
    ///     and other non-text segment types.
    ///   - logContext: Name of the calling extractor, for the
    ///     skipped-segment debug log.
    /// - Returns: The segments' text, joined with newlines ("" if none matched).
    private static func extractConcatenatedText(
        from segments: [Transcript.Segment], includeStructure: Bool, logContext: String
    ) -> String {
        let texts = segments.compactMap { segment -> String? in
            switch segment {
            case .text(let textSegment):
                return textSegment.content
            case .structure(let structuredSegment) where includeStructure:
                return structuredSegment.content.jsonString
            default:
                logger.debug("Skipping non-text segment in \(logContext, privacy: .public)")
                return nil
            }
        }
        return texts.joined(separator: "\n")
    }

    /// Extracts text content from transcript segments.
    ///
    /// Concatenates all text segments with newlines. By default, skips
    /// images, structured content, and other non-text segments -- this is
    /// the right behavior for `.instructions`/`.prompt` entries, which are
    /// system/user text and should stay `.text`-only.
    ///
    /// - Parameters:
    ///   - segments: Array of transcript segments
    ///   - includeStructure: When true, `.structure` segments are also
    ///     included (rendered as their JSON). Passed `true` for `.response`
    ///     entries, where a prior turn's guided/structured generation must
    ///     survive replay into the next turn's prompt rather than being
    ///     silently dropped; defaults to `false` for every other call site.
    /// - Returns: Concatenated text, or nil if no text content found
    private static func extractText(
        from segments: [Transcript.Segment], includeStructure: Bool = false
    ) -> String? {
        let combined = extractConcatenatedText(
            from: segments, includeStructure: includeStructure, logContext: "extractText")
        return combined.isEmpty ? nil : combined
    }

    /// Extracts a tool output's text content, unlike `extractText`:
    /// - Includes `.structure` segments (a tool's structured result, rendered
    ///   as its JSON), not just `.text`, since tool outputs commonly return
    ///   structured data rather than prose.
    /// - Always returns a `String` (never `nil`), including empty, so a
    ///   tool-output entry always produces a `Chat.Message.tool(...)` even
    ///   when the tool returned no content -- omitting the message entirely
    ///   would make a continuation round's rendered prompt identical to the
    ///   round that made the call.
    ///
    /// - Parameter segments: A tool output's transcript segments.
    /// - Returns: Concatenated text (and structured-content JSON), or "".
    private static func extractToolOutputText(from segments: [Transcript.Segment]) -> String {
        extractConcatenatedText(
            from: segments, includeStructure: true, logContext: "extractToolOutputText")
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

    /// The transcript entries that carry at least one image attachment
    /// segment: an `.instructions` or `.prompt` entry with an
    /// `.attachment(.image)` segment, or a `.toolOutput` entry whose result
    /// included one (e.g. a tool that returns a photo) -- the only entry
    /// kinds `extractImages` is ever consulted for; `.response`/`.toolCalls`
    /// never carry attachment segments in this converter's design.
    ///
    /// Used to populate `LanguageModelError.UnsupportedTranscriptContent`'s
    /// `unsupportedContent` when the local vision pipeline fails to process
    /// this round's image content, so the typed error names exactly which
    /// transcript entries carried it.
    ///
    /// - Parameter entries: Transcript entries from FoundationModels
    /// - Returns: The entries carrying image content, in transcript order
    static func entriesWithImages(for entries: some Collection<Transcript.Entry>)
        -> [Transcript.Entry]
    {
        entries.filter { entry in
            switch entry {
            case .instructions(let instructions):
                return !extractImages(from: instructions.segments).isEmpty
            case .prompt(let prompt):
                return !extractImages(from: prompt.segments).isEmpty
            case .toolOutput(let toolOutput):
                return !extractImages(from: toolOutput.segments).isEmpty
            default:
                return false
            }
        }
    }
}

#endif  // canImport(FoundationModels)
#endif  // FoundationModelsIntegration
