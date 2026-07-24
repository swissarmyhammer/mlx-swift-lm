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
    /// - Parameters:
    ///   - entries: Transcript entries from FoundationModels.
    ///   - toolCallFormat: The model's tool-call format, used only to choose
    ///     how a `.toolCalls` entry is replayed into the prompt. `nil` (the
    ///     default) and every value outside ``structuredToolCallFormats``
    ///     preserve the historical rendering: one assistant message per call
    ///     carrying the verbatim `{"name":…, "arguments":…}` envelope as text
    ///     content. Formats in ``structuredToolCallFormats`` (`.mistral`,
    ///     `.minimaxM2`, `.minimaxM3`) instead carry the calls as structured
    ///     tool-call metadata on a single assistant message (see the
    ///     `.toolCalls` case) so their sequence-validating chat templates
    ///     accept the exchange.
    ///   - replayReasoning: Whether a `.reasoning` entry's text should ride
    ///     on the `.response` entry it precedes as ``Chat/Message/reasoning``
    ///     (replayed into history renders as `reasoning_content`). `false`
    ///     (the default) preserves the historical behavior — reasoning
    ///     entries are dropped. Gated per model family by the executor on
    ///     `ReasoningConfig.historyPreservationKey`: only templates that
    ///     support preserved thinking (Qwen3.6's `preserve_thinking`) get
    ///     replay, keeping every other family's renders byte-identical.
    /// - Returns: Array of MLX Chat.Message objects
    static func mlxMessages(
        for entries: some Collection<Transcript.Entry>,
        toolCallFormat: ToolCallFormat? = nil,
        replayReasoning: Bool = false
    ) -> [Chat.Message] {
        var messages: [Chat.Message] = []
        // Reasoning text awaiting its response: the executor streams a
        // round's reasoning entry immediately before the response entry it
        // belongs to, so replay attaches a pending reasoning to the NEXT
        // `.response` — and to nothing else. Consecutive reasoning entries
        // accumulate; any other intervening entry discards the pending text.
        // In particular, think-then-call reasoning (a `.reasoning` entry
        // followed by `.toolCalls`) is deliberately dropped: the tool-call
        // turn is replayed as the verbatim envelope text the model actually
        // produced, and attaching Phase-1 reasoning there would change that
        // turn's render without real-weights verification of the seam.
        var pendingReasoning: String?
        for entry in entries {
            if replayReasoning, case .reasoning(let reasoning) = entry {
                // Accumulate rather than convert: the reasoning text becomes
                // message metadata on the response that follows, not a
                // message of its own. Multiple consecutive reasoning entries
                // join with newlines, exactly like segments within one entry.
                if let text = extractText(from: reasoning.segments) {
                    pendingReasoning = pendingReasoning.map { $0 + "\n" + text } ?? text
                }
                continue
            }
            var converted = convert(entry: entry, toolCallFormat: toolCallFormat)
            if case .response = entry, let reasoning = pendingReasoning, !converted.isEmpty {
                converted[0].reasoning = reasoning
            }
            // Consumed by the response above, or discarded: reasoning
            // belongs to the entry generated immediately after it and must
            // never be mis-attributed to a later, unrelated response.
            pendingReasoning = nil
            messages.append(contentsOf: converted)
        }
        return messages
    }

    /// Converts a single transcript entry to its chat message(s) — the
    /// per-entry body of ``mlxMessages(for:toolCallFormat:replayReasoning:)``,
    /// which owns the cross-entry reasoning-replay state around it.
    ///
    /// - Parameters:
    ///   - entry: The transcript entry to convert.
    ///   - toolCallFormat: See ``mlxMessages(for:toolCallFormat:replayReasoning:)``.
    /// - Returns: The entry's chat messages (empty for entries with nothing
    ///   to carry, including `.reasoning` when replay is disabled).
    private static func convert(
        entry: Transcript.Entry,
        toolCallFormat: ToolCallFormat?
    ) -> [Chat.Message] {
        switch entry {
        case .instructions(let instructions):
            // System message for model instructions. Labeled image
            // attachments ride along as message images, mirroring the
            // prompt path, so the `.vision` gate sees them and they are
            // not silently dropped.
            return makeTextImageMessage(
                from: instructions.segments,
                make: { Chat.Message.system(content: $0, images: $1) },
                emptyWarning: "Skipping instructions entry with no text or image content")

        case .prompt(let prompt):
            // User message for prompts. Labeled image attachments
            // (public `.attachment` segments) ride along as message
            // images; text is still concatenated as before.
            return makeTextImageMessage(
                from: prompt.segments,
                make: { Chat.Message.user(content: $0, images: $1) },
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
            return [Chat.Message.assistant(content: text)]

        case .toolCalls(let toolCalls):
            // Some chat templates validate the message sequence around
            // tool turns and reject a tool-call turn replayed as verbatim
            // assistant text content (the default below):
            //
            // - Mistral enforces strict user/assistant alternation and
            //   counts any assistant message *without* `tool_calls`
            //   toward it, so a completed round (user -> tool call ->
            //   tool result -> answer) renders as user, assistant,
            //   assistant and the template raises `TemplateException`.
            // - MiniMax-M2 requires every `tool` message to follow an
            //   assistant message carrying `tool_calls` (a plain
            //   assistant turn resets its `last_tool_call` tracker), so
            //   the tool result itself raises `TemplateException`.
            //
            // For these formats, carry the round's calls as structured
            // `tool_calls` on a single assistant message instead: the
            // template renders its own native tool-call shape (Mistral's
            // `[TOOL_CALLS]name[ARGS]args`, MiniMax's
            // `<minimax:tool_call><invoke …>`), the exact form those
            // models are trained to consume.
            if let toolCallFormat, Self.structuredToolCallFormats.contains(toolCallFormat) {
                return [
                    Chat.Message.assistant(
                        content: "", toolCalls: toolCalls.map(structuredToolCall(from:)))
                ]
            }
            // Every other family: one assistant message per tool call,
            // each carrying the exact `{"name": ..., "arguments": ...}`
            // envelope text the Executor's own tool-calling grammar
            // generates (see `unwrapToolCallMarkers` in
            // MLXLanguageModel.swift) -- replayed verbatim rather than
            // through Chat.Message's structured `toolCalls:` parameter, so
            // a continuation round sees byte-identical history to what it
            // actually produced, not a template's own `tool_calls`
            // re-rendering of it.
            return toolCalls.map { call in
                Chat.Message.assistant(
                    content:
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
                    content: extractToolOutputText(from: toolOutput.segments),
                    images: extractImages(from: toolOutput.segments),
                    id: toolOutput.id)
            ]

        case .reasoning:
            // Reasoning entries produce no message of their own. With
            // replay disabled (the historical default, per SKILL.md) the
            // chain-of-thought is dropped: the answer carries forward,
            // the thinking does not. With replay enabled, `mlxMessages`
            // consumes `.reasoning` entries BEFORE dispatching here and
            // attaches their text to the following response — so this
            // case only runs, and only drops, when replay is off.
            // Explicit rather than folded into the catch-all below so a
            // future SDK change is reviewed here.
            logger.debug("Skipping reasoning entry (not replayed into chat history)")
            return []

        @unknown default:
            // Skip unrecognized future entry types.
            logger.debug("Skipping unsupported entry type")
            return []
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
        if let data = try? JSONSerialization.data(
            withJSONObject: value, options: .fragmentsAllowed),
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

    /// The tool-call formats whose `.toolCalls` entries are replayed as
    /// structured `tool_calls` metadata (via `structuredToolCall(from:)`)
    /// instead of verbatim assistant text content, because their chat
    /// templates validate the message sequence around tool turns (see the
    /// `.toolCalls` case in `mlxMessages`).
    private static let structuredToolCallFormats: Set<ToolCallFormat> = [
        .mistral, .minimaxM2, .minimaxM3,
    ]

    /// Bridges a FoundationModels `Transcript.ToolCall` into MLXLMCommon's
    /// structured ``ToolCall`` so a `.toolCalls` entry of a format in
    /// ``structuredToolCallFormats`` can be replayed through the model's own
    /// `tool_calls` template branch (Mistral's `[TOOL_CALLS]name[ARGS]args`,
    /// MiniMax's `<minimax:tool_call><invoke …>`) rather than as verbatim
    /// assistant text content.
    ///
    /// The call's arguments arrive as a `GeneratedContent` JSON string;
    /// they are parsed back into a `[String: any Sendable]` object so the
    /// template re-serializes them as its native payload (Mistral's `[ARGS]`
    /// JSON, MiniMax's per-key `<parameter>` tags). A non-object or
    /// unparseable payload degrades to empty arguments rather than throwing
    /// -- the constrained tool-calling grammar makes malformed arguments
    /// unreachable in practice, and an empty `{}` still renders a valid turn.
    ///
    /// - Parameter call: The transcript tool call to bridge.
    /// - Returns: The equivalent structured ``ToolCall``.
    private static func structuredToolCall(from call: Transcript.ToolCall) -> ToolCall {
        let arguments: [String: any Sendable]
        if let data = call.arguments.jsonString.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: any Sendable]
        {
            arguments = object
        } else {
            arguments = [:]
        }
        return ToolCall(
            function: ToolCall.Function(name: call.toolName, arguments: arguments),
            id: call.id)
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
