// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import FoundationModels
import MLXLMCommon
import MLXVLM
import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite
struct TranscriptConverterTests {

    @Test
    func testConvertInstructionsToSystemMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let instructions = Transcript.Instructions(
            segments: [
                .text(Transcript.TextSegment(content: "You are a helpful assistant."))
            ],
            toolDefinitions: []
        )

        let entries: [Transcript.Entry] = [.instructions(instructions)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .system)
        #expect(message.content == "You are a helpful assistant.")
    }

    @Test
    func testConvertPromptToUserMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Hello!"))
            ],
            responseFormat: nil
        )

        let entries: [Transcript.Entry] = [.prompt(prompt)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .user)
        #expect(message.content == "Hello!")
    }

    @Test
    func testConvertResponseToAssistantMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let response = Transcript.Response(
            assetIDs: [],
            segments: [
                .text(Transcript.TextSegment(content: "Hi there!"))
            ]
        )

        let entries: [Transcript.Entry] = [.response(response)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .assistant)
        #expect(message.content == "Hi there!")
    }

    @Test
    func testStructuredResponseSegmentCarriesJSONContent() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // A prior turn's response was a guided/structured generation (a
        // Generable result carried as a `.structure` segment, not `.text`).
        // Replaying that turn into the next prompt must not silently drop
        // the structured content.
        let structured = Transcript.StructuredSegment(
            source: "assistant",
            content: try GeneratedContent(json: #"{"tempF": 72, "condition": "sunny"}"#))
        let response = Transcript.Response(
            assetIDs: [],
            segments: [.structure(structured)]
        )

        let entries: [Transcript.Entry] = [.response(response)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .assistant)
        #expect(message.content.contains("72"))
        #expect(message.content.contains("sunny"))
    }

    @Test
    func testMultipleSegmentsAreConcatenated() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Hello")),
                .text(Transcript.TextSegment(content: "world")),
            ],
            responseFormat: nil
        )

        let entries: [Transcript.Entry] = [.prompt(prompt)]
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        let message = messages.first!
        #expect(message.role == .user)
        #expect(message.content == "Hello\nworld")
    }

    @Test
    func testMultiTurnConversation() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: "Be helpful"))],
                    toolDefinitions: []
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Hi"))],
                    responseFormat: nil
                )),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "Hello"))]
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "How are you?"))],
                    responseFormat: nil
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 4)
        #expect(messages[0].role == .system)
        #expect(messages[1].role == .user)
        #expect(messages[2].role == .assistant)
        #expect(messages[3].role == .user)
    }

    @Test
    func testEmptyTranscriptReturnsEmptyArray() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = []
        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.isEmpty)
    }

    @Test
    func testUnsupportedEntryTypesAreSkipped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Reasoning is the only entry type still dropped by design (see
        // testReasoningEntryIsDropped); toolCalls/toolOutput are now
        // rendered (see the tool-call/tool-output tests below).
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Test"))],
                    responseFormat: nil
                ))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)
        #expect(messages.count == 1)
    }

    @Test
    func testReasoningEntryIsDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // A prior turn that contains reasoning between the prompt and response.
        // The reasoning must not be replayed into the chat history.
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "What is 2+2?"))],
                    responseFormat: nil
                )),
            .reasoning(
                Transcript.Reasoning(
                    segments: [
                        .text(Transcript.TextSegment(content: "Let me add: 2 plus 2 is 4."))
                    ]
                )),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "4"))]
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 2)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "What is 2+2?")
        #expect(messages[1].role == .assistant)
        #expect(messages[1].content == "4")
    }

    @Test
    func testMultipleReasoningEntriesAllDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = [
            .reasoning(
                Transcript.Reasoning(
                    segments: [.text(Transcript.TextSegment(content: "first thought"))]
                )),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Hi"))],
                    responseFormat: nil
                )),
            .reasoning(
                Transcript.Reasoning(
                    segments: [.text(Transcript.TextSegment(content: "second thought"))]
                )),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Hi")
    }

    @Test
    func testLabeledImageAttachmentBecomesUserMessageImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Describe this")),
                .attachment(attachment),
            ],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Describe this")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testImageOnlyPromptStillProducesMessageWithImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [.attachment(attachment)],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testInstructionsImageAttachmentBecomesSystemMessageImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "reference")
        let instructions = Transcript.Instructions(
            segments: [
                .text(Transcript.TextSegment(content: "Use this reference:")),
                .attachment(attachment),
            ],
            toolDefinitions: []
        )

        let messages = TranscriptConverter.mlxMessages(for: [.instructions(instructions)])

        #expect(messages.count == 1)
        #expect(messages[0].role == .system)
        #expect(messages[0].content == "Use this reference:")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testUrlBackedImageAttachmentYieldsDecodedCIImage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let url = try makeSolidImageFileURL()
        defer { try? FileManager.default.removeItem(at: url) }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(imageURL: url)),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [.attachment(attachment)],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].images.count == 1)
        // The SDK eagerly decodes a URL-backed attachment at construction, so
        // the converter hands over the in-memory CIImage rather than the URL —
        // passing the URL would force a redundant, failure-prone re-decode.
        guard let image = messages[0].images.first else {
            Issue.record("Expected one image input")
            return
        }
        guard case .ciImage = image else {
            Issue.record("URL-backed attachment should yield .ciImage, not .url")
            return
        }
    }

    @Test
    func testMultipleImageAttachmentsPreserveCountAndOrder() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Two distinguishable images (different dimensions) so order is checkable.
        let first = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage(width: 2, height: 2))),
            label: "first")
        let second = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage(width: 4, height: 4))),
            label: "second")
        let prompt = Transcript.Prompt(
            segments: [
                .text(Transcript.TextSegment(content: "Compare these")),
                .attachment(first),
                .attachment(second),
            ],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].images.count == 2)
        // Segment order is preserved: the 2x2 image precedes the 4x4 image.
        let widths = messages[0].images.compactMap { image -> CGFloat? in
            guard case .ciImage(let ciImage) = image else { return nil }
            return ciImage.extent.width
        }
        #expect(widths == [2, 4])
    }

    @Test
    func testToolCallsEntryBecomesAssistantEnvelopeMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let call = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let entries: [Transcript.Entry] = [.toolCalls(Transcript.ToolCalls(id: "tc-1", [call]))]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .assistant)
        // Assert on parsed structure rather than exact formatting/whitespace,
        // since `GeneratedContent.jsonString`'s serialization isn't a
        // contract this test should pin down.
        let envelope = try #require(
            try JSONSerialization.jsonObject(with: Data(messages[0].content.utf8))
                as? [String: Any])
        #expect(envelope["name"] as? String == "get_weather")
        let arguments = try #require(envelope["arguments"] as? [String: Any])
        #expect(arguments["location"] as? String == "Tokyo")
    }

    @Test
    func testMultipleToolCallsInOneEntryAllCarried() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let first = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let second = Transcript.ToolCall(
            id: "call-2", toolName: "get_time",
            arguments: try GeneratedContent(json: #"{"timezone": "JST"}"#))
        let entries: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [first, second]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 2)
        #expect(messages[0].role == .assistant)
        #expect(messages[0].content.contains("get_weather"))
        #expect(messages[1].role == .assistant)
        #expect(messages[1].content.contains("get_time"))
    }

    @Test
    func testToolOutputEntryBecomesToolRoleMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries: [Transcript.Entry] = [
            .toolOutput(
                Transcript.ToolOutput(
                    id: "call-1", toolName: "get_weather",
                    segments: [.text(Transcript.TextSegment(content: "72F and sunny"))]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .tool)
        #expect(messages[0].content == "72F and sunny")
    }

    @Test
    func testStructuredToolOutputCarriesJSONContent() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let structured = Transcript.StructuredSegment(
            source: "get_weather",
            content: try GeneratedContent(json: #"{"tempF": 72, "condition": "sunny"}"#))
        let entries: [Transcript.Entry] = [
            .toolOutput(
                Transcript.ToolOutput(
                    id: "call-1", toolName: "get_weather", segments: [.structure(structured)]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .tool)
        #expect(messages[0].content.contains("72"))
        #expect(messages[0].content.contains("sunny"))
    }

    @Test
    func testToolOutputImageAttachmentIsNotSilentlyDropped() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let entries: [Transcript.Entry] = [
            .toolOutput(
                Transcript.ToolOutput(
                    id: "call-1", toolName: "take_photo",
                    segments: [
                        .text(Transcript.TextSegment(content: "Here is the photo")),
                        .attachment(attachment),
                    ]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 1)
        #expect(messages[0].role == .tool)
        #expect(messages[0].content == "Here is the photo")
        #expect(messages[0].images.count == 1)
    }

    @Test
    func testToolCallAndOutputOrderingPreserved() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let call = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let entries: [Transcript.Entry] = [
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Weather in Tokyo?"))],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [call])),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "call-1", toolName: "get_weather",
                    segments: [.text(Transcript.TextSegment(content: "72F and sunny"))])),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "It's 72F and sunny."))])),
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries)

        #expect(messages.count == 4)
        #expect(messages[0].role == .user)
        #expect(messages[1].role == .assistant)
        #expect(messages[1].content.contains("get_weather"))
        #expect(messages[2].role == .tool)
        #expect(messages[2].content == "72F and sunny")
        #expect(messages[3].role == .assistant)
        #expect(messages[3].content == "It's 72F and sunny.")
    }

    @Test
    func testImageBeforeTextStillConcatenatesText() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Attachment position must not perturb text concatenation.
        let attachment = Transcript.AttachmentSegment(
            content: .image(Transcript.ImageAttachment(makeSolidCGImage())),
            label: "photo")
        let prompt = Transcript.Prompt(
            segments: [
                .attachment(attachment),
                .text(Transcript.TextSegment(content: "line one")),
                .text(Transcript.TextSegment(content: "line two")),
            ],
            responseFormat: nil
        )

        let messages = TranscriptConverter.mlxMessages(for: [.prompt(prompt)])

        #expect(messages.count == 1)
        #expect(messages[0].content == "line one\nline two")
        #expect(messages[0].images.count == 1)
    }

    // MARK: - Mistral3 strict-alternation rendering

    /// Builds a completed single-round tool exchange as transcript entries.
    ///
    /// The sequence is system instructions, a user prompt, the assistant's
    /// tool call, the tool's result, and the assistant's final answer. This is
    /// the shape Mistral3's chat template rejected before the format-aware
    /// rendering fix — a plain assistant-content tool call left two assistant
    /// turns adjacent.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func completedToolRoundEntries() throws -> [Transcript.Entry] {
        let call = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        return [
            .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: "Be helpful"))],
                    toolDefinitions: [])),
            .prompt(
                Transcript.Prompt(
                    segments: [.text(Transcript.TextSegment(content: "Weather in Tokyo?"))],
                    responseFormat: nil)),
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [call])),
            .toolOutput(
                Transcript.ToolOutput(
                    id: "call-1", toolName: "get_weather",
                    segments: [.text(Transcript.TextSegment(content: "72F and sunny"))])),
            .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: "It's 72F and sunny."))])),
        ]
    }

    /// Mirrors the alternation validator in Mistral3's `chat_template.jinja`.
    ///
    /// This is the check that raises the reported `TemplateException`,
    /// reproduced verbatim from Devstral-Small-2-24B-Instruct-2512-4bit's
    /// `chat_template.jinja`, whose validator loop (after slicing off a
    /// leading `system` message) is:
    ///
    /// ```jinja
    /// {%- set ns = namespace(index=0) %}
    /// {%- for message in loop_messages %}
    ///     {%- if message.role == 'user' or (message.role == 'assistant'
    ///            and (message.tool_calls is not defined
    ///                 or message.tool_calls is none
    ///                 or message.tool_calls | length == 0)) %}
    ///         {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}
    ///             {{- raise_exception('After the optional system message,
    ///                 conversation roles must alternate user and assistant
    ///                 roles except for tool calls and results.') }}
    ///         {%- endif %}
    ///         {%- set ns.index = ns.index + 1 %}
    ///     {%- endif %}
    /// {%- endfor %}
    /// ```
    ///
    /// i.e. a message counts toward the user/assistant alternation index only
    /// when it is a `user` message or an `assistant` message WITHOUT
    /// `tool_calls`; `tool` messages and assistant tool-call messages are
    /// excluded. Returns the offset of the first message that breaks
    /// alternation, or `nil` when the sequence is template-acceptable.
    ///
    /// This mirror is validated against the real template out-of-band: running
    /// the actual `chat_template.jinja` over the old (verbatim-content) shape
    /// raises exactly this exception, while the new structured-`tool_calls`
    /// shape renders cleanly. The full end-to-end leg (applying the template
    /// via the real tokenizer during Devstral generation) is the gated
    /// integration test, which requires the ~13GB model + GPU.
    private func firstAlternationViolation(in rawMessages: [Message]) -> Int? {
        var loop = rawMessages
        if loop.first?["role"] as? String == "system" { loop.removeFirst() }
        var index = 0
        for (offset, message) in loop.enumerated() {
            let role = message["role"] as? String ?? ""
            let toolCalls = message["tool_calls"] as? [Any]
            let counted =
                role == "user"
                || (role == "assistant" && (toolCalls == nil || toolCalls!.isEmpty))
            guard counted else { continue }
            if (role == "user") != (index % 2 == 0) { return offset }
            index += 1
        }
        return nil
    }

    @Test
    func testMistralToolRoundRendersTemplateAcceptableAlternation() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries = try completedToolRoundEntries()
        let messages = TranscriptConverter.mlxMessages(for: entries, toolCallFormat: .mistral)
        let raw = Mistral3MessageGenerator().generate(messages: messages)

        // The sequence must satisfy Mistral's strict user/assistant
        // alternation once tool-call and tool-result turns are excluded.
        #expect(firstAlternationViolation(in: raw) == nil)

        // Role sequence: system, user, assistant(tool call), tool, assistant.
        #expect(messages.map(\.role) == [.system, .user, .assistant, .tool, .assistant])

        // The assistant tool-call turn is carried as structured `tool_calls`
        // (not verbatim text content) so Mistral's template renders
        // `[TOOL_CALLS]name[ARGS]args` and excludes it from alternation.
        let toolCallMessage = raw[2]
        #expect(toolCallMessage["role"] as? String == "assistant")
        let calls = try #require(toolCallMessage["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 1)
        let function = try #require(calls[0]["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "get_weather")
        let arguments = try #require(function["arguments"] as? [String: any Sendable])
        #expect(arguments["location"] as? String == "Tokyo")
    }

    @Test
    func testVerbatimToolCallRenderingViolatesMistralAlternation() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Guards the fix's motivation: the default (verbatim-content) tool-call
        // rendering — correct for Qwen/GLM — is counted as a plain assistant
        // turn and therefore breaks Mistral's strict alternation, which is
        // exactly why `.mistral` must render structured `tool_calls` instead.
        let entries = try completedToolRoundEntries()
        let messages = TranscriptConverter.mlxMessages(for: entries)
        let raw = Mistral3MessageGenerator().generate(messages: messages)

        #expect(firstAlternationViolation(in: raw) != nil)
    }

    @Test
    func testMistralMultipleToolCallsFoldIntoOneAssistantMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let first = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let second = Transcript.ToolCall(
            id: "call-2", toolName: "get_time",
            arguments: try GeneratedContent(json: #"{"timezone": "JST"}"#))
        let entries: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [first, second]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries, toolCallFormat: .mistral)

        // Parallel calls in one round fold into a single assistant turn
        // carrying both tool calls, so Mistral renders one `<eos>`-terminated
        // assistant message rather than splitting an eos between the calls.
        #expect(messages.count == 1)
        let raw = Mistral3MessageGenerator().generate(message: messages[0])
        let calls = try #require(raw["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 2)
        let names = calls.compactMap {
            ($0["function"] as? [String: any Sendable])?["name"] as? String
        }
        #expect(names == ["get_weather", "get_time"])
    }

    /// Mirrors the tool-role validator in MiniMax-M2's `chat_template.jinja`.
    ///
    /// This is the check that raises the reported `TemplateException`,
    /// reproduced from mlx-community/MiniMax-M2-4bit's `chat_template.jinja`,
    /// whose message loop tracks the last assistant turn's tool calls:
    ///
    /// ```jinja
    /// {%- if message.tool_calls -%}
    ///     ...
    ///     {%- set last_tool_call.name = message.tool_calls[-1].name -%}
    /// {%- else -%}
    ///     {%- set last_tool_call.name = none -%}
    /// {%- endif -%}
    /// ...
    /// {%- elif message.role == 'tool' -%}
    ///     {%- if last_tool_call.name is none -%}
    ///         {{- raise_exception("Message has tool role, but there was no
    ///             previous assistant message with a tool call!") }}
    ///     {%- endif -%}
    /// ```
    ///
    /// i.e. every `tool` message must be preceded by an assistant message
    /// carrying non-empty `tool_calls`, with any *plain* assistant message in
    /// between resetting the tracker. Returns the offset of the first tool
    /// message that violates this, or `nil` when the sequence is
    /// template-acceptable.
    ///
    /// This mirror is validated against the real template out-of-band:
    /// running the actual `chat_template.jinja` (jinja2) over the old
    /// (verbatim-content) shape raises exactly this exception, while the
    /// structured-`tool_calls` shape renders cleanly as the native
    /// `<minimax:tool_call><invoke …>` replay. The full end-to-end leg is the
    /// gated integration run, which needs the ~119GB model + GPU.
    private func firstMiniMaxToolValidatorViolation(in rawMessages: [Message]) -> Int? {
        var loop = rawMessages
        if loop.first?["role"] as? String == "system" { loop.removeFirst() }
        var lastToolCallName: String?
        for (offset, message) in loop.enumerated() {
            switch message["role"] as? String ?? "" {
            case "assistant":
                let calls = message["tool_calls"] as? [[String: any Sendable]]
                if let calls, !calls.isEmpty {
                    lastToolCallName =
                        (calls.last?["function"] as? [String: any Sendable])?["name"] as? String
                } else {
                    lastToolCallName = nil
                }
            case "tool":
                if lastToolCallName == nil { return offset }
            default:
                break
            }
        }
        return nil
    }

    @Test
    func testMiniMaxToolRoundRendersTemplateAcceptableSequence() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let entries = try completedToolRoundEntries()
        let messages = TranscriptConverter.mlxMessages(for: entries, toolCallFormat: .minimaxM2)
        let raw = DefaultMessageGenerator().generate(messages: messages)

        // Every tool-role message must follow an assistant message carrying
        // structured `tool_calls`, or MiniMax's template raises
        // `TemplateException`.
        #expect(firstMiniMaxToolValidatorViolation(in: raw) == nil)

        // Role sequence: system, user, assistant(tool call), tool, assistant.
        #expect(messages.map(\.role) == [.system, .user, .assistant, .tool, .assistant])

        // The assistant tool-call turn is carried as structured `tool_calls`
        // (not verbatim text content) so MiniMax's template renders the
        // native `<minimax:tool_call><invoke …>` replay and sets its
        // `last_tool_call` tracker before the tool result arrives.
        let toolCallMessage = raw[2]
        #expect(toolCallMessage["role"] as? String == "assistant")
        let calls = try #require(toolCallMessage["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 1)
        let function = try #require(calls[0]["function"] as? [String: any Sendable])
        #expect(function["name"] as? String == "get_weather")
        // The template iterates `arguments.items()`, so the arguments must be
        // a structured object, not a JSON string.
        let arguments = try #require(function["arguments"] as? [String: any Sendable])
        #expect(arguments["location"] as? String == "Tokyo")
    }

    @Test
    func testVerbatimToolCallRenderingViolatesMiniMaxToolValidator() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Guards the fix's motivation: the default (verbatim-content) tool-call
        // rendering — correct for Qwen/GLM — carries no `tool_calls` metadata,
        // so MiniMax's template resets `last_tool_call` and raises on the tool
        // result, which is exactly why `.minimaxM2` must render structured
        // `tool_calls` instead.
        let entries = try completedToolRoundEntries()
        let messages = TranscriptConverter.mlxMessages(for: entries)
        let raw = DefaultMessageGenerator().generate(messages: messages)

        #expect(firstMiniMaxToolValidatorViolation(in: raw) != nil)
    }

    @Test
    func testMiniMaxMultipleToolCallsFoldIntoOneAssistantMessage() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let first = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let second = Transcript.ToolCall(
            id: "call-2", toolName: "get_time",
            arguments: try GeneratedContent(json: #"{"timezone": "JST"}"#))
        let entries: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [first, second]))
        ]

        let messages = TranscriptConverter.mlxMessages(for: entries, toolCallFormat: .minimaxM2)

        // Parallel calls in one round fold into a single assistant turn, so
        // MiniMax's template renders one `<minimax:tool_call>` block with one
        // `<invoke>` per call (its `tool_calls` loop), matching how the model
        // itself emits a parallel round.
        #expect(messages.count == 1)
        let raw = DefaultMessageGenerator().generate(message: messages[0])
        let calls = try #require(raw["tool_calls"] as? [[String: any Sendable]])
        #expect(calls.count == 2)
        let names = calls.compactMap {
            ($0["function"] as? [String: any Sendable])?["name"] as? String
        }
        #expect(names == ["get_weather", "get_time"])
    }

    @Test
    func testNonMistralToolCallRenderingUnchanged() throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        // Two calls in one round: enough to actually exercise the "one
        // assistant message *per call*" invariant. With a single call the
        // invariant holds trivially whether or not folding happens, so it must
        // be probed with more than one call.
        let first = Transcript.ToolCall(
            id: "call-1", toolName: "get_weather",
            arguments: try GeneratedContent(json: #"{"location": "Tokyo"}"#))
        let second = Transcript.ToolCall(
            id: "call-2", toolName: "get_time",
            arguments: try GeneratedContent(json: #"{"timezone": "JST"}"#))
        let entries: [Transcript.Entry] = [
            .toolCalls(Transcript.ToolCalls(id: "tc-1", [first, second]))
        ]

        // Regression guard: `nil` (default), `.json` (Qwen/Llama), and
        // `.glm4` (GLM) must all keep the historical rendering — one assistant
        // message per call carrying the verbatim envelope as text content,
        // with no structured tool metadata that would let a template re-render
        // it. Unlike `.mistral`/`.minimaxM2`, which fold parallel calls into a
        // single assistant turn, these formats never collapse multiple calls:
        // two calls yield two messages, one per call. Only the
        // structured-rendering formats deviate.
        let defaultMessages = TranscriptConverter.mlxMessages(for: entries)
        #expect(defaultMessages.count == 2)
        for format in [ToolCallFormat.json, .glm4] {
            let messages = TranscriptConverter.mlxMessages(for: entries, toolCallFormat: format)

            // The per-call invariant `.mistral` breaks by folding: two calls
            // stay two assistant messages, one per call, matching default.
            #expect(messages.count == 2)
            #expect(messages.map(\.role) == [.assistant, .assistant])
            #expect(messages.map(\.content) == defaultMessages.map(\.content))

            var toolNames: [String] = []
            for message in messages {
                let raw = DefaultMessageGenerator().generate(message: message)
                #expect(raw["tool_calls"] == nil)
                let envelope = try #require(
                    try JSONSerialization.jsonObject(
                        with: Data((raw["content"] as? String ?? "").utf8)) as? [String: Any])
                toolNames.append(try #require(envelope["name"] as? String))
            }
            #expect(toolNames == ["get_weather", "get_time"])
        }

        // Contrast: `.mistral` folds the same two calls into a single assistant
        // turn — the very behavior these formats must not adopt.
        let mistralMessages = TranscriptConverter.mlxMessages(
            for: entries, toolCallFormat: .mistral)
        #expect(mistralMessages.count == 1)
    }

}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
