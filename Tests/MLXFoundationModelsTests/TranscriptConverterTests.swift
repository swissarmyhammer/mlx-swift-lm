// Copyright © 2025 Apple Inc.

import CoreGraphics
import Foundation
import FoundationModels
import MLXLMCommon
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

}

#endif  // FoundationModelsIntegration && canImport(FoundationModels)
