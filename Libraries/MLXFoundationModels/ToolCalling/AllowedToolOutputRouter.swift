// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import MLXLMCommon

/// Detokenized routing for ordinary, unframed tool-call formats.
/// Framed protocols are selected by `ToolCallFormat` and bypass this router.
struct AllowedToolOutputRouter {
    enum Event: Sendable, Equatable {
        case reasoning(String)
        case response(String)
        case toolCall(MLXLMCommon.ToolCall)
        case rejectedToolCall(RejectedToolCall)
    }

    private var reasoningEmitter: ReasoningEventEmitter?
    private let toolProcessor: ToolCallProcessor

    init(
        format: ToolCallFormat,
        tools: [[String: any Sendable]],
        reasoning: (config: ReasoningConfig, primedInside: Bool)? = nil
    ) {
        self.toolProcessor = ToolCallProcessor(format: format, tools: tools)
        self.reasoningEmitter = reasoning.map {
            ReasoningEventEmitter(config: $0.config, primedInside: $0.primedInside)
        }
    }

    var isInsideReasoning: Bool {
        reasoningEmitter?.isInsideReasoning ?? false
    }

    mutating func process(_ chunk: String) -> [Event] {
        guard var emitter = reasoningEmitter else {
            return processResponse(chunk)
        }

        let segments = emitter.process(chunk)
        reasoningEmitter = emitter
        var events: [Event] = []
        for segment in segments {
            switch segment {
            case .reasoning(let text):
                events.append(.reasoning(text))
            case .response(let text):
                events.append(contentsOf: processResponse(text))
            }
        }
        return events
    }

    mutating func finish() -> [Event] {
        var events: [Event] = []
        if var emitter = reasoningEmitter {
            for segment in emitter.finalize() {
                switch segment {
                case .reasoning(let text): events.append(.reasoning(text))
                case .response(let text): events.append(contentsOf: processResponse(text))
                }
            }
            reasoningEmitter = emitter
        }
        events.append(contentsOf: route(toolProcessor.processEOSOutputs()))
        return events
    }

    private mutating func processResponse(_ text: String) -> [Event] {
        route(toolProcessor.processChunkOutputs(text))
    }

    /// Converts processor outputs to events, and joins adjacent responses.
    ///
    /// The recovery scanner of ``ToolCallProcessor`` can divide one span of
    /// response text at a `<` that does not start a call. The division is not
    /// a boundary that a consumer can see, thus contiguous response text must
    /// come out as one event.
    private func route(_ outputs: [ToolCallProcessor.Output]) -> [Event] {
        var events: [Event] = []
        for output in outputs {
            switch output {
            case .response(let text):
                if case .response(let previous) = events.last {
                    events[events.count - 1] = .response(previous + text)
                } else {
                    events.append(.response(text))
                }
            case .toolCall(let call):
                events.append(.toolCall(call))
            case .rejectedToolCall(let rejection):
                events.append(.rejectedToolCall(rejection))
            }
        }
        return events
    }
}

#endif
#endif
