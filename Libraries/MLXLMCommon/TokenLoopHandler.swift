// Copyright © 2026 Apple Inc.

import MLX

/// Controls whether the generation loop continues.
public enum TokenLoopDisposition: Sendable {
    /// Continue generating tokens.
    case more
    /// Stop at a semantic response boundary.
    case stop
    /// Stop because the consumer ended or cancelled.
    case cancelled

    var shouldContinue: Bool {
        if case .more = self { return true }
        return false
    }

    /// The reason the generation loop records for this disposition, or `nil` when it goes on.
    var stopReason: GenerateStopReason? {
        switch self {
        case .more: nil
        case .stop: .stop
        case .cancelled: .cancelled
        }
    }
}

/// Converts generated tokens into stream events.
///
/// Handlers run on the generation worker and may own non-Sendable decoding state.
/// Only their output crosses into the consumer's task.
/// The `emit` closure returns `false` when the consumer has ended the stream.
public protocol TokenLoopHandler: SendableMetatype {
    associatedtype Output: Sendable

    /// Semantic boundaries contributed by the response protocol handled by
    /// this consumer. Raw-token consumers intentionally contribute none.
    var additionalStopTokenIDs: Set<Int> { get }

    /// Whether semantic parsing needs to observe EOS tokens even though they
    /// are not included in the public output or generation token count.
    var receivesStopTokens: Bool { get }

    /// Return `.stop` for semantic generation stops, or `.cancelled` for consumer termination.
    ///
    /// `logProbabilities` stays on the GPU until you call
    /// ``DeferredTokenLogProbabilities/materialize()``.
    mutating func onToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition

    /// Called when `includeStopToken` is true or ``receivesStopTokens`` is true
    /// and a stop token was hit.
    mutating func onStopToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition

    /// Called after the token loop finishes, before the info event.
    mutating func onGenerationEnd(
        emit: (sending Output) -> Bool
    ) -> TokenLoopDisposition

    func infoEvent(_ info: GenerateCompletionInfo) -> Output
}

extension TokenLoopHandler {
    /// No semantic boundary: the generation stops only on the model's own stop tokens.
    public var additionalStopTokenIDs: Set<Int> { [] }

    /// The handler does not observe the stop tokens that stay out of the output.
    public var receivesStopTokens: Bool { false }
}

/// Emits `output`, and answers `.cancelled` when the consumer has ended the stream, else `.more`.
///
/// - Parameters:
///   - output: the event to emit.
///   - emit: the emit closure of the generation loop.
/// - Returns: the disposition the emit implies.
private func emitOrCancel<Output>(
    _ output: sending Output, emit: (sending Output) -> Bool
) -> TokenLoopDisposition {
    emit(output) ? .more : .cancelled
}

/// Decodes response text and tool calls into ``Generation`` events.
public struct TextToolTokenLoopHandler: TokenLoopHandler {
    public typealias Output = Generation

    private static let logger = MLXLogger(label: "TokenStreamProtocol")
    private var decoder: any TokenStreamDecoder
    var cachedPromptTokenCount = 0
    var speculativeDecodingFallbackReason: SpeculativeDecodingFallbackReason?

    public init(
        tokenizer: Tokenizer, stopStrings: Set<String> = [], format: ToolCallFormat,
        tools: [[String: any Sendable]]? = nil,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        self.decoder = format.makeTokenStreamDecoder(
            tokenizer: tokenizer, tools: tools, stopStrings: stopStrings,
            toolCallPolicy: toolCallPolicy)
    }

    /// Creates a handler with the stop strings and the tool-call format of
    /// `modelConfiguration` (JSON when the configuration names none).
    init(
        tokenizer: Tokenizer, modelConfiguration: ModelConfiguration,
        tools: [[String: any Sendable]]?, toolCallPolicy: ToolCallPolicy
    ) {
        self.init(
            tokenizer: tokenizer, stopStrings: modelConfiguration.effectiveStopStrings,
            format: modelConfiguration.toolCallFormat ?? .json, tools: tools,
            toolCallPolicy: toolCallPolicy)
    }

    public var additionalStopTokenIDs: Set<Int> { decoder.additionalStopTokenIDs }
    public var receivesStopTokens: Bool { decoder.receivesStopTokens }

    public mutating func onToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        process(token, emit: emit)
    }

    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        guard decoder.receivesStopTokens else { return .more }
        return process(token, emit: emit)
    }

    public mutating func onGenerationEnd(
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        drive(.finish, emit: emit).disposition
    }

    public func infoEvent(_ info: GenerateCompletionInfo) -> Generation {
        var info = info.withToolCallCounts(
            rejected: decoder.rejectedToolCallCount,
            recovered: decoder.recoveredToolCallCount)
        // The session's reason wins; else the reason the generation loop read off the iterator.
        info.speculativeDecodingFallbackReason =
            speculativeDecodingFallbackReason ?? info.speculativeDecodingFallbackReason
        return Generation.info(info).attributingCachedPromptTokens(cachedPromptTokenCount)
    }

    private mutating func process(
        _ token: Int,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        let (completed, disposition) = drive(.push(token), emit: emit)
        if disposition.shouldContinue {
            return completed ? .more : .cancelled
        }
        return disposition
    }

    /// What ``drive(_:emit:)`` feeds the decoder.
    private enum DecoderStep {
        /// A generated token.
        case push(Int)
        /// The end of the generation.
        case finish
    }

    /// Feeds the decoder `step`, handling each event it emits until an event stops the
    /// generation.
    ///
    /// The decoder is copied out of `self` for the run, so the event handler can mutate
    /// `self`, and is written back after it.
    ///
    /// - Parameters:
    ///   - step: what the decoder receives.
    ///   - emit: the emit closure of the generation loop.
    /// - Returns: what the decoder answered, and the disposition of the last handled event
    ///   (`.more` when the decoder emitted none).
    private mutating func drive(
        _ step: DecoderStep,
        emit: (sending Generation) -> Bool
    ) -> (completed: Bool, disposition: TokenLoopDisposition) {
        var decoder = self.decoder
        var disposition = TokenLoopDisposition.more
        func handle(_ event: TokenStreamEvent) -> Bool {
            disposition = process(event, emit: emit)
            return disposition.shouldContinue
        }
        let completed =
            switch step {
            case .push(let token): decoder.push(token, emit: handle)
            case .finish: decoder.finish(emit: handle)
            }
        self.decoder = decoder
        return (completed, disposition)
    }

    private mutating func process(
        _ event: TokenStreamEvent,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        switch event {
        case .reasoning:
            // The public Generation stream intentionally exposes only response
            // text and tool calls. Protocol-aware clients consume reasoning via
            // the package-level TokenStreamDecoder contract.
            return .more

        case .response(let response):
            return emitOrCancel(.chunk(response), emit: emit)

        case .toolCall(let toolCall):
            return emitOrCancel(.toolCall(toolCall), emit: emit)

        case .protocolError(let message):
            Self.logger.error("\(message)")
            return .more

        case .rejectedToolCall(let rejection):
            return emitOrCancel(.rejectedToolCall(rejection), emit: emit)

        case .stop:
            return .stop
        }
    }
}

/// Emits raw token IDs and completion information as ``TokenGeneration`` events.
public struct RawTokenLoopHandler: TokenLoopHandler {
    public typealias Output = TokenGeneration

    public let additionalStopTokenIDs: Set<Int>
    public let receivesStopTokens: Bool

    public init(additionalStopTokenIDs: Set<Int> = [], receivesStopTokens: Bool = false) {
        self.additionalStopTokenIDs = additionalStopTokenIDs
        self.receivesStopTokens = receivesStopTokens
    }

    public mutating func onToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending TokenGeneration) -> Bool
    ) -> TokenLoopDisposition {
        emitOrCancel(.token(token), emit: emit)
    }

    /// Emits the stop token the same way ``onToken(_:logProbabilities:emit:)`` emits an
    /// ordinary token.
    public mutating func onStopToken(
        _ token: Int,
        logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending TokenGeneration) -> Bool
    ) -> TokenLoopDisposition {
        onToken(token, logProbabilities: logProbabilities, emit: emit)
    }

    public mutating func onGenerationEnd(
        emit: (sending TokenGeneration) -> Bool
    ) -> TokenLoopDisposition { .more }

    public func infoEvent(_ info: GenerateCompletionInfo) -> TokenGeneration {
        .info(info)
    }
}
