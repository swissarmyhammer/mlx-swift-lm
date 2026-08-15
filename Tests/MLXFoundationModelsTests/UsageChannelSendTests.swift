// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import Testing

@testable import MLXFoundationModels

/// Proves that per-response usage reaches the FoundationModels channel.
///
/// A consumer reads usage from the framework, and the framework reads it from
/// the channel. The `generationObserver` task-local is a test-only mirror, and
/// `emitUsage` notifies it BEFORE the send. Thus a test that reads the observer
/// passes even when the send is absent, and that is how a missing send stayed
/// hidden while every consumer read an input count of zero. This suite reads
/// the channel itself, thus it fails the moment the send goes away again.
@Suite("Usage reaches the generation channel")
struct UsageChannelSendTests {

    /// Longest this suite waits for one channel event.
    ///
    /// The channel has no `finish()`, and a read of it blocks until a send
    /// arrives. A missing send must FAIL the test rather than stop the suite,
    /// thus each read is bounded. Five seconds is far longer than an in-process
    /// send needs, thus the bound never trips on a correct build.
    private static let channelReadTimeout = Duration.seconds(5)

    /// The prompt tokens of the emitted usage event.
    ///
    /// The four counts of this suite differ from each other, thus a swapped or
    /// a dropped field shows in the failure.
    private static let inputTokenCount = 128

    /// The reused prompt tokens of the emitted usage event.
    private static let inputCachedTokenCount = 96

    /// The generated tokens of the emitted usage event.
    private static let outputTokenCount = 32

    /// The reasoning subset of the generated tokens.
    private static let outputReasoningTokenCount = 8

    /// The transcript entry the usage event belongs to.
    private static let responseEntryID = "usage-entry"

    @Test("emitUsage sends the input and output token counts into the channel")
    func usageReachesTheChannel() async throws {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }

        let channel = LanguageModelExecutorGenerationChannel()
        let reader = ChannelReader(channel)
        defer { reader.stop() }

        await MLXLanguageModel.Executor.emitUsage(
            input: .init(
                totalTokenCount: Self.inputTokenCount,
                cachedTokenCount: Self.inputCachedTokenCount),
            output: .init(
                totalTokenCount: Self.outputTokenCount,
                reasoningTokenCount: Self.outputReasoningTokenCount),
            entryID: Self.responseEntryID,
            into: channel)

        var received = reader.events.makeAsyncIterator()
        let event = try #require(
            await received.next(),
            "The channel received no event. The usage send is absent.")

        let response = try #require(
            reflectedChannelPayload(
                of: event, caseLabel: "response",
                as: LanguageModelExecutorGenerationChannel.Response.self),
            "The channel event is not a response event.")
        let usage = try #require(
            reflectedChannelPayload(
                of: response.action, caseLabel: "updateUsage",
                as: LanguageModelExecutorGenerationChannel.Usage.self),
            "The response action is not an updateUsage action.")

        #expect(response.entryID == Self.responseEntryID)
        #expect(usage.input.totalTokenCount == Self.inputTokenCount)
        #expect(usage.input.cachedTokenCount == Self.inputCachedTokenCount)
        #expect(usage.output.totalTokenCount == Self.outputTokenCount)
        #expect(usage.output.reasoningTokenCount == Self.outputReasoningTokenCount)
    }

    /// A bounded read of the opaque generation channel.
    ///
    /// The channel is a rendezvous: a send blocks until a consumer takes the
    /// event, thus the consumer must run beside the send. The consumer relays
    /// each event into a stream this suite owns. A watchdog finishes that
    /// stream after ``channelReadTimeout``, thus a send that never arrives ends
    /// the read with `nil` instead of blocking the whole suite.
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private struct ChannelReader {

        /// Every event the channel delivered, in order.
        let events: AsyncStream<LanguageModelExecutorGenerationChannel.Event>

        private let consumer: Task<Void, Never>
        private let watchdog: Task<Void, Never>

        /// Starts consuming `channel` at once.
        init(_ channel: LanguageModelExecutorGenerationChannel) {
            let (events, continuation) =
                AsyncStream<LanguageModelExecutorGenerationChannel.Event>.makeStream()
            self.events = events
            self.consumer = Task {
                do {
                    for try await event in channel { continuation.yield(event) }
                } catch {
                    // Including cancellation. The finish below ends the read.
                }
                continuation.finish()
            }
            self.watchdog = Task {
                try? await Task.sleep(for: UsageChannelSendTests.channelReadTimeout)
                continuation.finish()
            }
        }

        /// Releases the consumer and the watchdog.
        func stop() {
            consumer.cancel()
            watchdog.cancel()
        }
    }
}

#endif  // FoundationModelsIntegration
