// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import FoundationModels
import Testing

@testable import MLXFoundationModels

/// Guided output emits a tool call when it is an envelope, and drops the tool call and says so
/// when it is not.
@Suite
struct RequiredToolCallEnvelopeTests {
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    private func emits(_ outputBuffer: String) async throws -> Bool {
        let executor = try makeMLXExecutor(for: makeStubModel("required-envelope"))
        let channel = LanguageModelExecutorGenerationChannel()
        // The channel is a rendezvous, thus a consumer must run beside the executor or every
        // send parks it.
        let consumer = Task<Void, Never> {
            do { for try await _ in channel {} } catch {}
        }
        defer { consumer.cancel() }
        return await executor.emitRequiredToolCallEvent(
            outputBuffer: outputBuffer, toolCallsEntryID: "entry", channel: channel)
    }

    @Test("an envelope with a name reports an emitted tool call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func namedEnvelopeIsEmitted() async throws {
        #expect(try await emits(#"{"name":"f","arguments":{}}"#))
    }

    @Test("output that is no JSON reports a dropped tool call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func malformedOutputIsDropped() async throws {
        #expect(try await !emits("no envelope"))
    }

    @Test("an envelope with no name reports a dropped tool call")
    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    func namelessEnvelopeIsDropped() async throws {
        #expect(try await !emits(#"{"arguments":{}}"#))
    }
}

#endif
