// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLXGuidedGeneration
import MLXLMCommon
import Synchronization

/// The MTP outcome of one `respond` call of an ``MLXLanguageModel`` that has
/// ``MLXLanguageModel/speculativeDecoding``, which the call reports in the metadata of its
/// response entry.
///
/// Every generation pass of the call feeds it, from the counts the pass already keeps: a
/// pass of the generation loop -- text, reasoning, allowed tools, the reasoning phase of
/// think-then-call -- through the speculative telemetry and the fallback reason of its
/// `GenerateCompletionInfo`, and a guided pass -- schema output, the grammar phase of a
/// required tool call -- through the MTP rounds and fallbacks of the
/// `GuidedGenerationDiagnosticSink` bound around it.
///
/// The metadata of a call with several passes reads:
///
/// - ``MetadataKey/speculativeDecoding``: `true` when some pass drafted in at least one
///   round.
/// - ``MetadataKey/fallbackReason``: the case name of the first fallback a pass of the call
///   reported, in the order the passes ran; absent when no pass fell back.
/// - ``MetadataKey/draftedTokenCount`` and ``MetadataKey/acceptedDraftTokenCount``: the sums
///   over every pass of the call.
///
/// A `Mutex` holds the tally: the passes of one call run in sequence, but across actor hops,
/// so each read and write takes the lock.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
final class SpeculativeDecodingOutcome: Sendable {

    /// The keys of the metadata a call reports. Consumers read these exact names.
    enum MetadataKey {
        /// Whether MTP drafted in at least one round of the call: `Bool`.
        static let speculativeDecoding = "speculativeDecoding"

        /// The case name of the first `SpeculativeDecodingFallbackReason` a pass of the call
        /// reported: `String`.
        static let fallbackReason = "speculativeDecodingFallbackReason"

        /// The drafts of the call that the target accepted: `Int`.
        static let acceptedDraftTokenCount = "acceptedDraftTokenCount"

        /// The drafts the call proposed: `Int`.
        static let draftedTokenCount = "draftedTokenCount"
    }

    /// The outcome of the call the current task runs, or `nil` outside a call whose model
    /// speculates.
    @TaskLocal static var current: SpeculativeDecodingOutcome?

    /// What the passes so far reported.
    private struct Tally {
        var draftedTokenCount = 0
        var acceptedDraftTokenCount = 0
        var fallbackReason: SpeculativeDecodingFallbackReason?
    }

    /// What one guided pass found on its sink.
    private struct GuidedReading {
        let draftedTokenCount: Int
        let acceptedDraftTokenCount: Int
        let fallbackCount: Int

        /// The counts `sink` holds now.
        init(of sink: GuidedGenerationDiagnosticSink) {
            draftedTokenCount = sink.draftedTokenCount
            acceptedDraftTokenCount = sink.acceptedDraftTokenCount
            fallbackCount = sink.speculationFallbacks.count
        }
    }

    private let tally = Mutex(Tally())

    /// Records what one pass of the generation loop reported when it ended.
    ///
    /// - Parameter info: The completion report of the pass.
    func record(_ info: GenerateCompletionInfo) {
        let telemetry = info.speculativeDecodingTelemetry
        record(
            drafted: telemetry?.draftTokenCount ?? 0,
            accepted: telemetry?.acceptedDraftTokenCount ?? 0,
            fallbackReason: info.speculativeDecodingFallbackReason)
    }

    /// Records a fallback of the whole call that no pass reports, such as a drafter the memory
    /// policy did not admit, which leaves every pass of the call to plain decode.
    ///
    /// - Parameter fallbackReason: Why the call decodes plainly.
    func record(fallbackReason: SpeculativeDecodingFallbackReason) {
        record(drafted: 0, accepted: 0, fallbackReason: fallbackReason)
    }

    /// Runs one guided pass and records the MTP rounds and fallbacks it reported to its sink.
    ///
    /// A sink a test bound stays bound, and only what the pass added to it counts. With no
    /// sink bound, the pass runs under a new sink that records no decoding.
    ///
    /// - Parameter pass: The guided pass.
    /// - Returns: What `pass` returned.
    /// - Throws: What `pass` threw; the rounds before the throw still count.
    func recordGuidedPass<Value>(_ pass: () throws -> Value) rethrows -> Value {
        let bound = GuidedGenerationDiagnosticSink.current
        let sink = bound ?? GuidedGenerationDiagnosticSink(recordsDecoding: false)
        let before = GuidedReading(of: sink)
        defer { record(guided: sink, since: before) }
        guard bound == nil else { return try pass() }
        return try GuidedGenerationDiagnosticSink.$current.withValue(sink, operation: pass)
    }

    /// Runs one guided pass, recording it into the outcome of the current call when there is
    /// one.
    ///
    /// - Parameter pass: The guided pass.
    /// - Returns: What `pass` returned.
    /// - Throws: What `pass` threw.
    static func recordingGuidedPass<Value>(_ pass: () throws -> Value) rethrows -> Value {
        guard let current else { return try pass() }
        return try current.recordGuidedPass(pass)
    }

    /// The metadata the response entry of the call reports.
    var metadata: [String: any ConvertibleToGeneratedContent & Sendable] {
        let counts = tally.withLock { $0 }
        var metadata: [String: any ConvertibleToGeneratedContent & Sendable] = [
            MetadataKey.speculativeDecoding: counts.draftedTokenCount > 0,
            MetadataKey.draftedTokenCount: counts.draftedTokenCount,
            MetadataKey.acceptedDraftTokenCount: counts.acceptedDraftTokenCount,
        ]
        if let fallbackReason = counts.fallbackReason {
            metadata[MetadataKey.fallbackReason] = String(describing: fallbackReason)
        }
        return metadata
    }

    /// Records what a guided pass added to `sink` since `before`.
    private func record(guided sink: GuidedGenerationDiagnosticSink, since before: GuidedReading) {
        let after = GuidedReading(of: sink)
        record(
            drafted: after.draftedTokenCount - before.draftedTokenCount,
            accepted: after.acceptedDraftTokenCount - before.acceptedDraftTokenCount,
            fallbackReason: sink.speculationFallbacks.dropFirst(before.fallbackCount).first)
    }

    /// Adds the counts of one pass, and keeps its fallback reason when it is the first one.
    private func record(
        drafted: Int, accepted: Int, fallbackReason: SpeculativeDecodingFallbackReason?
    ) {
        tally.withLock { tally in
            tally.draftedTokenCount += drafted
            tally.acceptedDraftTokenCount += accepted
            tally.fallbackReason = tally.fallbackReason ?? fallbackReason
        }
    }
}

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
