// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private let mtpFallbackLogger = MLXLogger(label: "mtp-fallback")

/// Why a generation configured with an MTP drafter decodes without speculation, with the reason
/// stated in words.
///
/// The generation functions put ``reason`` into
/// ``GenerateCompletionInfo/speculativeDecodingFallbackReason`` and ``statedReason`` into
/// ``GenerateCompletionInfo/passthroughReason``; guided generation records ``statedReason`` on
/// its diagnostic sink. ``MTPDrafterFallbackLog/shared`` logs each stated reason once.
package struct MTPDrafterFallback: Sendable, Equatable {

    /// The kind of the fallback.
    package let reason: SpeculativeDecodingFallbackReason

    /// The fallback in words: the drafter, the target, and why the drafter cannot draft for it.
    package let statedReason: String

    /// The fallback of `drafter` for `target`, logged once for each drafter and target, or `nil`
    /// when the drafter can draft for the target.
    ///
    /// - Parameters:
    ///   - drafter: The configured MTP drafter.
    ///   - target: The model the generation decodes with.
    /// - Returns: The ``SpeculativeDecodingFallbackReason/incompatibleDrafter`` fallback, or
    ///   `nil`.
    package static func incompatibility(
        of drafter: any MTPDrafterModel, with target: any LanguageModel
    ) -> MTPDrafterFallback? {
        guard !drafter.isCompatible(with: target) else { return nil }
        let fallback = MTPDrafterFallback(
            reason: .incompatibleDrafter,
            statedReason: statedIncompatibility(of: drafter, with: target))
        MTPDrafterFallbackLog.shared.report(fallback)
        return fallback
    }

    /// The words of an incompatibility: the two classes and the error of the drafter's check.
    private static func statedIncompatibility(
        of drafter: any MTPDrafterModel, with target: any LanguageModel
    ) -> String {
        let pair =
            "the MTP drafter \(type(of: drafter)) cannot draft for the target \(type(of: target)), "
            + "so the generation decodes without speculation"
        do {
            try drafter.validateCompatibility(with: target)
            return pair
        } catch {
            let detail = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
            return pair + ": " + detail
        }
    }
}

/// Writes each stated MTP drafter fallback once, so a configured drafter that every generation
/// skips leaves one line in the log rather than none, and rather than one line per generation.
// swiftlint:disable:next no_unchecked_sendable  every access to `reported` holds `lock`
package final class MTPDrafterFallbackLog: @unchecked Sendable {

    /// The log of the process, which writes a warning to the `mtp-fallback` logger.
    package static let shared = MTPDrafterFallbackLog { mtpFallbackLogger.warning($0) }

    /// Writes one message.
    private let write: @Sendable (String) -> Void

    /// Guards ``reported``.
    private let lock = NSLock()

    /// The stated reasons written so far. A stated reason names the drafter and the target, and
    /// the sizes that do not match, so each is one drafter and target.
    private var reported: Set<String> = []

    /// Makes a log that writes through `write`.
    ///
    /// - Parameter write: Writes one message.
    package init(write: @escaping @Sendable (String) -> Void) {
        self.write = write
    }

    /// Writes `fallback` when no earlier call wrote its stated reason.
    ///
    /// - Parameter fallback: The fallback of one generation.
    package func report(_ fallback: MTPDrafterFallback) {
        let isFirst = lock.withLock { reported.insert(fallback.statedReason).inserted }
        guard isFirst else { return }
        write("MTP speculation is off: \(fallback.statedReason)")
    }
}
