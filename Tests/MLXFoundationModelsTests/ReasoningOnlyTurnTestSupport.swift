// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels

/// The transcript entries of a turn that writes reasoning, shared by the transcript converter
/// tests of the reasoning a history replays.
@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
enum ReasoningOnlyTurnFixture {

    /// The reasoning the turn writes before its token budget runs out.
    static let reasoning = "Let me add: 2 plus"

    /// A prompt entry that asks `question`.
    ///
    /// - Parameter question: The text of the prompt.
    /// - Returns: The entry.
    static func prompt(_ question: String) -> Transcript.Entry {
        .prompt(
            Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: question))], responseFormat: nil))
    }

    /// The reasoning entry of the turn.
    static var reasoningEntry: Transcript.Entry {
        .reasoning(Transcript.Reasoning(segments: [.text(Transcript.TextSegment(content: reasoning))]))
    }

    /// The entries of the turn and the prompt after it, in the order the framework appends
    /// them: the response entry opens with the turn, and the reasoning entry opens when the
    /// first reasoning fragment arrives.
    ///
    /// - Parameter responseSegments: The segments of the response entry of the turn. Empty for
    ///   a turn that stopped inside its reasoning; a text segment for a turn that answered.
    /// - Returns: The entries.
    static func reasoningOnlyTurn(
        responseSegments: [Transcript.Segment] = []
    ) -> [Transcript.Entry] {
        [
            prompt("What is 2+2?"),
            .response(Transcript.Response(assetIDs: [], segments: responseSegments)),
            reasoningEntry,
            prompt("And 3+3?"),
        ]
    }
}

#endif
