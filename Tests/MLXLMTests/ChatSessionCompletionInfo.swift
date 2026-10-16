// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon

extension ChatSession {
    /// Runs one turn and returns its completion information.
    ///
    /// Shared by the `ChatSession` suites that read the prompt counts of a
    /// turn.
    ///
    /// - Parameter prompt: the user turn.
    /// - Returns: the completion information of the turn, or nil when the
    ///   stream gave none.
    func completionInfo(respondingTo prompt: String) async throws -> GenerateCompletionInfo? {
        var info: GenerateCompletionInfo?
        for try await item in streamDetails(to: prompt) {
            info = item.info ?? info
        }
        return info
    }
}
