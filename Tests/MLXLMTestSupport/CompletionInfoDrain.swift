// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Reads `stream` to its end, waits for `task`, and returns the completion information the
/// stream reported.
///
/// - Parameters:
///   - stream: the stream of one generation.
///   - task: the task that runs the generation.
/// - Returns: the completion information of the generation, or `nil` when the stream reported
///   none.
public func completionInfo<Success: Sendable>(
    of stream: AsyncStream<Generation>, task: Task<Success, Never>
) async -> GenerateCompletionInfo? {
    var info: GenerateCompletionInfo?
    for await event in stream {
        if case .info(let completion) = event { info = completion }
    }
    _ = await task.value
    return info
}
