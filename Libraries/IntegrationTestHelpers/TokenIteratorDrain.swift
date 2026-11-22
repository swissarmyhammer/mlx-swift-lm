// Copyright © 2026 Apple Inc.

import MLXLMCommon

/// Drains `iterator` and returns the tokens it emitted. `iterator` keeps its final state, thus a
/// caller can read its draft counts afterwards.
///
/// - Parameter iterator: The token iterator to run to its end.
/// - Returns: Every token `iterator` emitted, in order.
public func drain<Iterator: TokenIteratorProtocol>(_ iterator: inout Iterator) -> [Int] {
    var tokens: [Int] = []
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}
