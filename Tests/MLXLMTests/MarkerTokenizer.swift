// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon

/// A tokenizer that resolves the marker texts it is given and nothing else,
/// which is all the failable initializer of a committed-turn rule reads.
///
/// Shared by the rule suites of `DSMLCommittedTurnRule` and
/// `QwenCommittedTurnRule`.
struct MarkerTokenizer: Tokenizer {
    /// The identifier of each marker text this tokenizer knows.
    let identifierOfMarker: [String: Int]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { identifierOfMarker[token] }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
