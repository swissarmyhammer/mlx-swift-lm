// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Hands the drafts of rank 0 to every rank, inside a ``RemoteModelOp/speculate(_:)`` round.
///
/// Rank 0 drafts after the announce of the round, thus the other ranks learn the drafts from one
/// more all-sum: rank 0 passes the words of its drafts and every other rank passes nil, which
/// stands for zeros. Every rank gets the words of rank 0. Each rank of a round calls the exchange
/// once, with the same word count, or none calls it.
///
/// The words are the number of drafts, then the drafts, then zeros: ``wordCount(draftLimit:)``
/// words.
public struct RemoteDraftExchange {

    /// Sums the words of rank 0, or nil on another rank, across the ranks: given the words and
    /// their count, it gives the words of rank 0.
    public typealias Exchange = (_ words: [Int32]?, _ wordCount: Int) throws -> [Int32]

    /// The exchange.
    private let exchange: Exchange

    /// Makes an exchange of `exchange`.
    ///
    /// - Parameter exchange: Gives the words of rank 0 to each rank.
    public init(_ exchange: @escaping Exchange) {
        self.exchange = exchange
    }

    /// The exchange of a worker that runs the whole model on one rank: the words stay as they
    /// are.
    public static var alone: RemoteDraftExchange {
        RemoteDraftExchange { words, wordCount in
            words ?? [Int32](repeating: 0, count: wordCount)
        }
    }

    /// The exchange over `group`: one all-sum, as ``RemoteOpAnnouncement/exchange(_:group:)``
    /// sums an announce.
    ///
    /// - Parameter group: The group of the ranks.
    /// - Returns: The exchange.
    public static func group(_ group: MLXDistributed.Group) -> RemoteDraftExchange {
        RemoteDraftExchange { words, wordCount in
            try RemoteOpAnnouncement.sum(
                words ?? [Int32](repeating: 0, count: wordCount), group: group)
        }
    }

    /// The words of the drafts of a round of at most `draftLimit` drafts.
    ///
    /// - Parameter draftLimit: The most drafts of the round.
    /// - Returns: The word count.
    static func wordCount(draftLimit: Int) -> Int {
        draftLimit + 1
    }

    /// The drafts of rank 0 on every rank: rank 0 passes its drafts, any other rank nil.
    ///
    /// - Parameters:
    ///   - drafts: The drafts of rank 0, or nil on another rank.
    ///   - draftLimit: The most drafts of the round.
    /// - Returns: The drafts of rank 0.
    /// - Throws: The error of the exchange, or ``RemoteModelExecutorError/draftWords(count:)``
    ///   for words that do not hold at most `draftLimit` drafts.
    func drafts(_ drafts: [Int32]?, draftLimit: Int) throws -> [Int32] {
        let wordCount = Self.wordCount(draftLimit: draftLimit)
        let local = drafts.map { drafts in
            [Int32(drafts.count)] + drafts
                + [Int32](repeating: 0, count: wordCount - 1 - drafts.count)
        }
        let words = try exchange(local, wordCount)
        guard words.count == wordCount, let count = words.first.map(Int.init),
            (0 ... draftLimit).contains(count)
        else {
            throw RemoteModelExecutorError.draftWords(count: words.first.map(Int.init) ?? -1)
        }
        return Array(words[1 ..< 1 + count])
    }
}
