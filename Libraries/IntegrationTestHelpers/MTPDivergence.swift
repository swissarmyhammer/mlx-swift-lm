// Copyright © 2026 Apple Inc.

import MLX

/// Where MTP first parts from plain decoding, and the best logits of an uncached forward pass of
/// the target over the tokens both decodes share.
///
/// A greedy MTP round verifies its drafts in one batched forward pass, and plain decoding feeds
/// one token at a time, thus the two can round a near tie apart.
public struct MTPDivergence: CustomStringConvertible, Sendable {

    /// The best candidates of the forward pass that a divergence compares.
    public static let topCandidateCount = 2

    /// The index of the first token where the two part.
    public let step: Int

    /// The token MTP wrote there, or `nil` when MTP wrote no token there.
    public let mtpToken: Int?

    /// The best tokens of the forward pass, best first.
    public let topTokens: [Int]

    /// The logits of ``topTokens``.
    public let topLogits: [Float]

    /// Whether MTP wrote one of the two best tokens, and the two are within `gap` of each other.
    ///
    /// - Parameter gap: The largest gap of the two best logits that is a near tie.
    /// - Returns: `true` for a near tie MTP rounded the other way.
    public func isNearTie(within gap: Float) -> Bool {
        guard let mtpToken else { return false }
        return topTokens.contains(mtpToken) && topLogits[0] - topLogits[1] <= gap
    }

    /// Whether MTP wrote the second best token, less than `gap` below the best.
    ///
    /// - Parameter gap: The gap of the two best logits below which MTP may write the second.
    /// - Returns: `true` when MTP wrote the second best token of a near tie.
    public func wroteSecondBest(within gap: Float) -> Bool {
        mtpToken == topTokens[1] && topLogits[0] - topLogits[1] < gap
    }

    /// Whether MTP wrote the best token of the uncached forward pass, thus plain decoding's
    /// cached forward passes rounded the other way.
    public var wroteTheArgmax: Bool { mtpToken == topTokens[0] }

    /// One line that names the token where MTP parts from plain decoding, the token MTP wrote
    /// there, and the best tokens and logits of the forward pass.
    public var description: String {
        """
        MTP parts from plain decode at token \(step): MTP \(mtpToken ?? -1), plain best \
        \(topTokens) with logits \(topLogits)
        """
    }

    /// Where `mtp` first parts from `plain`.
    ///
    /// - Parameters:
    ///   - mtp: The tokens MTP wrote after `context`.
    ///   - plain: The tokens plain decoding wrote after `context`.
    ///   - context: The tokens both decodes followed.
    ///   - topCandidates: The best tokens and logits of an uncached forward pass of the target
    ///     over the given tokens, from its last row (see ``topCandidates(of:)``).
    /// - Returns: The divergence, or `nil` when the two are the same.
    /// - Throws: The error of `topCandidates`.
    public static func find(
        mtp: [Int], plain: [Int], context: [Int],
        topCandidates: ([Int]) async throws -> (tokens: [Int], logits: [Float])
    ) async rethrows -> MTPDivergence? {
        guard mtp != plain else { return nil }
        let comparedCount = Swift.min(mtp.count, plain.count)
        let step = (0 ..< comparedCount).first { mtp[$0] != plain[$0] } ?? comparedCount
        let (tokens, logits) = try await topCandidates(context + plain.prefix(step))
        return MTPDivergence(
            step: step, mtpToken: step < mtp.count ? mtp[step] : nil, topTokens: tokens,
            topLogits: logits)
    }

    /// The ``topCandidateCount`` best tokens of one row of logits and their logits, best first.
    ///
    /// - Parameter logits: The logits of one row, over the vocabulary.
    /// - Returns: The tokens and their float32 logits.
    public static func topCandidates(of logits: MLXArray) -> (tokens: [Int], logits: [Float]) {
        let row = logits.asType(.float32)
        let order = argSort(-row)[..<topCandidateCount]
        return (order.asArray(Int.self), row[order].asArray(Float.self))
    }
}
