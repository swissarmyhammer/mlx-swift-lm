// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Greedy decoding at exact logit ties: the lowest tied token id wins, whatever the element
/// type and the shape of the logits, and whether a local model computed them or a worker sent
/// them as ``RemoteLogits``.
///
/// A local and a clustered greedy run then agree token for token wherever their logits agree
/// bit for bit (^yc6jghy).
@Suite(.serialized)
struct ArgMaxSamplerTieTests {

    /// The two tied token ids of a case, in either order: the sampler must pick the lower.
    struct Tie: Sendable, CustomTestStringConvertible {
        /// The tied ids.
        let ids: [Int]

        /// The id greedy decoding must pick.
        var lowest: Int { ids.min() ?? 0 }

        var testDescription: String { ids.map(String.init).joined(separator: " = ") }
    }

    /// A logits layout the sampler reads.
    enum Layout: String, CaseIterable, Sendable {
        /// `[vocabulary]`.
        case row
        /// `[1, vocabulary]`, as a sampler of a decode step reads it.
        case batch
        /// `[1, 1, vocabulary]`, as a model returns the logits of one position.
        case position

        /// `row` in this layout.
        func shaped(_ row: MLXArray) -> MLXArray {
            switch self {
            case .row: row
            case .batch: row[.newAxis]
            case .position: row[.newAxis, .newAxis]
            }
        }
    }

    /// The vocabulary of the Qwen 3.5 family, where the parity suite found its tie.
    static let vocabularySize = 248_320

    /// The tied logit; every other logit is lower. Exact in each element type.
    static let tiedLogit: Float = 9.5

    /// The ties the suite checks.
    ///
    /// - 17 and 18: neighbours, the Qwen3.8 tie of the parity suite.
    /// - 40172 and 38: far apart, the higher id first, the Inkling tie.
    /// - 4096 and 5: the higher id is read by the first thread of the first SIMD lane on the
    ///   reduction's second pass, the lower id by the second lane on its first pass, so the
    ///   higher id reaches the lane reduction first.
    /// - 1 and the last id: the two ends of the vocabulary.
    /// - three ids at once.
    static let ties = [
        Tie(ids: [17, 18]), Tie(ids: [40172, 38]), Tie(ids: [4096, 5]),
        Tie(ids: [vocabularySize - 1, 1]), Tie(ids: [200_001, 131_072, 65_537]),
    ]

    /// The element types a model returns logits in.
    static let dtypes: [DType] = [.bfloat16, .float16, .float32]

    /// A float32 row of ``vocabularySize`` logits below ``tiedLogit``, with ``tiedLogit`` at
    /// each id of `tie`. Each logit is a multiple of 1/16 under 8 in magnitude, which every
    /// element type holds exactly.
    static func row(tiedAt tie: Tie) -> MLXArray {
        var values = (0 ..< vocabularySize).map { Float($0 % 97) / 16 - 8 }
        for id in tie.ids { values[id] = tiedLogit }
        return MLXArray(values)
    }

    /// The token ``ArgMaxSampler`` picks from `logits`.
    static func greedyToken(_ logits: MLXArray) -> Int {
        ArgMaxSampler().sample(logits: logits).item(Int.self)
    }

    @Test(arguments: dtypes, ties)
    func picksTheLowestTiedIdOfARow(dtype: DType, tie: Tie) {
        let logits = Self.row(tiedAt: tie).asType(dtype)
        #expect(Self.greedyToken(logits) == tie.lowest)
    }

    @Test(arguments: Layout.allCases, ties)
    func picksTheLowestTiedIdInEachLayout(layout: Layout, tie: Tie) {
        let logits = layout.shaped(Self.row(tiedAt: tie).asType(.bfloat16))
        #expect(Self.greedyToken(logits) == tie.lowest)
    }

    @Test(arguments: ties)
    func picksTheLowestTiedIdOfAFloat32CopyOfABFloat16Row(tie: Tie) {
        let logits = Self.row(tiedAt: tie).asType(.bfloat16).asType(.float32)
        #expect(Self.greedyToken(logits) == tie.lowest)
    }

    @Test(arguments: dtypes, ties)
    func picksTheLowestTiedIdOfRemoteLogits(dtype: DType, tie: Tie) throws {
        let local = Layout.position.shaped(Self.row(tiedAt: tie).asType(dtype))
        let remote = try RemoteLogits(local).array()
        #expect(Self.greedyToken(remote) == tie.lowest)
    }

    @Test(arguments: ties)
    func picksTheLowestTiedIdOfTheLastPositionOfAPrefill(tie: Tie) {
        let tiedRow = Self.row(tiedAt: tie).asType(.bfloat16)
        let prefill = stacked([tiedRow + 1, tiedRow])[.newAxis]
        #expect(Self.greedyToken(prefill[0..., -1, 0...]) == tie.lowest)
    }
}
