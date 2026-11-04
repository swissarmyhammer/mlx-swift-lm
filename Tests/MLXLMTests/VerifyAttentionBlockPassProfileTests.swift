// Copyright © 2026 Apple Inc.
//
// A profile of the attention of a speculative verify of Qwen3.8 over 16
// full-attention layers, each with its own key cache, at 2k, 32k and 100k
// keys. It times, for each verify row count:
//
//  - the block pass of ``VerifyAttentionKernel`` alone, and the block pass
//    with the combine pass, at each block count of a sweep,
//  - the SDPA of MLX with the causal mask, the route of a verify before the
//    kernel,
//  - one decode step through ``DecodeAttentionKernel``, the floor of one read
//    of the cache.
//
// The profile holds nothing of the times. It runs only when the environment
// sets `MLX_VERIFY_ATTENTION_PROFILE=1`; `MLX_VERIFY_ATTENTION_PROFILE_KEYS` picks the key
// counts:
//
//     MLX_VERIFY_ATTENTION_PROFILE=1 xcrun xctest \
//         -XCTest MLXLMTests.VerifyAttentionBlockPassProfileTests \
//         .build/out/Products/Debug/MLXLMTests.xctest
//
// Each measurement line starts with `VERIFY BLOCK PASS PROFILE:`.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["MLX_VERIFY_ATTENTION_PROFILE"] == "1"))
struct VerifyAttentionBlockPassProfileTests {

    /// The full-attention layers of Qwen3.8, each with its own cache, thus no
    /// layer reads a cache another layer left in the GPU caches.
    private static let layers = 16

    /// The key counts of the profile: `MLX_VERIFY_ATTENTION_PROFILE_KEYS`, a list of counts
    /// split by commas, or 2k, 32k and 100k when the environment does not set it.
    private static let keyCounts =
        ProcessInfo.processInfo.environment["MLX_VERIFY_ATTENTION_PROFILE_KEYS"]?
        .split(separator: ",").compactMap { Int($0) } ?? [2048, 32_768, 100_000]

    /// The verify row counts of the profile.
    private static let rowCounts = [2, 3, 4, 5]

    /// The block counts of the sweep.
    private static let blockCounts = [4, 8, 16, 32, 64, 128]

    /// The untimed passes before the timed ones.
    private static let warmUpPasses = 3

    /// The timed passes of one measurement; the median counts.
    private static let timedPasses = 9

    /// The prefix of each measurement line.
    private static let prefix = "VERIFY BLOCK PASS PROFILE:"

    /// The median time of one pass over every layer, in milliseconds.
    ///
    /// - Parameter pass: Builds the outputs of every layer.
    /// - Returns: The median time of the evaluation of those outputs.
    private static func milliseconds(_ pass: () -> [MLXArray]) -> Double {
        for _ in 0 ..< warmUpPasses { eval(pass()) }
        var times: [Double] = []
        for _ in 0 ..< timedPasses {
            let start = DispatchTime.now().uptimeNanoseconds
            eval(pass())
            times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
        }
        return times.sorted()[times.count / 2]
    }

    /// The key and value caches of every layer.
    private static func caches(keys: Int) -> [(keys: MLXArray, values: MLXArray)] {
        (0 ..< layers).map { layer in
            let shape = [
                1, AttentionKernelCase.qwen38KVHeads, keys, AttentionKernelCase.qwen38HeadDim,
            ]
            let pair = (
                MLXRandom.normal(shape, key: MLXRandom.key(UInt64(2 * layer))).asType(.bfloat16),
                MLXRandom.normal(shape, key: MLXRandom.key(UInt64(2 * layer + 1))).asType(
                    .bfloat16)
            )
            eval(pair.0, pair.1)
            return pair
        }
    }

    /// The queries of `rows` tokens, as the model gives them.
    private static func queries(rows: Int) -> MLXArray {
        let queries = MLXRandom.normal(
            [1, rows, AttentionKernelCase.qwen38QueryHeads, AttentionKernelCase.qwen38HeadDim],
            key: MLXRandom.key(99)
        ).asType(.bfloat16).swappedAxes(1, 2)
        eval(queries)
        return queries
    }

    @Test func profileTheVerifyBlockPass() {
        let scale = 1 / Float(AttentionKernelCase.qwen38HeadDim).squareRoot()
        for keyCount in Self.keyCounts {
            let caches = Self.caches(keys: keyCount)
            let bytes = Double(caches.count * 2 * caches[0].keys.nbytes)
            let decode = Self.queries(rows: 1)
            let decodeTime = Self.milliseconds {
                caches.map {
                    DecodeAttentionKernel.attention(
                        queries: decode, keys: $0.keys, values: $0.values, scale: scale)
                }
            }
            // swiftlint:disable:next no_direct_standard_out_logs — an opt-in profile; its output is the measurement.
            print(
                Self.prefix,
                "keys \(keyCount) decode step \(String(format: "%.2f", decodeTime)) ms",
                "(\(String(format: "%.0f", bytes / decodeTime / 1e6)) GB/s of K/V),",
                "default verify blocks \(VerifyAttentionKernel.blockCount(keys: keyCount)),",
                "decode blocks \(DecodeAttentionKernel.blockCount(keys: keyCount))")
            for rows in Self.rowCounts {
                let queries = Self.queries(rows: rows)
                let sdpa = Self.milliseconds {
                    caches.map {
                        MLXFast.scaledDotProductAttention(
                            queries: queries, keys: $0.keys, values: $0.values, scale: scale,
                            mask: .causal)
                    }
                }
                var line = "keys \(keyCount) rows \(rows): SDPA \(String(format: "%.2f", sdpa))"
                for blocks in Self.blockCounts {
                    let blockPass = Self.milliseconds {
                        caches.flatMap {
                            VerifyAttentionKernel.blockResults(
                                queries: queries, keys: $0.keys, values: $0.values, scale: scale,
                                causal: true, blocks: blocks)
                        }
                    }
                    let whole = Self.milliseconds {
                        caches.map {
                            VerifyAttentionKernel.attention(
                                queries: queries, keys: $0.keys, values: $0.values, scale: scale,
                                causal: true, blocks: blocks)
                        }
                    }
                    line +=
                        ", b\(blocks) \(String(format: "%.2f", whole))"
                        + " (block \(String(format: "%.2f", blockPass)))"
                }
                // swiftlint:disable:next no_direct_standard_out_logs — an opt-in profile; its output is the measurement.
                print(Self.prefix, line, "ms")
            }
        }
    }
}
