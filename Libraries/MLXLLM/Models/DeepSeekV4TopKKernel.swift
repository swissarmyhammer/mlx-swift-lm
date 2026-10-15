// Copyright © 2026 Apple Inc.
//
// The radix-select top-k Metal kernel of the DeepSeek-V4 indexer.
//
// The method follows the kernel `kernel_top_k_f32_i32` of llama.cpp,
// ggml/src/ggml-metal/kernels/argsort.metal: one threadgroup for each row,
// an order-preserving integer key for each float, and one histogram pass for
// each digit of the key, which fixes the digit of the cut. Manual
// transcription of the method; no git ancestry. Three details do not come
// from that kernel:
//
//  1. A NaN score takes the lowest key, and a negative zero takes the key of
//     a positive zero. That is the order of `argPartition`, whose sort puts a
//     NaN after every number and finds no order between the two zeros.
//  2. The last pass writes the picks in ascending index order: each thread
//     takes one contiguous run of the row, and a prefix sum over the threads
//     gives the place of the first pick of each run.
//  3. When more scores tie at the cut than the budget holds, the scores of
//     the lowest indices win. Thus the answer does not depend on the order
//     in which the threads run.

import Foundation
import MLX

/// One Metal kernel that picks the indices of the highest scores of each row.
///
/// `argPartition` in MLX is a full sort of the row. The indexer of a
/// DeepSeek-V4 compressed layer picks 512 of up to 8192 chunks for each
/// query, and a sort of 8192 values takes the multi-block sort of MLX, which
/// is several dispatches. This kernel finds the score at the cut in a fixed
/// number of passes over the row, and then writes every index above the cut.
///
/// The kernel takes one threadgroup for each row. Each pass builds a
/// histogram of one digit of the key of each score that matches the digits
/// the earlier passes fixed. A prefix sum over the bins, from the highest bin
/// down, finds the bin that holds the cut. After the last pass, the key at the
/// cut is known exactly, and one more pass writes the picks.
enum DeepSeekV4TopKKernel {

    /// The number of threads in the threadgroup of one row. It must not be
    /// less than the number of bins of one digit, because each bin takes one
    /// thread in the prefix sum over the bins. The prefix sum over the
    /// threads keeps one total for each SIMD group in the lanes of one SIMD
    /// group, thus this number must not be more than the square of
    /// ``DeepSeekV4MetalKernel/simdGroupWidth``.
    private static let threadsPerRow = 1024

    /// The number of key bits that one histogram pass fixes. A digit of this
    /// width gives a histogram of 256 bins, and a 32-bit key takes 4 passes.
    private static let digitBits = 8

    /// The axis of the scores that holds one row.
    private static let rowAxis = -1

    /// The Metal source. The template arguments are `COUNT`, the number of
    /// picks for each row; `THREADS`, the threadgroup width; `SIMD_WIDTH`,
    /// the number of lanes in one SIMD group; `KEY_BITS`, the width of the
    /// key of one score; and `DIGIT_BITS`, the width of one digit of the key.
    /// `DIGIT_BITS` must divide `KEY_BITS`. The row width comes from `scores_shape`, thus a
    /// pool that grows does not compile a new specialization.
    ///
    /// `prefix` holds the key digits the earlier passes fixed, and
    /// `prefixMask` holds the bits of those digits. `remaining` is the number
    /// of picks that the keys that match `prefix` must still give. The bin of
    /// the cut is the bin where the count of the keys in that bin and in every
    /// higher bin first reaches `remaining`.
    ///
    /// After the last pass, `prefix` is the key at the cut. Every key above it
    /// is a pick, and the first `remaining` keys equal to it, in index order,
    /// are picks.
    ///
    /// The source indexes `scores` and keeps no pointer into it, because a
    /// row can hold fewer values than
    /// ``DeepSeekV4MetalKernel/smallestDeviceInput``.
    private static let source = """
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        uint width = scores_shape[scores_ndim - 1];
        size_t rowFirst = size_t(row) * width;
        device uint* rowPicks = picks + size_t(row) * COUNT;

        constexpr uint bins = 1u << DIGIT_BITS;
        constexpr uint digitMask = bins - 1;
        constexpr uint passes = KEY_BITS / DIGIT_BITS;
        constexpr uint simdGroups = THREADS / SIMD_WIDTH;

        threadgroup atomic_uint histogram[bins];
        threadgroup uint totals[SIMD_WIDTH];
        threadgroup uint cutDigit;
        threadgroup uint cutAbove;

        uint prefix = 0;
        uint prefixMask = 0;
        uint remaining = COUNT;
        for (uint pass = 0; pass < passes; pass++) {
            uint shift = KEY_BITS - (pass + 1) * DIGIT_BITS;
            for (uint bin = lane; bin < bins; bin += THREADS) {
                atomic_store_explicit(&histogram[bin], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < width; i += THREADS) {
                uint key = ordered_key(scores[rowFirst + i]);
                if ((key & prefixMask) == prefix) {
                    atomic_fetch_add_explicit(
                        &histogram[(key >> shift) & digitMask], 1u, memory_order_relaxed);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint bin = digitMask - lane;
            uint count = lane < bins
                ? atomic_load_explicit(&histogram[bin], memory_order_relaxed) : 0u;
            uint higher = group_exclusive_sum(count, totals, lane, SIMD_WIDTH, simdGroups);
            if (lane < bins && higher < remaining && higher + count >= remaining) {
                cutDigit = bin;
                cutAbove = higher;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= cutDigit << shift;
            prefixMask |= digitMask << shift;
            remaining -= cutAbove;
        }

        uint span = (width + THREADS - 1) / THREADS;
        uint first = min(lane * span, width);
        uint end = min(first + span, width);
        uint above = 0;
        uint equal = 0;
        for (uint i = first; i < end; i++) {
            uint key = ordered_key(scores[rowFirst + i]);
            above += key > prefix ? 1u : 0u;
            equal += key == prefix ? 1u : 0u;
        }
        uint place = group_exclusive_sum(above, totals, lane, SIMD_WIDTH, simdGroups);
        uint equalBefore = group_exclusive_sum(equal, totals, lane, SIMD_WIDTH, simdGroups);
        place += min(equalBefore, remaining);
        for (uint i = first; i < end; i++) {
            uint key = ordered_key(scores[rowFirst + i]);
            bool tie = key == prefix;
            if (key > prefix || (tie && equalBefore < remaining)) {
                rowPicks[place++] = i;
            }
            equalBefore += tie ? 1u : 0u;
        }
        """

    /// The Metal header.
    ///
    /// `ordered_key` maps a float32 score to an unsigned key whose order is
    /// the order of the scores: a positive score sets the sign bit, and a
    /// negative score flips every bit. A NaN takes `nanKey`, the lowest key,
    /// which no number takes, because only a NaN has the bits that flip to
    /// it. A negative zero takes the key of a positive zero. The test reads
    /// the bits and not a float compare, because MLX compiles the kernel with
    /// fast math, which may drop an `isnan`.
    ///
    /// `group_exclusive_sum` gives each thread the sum of `value` over the
    /// threads of lower index. It keeps one total for each SIMD group in
    /// `totals`, and the last barrier lets the next call write `totals`
    /// again. Every thread of the threadgroup must call it.
    ///
    /// MLX writes the kernel signature directly after the header, thus the
    /// header ends in a line break.
    private static let header = """
        constant uint signBit = 0x80000000u;
        constant uint infinityBits = 0x7F800000u;
        constant uint nanKey = 0u;

        inline uint ordered_key(float score) {
            uint bits = as_type<uint>(score);
            if ((bits & ~signBit) > infinityBits) {
                return nanKey;
            }
            bits = (bits == signBit) ? 0u : bits;
            return (bits & signBit) ? ~bits : (bits | signBit);
        }

        inline uint group_exclusive_sum(
            uint value, threadgroup uint* totals, uint lane, uint simdWidth, uint simdGroups
        ) {
            uint simdGroup = lane / simdWidth;
            uint simdLane = lane % simdWidth;
            uint inclusive = simd_prefix_inclusive_sum(value);
            if (simdLane == simdWidth - 1) {
                totals[simdGroup] = inclusive;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simdGroup == 0) {
                uint total = simdLane < simdGroups ? totals[simdLane] : 0u;
                uint earlier = simd_prefix_exclusive_sum(total);
                if (simdLane < simdGroups) {
                    totals[simdLane] = earlier;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint answer = totals[simdGroup] + inclusive - value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return answer;
        }

        """

    /// The compiled kernel. MLX compiles one specialization for each set of
    /// template arguments and keeps it.
    private static let kernel = MLXFast.metalKernel(
        name: "deepseek_v4_radix_top_k",
        inputNames: ["scores"],
        outputNames: ["picks"],
        source: source,
        header: header)

    /// Tells if the kernel can run. It is a Metal kernel, thus it runs only
    /// when the default device is the GPU.
    static var canRun: Bool {
        DeepSeekV4MetalKernel.deviceIsGPU
    }

    /// The indices of the `count` highest scores of each row, in ascending
    /// index order.
    ///
    /// When no two scores of a row tie, the set of picks is the set of
    /// `argPartition(-scores, kth: count - 1)[..., 0 ..< count]`. A NaN score
    /// ranks below every number, as in `argPartition`. When more scores tie at
    /// the cut than the budget holds, the scores of the lowest indices win.
    ///
    /// - Parameters:
    ///   - scores: The scores, shape `(..., width)`. They move to float32
    ///     first.
    ///   - count: The number of picks for each row. It must be 1 or more, and
    ///     not more than the row width.
    /// - Returns: The indices of the picks, shape `(..., count)`, in uint32,
    ///   the dtype of the `argPartition` answer.
    static func topIndices(_ scores: MLXArray, count: Int) -> MLXArray {
        let width = scores.dim(rowAxis)
        precondition(
            count > 0 && count <= width,
            "the kernel picks 1 to \(width) scores of each row, not \(count)")
        let leading = Array(scores.shape.dropLast())
        let rows = leading.reduce(1, *)
        return kernel(
            [scores.asType(.float32)],
            template: [
                ("COUNT", count), ("THREADS", threadsPerRow),
                ("SIMD_WIDTH", DeepSeekV4MetalKernel.simdGroupWidth),
                ("KEY_BITS", UInt32.bitWidth), ("DIGIT_BITS", digitBits),
            ],
            grid: (rows * threadsPerRow, 1, 1),
            threadGroup: (threadsPerRow, 1, 1),
            outputShapes: [leading + [count]],
            outputDTypes: [.uint32])[0]
    }
}
