// Copyright © 2026 Apple Inc.
//
// The decode and verify attention kernels of a grouped-query model over an
// 8-bit affine `QuantizedKVCache`, for a head width of 256, as Qwen3.8 with
// `GenerateParameters(kvBits: 8)`.
//
// The quantized path of MLX (`quantizedScaledDotProductAttention`) runs a
// quantized matmul for the scores, a softmax and a quantized matmul for the
// values, and it writes the scores between them. At a 64k context a decode
// step over its 8-bit cache is slower than ``DecodeAttentionKernel`` over the
// bfloat16 cache of twice the bytes, and a verify of a few tokens is several
// times slower than ``VerifyAttentionKernel``. These kernels are those two
// kernels with readers of the packed rows: each number is
// `scale * code + bias`, with one scale and one bias for each group of 64
// numbers of a row, dequantized inline.
//
//  1. A decode step: the block pass of ``DecodeAttentionKernel``, whose lane
//     reads its 8 numbers of a row as 2 packed words and the scale and the
//     bias of their group, in float32.
//  2. A verify: the block pass of ``VerifyAttentionKernel``, whose loader
//     SIMD groups read the packed key and value tiles and write them
//     dequantized, in the dtype of the queries, to the threadgroup tiles that
//     the SIMD-group matrices read.
//  3. The combine pass of ``DecodeAttentionKernel`` for both.

import Foundation
import MLX

/// The decode and verify attention of a grouped-query model with a head width
/// of 256 over an 8-bit affine quantized key and value cache, in Metal kernels
/// that dequantize the packed rows inline.
///
/// The kernel output is not bit-exact with the quantized path of MLX: the
/// kernels add the keys in blocks with an online softmax, in float32, in
/// another order. The verify kernel also rounds the dequantized numbers and
/// the softmax weights to the dtype of the queries for its matrix products.
package enum QuantizedKVAttentionKernel {

    /// The `(packed numbers, scales, biases)` of the keys or of the values,
    /// as a `QuantizedKVCacheProtocol` cache gives them.
    package typealias QuantizedRows = (MLXArray, MLXArray, MLXArray?)

    /// The only bit width the kernels read.
    static let bits = 8

    /// The only group size the kernels read: the numbers of a row that share
    /// one scale and one bias.
    static let groupSize = 64

    /// The bits of one packed word.
    private static let wordBits = UInt32.bitWidth

    /// The numbers one packed word holds.
    private static let numbersPerWord = wordBits / bits

    /// The mask of the code of one number in a packed word.
    private static let codeMask = (1 << bits) - 1

    /// The packed words of one row of one head.
    private static let wordsPerRow = PrefillAttentionKernel.headWidth / numbersPerWord

    /// The groups of one row of one head.
    private static let groupsPerRow = PrefillAttentionKernel.headWidth / groupSize

    /// The Metal declarations of the packed layout, for the header of each
    /// kernel. Number `n` of a word stands in its bits from `n * KV_BITS`.
    private static let layoutDeclarations = """
        constexpr constant int KV_BITS = \(bits);
        constexpr constant int GROUP_SIZE = \(groupSize);
        constexpr constant int NUMBERS_PER_WORD = \(numbersPerWord);
        constexpr constant uint CODE_MASK = \(codeMask);

        inline float dequantized_number(uint packed, int number, float scale, float bias) {
            return scale * float((packed >> (number * KV_BITS)) & CODE_MASK) + bias;
        }

        """

    /// The header of the decode kernel: the header of
    /// ``DecodeAttentionKernel`` and a lane reader of packed rows.
    ///
    /// `QuantizedLaneRows` reads the numbers of one lane, which share one
    /// group, as ``DecodeAttentionKernel`` reads plain rows with `LaneRows`.
    /// With `UNIT_STEP` the words stand side by side.
    private static let decodeHeader =
        DecodeAttentionKernel.header + layoutDeclarations + """
            template <typename T>
            struct QuantizedLaneRows {
                const device uint32_t* words;
                const device T* scales;
                const device T* biases;
                int64_t rowStep;
                int64_t wordStep;
                int64_t scaleRowStep;
                int64_t biasRowStep;

                template <bool UNIT_STEP, int COUNT>
                inline void read(int row, thread float (&numbers)[COUNT]) const {
                    static_assert(COUNT % NUMBERS_PER_WORD == 0, "whole words for each lane");
                    const device uint32_t* source = words + row * rowStep;
                    float scale = float(scales[row * scaleRowStep]);
                    float bias = float(biases[row * biasRowStep]);
                    UNROLL_FULL for (int word = 0; word < COUNT / NUMBERS_PER_WORD; word++) {
                        uint packed = source[UNIT_STEP ? word : word * wordStep];
                        UNROLL_FULL for (int number = 0; number < NUMBERS_PER_WORD; number++) {
                            numbers[word * NUMBERS_PER_WORD + number] =
                                dequantized_number(packed, number, scale, bias);
                        }
                    }
                }
            };

            template <typename T, typename Strides>
            inline QuantizedLaneRows<T> quantized_lane_rows(
                const device uint32_t* words, const device T* scales, const device T* biases,
                Strides wordStrides, Strides scaleStrides, Strides biasStrides, uint sequence,
                uint kvHead, uint firstNumber
            ) {
                uint group = firstNumber / GROUP_SIZE;
                return {
                    words + sequence * wordStrides[SEQUENCE_AXIS] + kvHead * wordStrides[HEAD_AXIS]
                        + (firstNumber / NUMBERS_PER_WORD) * wordStrides[WIDTH_AXIS],
                    scales + sequence * scaleStrides[SEQUENCE_AXIS] + kvHead * scaleStrides[HEAD_AXIS]
                        + group * scaleStrides[WIDTH_AXIS],
                    biases + sequence * biasStrides[SEQUENCE_AXIS] + kvHead * biasStrides[HEAD_AXIS]
                        + group * biasStrides[WIDTH_AXIS],
                    wordStrides[ROW_AXIS], wordStrides[WIDTH_AXIS], scaleStrides[ROW_AXIS],
                    biasStrides[ROW_AXIS]};
            }

            """

    /// The rows part of the decode block pass: lane readers of the packed
    /// keys and values.
    private static let laneRowsSource = """

        static_assert(GROUP_SIZE % numbersPerLane == 0, "the numbers of one lane share one group");
        QuantizedLaneRows<T> keyRows = quantized_lane_rows<T>(
            keys, keyScales, keyBiases, keys_strides, keyScales_strides, keyBiases_strides,
            sequence, kvHead, laneFirstNumber);
        QuantizedLaneRows<T> valueRows = quantized_lane_rows<T>(
            values, valueScales, valueBiases, values_strides, valueScales_strides,
            valueBiases_strides, sequence, kvHead, laneFirstNumber);

        """

    /// The header of the verify kernel: the header of
    /// ``PrefillAttentionKernel`` and a tile loader of packed rows.
    ///
    /// `QuantizedTileRows` reads packed words and the scale and the bias of a
    /// group from the rows of one KV head. `copy_quantized_tiles` copies one
    /// key tile and one value tile to threadgroup memory, dequantized: each
    /// thread issues all its loads first, then all its stores. With `WIDE`,
    /// one load reads 16 bytes, 16 numbers of one group; this needs
    /// side-by-side words and rows on 16-byte steps. The `load_tiles`
    /// overload for `QuantizedTileRows` picks the form, thus the loop of
    /// ``PrefillAttentionKernel`` loads packed tiles with no change.
    private static let verifyHeader =
        PrefillAttentionKernel.header + layoutDeclarations + """
            template <typename T>
            struct QuantizedTileRows {
                const device uint32_t* words;
                const device T* scales;
                const device T* biases;
                int64_t rowStep;
                int64_t wordStep;
                int64_t scaleRowStep;
                int64_t scaleStep;
                int64_t biasRowStep;
                int64_t biasStep;

                template <typename Unit>
                inline Unit read(int row, int word) const {
                    return *(const device Unit*)(words + row * rowStep + word * wordStep);
                }

                inline float scale(int row, int group) const {
                    return float(scales[row * scaleRowStep + group * scaleStep]);
                }

                inline float bias(int row, int group) const {
                    return float(biases[row * biasRowStep + group * biasStep]);
                }
            };

            template <typename T, typename Strides>
            inline QuantizedTileRows<T> quantized_tile_rows(
                const device uint32_t* words, const device T* scales, const device T* biases,
                Strides wordStrides, Strides scaleStrides, Strides biasStrides, uint sequence,
                uint kvHead
            ) {
                return {
                    words + sequence * wordStrides[SEQUENCE_AXIS] + kvHead * wordStrides[HEAD_AXIS],
                    scales + sequence * scaleStrides[SEQUENCE_AXIS] + kvHead * scaleStrides[HEAD_AXIS],
                    biases + sequence * biasStrides[SEQUENCE_AXIS] + kvHead * biasStrides[HEAD_AXIS],
                    wordStrides[ROW_AXIS], wordStrides[WIDTH_AXIS], scaleStrides[ROW_AXIS],
                    scaleStrides[WIDTH_AXIS], biasStrides[ROW_AXIS], biasStrides[WIDTH_AXIS]};
            }

            inline uint unit_word(uint4 unit, int word) {
                return unit[word];
            }

            inline uint unit_word(uint unit, int) {
                return unit;
            }

            template <typename T, typename Unit>
            inline void store_dequantized(threadgroup T* destination, Unit unit, float scale, float bias) {
                constexpr int words = sizeof(Unit) / sizeof(uint);
                UNROLL_FULL for (int word = 0; word < words; word++) {
                    uint packed = unit_word(unit, word);
                    UNROLL_FULL for (int number = 0; number < NUMBERS_PER_WORD; number++) {
                        destination[word * NUMBERS_PER_WORD + number] =
                            T(dequantized_number(packed, number, scale, bias));
                    }
                }
            }

            template <typename T, int DIM, int ROWS, int THREADS, bool WIDE>
            inline void copy_quantized_tiles(
                threadgroup T* keyTile, threadgroup T* valueTile, QuantizedTileRows<T> keys,
                QuantizedTileRows<T> values, int first, int count, uint threadIndex
            ) {
                typedef typename metal::conditional<WIDE, uint4, uint>::type Unit;
                constexpr int perUnit = sizeof(Unit) / sizeof(uint) * NUMBERS_PER_WORD;
                constexpr int unitsPerRow = DIM / perUnit;
                constexpr int units = ROWS * unitsPerRow;
                constexpr int perThread = (units + THREADS - 1) / THREADS;
                constexpr int tileStride = DIM + TILE_PADDING;
                static_assert(GROUP_SIZE % perUnit == 0, "the numbers of one load share one group");
                Unit keyUnits[perThread];
                Unit valueUnits[perThread];
                float keyScales[perThread];
                float keyBiases[perThread];
                float valueScales[perThread];
                float valueBiases[perThread];
                UNROLL_FULL for (int i = 0; i < perThread; i++) {
                    int unit = threadIndex + i * THREADS;
                    int row = first + unit / unitsPerRow;
                    int column = (unit % unitsPerRow) * perUnit;
                    int group = column / GROUP_SIZE;
                    int word = column / NUMBERS_PER_WORD;
                    bool real = (units % THREADS == 0 || unit < units) && row < count;
                    keyUnits[i] = real ? keys.template read<Unit>(row, word) : Unit(0);
                    valueUnits[i] = real ? values.template read<Unit>(row, word) : Unit(0);
                    keyScales[i] = real ? keys.scale(row, group) : 0.0f;
                    keyBiases[i] = real ? keys.bias(row, group) : 0.0f;
                    valueScales[i] = real ? values.scale(row, group) : 0.0f;
                    valueBiases[i] = real ? values.bias(row, group) : 0.0f;
                }
                UNROLL_FULL for (int i = 0; i < perThread; i++) {
                    int unit = threadIndex + i * THREADS;
                    if (units % THREADS == 0 || unit < units) {
                        int at = (unit / unitsPerRow) * tileStride + (unit % unitsPerRow) * perUnit;
                        store_dequantized(keyTile + at, keyUnits[i], keyScales[i], keyBiases[i]);
                        store_dequantized(valueTile + at, valueUnits[i], valueScales[i], valueBiases[i]);
                    }
                }
            }

            template <typename T, int DIM, int ROWS, int THREADS>
            inline void load_tiles(
                bool wide, threadgroup T* keyTile, threadgroup T* valueTile, QuantizedTileRows<T> keys,
                QuantizedTileRows<T> values, int first, int count, uint threadIndex
            ) {
                if (wide) {
                    copy_quantized_tiles<T, DIM, ROWS, THREADS, true>(
                        keyTile, valueTile, keys, values, first, count, threadIndex);
                } else {
                    copy_quantized_tiles<T, DIM, ROWS, THREADS, false>(
                        keyTile, valueTile, keys, values, first, count, threadIndex);
                }
            }

            """

    /// The rows part of the verify block pass: tile readers of the packed
    /// keys and values, and the flag of their 16-byte loads.
    private static let tileRowsSource = """

        QuantizedTileRows<T> keyRows = quantized_tile_rows<T>(
            keys, keyScales, keyBiases, keys_strides, keyScales_strides, keyBiases_strides,
            sequence, kvHead);
        QuantizedTileRows<T> valueRows = quantized_tile_rows<T>(
            values, valueScales, valueBiases, values_strides, valueScales_strides,
            valueBiases_strides, sequence, kvHead);
        bool wide = wide_rows<uint32_t>(keys_strides) && wide_rows<uint32_t>(values_strides);

        """

    /// The names of the inputs of the block pass kernels: the queries, the
    /// packed keys and values with their scales and biases, the score factor
    /// and the keys of one block.
    private static let inputNames = [
        "queries", "keys", "keyScales", "keyBiases", "values", "valueScales", "valueBiases",
        "scale", "blockKeys",
    ]

    /// The names of the outputs of the block pass kernels, the block results
    /// that the combine pass of ``DecodeAttentionKernel`` reads.
    private static let outputNames = ["partials", "largestScores", "weightTotals"]

    /// The block pass kernel of a decode step. It reads the strides of each
    /// input, so a cache slice reaches it with no copy.
    private static let decodeKernel = MLXFast.metalKernel(
        name: "quantized_decode_attention_blocks",
        inputNames: inputNames,
        outputNames: outputNames,
        source: DecodeAttentionKernel.blockSetupSource + laneRowsSource
            + DecodeAttentionKernel.blockSoftmaxSource,
        header: decodeHeader,
        ensureRowContiguous: false)

    /// The block pass kernel of a verify. It reads the strides of each
    /// input, so a cache slice reaches it with no copy.
    private static let verifyKernel = MLXFast.metalKernel(
        name: "quantized_verify_attention_blocks",
        inputNames: inputNames,
        outputNames: outputNames,
        source: PrefillAttentionKernel.layoutSource + tileRowsSource
            + PrefillAttentionKernel.querySource + VerifyAttentionKernel.blockRangeSource
            + PrefillAttentionKernel.loopSource + VerifyAttentionKernel.resultSource,
        header: verifyHeader,
        ensureRowContiguous: false)

    /// The packed numbers, the scales and the biases of the keys or of the
    /// values of an affine quantized cache.
    private struct AffineRows {
        /// The packed numbers, shape `(batch, kvHeads, keys, words)`.
        let words: MLXArray
        /// The scale of each group, shape `(batch, kvHeads, keys, groups)`.
        let scales: MLXArray
        /// The bias of each group, of the shape of ``scales``.
        let biases: MLXArray

        /// Reads the rows of a quantized cache, or gives `nil` when they
        /// carry no biases.
        init?(_ rows: QuantizedRows) {
            guard let biases = rows.2 else { return nil }
            words = rows.0
            scales = rows.1
            self.biases = biases
        }

        /// The arrays in the order of the kernel inputs.
        var arrays: [MLXArray] {
            [words, scales, biases]
        }

        /// The number of cached keys.
        var count: Int {
            words.dim(HeadRowsAxis.row)
        }

        /// Tells if the rows hold the packed layout of the kernels for
        /// queries of one dtype: 8-bit codes of 256 numbers in 32-bit words,
        /// and one scale and one bias of that dtype for each group of 64.
        func holdTheLayout(for dtype: DType) -> Bool {
            guard words.ndim == HeadRowsAxis.count, words.dtype == .uint32,
                words.dim(HeadRowsAxis.width) == wordsPerRow
            else { return false }
            var groupShape = words.shape
            groupShape[HeadRowsAxis.width] = groupsPerRow
            return scales.shape == groupShape && biases.shape == groupShape
                && scales.dtype == dtype && biases.dtype == dtype
        }
    }

    /// Tells if the kernels run one attention call over a quantized cache:
    /// the default device is the GPU, the cache is affine with 8 bits and
    /// groups of 64, the heads are 256 numbers wide, the queries share the
    /// float dtype of the scales and the biases, and the call reads at least
    /// ``DecodeAttentionKernel/fewestKeys`` keys. A decode step of one query
    /// token takes no mask, and its query heads fit the lanes of
    /// ``DecodeAttentionKernel``. A verify of
    /// ``VerifyAttentionKernel/fewestQueries`` to
    /// ``VerifyAttentionKernel/mostQueries`` tokens takes a causal mask or no
    /// mask, and its query heads make whole head groups of
    /// ``PrefillAttentionKernel``.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The quantized keys, `(batch, kvHeads, keys, …)` each.
    ///   - values: The quantized values, of the shapes of `keys`.
    ///   - mask: The mask of the call.
    ///   - groupSize: The group size of the cache.
    ///   - bits: The bit width of the cache.
    ///   - mode: The quantization mode of the cache.
    /// - Returns: True when the kernels run the call.
    static func canRun(
        queries: MLXArray, keys: QuantizedRows, values: QuantizedRows,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, groupSize: Int, bits: Int,
        mode: QuantizationMode
    ) -> Bool {
        guard mode == .affine, bits == Self.bits, groupSize == Self.groupSize,
            let keyRows = AffineRows(keys), let valueRows = AffineRows(values),
            layoutHolds(queries: queries, keys: keyRows, values: valueRows),
            keyRows.count >= DecodeAttentionKernel.fewestKeys,
            PrefillAttentionKernel.causalFlag(of: mask) != nil
        else { return false }
        let queryHeads = queries.dim(HeadRowsAxis.head)
        let kvHeads = keyRows.words.dim(HeadRowsAxis.head)
        let tokens = queries.dim(HeadRowsAxis.row)
        if tokens == 1 {
            guard case .none = mask else { return false }
            return DecodeAttentionKernel.headShapeFits(
                queryHeads: queryHeads, kvHeads: kvHeads, headDim: PrefillAttentionKernel.headWidth)
        }
        return (VerifyAttentionKernel.fewestQueries ... VerifyAttentionKernel.mostQueries)
            .contains(tokens)
            && PrefillAttentionKernel.headGroupsFit(queryHeads: queryHeads, kvHeads: kvHeads)
    }

    /// Tells if the shapes and the dtypes of a call hold the layout of the
    /// kernels.
    private static func layoutHolds(
        queries: MLXArray, keys: AffineRows, values: AffineRows
    ) -> Bool {
        Device.defaultDevice().deviceType == .gpu
            && queries.ndim == HeadRowsAxis.count
            && queries.dim(HeadRowsAxis.width) == PrefillAttentionKernel.headWidth
            && PrefillAttentionKernel.floatTypes.contains(queries.dtype)
            && keys.holdTheLayout(for: queries.dtype) && values.holdTheLayout(for: queries.dtype)
            && values.words.shape == keys.words.shape
            && queries.dim(HeadRowsAxis.batch) == keys.words.dim(HeadRowsAxis.batch)
    }

    /// The attention output of each query token over the quantized cache:
    /// the decode kernel for one token, the verify kernel for more.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - keys: The quantized keys. Slices of the first rows of larger cache
    ///     buffers reach the kernels with no copy.
    ///   - values: The quantized values, of the shapes of `keys`.
    ///   - scale: The factor each score takes before the softmax.
    ///   - mask: The mask of the call: no mask for a decode step, a causal
    ///     mask or no mask for a verify.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    static func attention(
        queries: MLXArray, keys: QuantizedRows, values: QuantizedRows, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard
            canRun(
                queries: queries, keys: keys, values: values, mask: mask, groupSize: groupSize,
                bits: bits, mode: .affine),
            let keyRows = AffineRows(keys), let valueRows = AffineRows(values),
            let causal = PrefillAttentionKernel.causalFlag(of: mask)
        else {
            preconditionFailure(
                "the quantized attention kernels cannot read queries \(queries.shape) "
                    + "over keys \(keys.0.shape)")
        }
        let cache = keyRows.arrays + valueRows.arrays
        guard queries.dim(HeadRowsAxis.row) == 1 else {
            return DecodeAttentionKernel.combine(
                VerifyAttentionKernel.blockResults(
                    verifyKernel, queries: queries, keys: keyRows.words, cache: cache,
                    scale: scale, causal: causal, blocks: nil),
                like: queries)
        }
        return DecodeAttentionKernel.blockAttention(
            decodeKernel, cache: cache, queries: queries,
            kvHeads: keyRows.words.dim(HeadRowsAxis.head), keyCount: keyRows.count, scale: scale)
    }

    /// The scaled dot-product attention of a call over a quantized cache:
    /// these kernels when they run the call, else
    /// ``MLXLMCommon/quantizedCacheAttention(queries:quantizedKeys:quantizedValues:scale:mask:groupSize:bits:mode:sinks:)``.
    ///
    /// - Parameters:
    ///   - queries: The queries, shape `(batch, queryHeads, tokens, headDim)`.
    ///   - quantizedKeys: The quantized keys the cache gives.
    ///   - quantizedValues: The quantized values the cache gives.
    ///   - scale: The factor each score takes before the softmax.
    ///   - mask: The mask of the call.
    ///   - groupSize: The group size of the cache.
    ///   - bits: The bit width of the cache.
    ///   - mode: The quantization mode of the cache.
    /// - Returns: The attention output, of the shape and the dtype of
    ///   `queries`.
    package static func scaledDotProductAttention(
        queries: MLXArray, quantizedKeys: QuantizedRows, quantizedValues: QuantizedRows,
        scale: Float, mask: MLXFast.ScaledDotProductAttentionMaskMode, groupSize: Int,
        bits: Int, mode: QuantizationMode
    ) -> MLXArray {
        guard
            canRun(
                queries: queries, keys: quantizedKeys, values: quantizedValues, mask: mask,
                groupSize: groupSize, bits: bits, mode: mode)
        else {
            return quantizedCacheAttention(
                queries: queries, quantizedKeys: quantizedKeys, quantizedValues: quantizedValues,
                scale: scale, mask: mask, groupSize: groupSize, bits: bits, mode: mode)
        }
        return attention(
            queries: queries, keys: quantizedKeys, values: quantizedValues, scale: scale,
            mask: mask)
    }
}
