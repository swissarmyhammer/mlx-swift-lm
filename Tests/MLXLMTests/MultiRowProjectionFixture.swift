// Copyright © 2026 Apple Inc.
//
// The shared pieces of the tests that tell where a Qwen3.5 model runs
// ``MultiRowQuantizedMatvec``: only inside a speculative verify.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// The configuration, the quantization and the row check shared by the
/// multi-row projection tests of the Qwen3.5 text model and of the Qwen3.5
/// vision-language model. The seeded, quantized build serves the Qwen3-Next
/// MLP test as well.
enum MultiRowProjectionFixture {

    /// A tiny dense Qwen3.5 text configuration whose projections fit the
    /// kernel after a 4-bit quantization in groups of 64: each input width is
    /// a multiple of 512 and each output width a multiple of 8, but for the 4
    /// outputs of the unfused `in_proj_b` and `in_proj_a`. Layer 0 is a gated
    /// delta net layer, layer 1 a full-attention layer.
    static let textConfigurationJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": \(hiddenSize),
          "num_hidden_layers": 2,
          "intermediate_size": 1024,
          "num_attention_heads": \(attentionHeads),
          "num_key_value_heads": 1,
          "head_dim": \(headDimension),
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 128,
          "linear_value_head_dim": 128,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 64,
          "rope_theta": 100000.0,
          "partial_rotary_factor": 0.25,
          "max_position_embeddings": 64,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "full_attention_interval": 2,
          "num_experts": 0
        }
        """

    /// The width of the hidden state.
    static let hiddenSize = 512

    /// The attention heads of the full-attention layer.
    static let attentionHeads = 2

    /// The width of one attention head.
    static let headDimension = 256

    /// The token axis of the queries, keys and values, shape
    /// `(batch, heads, tokens, headDimension)`.
    static let headTokenAxis = 2

    /// The group size of the quantized weights.
    static let groupSize = 64

    /// The bits of each quantized weight.
    static let bits = 4

    /// The rows of a prompt tail and of each projection check: the fewest
    /// the kernel runs.
    static let rows = MultiRowQuantizedMatvec.fewestRows

    /// The verify lengths whose projections the kernel runs: from its fewest
    /// rows up to the most tokens the GDN kernel holds in one verify.
    static let kernelVerifyTokenCounts = Array(
        MultiRowQuantizedMatvec.fewestRows ... GatedDeltaDecodeKernel.mostTokens)

    /// The seed of the rows of each projection check.
    static let rowSeed: UInt64 = 20_261_031

    /// Quantizes the layers of `model` that `filter` takes, with stock
    /// projections or with projections that always run the matvec of MLX.
    ///
    /// - Parameters:
    ///   - model: The model to quantize.
    ///   - matvecOfMLX: True to quantize each linear layer as an
    ///     ``MLXMatvecQuantizedLinear``.
    ///   - filter: Tells, from its path and the module, if a layer is
    ///     quantized.
    static func quantize(
        _ model: Module, matvecOfMLX: Bool,
        filter: (String, Module) -> Bool = { _, _ in true }
    ) {
        if matvecOfMLX {
            MLXMatvecQuantizedLinear.quantize(
                model, groupSize: groupSize, bits: bits, filter: filter)
        } else {
            MLXNN.quantize(model: model, groupSize: groupSize, bits: bits, filter: filter)
        }
    }

    /// The module `make` builds on weights seeded with `seed`, cast to
    /// bfloat16, quantized, prepared, in inference mode and evaluated.
    ///
    /// - Parameters:
    ///   - seed: The seed of the weights.
    ///   - matvecOfMLX: True to give every projection the matvec of MLX.
    ///   - filter: Tells, from its path and the module, if a layer is
    ///     quantized.
    ///   - prepare: Runs on the quantized module before it is evaluated, as
    ///     the model loader prepares a model.
    ///   - make: Builds the module.
    /// - Returns: The quantized module, ready to run.
    static func seededQuantized<M: Module>(
        seed: UInt64, matvecOfMLX: Bool = false,
        filter: (String, Module) -> Bool = { _, _ in true },
        prepare: (M) throws -> Void = { _ in },
        _ make: () throws -> M
    ) throws -> M {
        let module = try withRandomState(MLXRandom.RandomState(seed: seed), body: make)
        module.update(parameters: module.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(module, matvecOfMLX: matvecOfMLX, filter: filter)
        try prepare(module)
        module.train(false)
        eval(module)
        return module
    }

    /// Runs `project` inside a speculative verify over ``rows`` seeded rows
    /// of `width` and over each row alone, and asserts the same bits for
    /// every row.
    ///
    /// - Parameters:
    ///   - width: The width of each row.
    ///   - axis: The token axis of the outputs of `project`.
    ///   - label: The name of the projection in a failure.
    ///   - project: The projections under test, over rows of shape
    ///     `(1, tokens, width)`.
    static func assertVerifyRowsMatchSingleRows(
        width: Int, axis: Int = 1, _ label: String,
        file: StaticString = #filePath, line: UInt = #line,
        project: (MLXArray) -> [MLXArray]
    ) {
        let x = MLXRandom.normal([1, rows, width], key: MLXRandom.key(rowSeed))
            .asType(.bfloat16)
        let (together, alone) = MultiRowQuantizedMatvec.run(asSpeculativeVerify: true) {
            (project(x), (0 ..< rows).map { project(x[0..., $0 ..< ($0 + 1), 0...]) })
        }
        for (index, got) in together.enumerated() {
            let want = concatenated(alone.map { $0[index] }, axis: axis)
            eval(got, want)
            XCTAssertEqual(
                got.shape, want.shape, "\(label) output \(index)", file: file, line: line)
            XCTAssertEqual(
                BitwiseComparison.mismatchCount(got, want), 0,
                "\(label) output \(index): verify rows differ from single-row decode",
                file: file, line: line)
        }
    }

    /// Asserts that every row of the verify `logits` has the bits of the
    /// single-row projection of its row of `hidden` by `lmHead`.
    ///
    /// - Parameters:
    ///   - logits: The logits of a verify, shape `(1, tokens, vocabulary)`.
    ///   - hidden: The hidden rows the LM head projected, shape
    ///     `(1, tokens, hiddenSize)`.
    ///   - lmHead: The LM head.
    ///   - label: The verify in a failure.
    static func assertLogitsMatchSingleRowProjections(
        _ logits: MLXArray, hidden: MLXArray, lmHead: Linear, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let rows = (0 ..< hidden.dim(1)).map { hidden[0..., $0 ..< ($0 + 1), 0...] }
        let want = concatenated(rows.map { lmHead($0) }, axis: 1)
        eval(want)
        XCTAssertEqual(
            BitwiseComparison.mismatchCount(logits, want), 0,
            "\(label): verify logits differ from the single-row projection of each row",
            file: file, line: line)
    }
}
