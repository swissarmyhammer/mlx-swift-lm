// Copyright © 2026 Apple Inc.
//
// Pins the fused single-token conv of a gated delta net (GDN) layer bit for bit
// against its general conv (MLX's `Convolution` kernel). Each GDN layer with a
// fused decode conv has its own copy of the conv, thus each one runs this one
// comparison; a mismatch means compiled decode silently diverges from prefill.

import MLX
import MLXNN
import XCTest

@testable import MLXLLM

/// A GDN layer whose single-token conv has a fused decode form beside the general form.
protocol GDNDecodeConvolutionLayer {
    /// The depthwise conv of the layer.
    var conv1d: Conv1d { get }
    /// The number of conv channels.
    var convDim: Int { get }
    /// The number of conv taps.
    var convKernelSize: Int { get }

    /// The conv of one token as multiply-adds, which a compiled decode segment can fuse.
    ///
    /// - Parameters:
    ///   - convState: The conv state before the token.
    ///   - qkv: The conv input of the token, shape `(batch, 1, convDim)`.
    /// - Returns: The conv output and the conv state after the token.
    func decodeConv(convState: MLXArray, qkv: MLXArray) -> (conv: MLXArray, state: MLXArray)

    /// The conv of the tokens through MLX's `Convolution` kernel.
    ///
    /// - Parameters:
    ///   - convState: The conv state before the tokens.
    ///   - qkv: The conv input of the tokens.
    /// - Returns: The conv output and the conv state after the tokens.
    func generalConv(convState: MLXArray, qkv: MLXArray) -> (conv: MLXArray, state: MLXArray)
}

extension Qwen35GatedDeltaNet: GDNDecodeConvolutionLayer {}

extension Qwen3NextGatedDeltaNet: GDNDecodeConvolutionLayer {}

/// The comparison of the fused decode conv of a GDN layer with its general conv.
enum GDNDecodeConvolutionComparison {
    /// Casts the conv weights of `layer` to `dtype`, draws one token row and a conv state from
    /// the current random state, and asserts that the fused decode conv gives the bits of the
    /// general conv, for the conv output and for the conv state.
    ///
    /// - Parameters:
    ///   - layer: The GDN layer.
    ///   - dtype: The dtype of the conv weights, the token row and the conv state.
    ///   - file: The file of the caller.
    ///   - line: The line of the caller.
    static func assertFusedMatchesGeneral(
        _ layer: some GDNDecodeConvolutionLayer,
        dtype: DType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        layer.conv1d.update(parameters: layer.conv1d.parameters().mapValues { $0.asType(dtype) })
        let qkv = MLXRandom.normal([1, 1, layer.convDim]).asType(dtype)
        let convState = MLXRandom.normal([1, layer.convKernelSize - 1, layer.convDim])
            .asType(dtype)
        eval(qkv, convState)

        let fused = layer.decodeConv(convState: convState, qkv: qkv)
        let general = layer.generalConv(convState: convState, qkv: qkv)

        XCTAssertEqual(
            ArraySnapshot(fused.conv), ArraySnapshot(general.conv),
            "conv output differs for \(dtype)", file: file, line: line)
        XCTAssertEqual(
            ArraySnapshot(fused.state), ArraySnapshot(general.state),
            "conv state differs for \(dtype)", file: file, line: line)
    }
}
