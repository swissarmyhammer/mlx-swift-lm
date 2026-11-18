// Copyright © 2026 Apple Inc.
//
// Quantizes the four input projections of a Qwen 3.5 gated delta net (GDN)
// layer, the text model's and the vision-language model's, so the layer can
// fuse them into one projection. The suites of the fused projection and of the
// compiled verify set up the same quantized layer through this one helper.

import MLXNN

@testable import MLXLLM
@testable import MLXVLM

/// A Qwen 3.5 GDN layer with four separate input projections.
protocol GDNInputProjectionLayer: Module {
    /// The projection of the conv input: the queries, the keys and the values.
    var inProjQKV: Linear { get }
    /// The projection of the output gate.
    var inProjZ: Linear { get }
    /// The projection of the beta input.
    var inProjB: Linear { get }
    /// The projection of the decay input.
    var inProjA: Linear { get }

    /// Fuses the four input projections into one.
    ///
    /// - Returns: Whether one matmul computes the four projections.
    /// - Throws: The error of the preparation.
    func prepareFusedInputProjection() throws -> Bool
}

extension Qwen35GatedDeltaNet: GDNInputProjectionLayer {}

extension Qwen35Language.GatedDeltaNet: GDNInputProjectionLayer {}

/// The quantization of the input projections of a GDN layer.
enum GDNInputProjectionQuantization {
    /// The group size of a quantized input projection that fuses.
    static let groupSize = 32

    /// The bits of a quantized input projection that fuses.
    static let bits = 4
}

extension GDNInputProjectionLayer {
    /// Replaces the four input projections with quantized ones, which fuse.
    ///
    /// - Parameters:
    ///   - groupSize: The group size of each quantized projection.
    ///   - bits: The bits of each quantized projection.
    /// - Throws: The error of the module update.
    func quantizeInputProjections(
        groupSize: Int = GDNInputProjectionQuantization.groupSize,
        bits: Int = GDNInputProjectionQuantization.bits
    ) throws {
        let quantized = { (linear: Linear) -> Linear in
            QuantizedLinear(linear, groupSize: groupSize, bits: bits)
        }
        try update(
            modules: ModuleChildren(values: [
                "in_proj_qkv": .value(quantized(inProjQKV)),
                "in_proj_z": .value(quantized(inProjZ)),
                "in_proj_b": .value(quantized(inProjB)),
                "in_proj_a": .value(quantized(inProjA)),
            ]), verify: [])
    }
}
