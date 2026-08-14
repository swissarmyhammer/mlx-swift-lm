// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

/// Characterization tests for ``LoRALinear`` and ``QLoRALinear``: freezing
/// semantics, adapter evaluation, and `fused()` round trips.
class LoRAAdapterTests: XCTestCase {

    /// Ensure the Metal library is reachable before any GPU work runs.
    /// Give `layer` a deterministic non-zero `lora_b` so the adapter term is
    /// active (a freshly created adapter has `lora_b == 0` and contributes
    /// nothing).
    private func activateAdapter(_ layer: Module, rank: Int, outputDimensions: Int) {
        let loraB = MLXRandom.normal([rank, outputDimensions]) * 0.1
        layer.update(parameters: ModuleParameters.unflattened([("lora_b", loraB)]))
    }

    /// A fresh ``LoRALinear`` freezes everything except the LoRA parameters.
    func testLoRALinearFreezesAllButLoRAParameters() throws {
        let adapter = LoRALinear.from(linear: Linear(8, 4))

        let trainableKeys = Set(adapter.trainableParameters().flattened().map { $0.0 })
        XCTAssertEqual(trainableKeys, ["lora_a", "lora_b"])
    }

    /// A fresh ``QLoRALinear`` freezes everything except the LoRA parameters,
    /// and `LoRALinear.from(linear:rank:scale:)` routes `QuantizedLinear`
    /// layers to ``QLoRALinear``.
    func testQLoRALinearFreezesAllButLoRAParameters() throws {
        let adapter = LoRALinear.from(linear: QuantizedLinear(64, 32))
        XCTAssertTrue(adapter is QLoRALinear)

        let trainableKeys = Set(adapter.trainableParameters().flattened().map { $0.0 })
        XCTAssertEqual(trainableKeys, ["lora_a", "lora_b"])
    }

    /// ``LoRALinear/fused()`` folds an active adapter into a plain `Linear`
    /// that computes the same outputs as the adapted layer.
    func testLoRALinearFusedMatchesAdapterOutput() throws {
        let base = Linear(8, 4)
        guard let adapter = LoRALinear.from(linear: base) as? LoRALinear else {
            XCTFail("expected a LoRALinear adapter")
            return
        }
        activateAdapter(adapter, rank: LoRALinear.defaultRank, outputDimensions: 4)

        let x = MLXRandom.normal([2, 8])
        let adapted = adapter(x)
        // the adapter term is active: the output must differ from the base layer
        XCTAssertFalse(allClose(adapted, base(x), atol: 1e-5).item(Bool.self))

        guard let fused = adapter.fused() as? Linear else {
            XCTFail("expected fused() to produce a Linear")
            return
        }
        XCTAssertFalse(fused is LoRALinear)
        XCTAssertTrue(allClose(fused(x), adapted, rtol: 1e-4, atol: 1e-5).item(Bool.self))
    }

    /// The relative root-mean-square difference between `lhs` and `rhs`,
    /// normalized by the magnitude of `rhs`.
    private func relativeRMSError(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let error = sqrt(mean(square(lhs - rhs))).item(Float.self)
        let magnitude = sqrt(mean(square(rhs))).item(Float.self)
        return error / magnitude
    }

    /// ``QLoRALinear/fused()`` folds an active adapter into a
    /// `QuantizedLinear` that computes the same outputs as the adapted layer
    /// up to requantization error.
    func testQLoRALinearFusedMatchesAdapterOutput() throws {
        let base = QuantizedLinear(64, 32)
        guard let adapter = LoRALinear.from(linear: base) as? QLoRALinear else {
            XCTFail("expected a QLoRALinear adapter")
            return
        }
        activateAdapter(adapter, rank: LoRALinear.defaultRank, outputDimensions: 32)

        let x = MLXRandom.normal([2, 64])
        let adapted = adapter(x)
        // the adapter term is active: the output must differ from the base layer
        XCTAssertGreaterThan(relativeRMSError(base(x), adapted), 0.3)

        guard let fused = adapter.fused() as? QuantizedLinear else {
            XCTFail("expected fused() to produce a QuantizedLinear")
            return
        }
        XCTAssertFalse(fused is QLoRALinear)
        // fused() requantizes the merged weight, so the outputs match the
        // adapted layer only up to 4-bit quantization noise -- compare the
        // normalized RMS error, which is an order of magnitude below this
        // bound for a correct fuse and well above it for a dropped or
        // garbled low-rank update
        XCTAssertLessThan(relativeRMSError(fused(x), adapted), 0.15)
    }
}
