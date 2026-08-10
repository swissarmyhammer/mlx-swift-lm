// Copyright © 2026 Apple Inc.
//
// Verifies `DeepseekV4Configuration` against the `config.json` that
// `mlx-community/DeepSeek-V4-Flash-4bit` publishes. The fixture in
// `Resources/DeepSeek-V4-Flash-4bit-config.json` is a copy of that file.
//
// DeepSeek-V4 gives a compress ratio to each layer and hash routing to the
// first layers. The derived helpers read those two keys, thus the tests
// examine them against the real values in the fixture.

import Foundation
import MLXLLM
import XCTest

final class DeepseekV4ConfigurationTests: XCTestCase {

    /// The fixture gives layer 0 a compress ratio of 0, and layers 2, 3 and 42
    /// a compress ratio of more than 0.
    private static let plainLayer = 0
    private static let indexedLayer = 2
    private static let compressedLayer = 3
    private static let lastCompressedLayer = 42

    private func decodeFixture() throws -> DeepseekV4Configuration {
        let name = "DeepSeek-V4-Flash-4bit-config"
        guard let url = Bundle.module.url(forResource: name, withExtension: "json") else {
            XCTFail("Missing fixture: \(name).json")
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(
            DeepseekV4Configuration.self, from: Data(contentsOf: url))
    }

    private func decode(_ json: String) throws -> DeepseekV4Configuration {
        try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
    }

    func testFixtureDecodesTransformerShape() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.numHiddenLayers, 43)
        XCTAssertEqual(config.numAttentionHeads, 64)
        XCTAssertEqual(config.numKeyValueHeads, 1)
        XCTAssertEqual(config.headDim, 512)
        XCTAssertEqual(config.maxPositionEmbeddings, 1_048_576)
        XCTAssertEqual(config.qLoraRank, 1024)
        XCTAssertEqual(config.oLoraRank, 1024)
    }

    func testFixtureDecodesMixtureOfExpertsShape() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.nRoutedExperts, 256)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        XCTAssertEqual(config.nSharedExperts, 1)
        XCTAssertEqual(config.moeIntermediateSize, 2048)
        XCTAssertEqual(config.scoringFunc, "sqrtsoftplus")
    }

    func testFixtureDecodesHyperConnectionAndIndexerKeys() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.hcMult, 4)
        XCTAssertEqual(config.hcSinkhornIters, 20)
        XCTAssertEqual(config.hcEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.indexNHeads, 64)
        XCTAssertEqual(config.indexHeadDim, 128)
        XCTAssertEqual(config.indexTopk, 512)
        XCTAssertEqual(config.slidingWindow, 128)
    }

    func testFixtureHashLayersStopAfterNumHashLayers() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.numHashLayers, 3)
        XCTAssertTrue(config.isHashLayer(0))
        XCTAssertTrue(config.isHashLayer(1))
        XCTAssertTrue(config.isHashLayer(2))
        XCTAssertFalse(config.isHashLayer(3))
    }

    func testFixtureCompressorLayersFollowCompressRatios() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.compressRatios.prefix(4), [0, 0, 4, 128])
        XCTAssertFalse(config.hasCompressor(layer: Self.plainLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.indexedLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.compressedLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.lastCompressedLayer))
    }

    func testFixtureRopeThetaFollowsCompressRatios() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.compressRopeTheta, 160_000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.plainLayer), 10000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.indexedLayer), 160_000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.compressedLayer), 160_000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.lastCompressedLayer), 160_000.0)
    }

    func testMinimalJSONDecodesToFlashDefaults() throws {
        let config = try decode(#"{ "model_type": "deepseek_v4" }"#)

        XCTAssertEqual(config.vocabSize, 129_280)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.numHiddenLayers, 43)
        XCTAssertEqual(config.hcMult, 4)
        XCTAssertEqual(config.swigluLimit, 10.0)
        XCTAssertTrue(config.useAttnSink)
        XCTAssertTrue(config.compressRatios.isEmpty)
        XCTAssertNil(config.ropeScaling)
    }

    /// A checkpoint with no `compress_ratios` runs every layer on the plain
    /// rope theta, because no layer then has a compressor.
    func testMissingCompressRatiosGivePlainRopeTheta() throws {
        let config = try decode(#"{ "model_type": "deepseek_v4" }"#)

        XCTAssertFalse(config.hasCompressor(layer: Self.compressedLayer))
        XCTAssertEqual(config.ropeTheta(forLayer: Self.compressedLayer), config.ropeTheta)
    }

    func testDsparkKeysDecode() throws {
        let config = try decode(
            """
            {
                "model_type": "deepseek_v4",
                "dspark_block_size": 8,
                "dspark_noise_token_id": 7,
                "dspark_target_layer_ids": [3, 5],
                "dspark_markov_rank": 128
            }
            """)

        XCTAssertEqual(config.dsparkBlockSize, 8)
        XCTAssertEqual(config.dsparkNoiseTokenId, 7)
        XCTAssertEqual(config.dsparkTargetLayerIds, [3, 5])
        XCTAssertEqual(config.dsparkMarkovRank, 128)
    }
}
