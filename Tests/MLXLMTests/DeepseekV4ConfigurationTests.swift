// Copyright © 2026 Apple Inc.
//
// Verifies `DeepseekV4Configuration` against the `config.json` that
// `mlx-community/DeepSeek-V4-Flash-4bit` publishes. The fixture in
// `Resources/DeepSeek-V4-Flash-4bit-config.json` is a copy of that file.
//
// The fixture alone cannot show that the decoder reads the file. Each default
// holds the DeepSeek-V4-Flash value, and the fixture is DeepSeek-V4-Flash, thus
// a wrong key string gives the default and the default is the number the
// fixture holds. The tests thus keep the two questions apart:
//
// - Does the decoder read the file? `distinctJSON` gives each of the 37 keys a
//   value that is different from the default of that key. A wrong key string,
//   or a `CodingKeys` entry that points at the wrong property, makes
//   `testDistinctJSONDecodesEveryKey` fail.
// - Do the defaults apply when a key is absent? `minimalJSON` gives no key at
//   all. A wrong `Default` value makes `testMinimalJSONDecodesToFlashDefaults`
//   fail.
//
// The published fixture keeps its own test, which shows that the real file
// decodes, and an encode-and-decode test shows that the synthesized encoder
// writes every key back.
//
// DeepSeek-V4 gives a compress ratio to each layer and hash routing to the
// first layers. The derived helpers read those two keys, thus the tests
// examine them against the real values in the fixture.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

/// The value each property of one decoded configuration must hold.
///
/// Three tables use this type: the values ``DeepseekV4ConfigurationTests``
/// writes into its distinct JSON, the values a file with no key falls back to,
/// and the values the published fixture gives. One ``assertMatches(_:)`` reads
/// all three, thus the 37 comparisons are written one time.
///
/// No property has a default value. The memberwise initializer thus asks for
/// all 37, and a table that forgets one does not compile.
private struct ExpectedValues {
    var vocabSize: Int
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var qkRopeHeadDim: Int
    var qLoraRank: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var oGroups: Int
    var oLoraRank: Int
    var nRoutedExperts: Int
    var numSharedExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var numHashLayers: Int
    var scoringFunction: String
    var normalizeTopkProb: Bool
    var routedScalingFactor: Float
    var swigluLimit: Float
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float
    var ropeTheta: Float
    var compressRopeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var slidingWindow: Int
    var compressRatios: [Int]
    var indexNHeads: Int
    var indexHeadDim: Int
    var indexTopK: Int
    var useAttnSink: Bool
    var dsparkBlockSize: Int
    var dsparkNoiseTokenId: Int
    var dsparkTargetLayerIds: [Int]
    var dsparkMarkovRank: Int

    /// Compares one decoded configuration against this table.
    ///
    /// XCTest reports a failure on the line of the comparison that failed, thus
    /// the report names the property without a message of its own.
    ///
    /// - Parameter config: The decoded configuration.
    func assertMatches(_ config: DeepseekV4Configuration) {
        XCTAssertEqual(config.vocabSize, vocabSize)
        XCTAssertEqual(config.hiddenSize, hiddenSize)
        XCTAssertEqual(config.numHiddenLayers, numHiddenLayers)
        XCTAssertEqual(config.numAttentionHeads, numAttentionHeads)
        XCTAssertEqual(config.numKeyValueHeads, numKeyValueHeads)
        XCTAssertEqual(config.headDim, headDim)
        XCTAssertEqual(config.qkRopeHeadDim, qkRopeHeadDim)
        XCTAssertEqual(config.qLoraRank, qLoraRank)
        XCTAssertEqual(config.rmsNormEps, rmsNormEps)
        XCTAssertEqual(config.maxPositionEmbeddings, maxPositionEmbeddings)
        XCTAssertEqual(config.oGroups, oGroups)
        XCTAssertEqual(config.oLoraRank, oLoraRank)
        XCTAssertEqual(config.nRoutedExperts, nRoutedExperts)
        XCTAssertEqual(config.numSharedExperts, numSharedExperts)
        XCTAssertEqual(config.numExpertsPerTok, numExpertsPerTok)
        XCTAssertEqual(config.moeIntermediateSize, moeIntermediateSize)
        XCTAssertEqual(config.numHashLayers, numHashLayers)
        XCTAssertEqual(config.scoringFunction, scoringFunction)
        XCTAssertEqual(config.normalizeTopkProb, normalizeTopkProb)
        XCTAssertEqual(config.routedScalingFactor, routedScalingFactor)
        XCTAssertEqual(config.swigluLimit, swigluLimit)
        XCTAssertEqual(config.hcMult, hcMult)
        XCTAssertEqual(config.hcSinkhornIters, hcSinkhornIters)
        XCTAssertEqual(config.hcEps, hcEps)
        XCTAssertEqual(config.ropeTheta, ropeTheta)
        XCTAssertEqual(config.compressRopeTheta, compressRopeTheta)
        XCTAssertEqual(config.ropeScaling, ropeScaling)
        XCTAssertEqual(config.slidingWindow, slidingWindow)
        XCTAssertEqual(config.compressRatios, compressRatios)
        XCTAssertEqual(config.indexNHeads, indexNHeads)
        XCTAssertEqual(config.indexHeadDim, indexHeadDim)
        XCTAssertEqual(config.indexTopK, indexTopK)
        XCTAssertEqual(config.useAttnSink, useAttnSink)
        XCTAssertEqual(config.dsparkBlockSize, dsparkBlockSize)
        XCTAssertEqual(config.dsparkNoiseTokenId, dsparkNoiseTokenId)
        XCTAssertEqual(config.dsparkTargetLayerIds, dsparkTargetLayerIds)
        XCTAssertEqual(config.dsparkMarkovRank, dsparkMarkovRank)
    }
}

extension ExpectedValues {

    /// The value of each key for a file that gives no key at all.
    ///
    /// This table pins all 36 defaults of `DeepseekV4Configuration`. A wrong
    /// default makes `testMinimalJSONDecodesToFlashDefaults` fail. `ropeScaling`
    /// has no default, thus the table gives `nil` for it.
    static let flashDefaults = ExpectedValues(
        vocabSize: 129_280,
        hiddenSize: 4096,
        numHiddenLayers: 43,
        numAttentionHeads: 64,
        numKeyValueHeads: 1,
        headDim: 512,
        qkRopeHeadDim: 64,
        qLoraRank: 1024,
        rmsNormEps: 1e-6,
        maxPositionEmbeddings: 1_048_576,
        oGroups: 8,
        oLoraRank: 1024,
        nRoutedExperts: 256,
        numSharedExperts: 1,
        numExpertsPerTok: 6,
        moeIntermediateSize: 2048,
        numHashLayers: 3,
        scoringFunction: "sqrtsoftplus",
        normalizeTopkProb: true,
        routedScalingFactor: 1.5,
        swigluLimit: 10.0,
        hcMult: 4,
        hcSinkhornIters: 20,
        hcEps: 1e-6,
        ropeTheta: 10000.0,
        compressRopeTheta: 160_000.0,
        ropeScaling: nil,
        slidingWindow: 128,
        compressRatios: [],
        indexNHeads: 64,
        indexHeadDim: 128,
        indexTopK: 512,
        useAttnSink: true,
        dsparkBlockSize: 0,
        dsparkNoiseTokenId: 0,
        dsparkTargetLayerIds: [],
        dsparkMarkovRank: 256
    )

    /// The value of each key in the published `DeepSeek-V4-Flash-4bit`
    /// `config.json`.
    ///
    /// That file gives the DeepSeek-V4-Flash value to each key it holds, thus
    /// this table is different from ``flashDefaults`` in two keys only: the file
    /// gives the 44 compress ratios and the YaRN rope scaling, and the defaults
    /// give an empty list and `nil`. The file gives no `use_attn_sink` key and
    /// no `dspark_*` key, thus those five keep the default.
    ///
    /// This table thus cannot show that the decoder reads a key.
    /// ``distinctValues`` does that.
    static let publishedFixture: ExpectedValues = {
        var expected = flashDefaults
        expected.compressRatios = publishedCompressRatios
        expected.ropeScaling = publishedRopeScaling
        return expected
    }()

    /// The 44 compress ratios of the published file. It holds one more ratio
    /// than the 43 layers, because the extra ratio belongs to the MTP layer.
    static let publishedCompressRatios: [Int] = [
        0, 0, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 0,
    ]

    /// The YaRN rope scaling of the published file.
    static let publishedRopeScaling: [String: StringOrNumber] = [
        "beta_fast": .int(32),
        "beta_slow": .int(1),
        "factor": .int(16),
        "original_max_position_embeddings": .int(65536),
        "type": .string("yarn"),
    ]

    /// The value of each key in `DeepseekV4ConfigurationTests.distinctJSON`.
    ///
    /// Each value is different from the default of that key, and no two keys
    /// share a value. A wrong key string, or a `CodingKeys` entry that points at
    /// the wrong property, thus makes `testDistinctJSONDecodesEveryKey` fail.
    static let distinctValues = ExpectedValues(
        vocabSize: 11,
        hiddenSize: 12,
        numHiddenLayers: 13,
        numAttentionHeads: 14,
        numKeyValueHeads: 15,
        headDim: 16,
        qkRopeHeadDim: 17,
        qLoraRank: 18,
        rmsNormEps: 0.25,
        maxPositionEmbeddings: 19,
        oGroups: 20,
        oLoraRank: 21,
        nRoutedExperts: 22,
        numSharedExperts: 23,
        numExpertsPerTok: 24,
        moeIntermediateSize: 25,
        numHashLayers: 26,
        scoringFunction: "distinct-scoring-func",
        normalizeTopkProb: false,
        routedScalingFactor: 0.5,
        swigluLimit: 0.75,
        hcMult: 27,
        hcSinkhornIters: 28,
        hcEps: 0.125,
        ropeTheta: 3.5,
        compressRopeTheta: 4.5,
        ropeScaling: ["type": .string("distinct-rope-type"), "factor": .int(29)],
        slidingWindow: 30,
        compressRatios: [0, 31],
        indexNHeads: 32,
        indexHeadDim: 33,
        indexTopK: 34,
        useAttnSink: false,
        dsparkBlockSize: 35,
        dsparkNoiseTokenId: 36,
        dsparkTargetLayerIds: [37, 38],
        dsparkMarkovRank: 39
    )
}

final class DeepseekV4ConfigurationTests: XCTestCase {

    /// The fixture gives layer 0 a compress ratio of 0, and layers 2, 3 and 42
    /// a compress ratio of more than 0.
    private static let plainLayer = 0
    private static let indexedLayer = 2
    private static let compressedLayer = 3
    private static let lastCompressedLayer = 42

    /// A file that gives no key of the configuration. Every property then falls
    /// back to its default.
    private static let minimalJSON = #"{ "model_type": "deepseek_v4" }"#

    /// A file that gives each of the 37 keys a value that is different from the
    /// default of that key. ``ExpectedValues/distinctValues`` holds the same
    /// values in the same order.
    private static let distinctJSON = """
        {
            "model_type": "deepseek_v4",
            "vocab_size": 11,
            "hidden_size": 12,
            "num_hidden_layers": 13,
            "num_attention_heads": 14,
            "num_key_value_heads": 15,
            "head_dim": 16,
            "qk_rope_head_dim": 17,
            "q_lora_rank": 18,
            "rms_norm_eps": 0.25,
            "max_position_embeddings": 19,
            "o_groups": 20,
            "o_lora_rank": 21,
            "n_routed_experts": 22,
            "n_shared_experts": 23,
            "num_experts_per_tok": 24,
            "moe_intermediate_size": 25,
            "num_hash_layers": 26,
            "scoring_func": "distinct-scoring-func",
            "norm_topk_prob": false,
            "routed_scaling_factor": 0.5,
            "swiglu_limit": 0.75,
            "hc_mult": 27,
            "hc_sinkhorn_iters": 28,
            "hc_eps": 0.125,
            "rope_theta": 3.5,
            "compress_rope_theta": 4.5,
            "rope_scaling": { "type": "distinct-rope-type", "factor": 29 },
            "sliding_window": 30,
            "compress_ratios": [0, 31],
            "index_n_heads": 32,
            "index_head_dim": 33,
            "index_topk": 34,
            "use_attn_sink": false,
            "dspark_block_size": 35,
            "dspark_noise_token_id": 36,
            "dspark_target_layer_ids": [37, 38],
            "dspark_markov_rank": 39
        }
        """

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

    func testDistinctJSONDecodesEveryKey() throws {
        let config = try decode(Self.distinctJSON)

        ExpectedValues.distinctValues.assertMatches(config)
    }

    func testDistinctJSONSurvivesEncodeAndDecode() throws {
        let config = try decode(Self.distinctJSON)

        let written = try JSONEncoder().encode(config)
        let readBack = try JSONDecoder().decode(DeepseekV4Configuration.self, from: written)

        ExpectedValues.distinctValues.assertMatches(readBack)
    }

    func testMinimalJSONDecodesToFlashDefaults() throws {
        let config = try decode(Self.minimalJSON)

        ExpectedValues.flashDefaults.assertMatches(config)
    }

    func testFixtureDecodesEveryKey() throws {
        let config = try decodeFixture()

        ExpectedValues.publishedFixture.assertMatches(config)
    }

    func testFixtureHashLayersStopAfterNumHashLayers() throws {
        let config = try decodeFixture()

        XCTAssertTrue(config.isHashLayer(0))
        XCTAssertTrue(config.isHashLayer(1))
        XCTAssertTrue(config.isHashLayer(2))
        XCTAssertFalse(config.isHashLayer(3))
    }

    func testFixtureCompressorLayersFollowCompressRatios() throws {
        let config = try decodeFixture()

        XCTAssertFalse(config.hasCompressor(layer: Self.plainLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.indexedLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.compressedLayer))
        XCTAssertTrue(config.hasCompressor(layer: Self.lastCompressedLayer))
    }

    func testFixtureRopeThetaFollowsCompressRatios() throws {
        let config = try decodeFixture()

        XCTAssertEqual(config.ropeTheta(forLayer: Self.plainLayer), 10000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.indexedLayer), 160_000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.compressedLayer), 160_000.0)
        XCTAssertEqual(config.ropeTheta(forLayer: Self.lastCompressedLayer), 160_000.0)
    }

    /// A checkpoint with no `compress_ratios` runs every layer on the plain
    /// rope theta, because no layer then has a compressor.
    func testMissingCompressRatiosGivePlainRopeTheta() throws {
        let config = try decode(Self.minimalJSON)

        XCTAssertFalse(config.hasCompressor(layer: Self.compressedLayer))
        XCTAssertEqual(config.ropeTheta(forLayer: Self.compressedLayer), config.ropeTheta)
    }
}
