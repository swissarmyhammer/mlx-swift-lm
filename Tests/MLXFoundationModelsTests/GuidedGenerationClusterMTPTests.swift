// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import FoundationModels
import MLX
import MLXLMCommon
import MLXLMTestSupport
import Testing

@testable import MLXFoundationModels
@testable import MLXGuidedGeneration

/// Guided generation on a ``ClusterLanguageModel`` speculates with the MTP drafter of rank 0
/// over the `speculate` op, and writes what plain guided decoding writes.
///
/// Rank 0 drafts without the grammar mask. The host accepts each draft that equals the token
/// plain guided decoding feeds there, and decides each other token from the verify row, so the
/// output is the output of plain guided decoding. Each test runs `GuidedGenerationLoop` on the
/// target of the shared Qwen 3.5 fixture behind an in-process worker of one follower rank.
@Suite(.serialized)
struct GuidedGenerationClusterMTPTests {

    /// The tokens of the prompt of each generation.
    private static let prompt = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The tokens of each MTP round: chains of three drafts.
    private static let blockSize = 4

    /// The ranks other than 0 each worker stands for.
    private static let followerCount = 1

    /// One guided generation on a cluster model, and the worker that ran its target.
    private struct ClusterRun {
        let run: GuidedFixtureRun
        let worker: InProcessRemoteModelWorker
    }

    /// Runs `schema` on a cluster model over the target of `fixture`, whose rank 0 holds the
    /// drafter of the fixture.
    ///
    /// - Parameters:
    ///   - schema: The schema of the generation.
    ///   - fixture: The fixture whose target and drafter the worker runs.
    ///   - speculates: Whether the generation speculates with the drafter of rank 0.
    ///   - kvCache: The KV-cache configuration of the generation, or `nil` for full precision.
    /// - Returns: What the generation wrote, and its worker.
    /// - Throws: The error of the worker handshake or of the generation.
    private static func generateOnCluster(
        _ schema: GuidedFixtureSchema, fixture: Qwen35MTPFixture, speculates: Bool,
        kvCache: KVCacheConfiguration? = nil
    ) throws -> ClusterRun {
        let worker = fixture.makeWorker(drafting: true, followerCount: followerCount)
        let model = try ClusterLanguageModel.connect(over: worker)
        let drafter: RemoteMTPDrafter = try #require(model.mtpDrafter)
        let run = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: prompt, fixture: fixture,
            blockSize: speculates ? blockSize : nil, drafter: drafter, target: model,
            kvCache: kvCache)
        return ClusterRun(run: run, worker: worker)
    }

    /// The handle of the one cache the generation of `worker` made.
    private static func cacheID(of worker: InProcessRemoteModelWorker) throws -> RemoteCacheID {
        let made = worker.requests.compactMap { request -> RemoteCacheID? in
            guard case .newCache(let id, _) = request.op else { return nil }
            return id
        }
        return try #require(made.first)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases)
    func aClusteredSchemaSpeculatesWithTheDrafterOfRankZero(schema: GuidedFixtureSchema) throws {
        let speculative = try Self.generateOnCluster(
            schema, fixture: Qwen35MTPFixture(), speculates: true)

        #expect(speculative.run.sink.draftedTokenCount > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases)
    func aClusteredSchemaReportsTheDraftsTheTargetAccepted(schema: GuidedFixtureSchema) throws {
        let speculative = try Self.generateOnCluster(
            schema, fixture: Qwen35MTPFixture(), speculates: true)

        #expect(speculative.run.sink.acceptedDraftTokenCount > 0)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases, Qwen35MTPFixture.DrafterMode.allCases)
    func aClusteredSchemaWritesThePlainClusteredOutput(
        schema: GuidedFixtureSchema, mode: Qwen35MTPFixture.DrafterMode
    ) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try Self.generateOnCluster(schema, fixture: fixture, speculates: false)
        let speculative = try Self.generateOnCluster(schema, fixture: fixture, speculates: true)

        #expect(!plain.run.text.isEmpty)
        #expect(speculative.run.text == plain.run.text)
        #expect(speculative.run.tokenCount == plain.run.tokenCount)
        #expect(speculative.run.incomplete == plain.run.incomplete)
        #expect(speculative.run.sink.decisions == plain.run.sink.decisions)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: GuidedFixtureSchema.allCases)
    func aClusteredSchemaWritesTheLocalPlainOutput(schema: GuidedFixtureSchema) throws {
        let fixture = try Qwen35MTPFixture()
        let local = try GuidedFixtureRun.generate(
            schema: schema.json, prompt: Self.prompt, fixture: fixture, blockSize: nil)
        let speculative = try Self.generateOnCluster(schema, fixture: fixture, speculates: true)

        #expect(!local.text.isEmpty)
        #expect(speculative.run.text == local.text)
        #expect(speculative.run.tokenCount == local.tokenCount)
    }

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test(arguments: Qwen35MTPFixture.DrafterMode.allCases)
    func everyRankEndsAClusteredSchemaAtThePlainOffset(mode: Qwen35MTPFixture.DrafterMode) throws {
        let fixture = try Qwen35MTPFixture(drafter: mode)
        let plain = try Self.generateOnCluster(.dayPlan, fixture: fixture, speculates: false)
        let speculative = try Self.generateOnCluster(.dayPlan, fixture: fixture, speculates: true)

        let plainOffsets = plain.worker.rankStatuses(of: try Self.cacheID(of: plain.worker))
            .map(\.offset)
        let offsets = speculative.worker.rankStatuses(
            of: try Self.cacheID(of: speculative.worker)
        ).map(\.offset)
        #expect(plainOffsets.count == Self.followerCount + 1)
        #expect(offsets == plainOffsets)
    }

    /// The fixture of an 8-bit KV cache: its attention heads are as wide as a quantization
    /// group.
    private static func eightBitFixture() throws -> Qwen35MTPFixture {
        try Qwen35MTPFixture(headDimension: AffineKVCacheConfiguration.eightBit.groupSize)
    }

    /// The affine 8-bit KV-cache configuration.
    private static let eightBit = KVCacheConfiguration(strategy: .affine(.eightBit))

    @available(iOS 27.0, macOS 27.0, visionOS 27.0, *)
    @Test
    func anEightBitClusteredSchemaWritesThePlainClusteredOutput() throws {
        let fixture = try Self.eightBitFixture()
        let plain = try Self.generateOnCluster(
            .dayPlan, fixture: fixture, speculates: false, kvCache: Self.eightBit)
        let speculative = try Self.generateOnCluster(
            .dayPlan, fixture: fixture, speculates: true, kvCache: Self.eightBit)

        #expect(!plain.run.text.isEmpty)
        #expect(speculative.run.sink.draftedTokenCount > 0)
        #expect(speculative.run.text == plain.run.text)
    }
}

#endif
