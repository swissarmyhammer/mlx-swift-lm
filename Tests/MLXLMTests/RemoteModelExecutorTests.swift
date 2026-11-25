// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMTestSupport
import MLXNN
import MLXVLM
import Testing

@testable import MLXLMCommon

/// The worker side of the remote model protocol runs each op on a real local model, and keeps
/// the rules of the stream.
///
/// The model is the target of the tiny Qwen 3.5 fixture: one recurrent and one attention layer,
/// on seeded weights.
@Suite("A remote model executor runs the op set", .serialized)
struct RemoteModelExecutorTests {

    /// The tokens of the prompt each forward runs.
    private static let prompt: [Int32] = [3, 14, 15, 9, 2, 6, 5, 35]

    /// The ranks a sharded test configuration names.
    private static let shardedRankCount = 2

    /// The first cache handle of a test.
    private static let cache = RemoteCacheID(rawValue: 7)

    /// The second cache handle of a test.
    private static let otherCache = RemoteCacheID(rawValue: 8)

    /// The request every test begins.
    private static let request = RemoteRequestID(rawValue: 1)

    /// A request no test begins.
    private static let otherRequest = RemoteRequestID(rawValue: 2)

    /// The `model_type` the configuration of a test executor names.
    private static let modelType = "qwen3_5_text"

    /// The quantization the configuration of a test executor names.
    private static let quantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)

    /// An executor over the target of a new fixture, in `dtype` when one is given.
    private static func makeExecutor(
        rankCount: Int = 1, dtype: DType? = nil
    ) throws -> RemoteModelExecutor {
        let target = try Qwen35MTPFixture().target
        if let dtype {
            target.apply { $0.asType(dtype) }
        }
        return RemoteModelExecutor(
            model: target,
            configuration: .init(
                vocabularySize: Qwen35MTPFixture.vocabularySize, rankCount: rankCount,
                modelType: modelType, quantization: quantization))
    }

    /// The text of the failed reply of `error`.
    private static func text(of error: RemoteModelExecutorError) -> String {
        error.localizedDescription
    }

    /// The text of the failed reply of `error`.
    private static func text(of error: RemoteModelProtocolError) -> String {
        error.localizedDescription
    }

    /// The requests of `ops`, numbered from `firstSequence`, all of request `requestID`.
    private static func requests(
        _ ops: [RemoteModelOp], from firstSequence: UInt64 = 1,
        requestID: RemoteRequestID = request
    ) -> [RemoteModelRequest] {
        ops.enumerated().map { index, op in
            RemoteModelRequest(
                requestID: requestID, sequence: firstSequence + UInt64(index), op: op)
        }
    }

    /// The ops that begin a request and make ``cache``.
    private static let opening: [RemoteModelOp] = [.begin, .newCache(cache)]

    /// Runs `ops` from the first sequence, and returns the reply of the last.
    private static func run(
        _ ops: [RemoteModelOp], on executor: RemoteModelExecutor
    ) throws -> RemoteModelReply {
        var reply = RemoteModelReply.done
        for request in requests(ops) {
            reply = try executor.handle(request)
        }
        return reply
    }

    /// The logits `reply` carries, or nil for a reply that carries none.
    private static func remoteLogits(of reply: RemoteModelReply?) -> RemoteLogits? {
        guard case .forwarded(_, let logits) = reply else { return nil }
        return logits
    }

    /// The logits of `reply` as floats, or nil for a reply that carries none.
    private static func logits(of reply: RemoteModelReply?) throws -> [Float]? {
        try remoteLogits(of: reply)?.array().asArray(Float.self)
    }

    /// The status of `reply`, or nil for a reply that carries none.
    private static func status(of reply: RemoteModelReply) -> RemoteCacheStatus? {
        switch reply {
        case .cache(let status), .trimmed(_, let status), .forwarded(let status, _): status
        case .capabilities, .done, .failed: nil
        }
    }

    /// The capabilities of `reply`, or nil for another reply.
    private static func capabilities(of reply: RemoteModelReply) -> RemoteModelCapabilities? {
        guard case .capabilities(let capabilities) = reply else { return nil }
        return capabilities
    }

    // MARK: - Capabilities

    @Test("the static capabilities name the shardable model families")
    func theStaticCapabilitiesNameTheShardableModelFamilies() {
        #expect(
            RemoteModelExecutor.capabilities
                == RemoteWorkerCapabilities(
                    protocolVersion: RemoteModelProtocol.version,
                    modelFamilies: RemoteModelExecutor.shardableModelFamilies,
                    supportsSharding: true))
    }

    @Test(
        "each shardable model family loads a model that shards",
        arguments: RemoteModelExecutor.shardableModelFamilies)
    func eachShardableModelFamilyLoadsAModelThatShards(family: String) async throws {
        let model = try await ShardableFamilyConfiguration.makeModel(of: family)

        #expect(model is TensorParallelShardable)
    }

    @Test("describe reports the model as the worker runs it")
    func describeReportsTheModelAsTheWorkerRunsIt() throws {
        let executor = try Self.makeExecutor(rankCount: Self.shardedRankCount)

        let reply = try executor.handle(
            RemoteModelRequest(requestID: .none, sequence: 1, op: .describe))

        let target = try Qwen35MTPFixture().target
        #expect(
            reply
                == .capabilities(
                    RemoteModelCapabilities(
                        protocolVersion: RemoteModelProtocol.version,
                        modelFamilies: RemoteModelExecutor.shardableModelFamilies,
                        supportsSharding: true, rankCount: Self.shardedRankCount,
                        vocabularySize: Qwen35MTPFixture.vocabularySize,
                        defaultPrefillStepSize: target.defaultPrefillStepSize,
                        maximumPrefillStepSize: target.maximumPrefillStepSize,
                        modelType: Self.modelType, quantization: Self.quantization)))
    }

    @Test("describe of a model that cannot shard reports no sharding")
    func describeOfAModelThatCannotShardReportsNoSharding() throws {
        let executor = RemoteModelExecutor(
            model: UnshardableModel(), configuration: .init(vocabularySize: 1))

        let reply = try executor.handle(
            RemoteModelRequest(requestID: .none, sequence: 1, op: .describe))

        #expect(Self.capabilities(of: reply)?.supportsSharding == false)
    }

    // MARK: - Configuration

    @Test("a configuration reads the vocabulary of a text configuration")
    func aConfigurationReadsTheVocabularyOfATextConfiguration() throws {
        let directory = try ConfigurationDirectory(json: #"{"vocab_size": 151936}"#)

        let configuration = try RemoteModelExecutor.Configuration(
            modelDirectory: directory.url, rankCount: Self.shardedRankCount)

        #expect(
            configuration == .init(vocabularySize: 151_936, rankCount: Self.shardedRankCount))
    }

    @Test("a configuration reads the vocabulary of a nested text configuration")
    func aConfigurationReadsTheVocabularyOfANestedTextConfiguration() throws {
        let directory = try ConfigurationDirectory(
            json: #"{"model_type": "inkling2_loop", "text_config": {"vocab_size": 201024}}"#)

        let configuration = try RemoteModelExecutor.Configuration(modelDirectory: directory.url)

        #expect(configuration.vocabularySize == 201_024)
    }

    @Test("a configuration reads the model type and the quantization")
    func aConfigurationReadsTheModelTypeAndTheQuantization() throws {
        let directory = try ConfigurationDirectory(
            json: #"""
                {"model_type": "qwen3_5_text", "vocab_size": 64,
                 "quantization": {"group_size": 64, "bits": 4, "model.embed_tokens": false}}
                """#)

        let configuration = try RemoteModelExecutor.Configuration(modelDirectory: directory.url)

        #expect(
            configuration
                == .init(
                    vocabularySize: Qwen35MTPFixture.vocabularySize, modelType: Self.modelType,
                    quantization: Self.quantization))
    }

    @Test("a configuration without a vocabulary is refused")
    func aConfigurationWithoutAVocabularyIsRefused() throws {
        let directory = try ConfigurationDirectory(json: #"{"model_type": "qwen3_5"}"#)

        #expect(throws: RemoteModelExecutorError.missingVocabularySize) {
            try RemoteModelExecutor.Configuration(modelDirectory: directory.url)
        }
    }

    // MARK: - Forward

    @Test("a forward returns the logits of the local model")
    func aForwardReturnsTheLogitsOfTheLocalModel() throws {
        let fixture = try Qwen35MTPFixture()
        let executor = RemoteModelExecutor(
            model: fixture.target,
            configuration: .init(vocabularySize: Qwen35MTPFixture.vocabularySize))
        let local = fixture.target.nextTokenLogits(
            LMInput.Text(tokens: MLXArray(Self.prompt)[.newAxis]),
            cache: try fixture.target.newCache(parameters: nil), state: nil
        ).logits

        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .lastPosition)
            ],
            on: executor)

        #expect(try Self.logits(of: reply) == local.asArray(Float.self))
    }

    @Test("a forward returns its logits in the dtype of the model")
    func aForwardReturnsItsLogitsInTheDtypeOfTheModel() throws {
        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .lastPosition)
            ],
            on: Self.makeExecutor(dtype: .bfloat16))

        #expect(Self.remoteLogits(of: reply)?.elementType == .bfloat16)
    }

    @Test("a forward of no logits returns none")
    func aForwardOfNoLogitsReturnsNone() throws {
        let reply = try Self.run(
            Self.opening + [.forward(tokens: Self.prompt, cache: Self.cache, logits: .none)],
            on: Self.makeExecutor())

        #expect(reply == .forwarded(try #require(Self.status(of: reply)), logits: nil))
    }

    @Test("a forward of all positions returns a row for each token")
    func aForwardOfAllPositionsReturnsARowForEachToken() throws {
        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .allPositions)
            ],
            on: Self.makeExecutor())

        #expect(
            try Self.logits(of: reply)?.count
                == Self.prompt.count * Qwen35MTPFixture.vocabularySize)
    }

    @Test("a forward advances the offset of its cache")
    func aForwardAdvancesTheOffsetOfItsCache() throws {
        let reply = try Self.run(
            Self.opening + [.forward(tokens: Self.prompt, cache: Self.cache, logits: .none)],
            on: Self.makeExecutor())

        #expect(Self.status(of: reply)?.offset == Self.prompt.count)
    }

    @Test("a forward of more than one chunk fails")
    func aForwardOfMoreThanOneChunkFails() throws {
        let tokens = [Int32](repeating: 1, count: RemoteModelProtocol.maximumChunkTokenCount + 1)

        let reply = try Self.run(
            Self.opening + [.forward(tokens: tokens, cache: Self.cache, logits: .none)],
            on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .chunkTooLong(tokenCount: tokens.count))))
    }

    @Test("a forward of no tokens fails")
    func aForwardOfNoTokensFails() throws {
        let reply = try Self.run(
            Self.opening + [.forward(tokens: [], cache: Self.cache, logits: .none)],
            on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .emptyForward)))
    }

    @Test("a forward on a handle the worker does not hold fails")
    func aForwardOnAnUnknownHandleFails() throws {
        let reply = try Self.run(
            [.begin, .forward(tokens: Self.prompt, cache: Self.cache, logits: .none)],
            on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .unknownCache(Self.cache))))
    }

    // MARK: - Cache ops

    @Test("a new cache on a held handle fails")
    func aNewCacheOnAHeldHandleFails() throws {
        let reply = try Self.run(Self.opening + [.newCache(Self.cache)], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .cacheExists(Self.cache))))
    }

    @Test("a trim drops the newest positions")
    func aTrimDropsTheNewestPositions() throws {
        let trimmedCount = 3

        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .none),
                .trim(Self.cache, count: trimmedCount),
            ], on: AttentionOnlyExecutor.make())

        let status = try #require(Self.status(of: reply))
        #expect(reply == .trimmed(trimmedCount, status: status))
        #expect(status.offset == Self.prompt.count - trimmedCount)
    }

    @Test("a copy keeps the position of its original")
    func aCopyKeepsThePositionOfItsOriginal() throws {
        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .none),
                .copy(Self.cache, into: Self.otherCache),
            ], on: Self.makeExecutor())

        #expect(Self.status(of: reply)?.offset == Self.prompt.count)
    }

    @Test("a copy decodes the logits its original decodes")
    func aCopyDecodesTheLogitsItsOriginalDecodes() throws {
        let executor = try Self.makeExecutor()
        let next: [Int32] = [12]
        let replies = try Self.requests(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .none),
                .copy(Self.cache, into: Self.otherCache),
                .forward(tokens: next, cache: Self.cache, logits: .lastPosition),
                .forward(tokens: next, cache: Self.otherCache, logits: .lastPosition),
            ]
        ).map { try executor.handle($0) }

        let original = try #require(try Self.logits(of: replies[4]))
        #expect(try Self.logits(of: replies[5]) == original)
    }

    @Test("a copy into a held handle fails")
    func aCopyIntoAHeldHandleFails() throws {
        let reply = try Self.run(
            Self.opening + [.copy(Self.cache, into: Self.cache)], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .cacheExists(Self.cache))))
    }

    @Test("a reset empties the cache")
    func aResetEmptiesTheCache() throws {
        let reply = try Self.run(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .none),
                .reset(Self.cache),
            ], on: Self.makeExecutor())

        #expect(Self.status(of: reply)?.offset == 0)
    }

    @Test("a release forgets the handle")
    func aReleaseForgetsTheHandle() throws {
        let reply = try Self.run(
            Self.opening + [.release(Self.cache), .reset(Self.cache)], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .unknownCache(Self.cache))))
    }

    // MARK: - The rules of the stream

    @Test("a sequence gap fails and names the expected and the found sequence")
    func aSequenceGapFails() throws {
        let executor = try Self.makeExecutor()
        _ = try executor.handle(RemoteModelRequest(requestID: .none, sequence: 1, op: .describe))

        let reply = try executor.handle(
            RemoteModelRequest(requestID: .none, sequence: 3, op: .describe))

        #expect(reply == .failed(Self.text(of: .sequenceGap(expected: 2, found: 3))))
    }

    @Test("a request with a gap does not move the sequence")
    func aRequestWithAGapDoesNotMoveTheSequence() throws {
        let executor = try Self.makeExecutor()
        _ = try executor.handle(RemoteModelRequest(requestID: .none, sequence: 2, op: .describe))

        let reply = try executor.handle(
            RemoteModelRequest(requestID: .none, sequence: 1, op: .describe))

        #expect(Self.capabilities(of: reply) != nil)
    }

    @Test("an op of another request than the open one fails")
    func anOpOfAnotherRequestFails() throws {
        let other = RemoteRequestID(rawValue: 2)
        let executor = try Self.makeExecutor()
        _ = try executor.handle(
            RemoteModelRequest(requestID: Self.request, sequence: 1, op: .begin))

        let reply = try executor.handle(
            RemoteModelRequest(requestID: other, sequence: 2, op: .newCache(Self.cache)))

        #expect(
            reply == .failed(Self.text(of: .requestMismatch(expected: Self.request, found: other))))
    }

    @Test("a begin while a request is open fails")
    func aBeginWhileARequestIsOpenFails() throws {
        let reply = try Self.run([.begin, .begin], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .requestStillOpen(Self.request))))
    }

    @Test("a begin must name a new request")
    func aBeginMustNameANewRequest() throws {
        let reply = try Self.run([.begin, .end, .begin], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .staleRequest(Self.request))))
    }

    @Test("an end without an open request fails")
    func anEndWithoutAnOpenRequestFails() throws {
        let reply = try Self.run([.end], on: Self.makeExecutor())

        #expect(reply == .failed(Self.text(of: .noOpenRequest)))
    }

    @Test("an op after an end carries the id of the ended request")
    func anOpAfterAnEndCarriesTheIdOfTheEndedRequest() throws {
        let reply = try Self.run([.begin, .end, .newCache(Self.cache)], on: Self.makeExecutor())

        #expect(Self.status(of: reply)?.offset == 0)
    }

    // MARK: - Announcements

    @Test("an announced op answers as the op itself does")
    func anAnnouncedOpAnswersAsTheOpItselfDoes() throws {
        let leader = try Self.makeExecutor()
        let member = try Self.makeExecutor()
        let requests = Self.requests(
            Self.opening + [
                .forward(tokens: Self.prompt, cache: Self.cache, logits: .lastPosition)
            ])

        let led = try requests.map { try leader.handle($0) }
        let applied = try requests.map {
            try member.apply(announcement: RemoteOpAnnouncement.encode($0))
        }

        #expect(applied == led)
    }

    @Test("a park answers nothing")
    func aParkAnswersNothing() throws {
        let executor = try Self.makeExecutor()
        _ = try Self.run([.begin], on: executor)

        let reply = try executor.apply(
            announcement: RemoteOpAnnouncement.park(requestID: Self.request))

        #expect(reply == nil)
    }

    @Test("a park of another request fails on a member")
    func aParkOfAnotherRequestFailsOnAMember() throws {
        let executor = try Self.makeExecutor()
        _ = try Self.run([.begin], on: executor)

        #expect(
            throws: RemoteModelExecutorError.requestMismatch(
                expected: Self.request, found: Self.otherRequest)
        ) {
            try executor.apply(
                announcement: RemoteOpAnnouncement.park(requestID: Self.otherRequest))
        }
    }

    @Test("a park of another request gets a failed reply on rank 0")
    func aParkOfAnotherRequestGetsAFailedReplyOnRankZero() throws {
        let executor = try Self.makeExecutor()
        _ = try Self.run([.begin], on: executor)

        let reply = executor.handlePark(requestID: Self.otherRequest)

        #expect(
            reply
                == .failed(
                    Self.text(
                        of: .requestMismatch(expected: Self.request, found: Self.otherRequest))))
    }

    @Test("a park of the open request is done on rank 0")
    func aParkOfTheOpenRequestIsDoneOnRankZero() throws {
        let executor = try Self.makeExecutor()
        _ = try Self.run([.begin], on: executor)

        #expect(executor.handlePark(requestID: Self.request) == .done)
    }

    @Test("a park leaves the cache and the sequence of the request as they were")
    func aParkLeavesTheCacheAndTheSequenceAsTheyWere() throws {
        let parked = try Self.makeExecutor()
        let unparked = try Self.makeExecutor()
        let before = Self.requests(
            Self.opening + [.forward(tokens: Self.prompt, cache: Self.cache, logits: .none)])
        let next = Self.requests(
            [.forward(tokens: [12], cache: Self.cache, logits: .lastPosition)],
            from: UInt64(before.count + 1))[0]
        for request in before {
            _ = try parked.apply(announcement: RemoteOpAnnouncement.encode(request))
            _ = try unparked.apply(announcement: RemoteOpAnnouncement.encode(request))
        }

        _ = try parked.apply(announcement: RemoteOpAnnouncement.park(requestID: Self.request))
        let resumed = try parked.apply(announcement: RemoteOpAnnouncement.encode(next))

        let straight = try unparked.apply(announcement: RemoteOpAnnouncement.encode(next))
        #expect(try Self.logits(of: resumed) == Self.logits(of: straight))
        #expect(try Self.logits(of: resumed) != nil)
    }
}

/// A directory that holds one `config.json`, removed when the value is.
private final class ConfigurationDirectory {

    /// The directory.
    let url: URL

    /// Writes `json` as the `config.json` of a new directory.
    init(json: String) throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "remote-executor-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: url.appendingPathComponent("config.json"))
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }
}

/// The small configuration of each shardable model family, made from the fixtures of its model.
private enum ShardableFamilyConfiguration {

    /// The model of `family`, from the LLM factory, or from the VLM factory for a type only it
    /// knows, as a worker loads it.
    static func makeModel(of family: String) async throws -> any LanguageModel {
        let configuration = Data(try json(of: family).utf8)
        do {
            return try await LLMTypeRegistry.shared.createModel(
                configuration: configuration, modelType: family)
        } catch ModelFactoryError.unsupportedModelType {
            return try await VLMTypeRegistry.shared.createModel(
                configuration: configuration, modelType: family)
        }
    }

    /// The fixture `config.json` of each family, and the model type the fixture names.
    private static let fixtures: [String: (json: String, modelType: String)] = [
        "qwen3_5": (Qwen35FixtureJSON.visionLanguage(mtpLayers: 0), "qwen3_5"),
        "qwen3_8": (Qwen35FixtureJSON.visionLanguage(mtpLayers: 0), "qwen3_5"),
        "qwen3_5_text": (Qwen35FixtureJSON.text(mtpLayers: 0), "qwen3_5_text"),
        "qwen3_8_text": (Qwen35FixtureJSON.text(mtpLayers: 0), "qwen3_5_text"),
        "inkling2": (Inkling2ModelTests.tinyJSON, "inkling2_loop"),
        "inkling2_loop": (Inkling2ModelTests.tinyJSON, "inkling2_loop"),
    ]

    /// A small `config.json` of `family`.
    private static func json(of family: String) throws -> String {
        let fixture = try #require(fixtures[family], "no fixture configuration for \(family)")
        return fixture.json.replacingOccurrences(
            of: "\"model_type\": \"\(fixture.modelType)\"",
            with: "\"model_type\": \"\(family)\"")
    }
}

/// An executor over a tiny Qwen 3.5 target whose layers are all full attention, so a cache
/// trims.
private enum AttentionOnlyExecutor {

    /// The seed of the weights.
    private static let seed: UInt64 = 21

    /// The decoder layers, every one full attention.
    private static let layerCount = 2

    /// Makes the executor.
    static func make() throws -> RemoteModelExecutor {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                Qwen35FixtureJSON.text(
                    mtpLayers: 0, hiddenLayers: layerCount, fullAttentionInterval: 1,
                    hiddenSize: Qwen35MTPFixture.hiddenSize,
                    vocabularySize: Qwen35MTPFixture.vocabularySize
                ).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: seed)) {
            Qwen35TextModel(configuration)
        }
        eval(model)
        return RemoteModelExecutor(
            model: model,
            configuration: .init(vocabularySize: Qwen35MTPFixture.vocabularySize))
    }
}

/// A model that does not conform to ``TensorParallelShardable``. Only its conformance is read.
private final class UnshardableModel: Module, LanguageModel {

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        []
    }
}
