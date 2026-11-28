// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// The op set of a remote model: each op survives its fixed-size announce words and its
/// Codable form, and the logits survive their bytes.
@Suite("The remote model op set round-trips")
struct RemoteModelProtocolTests {

    /// A request of `op` with a request id and a sequence number that need both 32-bit halves.
    private static func request(_ op: RemoteModelOp) -> RemoteModelRequest {
        RemoteModelRequest(
            requestID: RemoteRequestID(rawValue: 0x1_0000_0002),
            sequence: 0x3_0000_0004, op: op)
    }

    /// A cache handle that needs both 32-bit halves.
    private static let cache = RemoteCacheID(rawValue: 0xFFFF_0000_1234_5678)

    /// A second cache handle, for the target of a copy.
    private static let copyTarget = RemoteCacheID(rawValue: 0x8000_0000_0000_0001)

    /// One op of each kind.
    private static let everyOp: [RemoteModelOp] = [
        .describe,
        .begin,
        .end,
        .newCache(cache),
        .forward(tokens: [1, 2, 3], cache: cache, logits: .lastPosition),
        .forward(tokens: [7], cache: cache, logits: .allPositions),
        .forward(tokens: [4, 5], cache: cache, logits: .none),
        .trim(cache, count: 17),
        .copy(cache, into: copyTarget),
        .reset(cache),
        .release(cache),
        .speculate(round(draftLimit: 3)),
        .speculate(round(draftLimit: 0)),
    ]

    /// A round of ``cache`` with a token, rejected drafts and a confidence floor in each field.
    private static func round(draftLimit: Int) -> RemoteDraftRound {
        RemoteDraftRound(
            cache: cache, token: 41, rejectedCount: 2, draftLimit: draftLimit,
            minimumDraftProbability: 0.25)
    }

    @Test("every op survives its announce words", arguments: everyOp)
    func everyOpSurvivesItsAnnounceWords(op: RemoteModelOp) throws {
        let words = try RemoteOpAnnouncement.encode(Self.request(op))

        #expect(try RemoteOpAnnouncement.decode(words) == Self.request(op))
    }

    @Test("every announce has the same number of words", arguments: everyOp)
    func everyAnnounceHasTheSameNumberOfWords(op: RemoteModelOp) throws {
        #expect(
            try RemoteOpAnnouncement.encode(Self.request(op)).count
                == RemoteOpAnnouncement.wordCount)
    }

    @Test("a forward of a whole chunk survives its announce words")
    func aForwardOfAWholeChunkSurvivesItsAnnounceWords() throws {
        let tokens = (0 ..< RemoteModelProtocol.maximumChunkTokenCount).map { Int32($0) }
        let request = Self.request(.forward(tokens: tokens, cache: Self.cache, logits: .none))

        #expect(try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.encode(request)) == request)
    }

    @Test("a forward longer than a chunk has no announce")
    func aForwardLongerThanAChunkHasNoAnnounce() {
        let tokens = [Int32](repeating: 1, count: RemoteModelProtocol.maximumChunkTokenCount + 1)
        let request = Self.request(.forward(tokens: tokens, cache: Self.cache, logits: .none))

        #expect(throws: RemoteModelProtocolError.chunkTooLong(tokenCount: tokens.count)) {
            try RemoteOpAnnouncement.encode(request)
        }
    }

    @Test("a round whose verify does not fit a chunk has no announce")
    func aRoundWhoseVerifyDoesNotFitAChunkHasNoAnnounce() {
        let draftLimit = RemoteModelProtocol.maximumChunkTokenCount
        let request = Self.request(.speculate(Self.round(draftLimit: draftLimit)))

        #expect(throws: RemoteModelProtocolError.invalidCount(Int32(draftLimit))) {
            try RemoteOpAnnouncement.encode(request)
        }
    }

    @Test("the largest round whose verify fits a chunk survives its announce words")
    func theLargestRoundWhoseVerifyFitsAChunkSurvivesItsAnnounceWords() throws {
        let draftLimit = RemoteModelProtocol.maximumChunkTokenCount - 1
        let request = Self.request(.speculate(Self.round(draftLimit: draftLimit)))

        #expect(try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.encode(request)) == request)
    }

    @Test("a round whose verify does not fit a chunk fails validation")
    func aRoundWhoseVerifyDoesNotFitAChunkFailsValidation() {
        let draftLimit = RemoteModelProtocol.maximumChunkTokenCount

        #expect(throws: RemoteModelProtocolError.invalidCount(Int32(draftLimit))) {
            try Self.round(draftLimit: draftLimit).validate()
        }
    }

    @Test("a round that rewinds a negative count fails validation")
    func aRoundThatRewindsANegativeCountFailsValidation() {
        let round = RemoteDraftRound(
            cache: Self.cache, token: 1, rejectedCount: -1, draftLimit: 1,
            minimumDraftProbability: 0)

        #expect(throws: RemoteModelProtocolError.invalidCount(-1)) { try round.validate() }
    }

    @Test("a round that rewinds more than a word holds fails validation")
    func aRoundThatRewindsMoreThanAWordHoldsFailsValidation() {
        let round = RemoteDraftRound(
            cache: Self.cache, token: 1, rejectedCount: Int(Int32.max) + 1, draftLimit: 1,
            minimumDraftProbability: 0)

        #expect(throws: RemoteModelProtocolError.invalidCount(Int32.max)) {
            try round.validate()
        }
    }

    @Test("a round that rewinds a negative count has no announce")
    func aRoundThatRewindsANegativeCountHasNoAnnounce() {
        let round = RemoteDraftRound(
            cache: Self.cache, token: 1, rejectedCount: -1, draftLimit: 1,
            minimumDraftProbability: 0)

        #expect(throws: RemoteModelProtocolError.invalidCount(-1)) {
            try RemoteOpAnnouncement.encode(Self.request(.speculate(round)))
        }
    }

    @Test("only the end of a request raises the stop flag")
    func onlyTheEndOfARequestRaisesTheStopFlag() throws {
        let stops = try Self.everyOp.map {
            RemoteOpAnnouncement.stopsLoop(try RemoteOpAnnouncement.encode(Self.request($0)))
        }

        #expect(stops == Self.everyOp.map { $0 == .end })
    }

    @Test("no op announces a park", arguments: everyOp)
    func noOpAnnouncesAPark(op: RemoteModelOp) throws {
        #expect(!RemoteOpAnnouncement.isPark(try RemoteOpAnnouncement.encode(Self.request(op))))
    }

    @Test("a park announces a park")
    func aParkAnnouncesAPark() {
        #expect(
            RemoteOpAnnouncement.isPark(
                RemoteOpAnnouncement.park(requestID: Self.request(.begin).requestID)))
    }

    @Test("a park has the words of every announce")
    func aParkHasTheWordsOfEveryAnnounce() {
        #expect(
            RemoteOpAnnouncement.park(requestID: .none).count == RemoteOpAnnouncement.wordCount)
    }

    @Test("a park does not raise the stop flag")
    func aParkDoesNotRaiseTheStopFlag() {
        #expect(!RemoteOpAnnouncement.stopsLoop(RemoteOpAnnouncement.park(requestID: .none)))
    }

    @Test("a park decodes to no request")
    func aParkDecodesToNoRequest() {
        #expect(throws: RemoteModelProtocolError.parkHasNoRequest) {
            try RemoteOpAnnouncement.decode(RemoteOpAnnouncement.park(requestID: .none))
        }
    }

    @Test("rank 0 exchanges the words of its op with a group of one")
    func rankZeroExchangesTheWordsOfItsOp() throws {
        let words = try RemoteOpAnnouncement.encode(Self.request(.reset(Self.cache)))

        let exchanged = try RemoteOpAnnouncement.exchange(words, group: MLXDistributed.initialize())

        #expect(exchanged == words)
    }

    @Test("a member exchanges an announce of zeros with a group of one")
    func aMemberExchangesAnAnnounceOfZeros() throws {
        let exchanged = try RemoteOpAnnouncement.exchange(nil, group: MLXDistributed.initialize())

        #expect(exchanged == [Int32](repeating: 0, count: RemoteOpAnnouncement.wordCount))
    }

    @Test("words of the wrong length do not decode")
    func wordsOfTheWrongLengthDoNotDecode() {
        #expect(throws: RemoteModelProtocolError.announceLength(1)) {
            try RemoteOpAnnouncement.decode([0])
        }
    }

    @Test("an unknown opcode does not decode")
    func anUnknownOpcodeDoesNotDecode() throws {
        var words = try RemoteOpAnnouncement.encode(Self.request(.begin))
        words[0] = Int32.max

        #expect(throws: RemoteModelProtocolError.unknownOpcode(Int32.max)) {
            try RemoteOpAnnouncement.decode(words)
        }
    }

    @Test("every op survives JSON", arguments: everyOp)
    func everyOpSurvivesJSON(op: RemoteModelOp) throws {
        let data = try JSONEncoder().encode(Self.request(op))

        #expect(try JSONDecoder().decode(RemoteModelRequest.self, from: data) == Self.request(op))
    }

    @Test("a forward reply survives JSON")
    func aForwardReplySurvivesJSON() throws {
        let reply = RemoteModelReply.forwarded(
            Self.status,
            logits: RemoteLogits(shape: [1, 1, 2], elementType: .float32, bytes: Data([1, 2])))

        let data = try JSONEncoder().encode(reply)

        #expect(try JSONDecoder().decode(RemoteModelReply.self, from: data) == reply)
    }

    @Test("a drafted reply survives JSON")
    func aDraftedReplySurvivesJSON() throws {
        let reply = RemoteModelReply.drafted(
            Self.status, drafts: [4, 9],
            logits: RemoteLogits(shape: [1, 3, 2], elementType: .float32, bytes: Data([1, 2])))

        let data = try JSONEncoder().encode(reply)

        #expect(try JSONDecoder().decode(RemoteModelReply.self, from: data) == reply)
    }

    @Test("the capabilities of a worker with a drafter survive JSON")
    func theCapabilitiesOfAWorkerWithADrafterSurviveJSON() throws {
        let reply = RemoteModelReply.capabilities(
            RemoteModelCapabilities(
                protocolVersion: RemoteModelProtocol.version, modelFamilies: ["inkling2"],
                supportsSharding: true, rankCount: 2, vocabularySize: 64,
                defaultPrefillStepSize: 512, maximumPrefillStepSize: nil,
                preparesEveryPrompt: true, modelType: "inkling2", quantization: nil,
                drafter: RemoteDrafterCapabilities(
                    drafterID: "Inkling2DraftModel#1f", maximumBlockSize: 4,
                    preferredBlockSize: 3)))

        let data = try JSONEncoder().encode(reply)

        #expect(try JSONDecoder().decode(RemoteModelReply.self, from: data) == reply)
    }

    @Test("capabilities survive JSON")
    func capabilitiesSurviveJSON() throws {
        let reply = RemoteModelReply.capabilities(
            RemoteModelCapabilities(
                protocolVersion: RemoteModelProtocol.version, modelFamilies: ["qwen3_5"],
                supportsSharding: true, rankCount: 2, vocabularySize: 64,
                defaultPrefillStepSize: 512, maximumPrefillStepSize: 1024,
                preparesEveryPrompt: true, modelType: "qwen3_5",
                quantization: .init(groupSize: 64, bits: 4)))

        let data = try JSONEncoder().encode(reply)

        #expect(try JSONDecoder().decode(RemoteModelReply.self, from: data) == reply)
    }

    /// A status with a value in each field.
    private static let status = RemoteCacheStatus(
        offset: 9, maxSize: 64, evictedTokenCount: 1, isTrimmable: true, maxTrimCount: 8,
        rewindLimit: .within(40), canLoseRewind: true, residentByteCount: 4096)

    @Test("float32 logits survive their bytes")
    func float32LogitsSurviveTheirBytes() throws {
        let logits = MLXArray((0 ..< 6).map { Float($0) / 3 }, [1, 2, 3])

        let restored = try RemoteLogits(logits).array()

        #expect(restored.dtype == .float32)
        #expect(restored.shape == [1, 2, 3])
        #expect(restored.asArray(Float.self) == logits.asArray(Float.self))
    }

    @Test("bfloat16 logits survive their bytes")
    func bfloat16LogitsSurviveTheirBytes() throws {
        let logits = MLXArray((0 ..< 4).map { Float($0) * 0.25 }, [1, 1, 4]).asType(.bfloat16)

        let restored = try RemoteLogits(logits).array()

        #expect(restored.dtype == .bfloat16)
        #expect(restored.asType(.float32).asArray(Float.self) == [0, 0.25, 0.5, 0.75])
    }

    @Test("logits whose bytes do not fill their shape are refused")
    func logitsWhoseBytesDoNotFillTheirShapeAreRefused() {
        let logits = RemoteLogits(shape: [1, 1, 4], elementType: .float32, bytes: Data([0]))

        #expect(throws: RemoteModelProtocolError.logitsByteCount(expected: 16, found: 1)) {
            try logits.array()
        }
    }
}
