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
    ]

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

    @Test("capabilities survive JSON")
    func capabilitiesSurviveJSON() throws {
        let reply = RemoteModelReply.capabilities(
            RemoteModelCapabilities(
                protocolVersion: RemoteModelProtocol.version, modelFamilies: ["qwen3_5"],
                supportsSharding: true, rankCount: 2, vocabularySize: 64,
                defaultPrefillStepSize: 512, maximumPrefillStepSize: 1024))

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
