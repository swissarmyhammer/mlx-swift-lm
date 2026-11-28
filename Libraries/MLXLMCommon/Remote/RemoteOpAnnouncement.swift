// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// The fixed-size `Int32` form of a ``RemoteModelRequest``, which rank 0 announces to every
/// rank with one all-sum before each op.
///
/// Every announce holds ``wordCount`` words, whatever its op, so each rank sums a buffer of one
/// shape. Rank 0 writes the words of the op and every other rank writes zeros, thus the sum is
/// the announce of rank 0.
///
/// The words, in order:
///
/// | Word | Holds |
/// |---|---|
/// | 0 | The opcode: 1 describe, 2 begin, 3 end, 4 newCache, 5 forward, 6 trim, 7 copy, 8 reset, 9 release, 10 park, 11 speculate, 12 quantize. |
/// | 1 | The flags: bit 0 is the stop flag, which only ``RemoteModelOp/end`` raises; bits 1 and 2 hold the ``RemoteLogitsScope`` of a forward. |
/// | 2 | The count: the tokens of a forward, the positions of a trim, the draft limit of a round, the bits of the quantization of a new cache (0 for none), else 0. |
/// | 3, 4 | ``RemoteModelRequest/sequence``, low half then high half. |
/// | 5, 6 | ``RemoteModelRequest/requestID``, low half then high half. |
/// | 7, 8 | The cache handle of the op, low half then high half, else 0. |
/// | 9, 10 | The target handle of a copy, low half then high half, else 0. |
/// | 11 ... | The tokens of a forward, then zeros, ``RemoteModelProtocol/maximumChunkTokenCount`` words. A round puts its token, its rejected drafts and the bits of its confidence floor in the first three. A new cache puts the group size, the compression start and the compatibility of its quantization in the first three. |
///
/// No opcode is 0, thus a buffer of zeros never decodes to an op.
///
/// ## Park
///
/// A park (opcode 10) is no op and has no sequence number. Rank 0 announces it when a request
/// waits for its next op, so the other ranks can leave the all-sum loop and wait without
/// polling. A park changes no state: the next op keeps the request id and takes the next
/// sequence number. Build it with ``park(requestID:)`` and test for it with ``isPark(_:)``.
public enum RemoteOpAnnouncement {

    /// The position of each header word.
    private enum Word: Int, CaseIterable {
        case opcode, flags, count
        case sequenceLow, sequenceHigh
        case requestLow, requestHigh
        case cacheLow, cacheHigh
        case targetLow, targetHigh
    }

    /// The opcode of each op.
    private enum Opcode: Int32 {
        case describe = 1
        case begin, end, newCache, forward, trim, copy, reset, release, park, speculate, quantize
    }

    /// The payload words of a ``RemoteDraftRound``, after the header.
    private enum RoundWord: Int {
        case token, rejectedCount, minimumDraftProbability
    }

    /// The payload words of a ``RemoteKVCacheQuantization``, after the header. Its bits stand
    /// in the count word.
    private enum QuantizationWord: Int {
        case groupSize, compressionStart, compatibility
    }

    /// The stop flag in the flags word.
    private static let stopFlag: Int32 = 1

    /// The shift of the logits scope in the flags word.
    private static let logitsScopeShift: Int32 = 1

    /// The bits of the logits scope in the flags word, before the shift.
    private static let logitsScopeMask: Int32 = 0b11

    /// The bits of one half of a 64-bit value.
    private static let halfWidth: UInt64 = 32

    /// The words before the token payload.
    public static let headerWordCount = Word.allCases.count

    /// The words of every announce.
    public static let wordCount = headerWordCount + RemoteModelProtocol.maximumChunkTokenCount

    /// The announce words of `request`.
    ///
    /// - Parameter request: The request to announce.
    /// - Returns: ``wordCount`` words.
    /// - Throws: ``RemoteModelProtocolError/chunkTooLong(tokenCount:)`` for a forward of more
    ///   than ``RemoteModelProtocol/maximumChunkTokenCount`` tokens, and
    ///   ``RemoteModelProtocolError/invalidCount(_:)`` for a trim count outside `Int32`, a round
    ///   that fails ``RemoteDraftRound/validate()``, or a quantization value outside
    ///   `0 ... Int32.max`.
    public static func encode(_ request: RemoteModelRequest) throws -> [Int32] {
        var words = [Int32](repeating: 0, count: wordCount)
        func set(_ word: Word, _ value: Int32) { words[word.rawValue] = value }
        func set(_ low: Word, _ high: Word, _ value: UInt64) {
            write(value, into: &words, low: low, high: high)
        }
        set(.sequenceLow, .sequenceHigh, request.sequence)
        set(.requestLow, .requestHigh, request.requestID.rawValue)
        let opcode: Opcode
        switch request.op {
        case .describe: opcode = .describe
        case .begin: opcode = .begin
        case .end:
            opcode = .end
            set(.flags, stopFlag)
        case .newCache(let cache, let quantization):
            opcode = .newCache
            set(.cacheLow, .cacheHigh, cache.rawValue)
            if let quantization {
                try write(quantization, into: &words)
            }
        case .forward(let tokens, let cache, let logits):
            guard tokens.count <= RemoteModelProtocol.maximumChunkTokenCount else {
                throw RemoteModelProtocolError.chunkTooLong(tokenCount: tokens.count)
            }
            opcode = .forward
            set(.flags, logits.rawValue << logitsScopeShift)
            set(.count, Int32(tokens.count))
            set(.cacheLow, .cacheHigh, cache.rawValue)
            words.replaceSubrange(headerWordCount ..< headerWordCount + tokens.count, with: tokens)
        case .trim(let cache, let count):
            guard let count32 = Int32(exactly: count), count32 >= 0 else {
                throw RemoteModelProtocolError.invalidCount(Int32(clamping: count))
            }
            opcode = .trim
            set(.count, count32)
            set(.cacheLow, .cacheHigh, cache.rawValue)
        case .copy(let cache, let target):
            opcode = .copy
            set(.cacheLow, .cacheHigh, cache.rawValue)
            set(.targetLow, .targetHigh, target.rawValue)
        case .reset(let cache):
            opcode = .reset
            set(.cacheLow, .cacheHigh, cache.rawValue)
        case .release(let cache):
            opcode = .release
            set(.cacheLow, .cacheHigh, cache.rawValue)
        case .speculate(let round):
            opcode = .speculate
            try write(round, into: &words)
        case .quantize(let cache):
            opcode = .quantize
            set(.cacheLow, .cacheHigh, cache.rawValue)
        }
        set(.opcode, opcode.rawValue)
        return words
    }

    /// Writes the count, the cache and the payload words of `round` into `words`.
    ///
    /// - Throws: ``RemoteModelProtocolError/invalidCount(_:)`` for a round that fails
    ///   ``RemoteDraftRound/validate()``.
    private static func write(_ round: RemoteDraftRound, into words: inout [Int32]) throws {
        try round.validate()
        words[Word.count.rawValue] = Int32(round.draftLimit)
        write(round.cache.rawValue, into: &words, low: .cacheLow, high: .cacheHigh)
        words[payloadIndex(.token)] = round.token
        words[payloadIndex(.rejectedCount)] = Int32(round.rejectedCount)
        words[payloadIndex(.minimumDraftProbability)] = Int32(
            bitPattern: round.minimumDraftProbability.bitPattern)
    }

    /// The round of the count, the cache and the payload words of `words`.
    ///
    /// - Throws: ``RemoteModelProtocolError/invalidCount(_:)`` for a round that fails
    ///   ``RemoteDraftRound/validate()``.
    private static func round(of words: [Int32]) throws -> RemoteDraftRound {
        let round = RemoteDraftRound(
            cache: RemoteCacheID(rawValue: read(words, low: .cacheLow, high: .cacheHigh)),
            token: words[payloadIndex(.token)],
            rejectedCount: Int(words[payloadIndex(.rejectedCount)]),
            draftLimit: Int(words[Word.count.rawValue]),
            minimumDraftProbability: Float(
                bitPattern: UInt32(bitPattern: words[payloadIndex(.minimumDraftProbability)])))
        try round.validate()
        return round
    }

    /// The index in the announce words of the payload word `word` of a round.
    private static func payloadIndex(_ word: RoundWord) -> Int {
        headerWordCount + word.rawValue
    }

    /// Writes the bits and the payload words of `quantization` into `words`.
    ///
    /// - Throws: ``RemoteModelProtocolError/invalidCount(_:)`` with the offending value, clamped
    ///   to a word, for bits that are not positive or a value outside `0 ... Int32.max`.
    private static func write(
        _ quantization: RemoteKVCacheQuantization, into words: inout [Int32]
    ) throws {
        guard quantization.bits > 0 else {
            throw RemoteModelProtocolError.invalidCount(Int32(clamping: quantization.bits))
        }
        func word(_ value: Int) throws -> Int32 {
            guard let word = Int32(exactly: value), word >= 0 else {
                throw RemoteModelProtocolError.invalidCount(Int32(clamping: value))
            }
            return word
        }
        words[Word.count.rawValue] = try word(quantization.bits)
        words[payloadIndex(.groupSize)] = try word(quantization.groupSize)
        words[payloadIndex(.compressionStart)] = try word(quantization.compressionStart)
        words[payloadIndex(.compatibility)] = quantization.compatibility.rawValue
    }

    /// The quantization of the count and the payload words of `words`, or nil for a count of
    /// zero bits.
    ///
    /// - Throws: ``RemoteModelProtocolError/invalidCount(_:)`` for a negative value, and
    ///   ``RemoteModelProtocolError/unknownCompatibility(_:)`` for a compatibility that names no
    ///   policy.
    private static func quantization(of words: [Int32]) throws -> RemoteKVCacheQuantization? {
        let bits = words[Word.count.rawValue]
        guard bits != 0 else { return nil }
        let values = [
            bits, words[payloadIndex(.groupSize)], words[payloadIndex(.compressionStart)],
        ]
        if let negative = values.first(where: { $0 < 0 }) {
            throw RemoteModelProtocolError.invalidCount(negative)
        }
        let compatibilityValue = words[payloadIndex(.compatibility)]
        guard let compatibility = RemoteKVCacheCompatibility(rawValue: compatibilityValue) else {
            throw RemoteModelProtocolError.unknownCompatibility(compatibilityValue)
        }
        return RemoteKVCacheQuantization(
            bits: Int(bits), groupSize: Int(words[payloadIndex(.groupSize)]),
            compressionStart: Int(words[payloadIndex(.compressionStart)]),
            compatibility: compatibility)
    }

    /// The index in the announce words of the payload word `word` of a quantization.
    private static func payloadIndex(_ word: QuantizationWord) -> Int {
        headerWordCount + word.rawValue
    }

    /// The request of announce `words`.
    ///
    /// - Parameter words: The words of one announce.
    /// - Returns: The request.
    /// - Throws: ``RemoteModelProtocolError`` when the words have the wrong length or name no op.
    public static func decode(_ words: [Int32]) throws -> RemoteModelRequest {
        guard words.count == wordCount else {
            throw RemoteModelProtocolError.announceLength(words.count)
        }
        let word = { (word: Word) in words[word.rawValue] }
        let joined = { (low: Word, high: Word) in read(words, low: low, high: high) }
        guard let opcode = Opcode(rawValue: word(.opcode)) else {
            throw RemoteModelProtocolError.unknownOpcode(word(.opcode))
        }
        let cache = RemoteCacheID(rawValue: joined(.cacheLow, .cacheHigh))
        let count = word(.count)
        let op: RemoteModelOp
        switch opcode {
        case .describe: op = .describe
        case .begin: op = .begin
        case .end: op = .end
        case .newCache: op = .newCache(cache, quantization: try quantization(of: words))
        case .forward:
            guard (0 ... Int32(RemoteModelProtocol.maximumChunkTokenCount)).contains(count) else {
                throw RemoteModelProtocolError.invalidCount(count)
            }
            let scopeValue = (word(.flags) >> logitsScopeShift) & logitsScopeMask
            guard let scope = RemoteLogitsScope(rawValue: scopeValue) else {
                throw RemoteModelProtocolError.unknownLogitsScope(scopeValue)
            }
            let tokens = Array(words[headerWordCount ..< headerWordCount + Int(count)])
            op = .forward(tokens: tokens, cache: cache, logits: scope)
        case .trim:
            guard count >= 0 else { throw RemoteModelProtocolError.invalidCount(count) }
            op = .trim(cache, count: Int(count))
        case .copy:
            op = .copy(cache, into: RemoteCacheID(rawValue: joined(.targetLow, .targetHigh)))
        case .reset: op = .reset(cache)
        case .release: op = .release(cache)
        case .speculate: op = .speculate(try round(of: words))
        case .quantize: op = .quantize(cache)
        case .park: throw RemoteModelProtocolError.parkHasNoRequest
        }
        return RemoteModelRequest(
            requestID: RemoteRequestID(rawValue: joined(.requestLow, .requestHigh)),
            sequence: joined(.sequenceLow, .sequenceHigh), op: op)
    }

    /// The announce of a park in request `requestID`.
    ///
    /// - Parameter requestID: The open request.
    /// - Returns: ``wordCount`` words.
    public static func park(requestID: RemoteRequestID) -> [Int32] {
        var words = [Int32](repeating: 0, count: wordCount)
        words[Word.opcode.rawValue] = Opcode.park.rawValue
        write(requestID.rawValue, into: &words, low: .requestLow, high: .requestHigh)
        return words
    }

    /// Writes the halves of `value` into the words `low` and `high`.
    private static func write(_ value: UInt64, into words: inout [Int32], low: Word, high: Word) {
        words[low.rawValue] = Int32(bitPattern: UInt32(truncatingIfNeeded: value))
        words[high.rawValue] = Int32(bitPattern: UInt32(truncatingIfNeeded: value >> halfWidth))
    }

    /// The request a park names, or nil when `words` are no park.
    ///
    /// - Parameter words: The words of one announce.
    /// - Returns: The request id of a park, else nil.
    public static func parkedRequestID(_ words: [Int32]) -> RemoteRequestID? {
        guard isPark(words), words.count == wordCount else { return nil }
        return RemoteRequestID(rawValue: read(words, low: .requestLow, high: .requestHigh))
    }

    /// The 64-bit value whose halves the words `low` and `high` hold.
    private static func read(_ words: [Int32], low: Word, high: Word) -> UInt64 {
        UInt64(UInt32(bitPattern: words[low.rawValue]))
            | UInt64(UInt32(bitPattern: words[high.rawValue])) << halfWidth
    }

    /// Whether announce `words` are a park, which tells each rank other than rank 0 to leave
    /// the loop until the next op.
    ///
    /// - Parameter words: The words of one announce.
    /// - Returns: `true` for a park.
    public static func isPark(_ words: [Int32]) -> Bool {
        words.indices.contains(Word.opcode.rawValue)
            && words[Word.opcode.rawValue] == Opcode.park.rawValue
    }

    /// Sums announce words across `group`, as every rank does before each op.
    ///
    /// Rank 0 passes the words of its op, and every other rank passes nil, which stands for
    /// zeros. Every rank gets the words of rank 0. The sum runs on the CPU stream, where the
    /// ring and JACCL backends communicate. Every rank must call this the same number of times,
    /// in the same order.
    ///
    /// - Parameters:
    ///   - words: The words of the op on rank 0, or nil on every other rank.
    ///   - group: The group of the ranks.
    /// - Returns: The words of rank 0.
    /// - Throws: ``RemoteModelProtocolError/announceLength(_:)`` for words of another length
    ///   than ``wordCount``, or the error of the communication.
    public static func exchange(_ words: [Int32]?, group: MLXDistributed.Group) throws -> [Int32] {
        let local = words ?? [Int32](repeating: 0, count: wordCount)
        guard local.count == wordCount else {
            throw RemoteModelProtocolError.announceLength(local.count)
        }
        return try sum(local, group: group)
    }

    /// Sums `words` across `group` with one all-sum, and gives the sum to every rank.
    ///
    /// - Parameters:
    ///   - words: The words of this rank. Every rank passes as many.
    ///   - group: The group of the ranks.
    /// - Returns: The sum of the words of every rank.
    /// - Throws: The error of the communication.
    static func sum(_ words: [Int32], group: MLXDistributed.Group) throws -> [Int32] {
        try withError {
            let summed = MLXDistributed.allSum(MLXArray(words), group: group)
            try checkedEval(summed)
            return summed.asArray(Int32.self)
        }
    }

    /// Whether announce `words` raise the stop flag, which tells each rank to leave the loop of
    /// the current request.
    ///
    /// - Parameter words: The words of one announce.
    /// - Returns: `true` when the stop flag is set.
    public static func stopsLoop(_ words: [Int32]) -> Bool {
        words.indices.contains(Word.flags.rawValue)
            && words[Word.flags.rawValue] & stopFlag != 0
    }
}
