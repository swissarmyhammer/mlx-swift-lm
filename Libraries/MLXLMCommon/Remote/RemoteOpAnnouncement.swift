// Copyright © 2026 Apple Inc.

import Foundation

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
/// | 0 | The opcode: 1 describe, 2 begin, 3 end, 4 newCache, 5 forward, 6 trim, 7 copy, 8 reset, 9 release. |
/// | 1 | The flags: bit 0 is the stop flag, which only ``RemoteModelOp/end`` raises; bits 1 and 2 hold the ``RemoteLogitsScope`` of a forward. |
/// | 2 | The count: the tokens of a forward, the positions of a trim, else 0. |
/// | 3, 4 | ``RemoteModelRequest/sequence``, low half then high half. |
/// | 5, 6 | ``RemoteModelRequest/requestID``, low half then high half. |
/// | 7, 8 | The cache handle of the op, low half then high half, else 0. |
/// | 9, 10 | The target handle of a copy, low half then high half, else 0. |
/// | 11 ... | The tokens of a forward, then zeros, ``RemoteModelProtocol/maximumChunkTokenCount`` words. |
///
/// No opcode is 0, thus a buffer of zeros never decodes to an op.
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
        case begin, end, newCache, forward, trim, copy, reset, release
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
    ///   ``RemoteModelProtocolError/invalidCount(_:)`` for a trim count outside `Int32`.
    public static func encode(_ request: RemoteModelRequest) throws -> [Int32] {
        var words = [Int32](repeating: 0, count: wordCount)
        func set(_ word: Word, _ value: Int32) { words[word.rawValue] = value }
        func set(_ low: Word, _ high: Word, _ value: UInt64) {
            set(low, Int32(bitPattern: UInt32(truncatingIfNeeded: value)))
            set(high, Int32(bitPattern: UInt32(truncatingIfNeeded: value >> halfWidth)))
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
        case .newCache(let cache):
            opcode = .newCache
            set(.cacheLow, .cacheHigh, cache.rawValue)
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
        }
        set(.opcode, opcode.rawValue)
        return words
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
        let joined = { (low: Word, high: Word) in
            UInt64(UInt32(bitPattern: word(low)))
                | UInt64(UInt32(bitPattern: word(high))) << halfWidth
        }
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
        case .newCache: op = .newCache(cache)
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
        }
        return RemoteModelRequest(
            requestID: RemoteRequestID(rawValue: joined(.requestLow, .requestHigh)),
            sequence: joined(.sequenceLow, .sequenceHigh), op: op)
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
