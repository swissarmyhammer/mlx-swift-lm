// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXFoundationModels

/// The shape and the values of the fixture caches of ``PromptCacheSpoolFixtures``.
enum PromptCacheSpoolFixtureShape {

    /// The ledger of each entry.
    static let tokens = [1, 42, 7]

    /// The number of key/value heads of the cache.
    static let headCount = 2

    /// The head dimension of the cache.
    static let headDimension = 4

    /// The distance between the first values of two fixture arrays.
    static let valueStride: Float = 100
}

/// The fixture arrays, in the order of their first values.
enum PromptCacheSpoolFixtureArray: Int {
    case keys
    case values
    case nextToken
}

/// Fixtures for the suites that spill prompt cache entries to disk. A suite conforms to get
/// them as static members.
///
/// No weights are needed. Each entry carries one `KVCacheSimple` with fixed values. Each test
/// makes its own store in its own temporary folder.
protocol PromptCacheSpoolFixtures {}

extension PromptCacheSpoolFixtures {

    /// The ledger of each entry.
    static var tokens: [Int] { PromptCacheSpoolFixtureShape.tokens }

    /// Makes the keys or the values of a block of tokens, with consecutive values.
    ///
    /// - Parameters:
    ///   - tokenCount: The number of tokens of the block.
    ///   - array: The fixture array. It sets the first value.
    /// - Returns: An array of shape `(1, heads, tokens, headDimension)`, as `float16`.
    static func block(tokenCount: Int, array: PromptCacheSpoolFixtureArray) -> MLXArray {
        let shape = [
            1, PromptCacheSpoolFixtureShape.headCount, tokenCount,
            PromptCacheSpoolFixtureShape.headDimension,
        ]
        let count = shape.reduce(1, *)
        let start = Float(array.rawValue) * PromptCacheSpoolFixtureShape.valueStride
        return MLXArray((0 ..< count).map { start + Float($0) }).reshaped(shape).asType(.float16)
    }

    /// Makes an entry with one `KVCacheSimple` that holds ``tokens``.
    ///
    /// - Returns: The entry.
    static func entry() -> ExecutorPromptCacheEntry {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: block(tokenCount: tokens.count, array: .keys),
            values: block(tokenCount: tokens.count, array: .values))
        return ExecutorPromptCacheEntry(caches: [cache], tokens: tokens)
    }

    /// Makes the name of an empty temporary folder for one test. The folder is not made: the
    /// store makes it at its first write.
    ///
    /// - Returns: The URL of the folder.
    static func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(Self.self)-\(UUID().uuidString)")
    }

    /// Makes a store in `directory` with the budget `budget`.
    ///
    /// - Parameters:
    ///   - directory: The spool folder of the store.
    ///   - budget: The memory budget in bytes. Zero spills each check-in at once.
    ///   - writer: The writer of the spill files.
    /// - Returns: The store.
    static func store(
        in directory: URL, budget: Int = 0,
        writer: @escaping ExecutorPromptCacheFileWriter = ExecutorPromptCacheStore.writeSpillFile
    ) async -> ExecutorPromptCacheStore {
        let store = ExecutorPromptCacheStore(directory: directory, writer: writer)
        await store.configure(memoryBudgetBytes: budget)
        return store
    }

    /// The names of the spill files in `directory`, sorted. A folder that does not exist
    /// holds none.
    ///
    /// - Parameter directory: The spool folder.
    /// - Returns: The file names.
    static func fileNames(in directory: URL) throws -> [String] {
        guard FileManager.default.fileExists(atPath: directory.path(percentEncoded: false))
        else {
            return []
        }
        return try FileManager.default.contentsOfDirectory(atPath: directory.path).sorted()
    }

    /// Records an issue unless `store` holds nothing for `key` and no file stays in `directory`:
    /// no entry in memory, no write that has not ended, no disk record and no byte on any tier.
    ///
    /// - Parameters:
    ///   - key: The key that the test removed.
    ///   - store: The store.
    ///   - directory: The spool folder of the store.
    static func expectNothingStored(
        for key: ExecutorPromptCacheKey, in store: ExecutorPromptCacheStore, directory: URL
    ) async throws {
        #expect(await store.retainedByteCount == 0)
        #expect(await store.spillingByteCount == 0)
        #expect(await store.diskByteCount == 0)
        #expect(try fileNames(in: directory).isEmpty)
        #expect(await store.checkOut(key) == .none)
    }
}

/// A spill writer that holds each write until the test releases it, and keeps a copy of each
/// file that it writes.
///
/// The store deletes a file that nobody needs when its write ends. The copy lets a test read
/// the bytes that the writer wrote all the same.
final class HeldWriter: Sendable {

    /// The URL of each write, in the order the writes start.
    let startedWrites: AsyncStream<URL>

    /// Receives the URL of each write that starts.
    private let starts: AsyncStream<URL>.Continuation

    /// Holds each write until ``release()``.
    private let releases = DispatchSemaphore(value: 0)

    /// The folder of the copies.
    let copies: URL

    /// Creates a writer that keeps its copies in `copies`.
    ///
    /// - Parameter copies: The folder of the copies. The writer makes it.
    init(copies: URL) throws {
        (startedWrites, starts) = AsyncStream.makeStream(of: URL.self)
        self.copies = copies
        try FileManager.default.createDirectory(at: copies, withIntermediateDirectories: true)
    }

    /// Tells the test that a write started, waits for a release, and writes the file and its
    /// copy.
    ///
    /// - Parameters:
    ///   - input: The prepared entry.
    ///   - url: The URL of the file.
    func write(_ input: PromptCacheSaveInput, to url: URL) throws {
        starts.yield(url)
        releases.wait()
        try ExecutorPromptCacheStore.writeSpillFile(input, to: url)
        try FileManager.default.copyItem(at: url, to: copy(of: url))
    }

    /// Lets one held write run.
    func release() {
        releases.signal()
    }

    /// The URL of the copy of `url`.
    ///
    /// - Parameter url: The URL of a file that the writer wrote.
    /// - Returns: The URL of its copy.
    func copy(of url: URL) -> URL {
        copies.appendingPathComponent(url.lastPathComponent)
    }
}

#endif
