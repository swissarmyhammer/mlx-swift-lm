// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// A ``Downloader`` that records each request and downloads nothing.
///
/// Each request answers with the directory of its repository under ``directory``, see
/// ``directory(for:)``, so a test can tell the model directory from the tokenizer directory, or
/// put the files of a load there before the load.
public struct RecordingDownloader: Downloader {

    /// One request to the downloader.
    public struct Call: Equatable, Sendable {
        /// The repository.
        public let id: String
        /// The revision, `nil` for the default revision.
        public let revision: String?
        /// The file patterns of the request.
        public let patterns: [String]

        /// Creates a request.
        ///
        /// - Parameters:
        ///   - id: the repository.
        ///   - revision: the revision, `nil` for the default revision.
        ///   - patterns: the file patterns of the request.
        public init(id: String, revision: String?, patterns: [String]) {
            self.id = id
            self.revision = revision
            self.patterns = patterns
        }
    }

    /// The parent of the repository directories that the requests answer with.
    public let directory: URL

    /// The requests, in order.
    private let recorded = RecordedCalls()

    /// Creates a downloader whose requests answer with directories under `directory`.
    ///
    /// - Parameter directory: the parent of the repository directories.
    public init(directory: URL = URL(filePath: "/mock")) {
        self.directory = directory
    }

    /// The requests so far, in order.
    public var calls: [Call] {
        recorded.calls
    }

    /// The directory that a request for `id` answers with.
    ///
    /// - Parameter id: the repository.
    /// - Returns: the directory under ``directory`` that the repository name names, with `_`
    ///   for each `/`.
    public func directory(for id: String) -> URL {
        directory.appending(component: id.replacingOccurrences(of: "/", with: "_"))
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        recorded.append(Call(id: id, revision: revision, patterns: patterns))
        return directory(for: id)
    }
}

/// The requests of a ``RecordingDownloader``, behind a lock, because a load can download from
/// any thread.
private final class RecordedCalls: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [RecordingDownloader.Call] = []

    /// The requests so far, in order.
    var calls: [RecordingDownloader.Call] {
        lock.withLock { values }
    }

    /// Records `call` after the requests so far.
    ///
    /// - Parameter call: the request.
    func append(_ call: RecordingDownloader.Call) {
        lock.withLock { values.append(call) }
    }
}
