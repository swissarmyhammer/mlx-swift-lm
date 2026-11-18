// Copyright © 2026 Apple Inc.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

import Foundation

/// The free space of one volume, as the file system reports it under two keys.
///
/// The prompt cache store takes its default disk budget from it, and
/// `MLXLanguageModel.freeDiskSpaceBytes` reports it for the download pre-flight. Both read
/// the volume through ``nearestExistingFolder(to:)`` and ``read(of:)``, and both take
/// ``usableFreeSpace``.
struct ExecutorPromptCacheVolumeCapacity: Sendable {

    /// The value of `volumeAvailableCapacityForImportantUsage`: the free space for a file the
    /// system thinks important, purgeable space included. Nil when the system gives none.
    let importantUsage: Int64?

    /// The value of `volumeAvailableCapacity`: the free space of the file system, without the
    /// purgeable space. Nil when the system gives none.
    let available: Int?

    /// The free space a caller can use, and the key it comes from.
    ///
    /// The important-usage value counts when it is positive. macOS can report zero for it,
    /// or no value, while the volume has room: after a restart, before it accounts for the
    /// purgeable space again. The plain value then counts. When neither key gives a value,
    /// there is no usable free space.
    var usableFreeSpace: (bytes: Int, key: URLResourceKey)? {
        if let importantUsage, importantUsage > 0 {
            return (Int(clamping: importantUsage), .volumeAvailableCapacityForImportantUsageKey)
        }
        if let available {
            return (available, .volumeAvailableCapacityKey)
        }
        return importantUsage.map {
            (Int(clamping: $0), .volumeAvailableCapacityForImportantUsageKey)
        }
    }

    /// Reads the free space of the volume that holds `folder` under both keys. This is the
    /// default ``ExecutorPromptCacheVolumeCapacityReader``.
    ///
    /// - Parameter folder: a folder that exists.
    /// - Returns: the two values that the file system reports.
    /// - Throws: the error of the file system.
    static func read(of folder: URL) throws -> ExecutorPromptCacheVolumeCapacity {
        let values = try folder.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey,
        ])
        return ExecutorPromptCacheVolumeCapacity(
            importantUsage: values.volumeAvailableCapacityForImportantUsage,
            available: values.volumeAvailableCapacity)
    }

    /// The nearest folder at or above `url` that exists, or the root of `url` when none does.
    ///
    /// A folder that is not made yet (a spool folder before its first write, the weights
    /// folder of a model before its download) has no volume to read, thus a reader reads
    /// the nearest folder above it that exists.
    ///
    /// - Parameter url: the folder whose volume a caller reads.
    /// - Returns: the folder to read.
    static func nearestExistingFolder(to url: URL) -> URL {
        var folder = url
        while !FileManager.default.fileExists(atPath: folder.path(percentEncoded: false)),
            folder.pathComponents.count > 1
        {
            folder = folder.deletingLastPathComponent()
        }
        return folder
    }
}

/// Reads the free space of the volume that holds a folder that exists.
typealias ExecutorPromptCacheVolumeCapacityReader =
    @Sendable (URL) throws -> ExecutorPromptCacheVolumeCapacity

#endif  // canImport(FoundationModels, _version: 2)
#endif  // FoundationModelsIntegration
