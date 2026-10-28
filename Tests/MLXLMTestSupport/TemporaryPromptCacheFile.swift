// Copyright © 2026 Apple Inc.

import Foundation

/// Paths of prompt cache files that a test writes and removes.
public enum TemporaryPromptCacheFile {

    /// The extension a prompt cache file must have.
    private static let pathExtension = "safetensors"

    /// A new prompt cache file path in the temporary directory, which no file holds yet.
    ///
    /// - Returns: The path. The caller removes the file it writes there.
    public static func makeURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(pathExtension)
    }
}
