// Copyright © 2026 Apple Inc.

import os

/// The unified log every Inkling 2 integration suite writes its result lines to.
enum Inkling2ResultLog {

    /// The log the result lines go to.
    private static let log = Logger(
        subsystem: "com.apple.FoundationModels-MLX", category: "Inkling2")

    /// Writes one result line under `tag`, which the unified log of the run can be searched for.
    static func report(_ tag: String, _ line: String) {
        log.info("[\(tag, privacy: .public)] \(line, privacy: .public)")
    }
}
