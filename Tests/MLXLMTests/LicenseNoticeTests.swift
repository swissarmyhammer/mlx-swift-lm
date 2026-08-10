// Copyright © 2026 Apple Inc.
//
// Guards the third-party notice file the attribution decision in
// CONTRIBUTING.md depends on. The decision says ported DeepSeek-V4 code is
// covered by per-file headers plus one notice file at the root of the
// repository. If that notice file goes away, the attribution is incomplete,
// and these tests fail.

import Foundation
import Testing

@Suite
struct LicenseNoticeTests {

    /// The root of the repository, found from the path of this source file.
    ///
    /// This file is at `Tests/MLXLMTests/LicenseNoticeTests.swift`, so the root
    /// is three path components above it.
    private static let repositoryRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/MLXLMTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repository root
    }()

    /// The third-party notice file that CONTRIBUTING.md names.
    private static let noticeFile: URL =
        repositoryRoot.appendingPathComponent("THIRD-PARTY-NOTICES.md", isDirectory: false)

    /// Reads the notice file as text.
    private static func noticeContents() throws -> String {
        try String(contentsOf: noticeFile, encoding: .utf8)
    }

    @Test func thirdPartyNoticesFileExists() {
        #expect(FileManager.default.fileExists(atPath: Self.noticeFile.path))
    }

    @Test func thirdPartyNoticesFileIsNotEmpty() throws {
        let contents = try Self.noticeContents()
        #expect(!contents.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Both reference repositories the decision names must appear in the notice,
    /// each with the MIT license text that covers it.
    @Test func thirdPartyNoticesNamesBothReferenceRepositories() throws {
        let contents = try Self.noticeContents()
        #expect(contents.contains("scouzi1966/mlx-swift-lm"))
        #expect(contents.contains("scouzi1966/maclocal-api"))
        #expect(contents.contains("MIT License"))
    }
}
