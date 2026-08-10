// Copyright © 2026 Apple Inc.
//
// Guards the third-party notice file the attribution decision in
// CONTRIBUTING.md depends on. The decision says that per-file headers and one
// notice file at the root of the repository give the attribution for ported
// DeepSeek-V4 code. If that notice file does not exist, the attribution is
// incomplete, and these tests fail.

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

    /// Each repository the attribution decision names must have its own section
    /// in the notice. `osaurus-ai/vmlx-swift-lm` is the source of each ported
    /// file, so the notice must name it too.
    @Test func thirdPartyNoticesHasOneSectionForEachRepository() throws {
        let contents = try Self.noticeContents()
        #expect(contents.contains("## osaurus-ai/vmlx-swift-lm"))
        #expect(contents.contains("## scouzi1966/mlx-swift-lm"))
        #expect(contents.contains("## scouzi1966/maclocal-api"))
    }

    /// The MIT license makes the copyright notice necessary, so the notice must
    /// give the MIT text and the copyright line of each repository.
    @Test func thirdPartyNoticesHasTheCopyrightLineOfEachRepository() throws {
        let contents = try Self.noticeContents()
        #expect(contents.contains("MIT License"))
        #expect(contents.contains("Copyright (c) 2024 ml-explore"))
        #expect(contents.contains("Copyright (c) 2026 Osaurus contributors"))
        #expect(contents.contains("Copyright (c) 2025 MacLocalAPI Contributors"))
    }
}
