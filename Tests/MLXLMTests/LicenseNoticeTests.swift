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

    /// The marker that starts and ends a fenced block in Markdown.
    private static let fenceMarker = "```"

    /// The prose sentences of the notice file, without the fenced blocks.
    ///
    /// Each fenced block is a verbatim quotation of a LICENSE or of a file
    /// header. A quotation is not a statement of this repository, so the checks
    /// below must not read it.
    private static func noticeSentences() throws -> [String] {
        var prose: [String] = []
        var insideFence = false
        for line in try noticeContents().components(separatedBy: .newlines) {
            if line.hasPrefix(fenceMarker) {
                insideFence.toggle()
                continue
            }
            if !insideFence {
                prose.append(line)
            }
        }
        return
            prose
            .joined(separator: " ")
            .components(separatedBy: ". ")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
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

    /// Only the DeepSeek-V4 files in `osaurus-ai/vmlx-swift-lm` have the
    /// `Osaurus AI` header. The other files in that project have an Apple Inc.
    /// header. A notice file is a legal record, thus each sentence that gives
    /// the name `Osaurus AI` must also name the DeepSeek-V4 files that the
    /// statement is about.
    @Test func thirdPartyNoticesLimitsTheOsaurusAiNameToDeepSeekV4Files() throws {
        for sentence in try Self.noticeSentences() where sentence.contains("Osaurus AI") {
            #expect(
                sentence.contains("DeepSeek-V4"),
                "This sentence does not limit the name `Osaurus AI` to DeepSeek-V4: \(sentence)")
        }
    }
}
