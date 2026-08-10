// Copyright © 2026 Apple Inc.
//
// Guards the two documents that hold the attribution decision: CONTRIBUTING.md
// and THIRD-PARTY-NOTICES.md. The decision says that per-file headers and one
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

    /// The contributing guide that holds the attribution decision.
    private static let contributingFile: URL =
        repositoryRoot.appendingPathComponent("CONTRIBUTING.md", isDirectory: false)

    /// Reads one of the two documents above as text.
    private static func contents(of file: URL) throws -> String {
        try String(contentsOf: file, encoding: .utf8)
    }

    /// The marker that starts and ends a fenced block in Markdown.
    private static let fenceMarker = "```"

    /// The prose of one of the two documents above, without the fenced blocks,
    /// and with each run of whitespace made into one space.
    ///
    /// Each fenced block is a verbatim quotation of a LICENSE or of a file
    /// header. A quotation is not a statement of this repository, and a person
    /// must not change it, so the checks below must not read it.
    ///
    /// Both documents wrap their lines, thus a sentence can go across two
    /// lines. A search for a full sentence must not see the line break.
    private static func proseContents(of file: URL) throws -> String {
        var prose: [String] = []
        var insideFence = false
        for line in try contents(of: file).components(separatedBy: .newlines) {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix(fenceMarker) {
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
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    @Test func thirdPartyNoticesFileExists() {
        #expect(FileManager.default.fileExists(atPath: Self.noticeFile.path))
    }

    @Test func thirdPartyNoticesFileIsNotEmpty() throws {
        let contents = try Self.contents(of: Self.noticeFile)
        #expect(!contents.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Each repository the attribution decision names must have its own section
    /// in the notice. `osaurus-ai/vmlx-swift-lm` is the source of each ported
    /// file, so the notice must name it too.
    @Test func thirdPartyNoticesHasOneSectionForEachRepository() throws {
        let contents = try Self.contents(of: Self.noticeFile)
        #expect(contents.contains("## osaurus-ai/vmlx-swift-lm"))
        #expect(contents.contains("## scouzi1966/mlx-swift-lm"))
        #expect(contents.contains("## scouzi1966/maclocal-api"))
    }

    /// The MIT license makes the copyright notice necessary, so the notice must
    /// give the MIT text and the copyright line of each repository.
    @Test func thirdPartyNoticesHasTheCopyrightLineOfEachRepository() throws {
        let contents = try Self.contents(of: Self.noticeFile)
        #expect(contents.contains("MIT License"))
        #expect(contents.contains("Copyright (c) 2024 ml-explore"))
        #expect(contents.contains("Copyright (c) 2026 Osaurus contributors"))
        #expect(contents.contains("Copyright (c) 2025 MacLocalAPI Contributors"))
    }

    /// `Osaurus AI` and `Osaurus contributors` are two different names from two
    /// different files. `Osaurus AI` comes from the header of a source file, and
    /// `Osaurus contributors` comes from a LICENSE. A reader can see one of the
    /// two names as an error and change it into the other. Both documents must
    /// keep the two names, and both must keep the instruction that stops that
    /// change.
    ///
    /// This test reads both documents, because both documents give the two
    /// names. It reads the prose only: the two names are also in the verbatim
    /// blocks, thus a check of the full text would pass after a change to the
    /// prose.
    ///
    /// No test here makes a rule true about the headers of the files in
    /// `osaurus-ai/vmlx-swift-lm`, because the two documents give no such rule.
    /// A person who ports a file reads the header and the commit SHA in that
    /// file.
    @Test func bothDocumentsKeepTheTwoOsaurusNamesSeparate() throws {
        for file in [Self.noticeFile, Self.contributingFile] {
            let contents = try Self.proseContents(of: file)
            let name = file.lastPathComponent
            #expect(
                contents.contains("Osaurus AI"),
                "\(name) does not have the name `Osaurus AI`")
            #expect(
                contents.contains("Osaurus contributors"),
                "\(name) does not have the name `Osaurus contributors`")
            #expect(
                contents.contains("Do not change one name into the other."),
                "\(name) does not have the instruction that keeps the two names separate")
        }
    }
}
