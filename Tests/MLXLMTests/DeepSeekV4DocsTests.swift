// Copyright © 2026 Apple Inc.
//
// Guards the DeepSeek-V4 scope document, docs/deepseek-v4-support.md. The
// document records what the port supports and the seven deferred items. If a
// later edit removes the document, a deferred item, or the provenance record,
// these tests fail.

import Foundation
import Testing

@Suite
struct DeepSeekV4DocsTests {

    /// The root of the repository, found from the path of this source file.
    ///
    /// This file is at `Tests/MLXLMTests/DeepSeekV4DocsTests.swift`, so the
    /// root is three path components above it.
    private static let repositoryRoot: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Tests/MLXLMTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repository root
    }()

    /// The DeepSeek-V4 scope document these tests guard.
    private static let supportDocument: URL =
        repositoryRoot
        .appendingPathComponent("docs", isDirectory: true)
        .appendingPathComponent("deepseek-v4-support.md", isDirectory: false)

    /// Reads the scope document as text.
    private static func contents() throws -> String {
        try String(contentsOf: supportDocument, encoding: .utf8)
    }

    @Test func supportDocumentExists() {
        #expect(FileManager.default.fileExists(atPath: Self.supportDocument.path))
    }

    /// Each deferred item must keep its name in the document. A name that is
    /// gone means the record of that gap is gone with it.
    @Test func supportDocumentNamesEachDeferredItem() throws {
        let contents = try Self.contents()
        for marker in [
            "DSpark", "ActivationQuant", "mxtq", "DeepSeek-V4-Pro", "mxfp4",
            "deepseek_v32", "maclocal-api",
        ] {
            #expect(
                contents.contains(marker),
                "the document does not have the marker `\(marker)`")
        }
    }

    /// The document must record the MIT license and the three reference
    /// repositories of the provenance chain.
    @Test func supportDocumentNamesTheLicenseAndEachReferenceRepository() throws {
        let contents = try Self.contents()
        for marker in [
            "MIT", "osaurus-ai/vmlx-swift-lm", "scouzi1966/mlx-swift-lm",
            "scouzi1966/maclocal-api",
        ] {
            #expect(
                contents.contains(marker),
                "the document does not have the marker `\(marker)`")
        }
    }
}
