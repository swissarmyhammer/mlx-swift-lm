// Copyright © 2026 Apple Inc.

import MLXHuggingFaceMacros
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

/// `#huggingFaceLanguageModel` forwards a speculative decoding that carries a memory policy and a
/// drafter byte estimate to `MLXLanguageModel` as it was written, so the policy reaches the model.
final class LanguageModelMacroMemoryPolicyTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLanguageModel": LanguageModelMacro.self
    ]

    func testSpeculativeDecodingWithAMemoryPolicyIsForwarded() {
        assertMacroExpansion(
            """
            let model = #huggingFaceLanguageModel(configuration: config, speculativeDecoding: try MLXSpeculativeDecoding(drafterConfiguration: drafter, from: downloader, using: loader, registration: .qwen35Text, drafterBytes: drafterBytes, memoryPolicy: .recommendedWorkingSet))
            """,
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    speculativeDecoding: try MLXSpeculativeDecoding(drafterConfiguration: drafter, from: downloader, using: loader, registration: .qwen35Text, drafterBytes: drafterBytes, memoryPolicy: .recommendedWorkingSet),
                    weightsLocation: { id in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: id) else {
                            return cache.cacheDirectory
                        }
                        if let commit = cache.resolveRevision(repo: repo, kind: .model, ref: "main"),
                            let snapshot = try? cache.snapshotPath(repo: repo, kind: .model, commitHash: commit) {
                            return snapshot
                        }
                        return cache.repoDirectory(repo: repo, kind: .model)
                    },
                    load: { configuration, progressHandler in
                        try await loadModelContainer(
                            from: #hubDownloader(),
                            using: #huggingFaceTokenizerLoader(),
                            configuration: configuration,
                            progressHandler: progressHandler)
                    })
                """,
            macros: testMacros)
    }
}
