// Copyright © 2026 Apple Inc.

import MLXHuggingFaceMacros
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class LanguageModelMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLanguageModel": LanguageModelMacro.self
    ]

    func testConfigurationOnly() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config)",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    weightsLocation: { ID in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: ID) else {
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

    func testExplicitCapabilities() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, capabilities: [.guidedGeneration, .toolCalling])",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    capabilities: [.guidedGeneration, .toolCalling],
                    weightsLocation: { ID in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: ID) else {
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

    func testExplicitConfigurationResolver() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, configurationResolver: MyResolver())",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    configurationResolver: MyResolver(),
                    weightsLocation: { ID in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: ID) else {
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

    func testExplicitCapabilitiesAndConfigurationResolver() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel(configuration: config, capabilities: [.guidedGeneration, .toolCalling], configurationResolver: MyResolver())",
            expandedSource: """
                let model = MLXLanguageModel(
                    configuration: config,
                    capabilities: [.guidedGeneration, .toolCalling],
                    configurationResolver: MyResolver(),
                    weightsLocation: { ID in
                        let cache = HuggingFace.HubCache.default
                        guard let repo = HuggingFace.Repo.ID(rawValue: ID) else {
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

    func testMissingConfigurationDiagnoses() {
        assertMacroExpansion(
            "let model = #huggingFaceLanguageModel()",
            expandedSource: "let model = #huggingFaceLanguageModel()",
            diagnostics: [
                DiagnosticSpec(
                    message: "#huggingFaceLanguageModel requires a configuration",
                    line: 1,
                    column: 13)
            ],
            macros: testMacros)
    }
}

final class TokenizerAdaptorMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "adaptHuggingFaceTokenizer": TokenizerAdaptorMacro.self
    ]

    /// The generated bridge must implement the optional
    /// `applyChatTemplate(messages:tools:additionalContext:addGenerationPrompt:)`
    /// capability by delegating to swift-transformers' full protocol requirement
    /// (verified against the pinned 1.3.3 checkout), forwarding
    /// `addGenerationPrompt` so callers can render past turns without the
    /// template's generation-priming region.
    func testBridgeExpansion() {
        assertMacroExpansion(
            "let tokenizer = #adaptHuggingFaceTokenizer(t)",
            expandedSource: """
                let tokenizer = // make sure you:
                //
                // import Tokenizers
                //
                { (huggingFaceTokenizer: Tokenizers.Tokenizer) -> MLXLMCommon.Tokenizer in
                    struct TokenizerBridge: MLXLMCommon.Tokenizer {
                        private let upstream: any Tokenizers.Tokenizer

                        init(_ upstream: any Tokenizers.Tokenizer) {
                            self.upstream = upstream
                        }

                        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
                            upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
                        }

                        // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
                        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
                            upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
                        }

                        func convertTokenToId(_ token: String) -> Int? {
                            upstream.convertTokenToId(token)
                        }

                        func convertIdToToken(_ id: Int) -> String? {
                            upstream.convertIdToToken(id)
                        }

                        var bosToken: String? {
                            upstream.bosToken
                        }
                        var eosToken: String? {
                            upstream.eosToken
                        }
                        var unknownToken: String? {
                            upstream.unknownToken
                        }

                        func applyChatTemplate(
                            messages: [[String: any Sendable]],
                            tools: [[String: any Sendable]]?,
                            additionalContext: [String: any Sendable]?
                        ) throws -> [Int] {
                            do {
                                return try upstream.applyChatTemplate(
                                    messages: messages, tools: tools, additionalContext: additionalContext)
                            } catch Tokenizers.TokenizerError.missingChatTemplate {
                                throw MLXLMCommon.TokenizerError.missingChatTemplate
                            }
                        }

                        // Renders with explicit control over the generation prompt by
                        // delegating to swift-transformers' full applyChatTemplate
                        // protocol requirement.
                        func applyChatTemplate(
                            messages: [[String: any Sendable]],
                            tools: [[String: any Sendable]]?,
                            additionalContext: [String: any Sendable]?,
                            addGenerationPrompt: Bool
                        ) throws -> [Int]? {
                            do {
                                return try upstream.applyChatTemplate(
                                    messages: messages,
                                    chatTemplate: nil,
                                    addGenerationPrompt: addGenerationPrompt,
                                    truncation: false,
                                    maxLength: nil,
                                    tools: tools,
                                    additionalContext: additionalContext)
                            } catch Tokenizers.TokenizerError.missingChatTemplate {
                                throw MLXLMCommon.TokenizerError.missingChatTemplate
                            }
                        }
                    }

                    return TokenizerBridge(huggingFaceTokenizer)
                }(t)
                """,
            macros: testMacros)
    }
}

final class LoadContainerMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLoadModelContainer": LoadContainerMacro.self
    ]

    func testLabeledConfigurationExpands() {
        assertMacroExpansion(
            "let container = #huggingFaceLoadModelContainer(configuration: config)",
            expandedSource: """
                let container = loadModelContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progressHandler: { _ in
                    })
                """,
            macros: testMacros)
    }

    func testProgressHandlerForwarded() {
        assertMacroExpansion(
            "let container = #huggingFaceLoadModelContainer(configuration: config, progressHandler: handler)",
            expandedSource: """
                let container = loadModelContainer(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progressHandler: handler)
                """,
            macros: testMacros)
    }

    /// A positional (unlabeled) first argument must be rejected — the macro
    /// requires the labeled `configuration` argument, matching
    /// `#huggingFaceLanguageModel`.
    func testPositionalConfigurationDiagnoses() {
        assertMacroExpansion(
            "let container = #huggingFaceLoadModelContainer(config)",
            expandedSource: "let container = #huggingFaceLoadModelContainer(config)",
            diagnostics: [
                DiagnosticSpec(
                    message: "#huggingFaceLoadModelContainer requires a configuration",
                    line: 1,
                    column: 17)
            ],
            macros: testMacros)
    }
}

final class LoadContextMacroTests: XCTestCase {
    let testMacros: [String: Macro.Type] = [
        "huggingFaceLoadModel": LoadContextMacro.self
    ]

    func testLabeledConfigurationExpands() {
        assertMacroExpansion(
            "let context = #huggingFaceLoadModel(configuration: config)",
            expandedSource: """
                let context = loadModel(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progressHandler: { _ in
                    })
                """,
            macros: testMacros)
    }

    func testProgressHandlerForwarded() {
        assertMacroExpansion(
            "let context = #huggingFaceLoadModel(configuration: config, progressHandler: handler)",
            expandedSource: """
                let context = loadModel(
                    from: #hubDownloader(),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: config,
                    progressHandler: handler)
                """,
            macros: testMacros)
    }

    /// A positional (unlabeled) first argument must be rejected — the macro
    /// requires the labeled `configuration` argument, matching
    /// `#huggingFaceLanguageModel`.
    func testPositionalConfigurationDiagnoses() {
        assertMacroExpansion(
            "let context = #huggingFaceLoadModel(config)",
            expandedSource: "let context = #huggingFaceLoadModel(config)",
            diagnostics: [
                DiagnosticSpec(
                    message: "#huggingFaceLoadModel requires a configuration",
                    line: 1,
                    column: 15)
            ],
            macros: testMacros)
    }
}
