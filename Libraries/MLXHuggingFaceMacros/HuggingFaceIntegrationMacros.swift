import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

@main
struct Macros: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        DownloaderMacro.self,
        TokenizerAdaptorMacro.self,
        TokenizerLoaderMacro.self,
        LoadContainerMacro.self,
        LoadContextMacro.self,
        LanguageModelMacro.self,
    ]
}

extension FreestandingMacroExpansionSyntax {
    /// Returns the expression passed for the given argument label, or `nil`
    /// when the caller did not supply that labeled argument.
    fileprivate func argument(_ label: String) -> ExprSyntax? {
        arguments.first(where: { $0.label?.text == label })?.expression
    }

    /// The source text of the `progressHandler` argument, defaulting to a
    /// no-op closure when the caller did not supply one.
    ///
    /// Shared by ``LoadContainerMacro`` and ``LoadContextMacro`` so the two
    /// expansions cannot drift.
    fileprivate var progressHandlerArgument: String {
        argument("progressHandler")?.description ?? "{ _ in }"
    }
}

/// Implements `#hubDownloader`: wraps a `HuggingFace.HubClient` (defaulting to
/// `HubClient()`) in an `MLXLMCommon.Downloader` bridge so model weights can
/// be fetched from the Hugging Face Hub without MLXLMCommon depending on the
/// HuggingFace package.
public struct DownloaderMacro: ExpressionMacro {
    /// Expands to an immediately-applied closure that defines a `HubBridge`
    /// struct conforming to `MLXLMCommon.Downloader` and returns it wrapping
    /// the caller's hub client (or a default `HubClient()`).
    ///
    /// - Parameters:
    ///   - node: the macro invocation; its optional first argument is the hub client
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the bridged downloader expression
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        let argument = node.arguments.first?.expression.description ?? "HubClient()"

        return
            """
            // make sure you:
            //
            // import HuggingFace
            //
            { (hubApi: HuggingFace.HubClient) -> MLXLMCommon.Downloader in
                struct HubBridge: MLXLMCommon.Downloader {
                    private let upstream: HuggingFace.HubClient

                    init(_ upstream: HuggingFace.HubClient) {
                        self.upstream = upstream
                    }

                    public func download(
                        id: String,
                        revision: String?,
                        matching patterns: [String],
                        useLatest: Bool,
                        progressHandler: @Sendable @escaping (Foundation.Progress) -> Void
                    ) async throws -> URL {                        
                        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
                            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
                        }
                        let revision = revision ?? "main"

                        return try await upstream.downloadSnapshot(
                            of: repoID,
                            revision: revision,
                            matching: patterns,
                            progressHandler: { @MainActor progress in
                                progressHandler(progress)
                            }
                        )
                    }                    
                }

                return HubBridge(hubApi)
            }(\(raw: argument))
            """
    }
}

/// Implements `#adaptHuggingFaceTokenizer`: wraps a swift-transformers
/// `Tokenizers.Tokenizer` in an `MLXLMCommon.Tokenizer` bridge, mapping the
/// naming and error differences between the two protocols.
public struct TokenizerAdaptorMacro: ExpressionMacro {
    /// Expands to an immediately-applied closure that defines a
    /// `TokenizerBridge` struct conforming to `MLXLMCommon.Tokenizer` —
    /// including the optional generation-prompt-controlled chat-template
    /// render — and returns it wrapping the caller's tokenizer.
    ///
    /// - Parameters:
    ///   - node: the macro invocation; its required first argument is the
    ///     upstream tokenizer
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the bridged tokenizer expression
    /// - Throws: `MacroExpansionError` when the tokenizer argument is missing
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let argument = node.arguments.first?.expression else {
            throw MacroExpansionError.message("#adaptHuggingFaceTokenizer requires an argument")
        }

        return
            """
            // make sure you:
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

                    var bosToken: String? { upstream.bosToken }
                    var eosToken: String? { upstream.eosToken }
                    var unknownToken: String? { upstream.unknownToken }

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
            }(\(argument))
            """
    }
}

/// Implements `#huggingFaceTokenizerLoader`: produces an
/// `MLXLMCommon.TokenizerLoader` that loads a tokenizer from a model
/// directory via swift-transformers' `AutoTokenizer` and bridges it with
/// `#adaptHuggingFaceTokenizer`.
public struct TokenizerLoaderMacro: ExpressionMacro {
    /// Expands to an immediately-applied closure that defines a
    /// `TransformersLoader` struct conforming to `MLXLMCommon.TokenizerLoader`
    /// and returns an instance of it.
    ///
    /// - Parameters:
    ///   - node: the macro invocation (takes no arguments)
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the tokenizer-loader expression
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        return
            """
            { () -> MLXLMCommon.TokenizerLoader in
                struct TransformersLoader: MLXLMCommon.TokenizerLoader {
                    public init() {}

                    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
                        // make sure you:
                        //
                        // import Tokenizers
                        //
                        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
                        return #adaptHuggingFaceTokenizer(upstream)
                    }
                }

                return TransformersLoader()
            }()
            """
    }
}

/// Implements `#huggingFaceLoadModelContainer`: loads a `ModelContainer` for
/// the given configuration using the default hub downloader
/// (`#hubDownloader`) and tokenizer loader (`#huggingFaceTokenizerLoader`),
/// with an optional progress handler.
public struct LoadContainerMacro: ExpressionMacro {
    /// Expands to a `loadModelContainer` call wired to the default hub
    /// downloader and tokenizer loader, forwarding the caller's labeled
    /// `configuration` and optional `progressHandler` arguments.
    ///
    /// - Parameters:
    ///   - node: the macro invocation; requires a labeled `configuration`
    ///     argument and accepts an optional `progressHandler`
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the `loadModelContainer` call expression
    /// - Throws: `MacroExpansionError` when the labeled `configuration`
    ///   argument is missing
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.argument("configuration") else {
            throw MacroExpansionError.message(
                "#huggingFaceLoadModelContainer requires a configuration")
        }

        return
            """
            loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: node.progressHandlerArgument))
            """
    }
}

/// Implements `#huggingFaceLoadModel`: loads a `ModelContext` for the given
/// configuration using the default hub downloader (`#hubDownloader`) and
/// tokenizer loader (`#huggingFaceTokenizerLoader`), with an optional
/// progress handler.
public struct LoadContextMacro: ExpressionMacro {
    /// Expands to a `loadModel` call wired to the default hub downloader and
    /// tokenizer loader, forwarding the caller's labeled `configuration` and
    /// optional `progressHandler` arguments.
    ///
    /// - Parameters:
    ///   - node: the macro invocation; requires a labeled `configuration`
    ///     argument and accepts an optional `progressHandler`
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the `loadModel` call expression
    /// - Throws: `MacroExpansionError` when the labeled `configuration`
    ///   argument is missing
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let configuration = node.argument("configuration") else {
            throw MacroExpansionError.message("#huggingFaceLoadModel requires a configuration")
        }

        return
            """
            loadModel(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: \(configuration),
                progressHandler: \(raw: node.progressHandlerArgument))
            """
    }
}

/// Implements `#huggingFaceLanguageModel`: builds an `MLXLanguageModel` for
/// the given configuration whose weights resolve from the Hugging Face hub
/// cache and whose loading is wired to the default hub downloader and
/// tokenizer loader.
public struct LanguageModelMacro: ExpressionMacro {
    /// Expands to an `MLXLanguageModel` initializer call, forwarding the
    /// caller's labeled `configuration` (required) plus `capabilities` and
    /// `configurationResolver` when supplied, and generating the
    /// `weightsLocation` and `load` closures.
    ///
    /// - Parameters:
    ///   - node: the macro invocation; requires a labeled `configuration`
    ///     argument and accepts optional `capabilities` and
    ///     `configurationResolver`
    ///   - context: the expansion context supplied by the compiler
    /// - Returns: the `MLXLanguageModel` initializer expression
    /// - Throws: `MacroExpansionError` when the labeled `configuration`
    ///   argument is missing
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        func argument(_ label: String) -> String? {
            node.argument(label)?.description
        }

        guard let configuration = argument("configuration") else {
            throw MacroExpansionError.message(
                "#huggingFaceLanguageModel requires a configuration")
        }

        // Forward the caller's configuration verbatim, then forward
        // capabilities / configurationResolver only when the caller supplied
        // them so the MLXLanguageModel init's own defaults apply otherwise.
        var arguments = ["configuration: \(configuration)"]
        if let capabilities = argument("capabilities") {
            arguments.append("capabilities: \(capabilities)")
        }
        if let resolver = argument("configurationResolver") {
            arguments.append("configurationResolver: \(resolver)")
        }
        arguments.append(
            """
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
                }
            """)
        arguments.append(
            """
            load: { configuration, progressHandler in
                    try await loadModelContainer(
                        from: #hubDownloader(),
                        using: #huggingFaceTokenizerLoader(),
                        configuration: configuration,
                        progressHandler: progressHandler)
                }
            """)

        let argumentList = arguments.joined(separator: ",\n    ")
        return
            """
            MLXLanguageModel(
                \(raw: argumentList))
            """
    }
}

enum MacroExpansionError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}
