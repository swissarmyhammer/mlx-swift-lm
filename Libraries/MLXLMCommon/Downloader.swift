import Foundation

/// A protocol for downloading model repository snapshots to local directories.
///
/// Conforming types encapsulate the full download lifecycle — cache check, network
/// download, and fallback to cache on failure. Each conformance owns its own caching
/// strategy. The return value is always a local directory URL containing the requested files.
///
/// The protocol is provider-agnostic. `id` is a plain `String` that each conformance
/// interprets however it wants (e.g. `"org/model"` for Hugging Face, a four-part
/// Kaggle handle, an S3 path). `revision` is optional for providers without versioning.
///
/// ## See Also
/// - ``ResolvedModelConfiguration``
/// - ``TokenizerSource``
public protocol Downloader: Sendable {
    /// Download (or retrieve from cache) a snapshot of a repository.
    ///
    /// - Parameters:
    ///   - id: Provider-specific repository identifier
    ///   - revision: Optional revision (branch, tag, commit hash, version number).
    ///     Providers without versioning receive `nil`.
    ///   - patterns: Glob patterns to filter which files to download
    ///     (e.g. `["*.safetensors", "*.json", "*.jinja"]`)
    ///   - useLatest: When `true`, check the provider for updates even if a cached
    ///     version exists. When `false`, return the cached version if available.
    ///   - progressHandler: Callback for download progress
    /// - Returns: Local directory URL containing the downloaded files
    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL
}

/// Identifies where a tokenizer should be loaded from.
///
/// Used by ``ModelConfiguration`` to specify an alternate tokenizer source
/// when the tokenizer files are not co-located with the model weights.
///
/// - ``id(_:revision:)`` downloads tokenizer files from a remote provider (requires a ``Downloader``)
/// - ``directory(_:)`` loads tokenizer files from a local path
///
/// When `nil` on a ``ModelConfiguration``, the tokenizer is loaded from the
/// same directory as the model.
public enum TokenizerSource: Sendable, Equatable {
    /// A provider-specific repository identifier for downloading tokenizer files.
    /// - Parameters:
    ///   - id: The repository identifier (e.g. `"org/tokenizer-name"`).
    ///   - revision: Optional revision (branch, tag, commit hash). When `nil`,
    ///     the ``Downloader`` decides the default (typically `"main"`).
    case id(String, revision: String? = nil)
    /// A local directory containing tokenizer files.
    case directory(URL)
}

/// A fully resolved model configuration where all sources have been resolved
/// to local directory paths.
///
/// Created by resolving a ``ModelConfiguration`` — downloading remote sources
/// via a ``Downloader`` and mapping behavioral properties. Factory implementations
/// receive this type in their `_load` method, so they work purely with local files.
///
/// ## See Also
/// - ``ModelConfiguration/resolved(modelDirectory:tokenizerDirectory:)``
/// - ``Downloader``
public struct ResolvedModelConfiguration: Sendable {
    /// The local directory holding the model's configuration and weights.
    public var modelDirectory: URL

    /// The local directory holding the tokenizer files, which is ``modelDirectory`` unless
    /// ``ModelConfiguration/tokenizerSource`` names another source.
    public var tokenizerDirectory: URL

    /// The name of the model, which errors and logs report.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/name``
    public var name: String

    /// The Hub repository id of the model, which the chat-convention resolvers key on.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/repositoryID``
    public var repositoryID: String

    /// The prompt that examples and tools use when the caller gives none.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/defaultPrompt``
    public var defaultPrompt: String

    /// Token strings that end generation in addition to the tokenizer's EOS token.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/extraEOSTokens``
    public var extraEOSTokens: Set<String>

    /// Strings that end generation when they appear in the decoded text.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/effectiveStopStrings``
    public var stopStrings: Set<String>

    /// Token ids that end generation, loaded from `config.json` and `generation_config.json`.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/eosTokenIds``
    public var eosTokenIds: Set<Int>

    /// The tool-call format the model emits, or `nil` to infer it from the model type.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/toolCallFormat``
    public var toolCallFormat: ToolCallFormat?

    /// How the model marks its reasoning, or `nil` for a model without reasoning markers.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/reasoningConfig``
    public var reasoningConfig: ReasoningConfig?

    /// The message generator that turns chat messages into the model's input, or `nil` for the
    /// model's own.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/messageGenerator``
    public var messageGenerator: (any MessageGenerator)?

    /// The sampling defaults of the checkpoint (`generation_config.json`), if it has them.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/generationConfig``
    public var generationConfig: GenerationConfigFile?

    /// How to choose the safetensors files holding the model's weights.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/weightFileSelection``
    public var weightFileSelection: WeightFileSelection

    /// When the weights of the load evaluate. A sharded load defers them, to evaluate the
    /// model layer by layer as it splits.
    public var weightEvaluation: WeightEvaluation = .atLoad

    /// Components to load the model without.
    ///
    /// ## See Also
    /// - ``ModelConfiguration/excludedComponents``
    public var excludedComponents: Set<ModelComponent>

    /// Creates a resolved configuration from local directories.
    ///
    /// - Parameters:
    ///   - modelDirectory: the directory holding the model's configuration and weights.
    ///   - tokenizerDirectory: the directory holding the tokenizer files.
    ///   - name: the name of the model.
    ///   - repositoryID: the Hub repository id, `nil` to read it from a Hub cache path of
    ///     `modelDirectory`, or else to use `name`.
    ///   - defaultPrompt: the prompt used when the caller gives none.
    ///   - extraEOSTokens: token strings that end generation.
    ///   - stopStrings: strings that end generation, `nil` for `extraEOSTokens`.
    ///   - eosTokenIds: token ids that end generation.
    ///   - toolCallFormat: the tool-call format, `nil` to infer it from the model type.
    ///   - reasoningConfig: the reasoning markers, `nil` for none.
    ///   - messageGenerator: the message generator, `nil` for the model's own.
    ///   - generationConfig: the sampling defaults of the checkpoint.
    ///   - weightFileSelection: how to choose the weight files.
    ///   - excludedComponents: components to load the model without.
    public init(
        modelDirectory: URL,
        tokenizerDirectory: URL,
        name: String,
        repositoryID: String? = nil,
        defaultPrompt: String,
        extraEOSTokens: Set<String>,
        stopStrings: Set<String>? = nil,
        eosTokenIds: Set<Int>,
        toolCallFormat: ToolCallFormat?,
        reasoningConfig: ReasoningConfig? = nil,
        messageGenerator: (any MessageGenerator)? = nil,
        generationConfig: GenerationConfigFile? = nil,
        weightFileSelection: WeightFileSelection = .automatic,
        excludedComponents: Set<ModelComponent> = []
    ) {
        self.modelDirectory = modelDirectory
        self.tokenizerDirectory = tokenizerDirectory
        self.name = name
        self.repositoryID =
            repositoryID ?? ModelConfiguration.hubCacheRepositoryID(of: modelDirectory) ?? name
        self.defaultPrompt = defaultPrompt
        self.extraEOSTokens = extraEOSTokens
        self.stopStrings = stopStrings ?? extraEOSTokens
        self.eosTokenIds = eosTokenIds
        self.toolCallFormat = toolCallFormat
        self.reasoningConfig = reasoningConfig
        self.messageGenerator = messageGenerator
        self.generationConfig = generationConfig
        self.weightFileSelection = weightFileSelection
        self.excludedComponents = excludedComponents
    }
}

extension ResolvedModelConfiguration {
    /// Convenience for loading everything from a single local directory.
    public init(directory: URL) {
        self.init(
            modelDirectory: directory,
            tokenizerDirectory: directory,
            name: directory.deletingLastPathComponent().lastPathComponent + "/"
                + directory.lastPathComponent,
            defaultPrompt: "",
            extraEOSTokens: [],
            stopStrings: [],
            eosTokenIds: [],
            toolCallFormat: nil,
            reasoningConfig: nil,
            messageGenerator: nil,
            generationConfig: nil)
    }
}
