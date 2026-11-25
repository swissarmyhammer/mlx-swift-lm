// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Registry of `model_type` strings (e.g. `"gemma4_assistant"`) to creator
/// closures that instantiate ``MTPDrafterModel`` instances from `config.json`
/// data.
///
/// Empty at bootstrap because drafter implementations live in MLXVLM —
/// importing them here would create a circular dependency. Downstream
/// modules call `registerDrafter(modelTypes:configuration:create:)` (or the
/// variant with a `matches` predicate, for a `model_type` that several
/// architectures share) on the shared registry at app/import time. See
/// `Gemma4AssistantRegistration` in MLXVLM.
///
/// Note: ``ModelTypeRegistry`` is an `actor`; registration is `async`.
public enum MTPDrafterTypeRegistry {
    /// Shared registry. Empty until a downstream module registers a drafter
    /// type via `await MTPDrafterTypeRegistry.shared.registerDrafter(...)`.
    public static let shared: ModelTypeRegistry<any MTPDrafterModel> = .init()

    /// Registry for drafters whose forward pass consumes multimodal position
    /// state. It is separate from ``shared`` because standalone Qwen MTP
    /// checkpoints intentionally ship an empty `vision_config`, so their
    /// target architecture cannot be inferred from drafter metadata alone.
    public static let visionLanguage: ModelTypeRegistry<any MTPDrafterModel> = .init()
}

extension ModelTypeRegistry where T == any MTPDrafterModel {
    /// Registers a drafter for each of `modelTypes`, read from a `config.json` that decodes as
    /// `Configuration`.
    ///
    /// A `config.json` matches when it decodes and `matches` holds for its configuration. The
    /// creator is `drafterCreator(_:create:)`, so that every drafter registered this way reads
    /// its configuration the same way.
    ///
    /// - Parameters:
    ///   - modelTypes: The `model_type` values of the checkpoints that ship the drafter.
    ///   - configuration: The configuration the drafter is built from.
    ///   - matches: Whether a decoded configuration is one of this drafter.
    ///   - create: The drafter of a validated configuration.
    public func registerDrafter<Configuration: Decodable>(
        modelTypes: [String], configuration: Configuration.Type,
        matches: @escaping (Configuration) -> Bool,
        create: @escaping (Configuration) throws -> any MTPDrafterModel
    ) {
        let creator = Self.drafterCreator(configuration, create: create)
        for modelType in modelTypes {
            registerModelType(
                modelType,
                matches: { data in
                    (try? JSONDecoder.json5().decode(Configuration.self, from: data))
                        .map(matches) ?? false
                },
                creator: creator)
        }
    }

    /// Registers the only drafter of each of `modelTypes`, read from a `config.json` that decodes
    /// as `Configuration`, in place of any creator registered for those model types before.
    ///
    /// The creator is `drafterCreator(_:create:)`. Registering again replaces the creator, so a
    /// registration that runs more than once leaves one creator.
    ///
    /// - Parameters:
    ///   - modelTypes: The `model_type` values of the checkpoints that ship the drafter.
    ///   - configuration: The configuration the drafter is built from.
    ///   - create: The drafter of a validated configuration.
    public func registerDrafter<Configuration: Decodable>(
        modelTypes: [String], configuration: Configuration.Type,
        create: @escaping (Configuration) throws -> any MTPDrafterModel
    ) {
        let creator = Self.drafterCreator(configuration, create: create)
        for modelType in modelTypes {
            registerModelType(modelType, creator: creator)
        }
    }

    /// The creator both `registerDrafter` variants register: it decodes the `config.json` as
    /// `Configuration`, throwing the decoding error of one that does not decode, validates the
    /// configuration when `Configuration` is ``ModelConfigurationValidating``, and builds the
    /// drafter from it.
    ///
    /// - Parameters:
    ///   - configuration: The configuration the drafter is built from.
    ///   - create: The drafter of a validated configuration.
    /// - Returns: The creator of the drafter of a `config.json`.
    static func drafterCreator<Configuration: Decodable>(
        _ configuration: Configuration.Type,
        create: @escaping (Configuration) throws -> any MTPDrafterModel
    ) -> (Data) throws -> any MTPDrafterModel {
        { data in
            let decoded = try JSONDecoder.json5().decode(Configuration.self, from: data)
            try (decoded as? any ModelConfigurationValidating)?.validateModelConfiguration()
            return try create(decoded)
        }
    }
}

/// Registry of model id (e.g. `"mlx-community/gemma-4-31B-it-assistant-bf16"`)
/// to ``ModelConfiguration``. Drafters don't have prompts, so the
/// configurations omit `defaultPrompt`.
///
/// The class adds no stored state. It restates the `@unchecked Sendable` conformance of
/// ``AbstractModelRegistry``, whose lock guards every read and write of the registered
/// configurations.
public final class MTPDrafterRegistry: AbstractModelRegistry, @unchecked Sendable {  // swiftlint:disable:this no_unchecked_sendable The class adds no stored state, and the lock of AbstractModelRegistry guards every read and write of the registered configurations.
    /// The registry of ``MTPDrafterModelFactory/shared`` and
    /// ``MTPDrafterModelFactory/visionLanguage``, holding every checkpoint constant below.
    public static let shared = MTPDrafterRegistry(modelConfigurations: all())

    /// The bf16 assistant drafter of Gemma 4 26B-A4B-it.
    public static let gemma4_26B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    )
    /// The bf16 assistant drafter of Gemma 4 31B-it.
    public static let gemma4_31B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-31B-it-assistant-bf16"
    )
    /// The 4-bit MTP head of `mlx-community/Qwen3.8-27B-4bit`.
    public static let qwen3_8_27b_mtp_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.8-27B-MTP-4bit"
    )
    /// The 8-bit MTP head of `mlx-community/Qwen3.8-27B-8bit`.
    public static let qwen3_8_27b_mtp_8bit = ModelConfiguration(
        id: "mlx-community/Qwen3.8-27B-MTP-8bit"
    )
    /// The DSpark drafter of LFM2.5 1.2B Instruct.
    public static let lfm2_1_2B_instruct_dspark = ModelConfiguration(
        id: "LiquidAI/LFM2.5-1.2B-Instruct-DSpark"
    )
    /// The DSpark drafter of LFM2.5 2.6B.
    public static let lfm2_2_6B_dspark = ModelConfiguration(
        id: "LiquidAI/LFM2.5-2.6B-DSpark"
    )
    /// The DSpark drafter of LFM2.5 8B-A1B.
    public static let lfm2_8B_A1B_dspark = ModelConfiguration(
        id: "LiquidAI/LFM2.5-8B-A1B-DSpark"
    )
    /// The DSpark head (`mtp.safetensors`) of Inkling 2 Mini, which this checkpoint ships beside
    /// its target weights. The drafter loads the head alone, thus a download fetches the head and
    /// `config.json`, the one file of configuration the drafter reads, and leaves the target
    /// weights and the tokenizer files out.
    public static let inkling2_mini_oq4e_mtp: ModelConfiguration = {
        var configuration = ModelConfiguration(id: "mlx-community/Inkling-2-Mini-oQ4e-mtp")
        configuration.downloadPatterns = ["mtp.safetensors", "config.json"]
        return configuration
    }()

    private static func all() -> [ModelConfiguration] {
        [
            gemma4_26B_assistant_bf16,
            gemma4_31B_assistant_bf16,
            qwen3_8_27b_mtp_4bit,
            qwen3_8_27b_mtp_8bit,
            lfm2_1_2B_instruct_dspark,
            lfm2_2_6B_dspark,
            lfm2_8B_A1B_dspark,
            inkling2_mini_oq4e_mtp,
        ]
    }
}

/// Loader for ``MTPDrafterModel`` checkpoints. Mirrors `LLMModelFactory`
/// in shape, but produces an ``MTPDrafterContext`` (no tokenizer, no user
/// input processor, no message generator — drafters borrow their target's
/// tokenizer).
public final class MTPDrafterModelFactory: GenericModelFactory {
    /// The context that `_load(configuration:tokenizerLoader:)` gives back: the drafter
    /// model and its configuration, with no tokenizer.
    public typealias ContextType = MTPDrafterContext

    /// The `Sendable` container that `_wrap(_:)` puts a drafter context in.
    public typealias ContainerType = MTPDrafterContainer

    /// Loader for text drafters, with the drafter model types of ``MTPDrafterTypeRegistry/shared``
    /// and the checkpoints of ``MTPDrafterRegistry/shared``.
    public static let shared = MTPDrafterModelFactory(
        typeRegistry: MTPDrafterTypeRegistry.shared,
        modelRegistry: MTPDrafterRegistry.shared
    )

    /// Loader for multimodal drafters. Use this factory when the verifier was
    /// loaded by `VLMModelFactory` and can emit multimodal RoPE deltas.
    public static let visionLanguage = MTPDrafterModelFactory(
        typeRegistry: MTPDrafterTypeRegistry.visionLanguage,
        modelRegistry: MTPDrafterRegistry.shared
    )

    /// registry of drafter model type, e.g. configuration value `qwen3_5_mtp` -> configuration
    /// and init methods
    public let typeRegistry: ModelTypeRegistry<any MTPDrafterModel>

    /// registry of model id to configuration, e.g. `mlx-community/gemma-4-31B-it-assistant-bf16`
    public let modelRegistry: AbstractModelRegistry

    /// Creates a factory that makes drafter models from the types of `typeRegistry` and looks
    /// up model ids in `modelRegistry`.
    ///
    /// - Parameters:
    ///   - typeRegistry: The drafter model types, keyed by the `model_type` of `config.json`.
    ///   - modelRegistry: The drafter checkpoints, keyed by model id.
    public init(
        typeRegistry: ModelTypeRegistry<any MTPDrafterModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Loads the drafter model of a resolved configuration into an ``MTPDrafterContext``: reads
    /// `config.json`, creates the model of its `model_type`, and loads the weights. The context
    /// holds no tokenizer, because a drafter borrows the tokenizer of its target.
    ///
    /// - Parameters:
    ///   - configuration: The resolved configuration, whose directory holds the checkpoint.
    ///   - tokenizerLoader: Unused, because a drafter loads no tokenizer.
    /// - Returns: The context that holds the loaded drafter model.
    /// - Throws: ``ModelFactoryError`` when `config.json` cannot be read or decoded, or an error
    ///   of the weight load.
    public func _load(
        configuration: ResolvedModelConfiguration,
        tokenizerLoader: any TokenizerLoader
    ) async throws -> MTPDrafterContext {
        let modelDirectory = configuration.modelDirectory
        let configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            configData = try Data(contentsOf: configurationURL)
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, configuration.name, error)
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: any MTPDrafterModel
        do {
            model = try await typeRegistry.createModel(
                configuration: configData, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization,
            weightFileSelection: configuration.weightFileSelection,
            excludedComponents: configuration.excludedComponents
        )

        let modelConfig = ModelConfiguration(
            directory: modelDirectory,
            tokenizerSource: nil,
            defaultPrompt: ""
        )
        return MTPDrafterContext(configuration: modelConfig, model: model)
    }

    /// Wraps a drafter context in an ``MTPDrafterContainer``.
    ///
    /// - Parameter context: The context that ``_load(configuration:tokenizerLoader:)`` gave back.
    /// - Returns: The `Sendable` container that holds the context.
    public func _wrap(_ context: MTPDrafterContext) -> MTPDrafterContainer {
        .init(context: context)
    }
}

/// The factory that loads a family of MTP drafters, with the step that registers the family's
/// drafter model types in the registry of that factory.
///
/// A module that implements drafters publishes its families as static members, such as
/// `qwen35Text` of MLXLLM and `qwen35VLM` of MLXVLM, and a deferred
/// ``SpeculativeDecodingConfig`` takes one to load its drafter.
///
/// One checkpoint can serve several families, as a Qwen3.5-family MTP checkpoint serves the text
/// target and the vision-language target. ``matchingTarget(_:_:)`` combines such families, and a
/// configuration that takes the combination loads the family that drafts for the target it
/// speculates for, however that target loaded.
public struct MTPDrafterRegistration: Sendable {

    /// The factory that loads the drafter.
    public let factory: MTPDrafterModelFactory

    /// Registers the drafter model types with the registry of ``factory`` before a load.
    public let registerDrafterTypes: @Sendable () async -> Void

    /// Whether the drafters of this family draft for a target.
    private let draftsForTarget: @Sendable (any LanguageModel) -> Bool

    /// The families a combination chooses from, in order, or empty for one family.
    private let candidates: [MTPDrafterRegistration]

    /// Creates the registration of a drafter family.
    ///
    /// - Parameters:
    ///   - factory: the factory that loads the drafter
    ///   - registerDrafterTypes: registers the drafter model types with the registry of
    ///     `factory` before a load; the default registers nothing, for types already registered
    ///   - draftsForTarget: whether the drafters of the family draft for a target, as the target
    ///     loaded; the default takes every target
    public init(
        factory: MTPDrafterModelFactory = .shared,
        registerDrafterTypes: @escaping @Sendable () async -> Void = {},
        draftsForTarget: @escaping @Sendable (any LanguageModel) -> Bool = { _ in true }
    ) {
        self.init(
            factory: factory, registerDrafterTypes: registerDrafterTypes,
            draftsForTarget: draftsForTarget, candidates: [])
    }

    /// The combination of drafter families that loads, for each target, the first family that
    /// drafts for it, and the first family for a target none drafts for.
    ///
    /// A load that names no target loads the first family.
    ///
    /// - Parameters:
    ///   - first: the first family.
    ///   - others: the other families, in order.
    /// - Returns: the combination.
    public static func matchingTarget(
        _ first: MTPDrafterRegistration, _ others: MTPDrafterRegistration...
    ) -> MTPDrafterRegistration {
        let candidates = ([first] + others).flatMap { $0.families }
        return MTPDrafterRegistration(
            factory: first.factory, registerDrafterTypes: first.registerDrafterTypes,
            draftsForTarget: { target in candidates.contains { $0.drafts(for: target) } },
            candidates: candidates)
    }

    /// Creates a registration that chooses among `candidates`, or a single family when
    /// `candidates` is empty, and loads through `factory` when no target picks a family.
    private init(
        factory: MTPDrafterModelFactory,
        registerDrafterTypes: @escaping @Sendable () async -> Void,
        draftsForTarget: @escaping @Sendable (any LanguageModel) -> Bool,
        candidates: [MTPDrafterRegistration]
    ) {
        self.factory = factory
        self.registerDrafterTypes = registerDrafterTypes
        self.draftsForTarget = draftsForTarget
        self.candidates = candidates
    }

    /// The single families of this registration: the families of a combination, or itself.
    private var families: [MTPDrafterRegistration] {
        candidates.isEmpty ? [self] : candidates
    }

    /// Whether the drafters of this registration draft for `target`, as it loaded.
    ///
    /// - Parameter target: the model a drafter would speculate for.
    /// - Returns: `true` when a family of the registration drafts for `target`.
    public func drafts(for target: any LanguageModel) -> Bool {
        draftsForTarget(target)
    }

    /// The family of this registration that loads the drafter of `target`: the first family that
    /// drafts for it, else the first family.
    ///
    /// - Parameter target: the model the drafter speculates for.
    /// - Returns: the family.
    public func registration(for target: any LanguageModel) -> MTPDrafterRegistration {
        families.first { $0.drafts(for: target) } ?? candidates.first ?? self
    }

    /// The loader of the drafter that drafts for `target`: it registers the drafter model types
    /// of the family that ``registration(for:)`` picks for `target`, then loads a drafter
    /// configuration with the factory of that family.
    ///
    /// - Parameters:
    ///   - target: the model the drafter speculates for, as it loaded
    ///   - downloader: the downloader that fetches a remote checkpoint
    ///   - tokenizerLoader: the tokenizer loader the factory takes; an MTP drafter borrows the
    ///     tokenizer of its target, so the load reads no tokenizer
    /// - Returns: the loader, which takes the drafter checkpoint
    public func drafterLoader(
        for target: any LanguageModel, from downloader: any Downloader,
        using tokenizerLoader: any TokenizerLoader
    ) -> @Sendable (ModelConfiguration) async throws -> MTPDrafterContainer {
        let family = registration(for: target)
        return { configuration in
            try await family.loadContainer(
                from: downloader, using: tokenizerLoader, configuration: configuration)
        }
    }

    /// Registers the drafter model types, then loads the drafter of `configuration` with
    /// ``factory``.
    ///
    /// - Parameters:
    ///   - downloader: the downloader that fetches a remote checkpoint
    ///   - tokenizerLoader: the tokenizer loader the factory takes; an MTP drafter borrows the
    ///     tokenizer of its target, so the load reads no tokenizer
    ///   - configuration: the drafter checkpoint, as a model id or a local directory
    /// - Returns: the loaded drafter
    /// - Throws: the error of the load
    public func loadContainer(
        from downloader: any Downloader, using tokenizerLoader: any TokenizerLoader,
        configuration: ModelConfiguration
    ) async throws -> MTPDrafterContainer {
        await registerDrafterTypes()
        return try await factory.loadContainer(
            from: downloader, using: tokenizerLoader, configuration: configuration)
    }
}
