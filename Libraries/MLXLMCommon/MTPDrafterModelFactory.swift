// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Registry of `model_type` strings (e.g. `"gemma4_assistant"`) to creator
/// closures that instantiate ``MTPDrafterModel`` instances from `config.json`
/// data.
///
/// Empty at bootstrap because drafter implementations live in MLXVLM —
/// importing them here would create a circular dependency. Downstream
/// modules call ``ModelTypeRegistry/registerModelType(_:creator:)`` on the
/// shared registry at app/import time. See `Gemma4AssistantRegistration`
/// in MLXVLM.
///
/// Note: ``ModelTypeRegistry`` is an `actor`; registration is `async`.
public enum MTPDrafterTypeRegistry {
    /// Shared registry. Empty until a downstream module registers a drafter
    /// type via `await MTPDrafterTypeRegistry.shared.registerModelType(...)`.
    public static let shared: ModelTypeRegistry<any MTPDrafterModel> = .init()

    /// Registry for drafters whose forward pass consumes multimodal position
    /// state. It is separate from ``shared`` because standalone Qwen MTP
    /// checkpoints intentionally ship an empty `vision_config`, so their
    /// target architecture cannot be inferred from drafter metadata alone.
    public static let visionLanguage: ModelTypeRegistry<any MTPDrafterModel> = .init()
}

/// Registry of model id (e.g. `"mlx-community/gemma-4-31B-it-assistant-bf16"`)
/// to ``ModelConfiguration``. Drafters don't have prompts, so the
/// configurations omit `defaultPrompt`.
public class MTPDrafterRegistry: AbstractModelRegistry, @unchecked Sendable {
    public static let shared = MTPDrafterRegistry(modelConfigurations: all())

    public static let gemma4_26B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-26B-A4B-it-assistant-bf16"
    )
    public static let gemma4_31B_assistant_bf16 = ModelConfiguration(
        id: "mlx-community/gemma-4-31B-it-assistant-bf16"
    )
    public static let qwen3_8_27b_mtp_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3.8-27B-MTP-4bit"
    )

    private static func all() -> [ModelConfiguration] {
        [gemma4_26B_assistant_bf16, gemma4_31B_assistant_bf16, qwen3_8_27b_mtp_4bit]
    }
}

/// Loader for ``MTPDrafterModel`` checkpoints. Mirrors `LLMModelFactory`
/// in shape, but produces an ``MTPDrafterContext`` (no tokenizer, no user
/// input processor, no message generator — drafters borrow their target's
/// tokenizer).
public final class MTPDrafterModelFactory: GenericModelFactory {
    public typealias ContextType = MTPDrafterContext
    public typealias ContainerType = MTPDrafterContainer

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

    public let typeRegistry: ModelTypeRegistry<any MTPDrafterModel>
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry<any MTPDrafterModel>,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

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
