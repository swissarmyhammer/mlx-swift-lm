// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Registers the DFlash2 block drafter with an MTP drafter type registry.
///
/// DFlash2 checkpoints report `model_type: qwen3`; the `DFlash2DraftModel` architecture tells
/// them apart from a Qwen3 language model and from a DSpark drafter. The drafter runs as an MTP
/// drafter, so `MTPDrafterModelFactory` loads it and ``MTPSpeculativeTokenIterator``,
/// `ChatSession` and the `generate(mtpDrafter:)` overloads speculate with it.
public enum DFlash2Registration {

    /// The architecture name of a DFlash2 checkpoint.
    static let architecture = "DFlash2DraftModel"

    /// The registry of the drafters that quantize to ``DFlash2DraftModel/fourBit`` at load.
    public static let fourBitRegistry: ModelTypeRegistry<any MTPDrafterModel> = .init()

    /// Registers the DFlash2 model type.
    ///
    /// - Parameters:
    ///   - registry: The registry to register with.
    ///   - quantization: The quantization a drafter applies to its linear layers at load, or
    ///     `nil` to keep the checkpoint's precision.
    public static func register(
        in registry: ModelTypeRegistry<any MTPDrafterModel> = MTPDrafterTypeRegistry.shared,
        quantization: BaseConfiguration.Quantization? = nil
    ) async {
        await registry.registerModelType(
            "qwen3", matches: isDFlash2Configuration,
            creator: { data in
                let configuration = try JSONDecoder.json5().decode(
                    DFlash2Configuration.self, from: data)
                return DFlash2DraftModel(configuration, quantization: quantization)
            })
    }

    /// Whether `data` is the configuration of a DFlash2 checkpoint.
    private static func isDFlash2Configuration(_ data: Data) -> Bool {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let architectures = root["architectures"] as? [String]
        else { return false }
        return architectures.contains(architecture)
    }
}

extension MTPDrafterRegistration {

    /// Whether a DFlash2 drafter can speculate for `target`: a target that writes its layer
    /// outputs for a block drafter.
    private static func draftsDFlash2(for target: any LanguageModel) -> Bool {
        target is any DFlash2TargetModel
    }

    /// The DFlash2 drafters in the precision of their checkpoints, loaded through
    /// `MTPDrafterModelFactory.shared`.
    public static let dflash2 = MTPDrafterRegistration(
        factory: .shared, registerDrafterTypes: { await DFlash2Registration.register() },
        draftsForTarget: draftsDFlash2(for:))

    /// The DFlash2 drafters quantized to ``DFlash2DraftModel/fourBit`` at load, through a
    /// factory of their own registry, so the bf16 and the 4-bit drafters load side by side.
    public static let dflash2FourBit = MTPDrafterRegistration(
        factory: MTPDrafterModelFactory(
            typeRegistry: DFlash2Registration.fourBitRegistry,
            modelRegistry: MTPDrafterRegistry.shared),
        registerDrafterTypes: {
            await DFlash2Registration.register(
                in: DFlash2Registration.fourBitRegistry, quantization: DFlash2DraftModel.fourBit)
        },
        draftsForTarget: draftsDFlash2(for:))
}
