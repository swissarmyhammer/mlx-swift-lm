// Copyright © 2026 Apple Inc.

import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

@Suite("Speculative decoding configuration")
struct SpeculativeDecodingConfigTests {
    @Test("MTP configuration uses the default verification block and confidence floor")
    func mtpDefaults() throws {
        let drafter = makeMTPDrafterContainer()

        let configuration = try SpeculativeDecodingConfig(mtpDrafter: drafter)

        #expect(configuration.draftModel == nil)
        #expect(configuration.numDraftTokens == SpeculativeDecodingConfig.defaultMTPBlockSize - 1)
        #expect(configuration.memoryPolicy == nil)

        switch configuration.strategy {
        case .mtp(.loaded(let configuredDrafter), let blockSize, let minimumDraftProbability):
            #expect(configuredDrafter === drafter)
            #expect(blockSize == SpeculativeDecodingConfig.defaultMTPBlockSize)
            #expect(
                minimumDraftProbability
                    == SpeculativeDecodingConfig.defaultMTPMinimumDraftProbability)
        case .mtp(.deferred, _, _), .mtp(.deferredMatchingTarget, _, _), .draftModel:
            Issue.record("Expected the eager MTP speculative-decoding strategy")
        }
    }

    @Test("MTP configuration keeps its explicit confidence floor")
    func mtpExplicitMinimumDraftProbability() throws {
        let floor: Float = 0.75

        let configuration = try SpeculativeDecodingConfig(
            mtpDrafter: makeMTPDrafterContainer(), minimumDraftProbability: floor)

        guard case .mtp(_, _, let minimumDraftProbability) = configuration.strategy else {
            Issue.record("Expected the MTP speculative-decoding strategy")
            return
        }
        #expect(minimumDraftProbability == floor)
    }

    @Test(
        "MTP configuration rejects a confidence floor outside zero to one",
        arguments: [Float(-0.25), 1.25, .nan]
    )
    func mtpRejectsInvalidMinimumDraftProbability(floor: Float) {
        #expect(throws: SpeculativeDecodingConfigurationError.self) {
            try SpeculativeDecodingConfig(
                mtpDrafter: makeMTPDrafterContainer(), minimumDraftProbability: floor)
        }
    }

    @Test(
        "MTP configuration rejects verification blocks smaller than two",
        arguments: [-1, 0, 1]
    )
    func mtpRejectsInvalidBlockSize(blockSize: Int) {
        let drafter = makeMTPDrafterContainer()

        #expect(
            throws: SpeculativeDecodingConfigurationError.invalidMTPBlockSize(blockSize)
        ) {
            try SpeculativeDecodingConfig(mtpDrafter: drafter, blockSize: blockSize)
        }
    }

    @Test("MTP configuration exposes its explicit verification block")
    func mtpExplicitBlockSize() throws {
        let drafter = makeMTPDrafterContainer()

        let configuration = try SpeculativeDecodingConfig(
            mtpDrafter: drafter,
            blockSize: 4
        )

        #expect(configuration.numDraftTokens == 3)
        guard case .mtp(.loaded(let configuredDrafter), let blockSize, _) = configuration.strategy
        else {
            Issue.record("Expected the MTP speculative-decoding strategy")
            return
        }
        #expect(configuredDrafter === drafter)
        #expect(blockSize == 4)
    }

    @Test("Eager draft-model configuration remains source compatible")
    func eagerDraftModelConfiguration() {
        let draftModel = makeDraftModelContainer()
        let memoryPolicy = SpeculativeDecodingMemoryPolicy(
            limitBytes: 1_024,
            additionalBytes: 128,
            action: .allow
        )

        let configuration = SpeculativeDecodingConfig(
            draftModel: draftModel,
            numDraftTokens: 7,
            memoryPolicy: memoryPolicy
        )

        #expect(configuration.draftModel === draftModel)
        #expect(configuration.numDraftTokens == 7)
        #expect(configuration.memoryPolicy == memoryPolicy)
        guard case .draftModel(.loaded(let configuredDraftModel)) = configuration.strategy else {
            Issue.record("Expected the eager draft-model strategy")
            return
        }
        #expect(configuredDraftModel === draftModel)
    }

    @Test("Deferred draft-model configuration remains source compatible")
    func deferredDraftModelConfiguration() {
        let draftModel = makeDraftModelContainer()
        let memoryPolicy = SpeculativeDecodingMemoryPolicy(
            limitBytes: 2_048,
            action: .fallbackToDefault
        )

        let configuration = SpeculativeDecodingConfig(
            draftModelBytes: -1,
            numDraftTokens: 6,
            memoryPolicy: memoryPolicy
        ) {
            draftModel
        }

        #expect(configuration.draftModel == nil)
        #expect(configuration.numDraftTokens == 6)
        #expect(configuration.memoryPolicy == memoryPolicy)
        #expect(configuration.estimatedDraftModelBytes == 0)
        guard case .draftModel(.deferred(let bytes, _)) = configuration.strategy else {
            Issue.record("Expected the deferred draft-model strategy")
            return
        }
        #expect(bytes == 0)
    }

    @Test("A memory policy admits speculation that fits its budget")
    func memoryPolicyAdmitsWithinBudget() throws {
        let policy = SpeculativeDecodingMemoryPolicy(
            limitBytes: memoryPolicyLimitBytes, action: .fail)

        #expect(
            try policy.admitsSpeculation(
                mainModelBytes: mainModelBytes, draftModelBytes: fittingDraftBytes))
    }

    @Test(
        "A memory policy over its budget admits or falls back by its action",
        arguments: [
            (SpeculativeDecodingMemoryAction.allow, true),
            (SpeculativeDecodingMemoryAction.fallbackToDefault, false),
        ]
    )
    func memoryPolicyOverBudgetFollowsAction(
        action: SpeculativeDecodingMemoryAction, admits: Bool
    ) throws {
        let policy = SpeculativeDecodingMemoryPolicy(
            limitBytes: memoryPolicyLimitBytes, action: action)

        #expect(
            try policy.admitsSpeculation(
                mainModelBytes: mainModelBytes, draftModelBytes: oversizedDraftBytes)
                == admits)
    }

    @Test("A failing memory policy over its budget throws its evaluation")
    func memoryPolicyOverBudgetFails() {
        let policy = SpeculativeDecodingMemoryPolicy(
            limitBytes: memoryPolicyLimitBytes, action: .fail)

        let error = #expect(throws: SpeculativeDecodingMemoryError.self) {
            try policy.admitsSpeculation(
                mainModelBytes: mainModelBytes, draftModelBytes: oversizedDraftBytes)
        }

        #expect(
            error?.evaluation
                == policy.evaluate(
                    mainModelBytes: mainModelBytes, draftModelBytes: oversizedDraftBytes))
    }
}

/// The budget of the memory policies in the admission tests.
private let memoryPolicyLimitBytes = 1_024
/// The main-model bytes of the admission tests.
private let mainModelBytes = 512
/// Draft bytes that, with `mainModelBytes`, fit `memoryPolicyLimitBytes`.
private let fittingDraftBytes = 256
/// Draft bytes that, with `mainModelBytes`, exceed `memoryPolicyLimitBytes`.
private let oversizedDraftBytes = 1_024

private func makeMTPDrafterContainer() -> MTPDrafterContainer {
    MTPDrafterContainer(
        context: MTPDrafterContext(
            configuration: ModelConfiguration(id: "test/mock-mtp-drafter"),
            model: ConfigTestMTPDrafter()
        )
    )
}

private func makeDraftModelContainer() -> ModelContainer {
    let processor = TestInputProcessor()
    return ModelContainer(
        context: ModelContext(
            configuration: processor.configuration,
            model: ConfigTestLanguageModel(),
            processor: processor,
            tokenizer: processor.tokenizer
        )
    )
}

private final class ConfigTestMTPDrafter: Module, MTPDrafterModel {
    func validateCompatibility(with _: any LanguageModel) throws {}

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        MLXArray.zeros([lastToken.dim(0), blockSize - 1], dtype: .int32)
    }
}

private final class ConfigTestLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [] }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1])
    }
}
