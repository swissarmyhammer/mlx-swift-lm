import Foundation
import MLX
import MLXLMTestSupport
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

@Test
func testQwen35TextConfigurationDecodesMTPFields() throws {
    let configuration = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))

    #expect(configuration.mtpNumHiddenLayers == 1)
    #expect(configuration.mtpUseDedicatedEmbeddings == false)
}

@Test
func testQwen35VLMTextConfigurationDecodesMTPFields() throws {
    let configuration = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))

    #expect(configuration.mtpNumHiddenLayers == 1)
    #expect(configuration.mtpUseDedicatedEmbeddings == false)
}

@Suite
struct Qwen35MTPCompatibilityTests {
    @Test
    func validatesTargetFamily() throws {
        let textConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let textTarget = MLXLLM.Qwen35TextModel(textConfig)
        let textDrafter = MLXLLM.Qwen35MTPDraftModel(textConfig)
        let wrappedTextConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8))

        let vlmConfig = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let vlmTarget = MLXVLM.Qwen35(vlmConfig)
        let vlmDrafter = MLXVLM.Qwen35VLMNextNDraftModel(vlmConfig)

        try textDrafter.validateCompatibility(with: textTarget)
        try textDrafter.validateCompatibility(with: MLXLLM.Qwen35Model(wrappedTextConfig))
        try vlmDrafter.validateCompatibility(with: vlmTarget)

        #expect(throws: MTPDrafterCompatibilityError.self) {
            try textDrafter.validateCompatibility(with: vlmTarget)
        }
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try vlmDrafter.validateCompatibility(with: textTarget)
        }
    }

    @Test
    func textDrafterRejectsSameFamilyDimensionMismatches() throws {
        let drafterConfig = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let drafter = MLXLLM.Qwen35MTPDraftModel(drafterConfig)

        let hiddenSizeMismatch = try JSONDecoder().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1, hiddenSize: 32).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXLLM.Qwen35TextModel(hiddenSizeMismatch))
        }

        let vocabularySizeMismatch = try JSONDecoder().decode(
            MLXLLM.Qwen35Configuration.self,
            from: Data(
                Qwen35FixtureJSON.wrappedText(
                    modelType: "qwen3_5",
                    textConfigJSON: Qwen35FixtureJSON.text(mtpLayers: 1, vocabularySize: 17)
                ).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXLLM.Qwen35Model(vocabularySizeMismatch))
        }
    }

    @Test
    func vlmDrafterRejectsSameFamilyDimensionMismatches() throws {
        let drafterConfig = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let drafter = MLXVLM.Qwen35VLMNextNDraftModel(drafterConfig)

        let hiddenSizeMismatch = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1, hiddenSize: 32).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXVLM.Qwen35(hiddenSizeMismatch))
        }

        let vocabularySizeMismatch = try JSONDecoder().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1, vocabularySize: 17).utf8))
        #expect(throws: MTPDrafterCompatibilityError.self) {
            try drafter.validateCompatibility(
                with: MLXVLM.Qwen35(vocabularySizeMismatch))
        }
    }
}

@Test
func testQwen35MTPDraftSanitizeKeepsAndShiftsMTPNorms() throws {
    let configuration = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)

    let sanitized = try drafter.sanitize(weights: [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "mtp.pre_fc_norm_embedding.weight": MLXArray.zeros([16]),
        "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 16]),
        "mtp.layers.0.mlp.experts.gate_up_proj": MLXArray.zeros([2, 32, 16]),
        "mtp.layers.0.mlp.experts.down_proj": MLXArray.zeros([2, 16, 16]),
        "model.embed_tokens.weight": MLXArray.zeros([16, 16]),
    ])

    #expect(sanitized["model.embed_tokens.weight"] == nil)
    #expect(sanitized["mtp.layers.0.self_attn.q_proj.weight"] != nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.gate_up_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.experts.down_proj"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
    let norm = try #require(sanitized["mtp.norm.weight"])
    let embeddingNorm = try #require(sanitized["mtp.pre_fc_norm_embedding.weight"])
    eval(norm, embeddingNorm)
    #expect(allClose(norm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
    #expect(allClose(embeddingNorm, MLXArray.ones([16]), rtol: 0, atol: 0).item(Bool.self))
}

@Test(arguments: [false, true])
func testQwen35MTPDraftSanitizeRejectsCompetingComponents(vision: Bool) throws {
    let data = Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8)
    let drafter: any BaseLanguageModel
    if vision {
        drafter = MLXVLM.Qwen35VLMNextNDraftModel(
            try JSONDecoder().decode(MLXVLM.Qwen35Configuration.TextConfiguration.self, from: data))
    } else {
        drafter = MLXLLM.Qwen35MTPDraftModel(
            try JSONDecoder().decode(MLXLLM.Qwen35TextConfiguration.self, from: data))
    }
    let weights = [
        "mtp.norm.weight": MLXArray.zeros([16]),
        "language_model.mtp.norm.weight": MLXArray.ones([16]),
    ]
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights)
    }
    #expect(throws: CheckpointComponent.SelectionError.self) {
        try drafter.sanitize(weights: weights, metadata: [:])
    }
}

@Test
func testQwen35StandaloneMTPDoesNotDoubleShiftConvertedNorms() throws {
    let configuration = try JSONDecoder().decode(
        MLXLLM.Qwen35Configuration.self,
        from: Data(Qwen35FixtureJSON.standaloneMTP().utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(configuration, preconvertedNorms: true)

    let weight = MLXArray.zeros([16])
    let sanitized = try drafter.sanitize(weights: ["mtp.norm.weight": weight])
    let norm = try #require(sanitized["mtp.norm.weight"])
    eval(norm)
    #expect(allClose(norm, weight, rtol: 0, atol: 0).item(Bool.self))
}

@Test
func testQwen35MTPDraftSanitizeStacksPerExpertMoEWeights() throws {
    let configuration = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(Qwen35FixtureJSON.text(mtpLayers: 1, numExperts: 2).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)

    let weights: [String: MLXArray] = [
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": MLXArray.ones([16, 16]),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": MLXArray.zeros([16, 16]),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": MLXArray.ones([16, 16]),
    ]

    let sanitized = try drafter.sanitize(weights: weights)

    #expect(sanitized["mtp.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [2, 16, 16])
    #expect(sanitized["mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [2, 16, 16])
}

@Test
func testQwen35MTPDraftInstantiatesDedicatedEmbeddingWhenConfigured() throws {
    let configuration = try JSONDecoder().decode(
        MLXLLM.Qwen35TextConfiguration.self,
        from: Data(
            Qwen35FixtureJSON.text(mtpLayers: 1, mtpUseDedicatedEmbeddings: true).utf8))
    let drafter = MLXLLM.Qwen35MTPDraftModel(configuration)

    #expect(drafter.mtp.embedTokens != nil)
    let sanitized = try drafter.sanitize(weights: [
        "mtp.embed_tokens.weight": MLXArray.zeros([16, 16]),
        "model.embed_tokens.weight": MLXArray.ones([16, 16]),
    ])
    #expect(sanitized["mtp.embed_tokens.weight"] != nil)
    #expect(sanitized["model.embed_tokens.weight"] == nil)
}

@Suite(.serialized)
struct Qwen35MTPRegistrationTests {
    @Test
    func registrationsCreateTextAndVLMDrafters() async throws {
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8),
            modelType: "qwen3_5_text")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)

        await MLXVLM.Qwen35VLMMTPRegistration.register()

        let wrappedTextModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(wrappedTextModel is MLXLLM.Qwen35MTPDraftModel)

        let vlmModel = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let standalone = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.standaloneMTP().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standalone is MLXLLM.Qwen35MTPDraftModel)
        #expect(standalone.maximumBlockSize == GatedDeltaDecodeKernel.mostTokens)
        #expect(standalone.requiresPromptPrefill)
        #expect(!standalone.requiresSharedTargetKV)
        #expect(!standalone.requiresGreedySampling)

        let standaloneVLM = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.standaloneMTP().utf8),
            modelType: "qwen3_5_mtp")
        #expect(standaloneVLM is MLXVLM.Qwen35VLMNextNDraftModel)

        let textConfig = try JSONDecoder.json5().decode(
            MLXLLM.Qwen35TextConfiguration.self,
            from: Data(Qwen35FixtureJSON.text(mtpLayers: 1).utf8))
        let vlmConfig = try JSONDecoder.json5().decode(
            MLXVLM.Qwen35Configuration.self,
            from: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8))
        let textTarget = MLXLLM.Qwen35TextModel(textConfig)
        let vlmTarget = MLXVLM.Qwen35(vlmConfig)

        #expect(standalone.isCompatible(with: textTarget))
        #expect(!standalone.isCompatible(with: vlmTarget))
        #expect(standaloneVLM.isCompatible(with: vlmTarget))
        #expect(!standaloneVLM.isCompatible(with: textTarget))
    }

    @Test
    func registrationsAreIsolatedByTargetArchitecture() async throws {
        await MLXVLM.Qwen35VLMMTPRegistration.register()
        await MLXLLM.Qwen35TextMTPRegistration.register()

        let vlmModel = try await MTPDrafterTypeRegistry.visionLanguage.createModel(
            configuration: Data(Qwen35FixtureJSON.visionLanguage(mtpLayers: 1).utf8),
            modelType: "qwen3_5")
        #expect(vlmModel is MLXVLM.Qwen35VLMNextNDraftModel)

        let textModel = try await MTPDrafterTypeRegistry.shared.createModel(
            configuration: Data(Qwen35FixtureJSON.wrappedText(modelType: "qwen3_5").utf8),
            modelType: "qwen3_5")
        #expect(textModel is MLXLLM.Qwen35MTPDraftModel)
    }
}

@Suite(.serialized)
struct Qwen35CheckpointLoadingTests {
    @Test(arguments: [false, true])
    func convertedEmbeddedFactoryPreservesMixedPrecision(vision: Bool) async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: false, prefix: "language_model.mtp.", bits: 4)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let fusionProjection = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fusionProjection.bits == 8)
        #expect(
            context.model.modules().compactMap { $0 as? QuantizedLinear }.contains { $0.bits == 4 })
    }

    /// A head embedded in a target checkpoint reads only its own tensors; a standalone head
    /// reads every tensor.
    @Test(arguments: [false, true])
    func drafterDoesNotReadTargetTensors(vision: Bool) async throws {
        for standalone in [false, true] {
            let fixture = try await Qwen35MTPCheckpointFixture(
                vision: vision, standalone: standalone, prefix: standalone ? "" : "mtp.")
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let context = try await fixture.factory.load(
                from: fixture.directory, using: UnusedTokenizerLoader())

            let url = fixture.directory.appendingPathComponent("model.safetensors")
            let all = try loadModelCheckpoint(urls: [url]).weights.keys
            let read = try loadModelCheckpoint(
                urls: [url], excludedNamespaces: context.model.excludedCheckpointNamespaces
            ).weights.keys
            if standalone {
                #expect(Set(read) == Set(all))
            } else {
                #expect(all.contains("model.embed_tokens.weight"))
                #expect(Set(read) == Set(all.filter { $0.hasPrefix("mtp.") }))
            }
        }
    }

    @Test(arguments: [false, true])
    func targetLoaderExcludesConvertedMTPComponent(vision: Bool) async throws {
        let data = try Qwen35MTPCheckpointFixture.configuration(
            vision: vision, standalone: false, bits: nil)
        let target: any BaseLanguageModel
        if vision {
            target = MLXVLM.Qwen35(
                try JSONDecoder().decode(MLXVLM.Qwen35Configuration.self, from: data))
        } else {
            target = MLXLLM.Qwen35Model(
                try JSONDecoder().decode(MLXLLM.Qwen35Configuration.self, from: data))
        }
        var weights = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        let expectedKeys = Set(weights.keys)
        weights["language_model.mtp.norm.weight"] = MLXArray.zeros([64])
        weights["language_model.mtp.fc.weight"] = MLXArray.zeros([64, 128])
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        try await loadWeights(
            modelDirectory: directory, model: target, componentLoading: .immediate)
        #expect(Set(target.parameters().flattened().map(\.0)) == expectedKeys)
    }

    @Test
    func conversionRoundTripRetainsNormConvention() async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: false, standalone: false, prefix: "mtp.")
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        defer {
            try? FileManager.default.removeItem(at: fixture.directory)
            try? FileManager.default.removeItem(at: output)
        }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        _ = try await convert(
            modelDirectory: fixture.directory, model: context.model, to: output,
            bits: 4, groupSize: 32)
        let (_, metadata) = try loadArraysAndMetadata(
            url: output.appendingPathComponent("model.safetensors"))
        #expect(metadata[Qwen35CheckpointPolicy.normMetadataKey] == "scale")
        let reloaded = try await fixture.factory.load(from: output, using: UnusedTokenizerLoader())
        let norm = try #require(
            reloaded.model.parameters().flattened().first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func expertParametersAndSettingsAreTransformedTogether() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp"
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for (parameter, shape) in [
                ("weight", [64, 8]), ("scales", [64, 2]), ("biases", [64, 2]),
            ] {
                weights["\(prefix).experts.\(expert).up_proj.\(parameter)"] = MLXArray.zeros(shape)
            }
        }
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).experts.0.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                    "\(prefix).experts.1.up_proj": .quantize(.init(groupSize: 32, bits: 8)),
                ]))
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.weight"]?.shape == [2, 64, 8])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.scales"]?.shape == [2, 64, 2])
        #expect(prepared.weights["\(prefix).switch_mlp.up_proj.biases"]?.shape == [2, 64, 2])
        #expect(
            prepared.perLayerQuantization?.quantization(layer: "\(prefix).switch_mlp.up_proj")?.bits
                == 8)
        #expect(
            prepared.perLayerQuantization?.perLayerQuantization["\(prefix).experts.0.up_proj"]
                == nil)
    }

    @Test(arguments: [false, true], [4, 5, 8])
    func standaloneFactoryLoadsPublishedLayout(vision: Bool, bits: Int) async throws {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: true, bits: bits)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let parameters = context.model.parameters().flattened()
        #expect(parameters.count == 31)
        let norm = try #require(parameters.first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
        let fusionProjection = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fusionProjection.bits == 8)
        let projections = context.model.modules().compactMap { $0 as? MLXNN.QuantizedLinear }
        #expect(projections.contains { $0.bits == bits })
    }

    @Test(arguments: [false, true], [false, true])
    func registeredFactoriesLoadQwen38Components(vision: Bool, standalone: Bool) async throws {
        await Qwen35TextMTPRegistration.register()
        await Qwen35VLMMTPRegistration.register()
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: standalone,
            prefix: standalone ? "" : "language_model.mtp.", bits: 4, family: "qwen3_8")
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let factory = vision ? MTPDrafterModelFactory.visionLanguage : .shared
        let context = try await factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let parameters = context.model.parameters().flattened()
        #expect(parameters.count == 31)
        let norm = try #require(parameters.first { $0.0 == "mtp.norm.weight" }?.1)
        #expect(norm.asArray(Float.self) == Array(repeating: 1, count: 64))
        let fusionProjection = try #require(
            (context.model as? MLXLLM.Qwen35MTPDraftModel)?.mtp.fc as? QuantizedLinear
                ?? (context.model as? MLXVLM.Qwen35VLMNextNDraftModel)?.mtp.fc as? QuantizedLinear)
        #expect(fusionProjection.bits == 8)
        #expect(
            context.model.modules().compactMap { $0 as? QuantizedLinear }.contains { $0.bits == 4 })
    }

    @Test(arguments: ["qwen3_5", "qwen3_8"])
    func standaloneVLMRegistrationUsesOuterModelType(family: String) async throws {
        await Qwen35VLMMTPRegistration.register()
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: true, standalone: true, bits: 4, family: family)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await MTPDrafterModelFactory.visionLanguage.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        #expect(context.model is MLXVLM.Qwen35VLMNextNDraftModel)
        #expect(context.model.parameters().flattened().count == 31)
    }

    @Test(arguments: [false, true], ["mtp.", "language_model.mtp."])
    func embeddedFactoryLoadsComponentWithoutTargetWeights(vision: Bool, prefix: String)
        async throws
    {
        let fixture = try await Qwen35MTPCheckpointFixture(
            vision: vision, standalone: false, prefix: prefix)
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let context = try await fixture.factory.load(
            from: fixture.directory, using: UnusedTokenizerLoader())
        let weights = Dictionary(uniqueKeysWithValues: context.model.parameters().flattened())
        #expect(weights.count == 15)
        #expect(weights["model.embed_tokens.weight"] == nil)
        #expect(weights["mtp.norm.weight"]?.asArray(Float.self) == Array(repeating: 1, count: 64))
    }

    @Test
    func targetMetadataDoesNotClassifyTheRawMTPShard() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let checkpoint = ModelCheckpoint(
            weights: ["mtp.norm.weight": MLXArray.zeros([64])],
            metadata: ["format": "mlx"], weightMetadata: ["mtp.norm.weight": [:]])
        let prepared = try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(
            prepared.weights["mtp.norm.weight"]?.asArray(Float.self)
                == Array(repeating: 1, count: 64))
    }

    @Test
    func explicitNormConventionOverridesLegacyLayout() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let raw = ModelCheckpoint(
            weights: ["language_model.mtp.norm.weight": MLXArray.zeros([4])],
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "offset"])
        let prepared = try policy.prepare(raw, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(prepared.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
        let scale = ModelCheckpoint(
            weights: prepared.weights,
            metadata: [Qwen35CheckpointPolicy.normMetadataKey: "scale"])
        let repeated = try policy.prepare(scale, mtpNumHiddenLayers: 1, numExperts: 0)
        #expect(repeated.weights["mtp.norm.weight"]?.asArray(Float.self) == [1, 1, 1, 1])
    }

    @Test
    func missingAndCompetingComponentsAreRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        for weights in [
            ["model.layers.0.weight": MLXArray(1)],
            ["mtp.norm.weight": MLXArray(1), "language_model.mtp.fc.weight": MLXArray(1)],
        ] {
            #expect(throws: CheckpointComponent.SelectionError.self) {
                try policy.prepare(.init(weights: weights), mtpNumHiddenLayers: 1, numExperts: 0)
            }
        }
    }

    @Test
    func conflictingAndUnknownNormMetadataIsRejected() {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let weights = [
            "mtp.norm.weight": MLXArray.zeros([4]),
            "mtp.layers.0.input_layernorm.weight": MLXArray.zeros([4]),
        ]
        for metadata in [
            ["mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "unknown"]],
            [
                "mtp.norm.weight": [Qwen35CheckpointPolicy.normMetadataKey: "scale"],
                "mtp.layers.0.input_layernorm.weight": [
                    Qwen35CheckpointPolicy.normMetadataKey: "offset"
                ],
            ],
        ] {
            #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
                try policy.prepare(
                    .init(weights: weights, weightMetadata: metadata), mtpNumHiddenLayers: 1,
                    numExperts: 0)
            }
        }
    }

    @Test
    func targetLayoutsShareComponentSelectionAndQuantizationMapping() throws {
        let checkpoint = ModelCheckpoint(
            weights: [
                "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray(1),
                "mtp.fc.weight": MLXArray(2),
                "lm_head.weight": MLXArray(3),
                "notmtp.weight": MLXArray(4),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "model.language_model.layers.0.self_attn.q_proj": .quantize(
                    .init(groupSize: 32, bits: 8))
            ]))
        for layout in [Qwen35CheckpointPolicy.TargetLayout.text, .wrappedText, .vision] {
            let prepared = try Qwen35CheckpointPolicy.prepareTarget(
                checkpoint, layout: layout, tiedWordEmbeddings: true)
            let prefix = layout == .text ? "model." : "language_model.model."
            let projection = prefix + "layers.0.self_attn.q_proj"
            #expect(prepared.weights[projection + ".weight"] != nil)
            #expect(prepared.perLayerQuantization?.quantization(layer: projection)?.bits == 8)
            #expect(prepared.weights.count == 2)
            #expect(prepared.weights["notmtp.weight"] != nil)
        }
    }

    @Test(arguments: ["qwen3_5", "qwen3_8"])
    func unknownAndMissingWeightsStillFailStrictFactoryLoading(family: String) async throws {
        for missing in [false, true] {
            let fixture = try await Qwen35MTPCheckpointFixture(
                vision: false, standalone: true, family: family)
            defer { try? FileManager.default.removeItem(at: fixture.directory) }
            let url = fixture.directory.appendingPathComponent("model.safetensors")
            var weights = try loadArrays(url: url)
            eval(Array(weights.values))
            if missing {
                weights.removeValue(forKey: "norm.weight")
            } else {
                weights["unexpected.weight"] = MLXArray(1)
            }
            try save(arrays: weights, metadata: ["format": "mlx"], url: url)
            let error = await #expect(throws: UpdateError.self) {
                try await fixture.factory.load(
                    from: fixture.directory, using: UnusedTokenizerLoader())
            }
            // The strict update names the tensor it rejects: a missing required one, or one
            // the drafter has no parameter for.
            switch error {
            case .keyNotFound(let path, _) where missing:
                #expect(path == ["mtp", "norm", "weight"], "\(String(describing: error))")
            case .unhandledKeys(_, _, let keys) where !missing:
                #expect(keys.contains("unexpected"), "\(String(describing: error))")
            default:
                Issue.record("unexpected loading error: \(String(describing: error))")
            }
        }
    }

    @Test
    func expertPrecisionCannotChangeDuringStacking() throws {
        let policy = Qwen35CheckpointPolicy(layout: .embedded, preconvertedNorms: false)
        let prefix = "mtp.layers.0.mlp.experts"
        let weights = [
            "\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64]),
            "\(prefix).1.up_proj.weight": MLXArray.zeros([64, 64]),
        ]
        let checkpoint = ModelCheckpoint(
            weights: weights,
            perLayerQuantization: .init(
                quantization: .init(groupSize: 32, bits: 4),
                perLayerQuantization: [
                    "\(prefix).0.up_proj": .quantize(.init(groupSize: 32, bits: 8))
                ]))
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(checkpoint, mtpNumHiddenLayers: 1, numExperts: 2)
        }
        #expect(throws: Qwen35CheckpointPolicy.LoadingError.self) {
            try policy.prepare(
                .init(weights: ["\(prefix).0.up_proj.weight": MLXArray.zeros([64, 64])]),
                mtpNumHiddenLayers: 1, numExperts: 2)
        }
    }
}
