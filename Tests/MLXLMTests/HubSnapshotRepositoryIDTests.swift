// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXScriptedLM
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// A snapshot directory in the layout of the Hugging Face hub cache.
private func hubSnapshot(
    root: URL = URL(fileURLWithPath: "/cache/hub"), repository: String,
    revision: String = "b41f2b65685e95418f1ac809bb022d4f79e1ab27"
) -> URL {
    root.appending(component: "models--" + repository.replacingOccurrences(of: "/", with: "--"))
        .appending(component: "snapshots").appending(component: revision)
}

@Suite
struct HubSnapshotRepositoryIDTests {

    @Test("A Hub id names its own repository")
    func idNamesItsRepository() {
        #expect(
            ModelConfiguration(id: "LiquidAI/LFM2.5-2.6B-MLX").repositoryID
                == "LiquidAI/LFM2.5-2.6B-MLX")
    }

    @Test("A hub-cache snapshot directory names the repository its layout encodes")
    func snapshotNamesItsRepository() {
        let configuration = ModelConfiguration(
            directory: hubSnapshot(repository: "LiquidAI/LFM2.5-8B-A1B-MLX-4bit"))
        #expect(configuration.repositoryID == "LiquidAI/LFM2.5-8B-A1B-MLX-4bit")
        #expect(configuration.name != configuration.repositoryID)
    }

    @Test("A variant directory inside a snapshot names the repository of the snapshot")
    func variantInsideSnapshotNamesItsRepository() {
        let directory = hubSnapshot(repository: "org/model").appending(component: "4bit")
        #expect(ModelConfiguration(directory: directory).repositoryID == "org/model")
    }

    @Test("A resolved snapshot directory names the repository its layout encodes")
    func resolvedSnapshotNamesItsRepository() {
        let directory = hubSnapshot(repository: "LiquidAI/LFM2.5-2.6B-MLX")
        #expect(
            ResolvedModelConfiguration(directory: directory).repositoryID
                == "LiquidAI/LFM2.5-2.6B-MLX")
        #expect(
            ModelConfiguration(directory: directory)
                .resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .repositoryID == "LiquidAI/LFM2.5-2.6B-MLX")
    }

    @Test("A resolved Hub id keeps its own id wherever its files are")
    func resolvedIDKeepsItsRepository() {
        let directory = URL(fileURLWithPath: "/downloads/elsewhere")
        #expect(
            ModelConfiguration(id: "org/model")
                .resolved(modelDirectory: directory, tokenizerDirectory: directory)
                .repositoryID == "org/model")
    }

    @Test(
        "A directory outside the hub-cache layout names its path",
        arguments: [
            "/models/LiquidAI/LFM2.5-2.6B-MLX", "/cache/models--org/other/rev",
            "/cache/models--/snapshots/rev", "/cache/models--org--name/snapshots",
        ])
    func otherDirectoryNamesItsPath(path: String) {
        let configuration = ModelConfiguration(directory: URL(fileURLWithPath: path))
        #expect(configuration.repositoryID == configuration.name)
    }

    private struct TokenizerLoader: MLXLMCommon.TokenizerLoader {
        func load(from directory: URL) async throws -> any Tokenizer {
            PseudoWordTokenizer()
        }
    }

    /// A factory load of a hub-cache snapshot of an LFM2.5 reasoning checkpoint resolves the
    /// reasoning protocol of LFM2.5 from the repository id, as a load by that id does.
    @Test("A factory load of an LFM2.5 hub snapshot directory gets the LFM2.5 reasoning")
    func factoryLoadOfSnapshotResolvesTheRepositoryConventions() async throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "hub-snapshot-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = hubSnapshot(root: root, repository: "LiquidAI/LFM2.5-2.6B-MLX")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(tinyLFM2DSparkTargetJSON.utf8).write(
            to: directory.appending(component: "config.json"))
        let model = LFM2Model(
            try JSONDecoder().decode(
                LFM2Configuration.self, from: Data(tinyLFM2DSparkTargetJSON.utf8)))
        eval(model)
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appending(component: "model.safetensors"))

        let context = try await LLMModelFactory.shared._load(
            configuration: ResolvedModelConfiguration(directory: directory),
            tokenizerLoader: TokenizerLoader())

        #expect(context.configuration.reasoningConfig?.promptStrategy == .alwaysOn)
        #expect(context.configuration.reasoningConfig?.isSpecialToken == false)
    }
}
