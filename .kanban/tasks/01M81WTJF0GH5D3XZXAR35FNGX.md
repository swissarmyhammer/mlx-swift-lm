---
assignees:
- claude-code
position_column: done
position_ordinal: ffffd680
title: 'MTP drafter download pulls the whole checkpoint (Inkling oQ4e: 15 GB for a 0.59 GB mtp.safetensors)'
---
Reported via FoundationModelsACPAgent from Extras (their card 6a9p51q). `resolve(configuration:...)` in Libraries/MLXLMCommon/ModelFactory.swift downloads with the package-wide `modelDownloadPatterns` (`*.safetensors`, ...) for drafters too. `mlx-community/Inkling-2-Mini-oQ4e-mtp` holds the full oQ4e model (model-0000{1..4}-of-00004.safetensors) beside `mtp.safetensors`, so a fresh machine downloads ~15 GB to load a 0.59 GB drafter.

- [x] Let a drafter configuration (or the drafter registration / `AdditionalWeightFilesProviding`) state its own download patterns: for Inkling, `mtp.safetensors` + `*.json` + `*.jinja` (check which config/tokenizer files the drafter load actually reads).
- [x] Keep the default patterns for drafters that ship as their own checkpoint (Qwen MTP, LFM2 DSpark, Gemma assistant).
- [x] Unit test with a stub downloader that records the requested patterns; integration check that a drafter-only load of the oQ4e repo into an empty cache fetches only those files.

Extras already sizes the drafter from mtp.safetensors alone (Extras 159f080).

#inkling

## Resolution

Commits 358e2957, 5581bd20 (branch inkling2-mini).

- `ModelConfiguration.downloadPatterns: [String]?` (nil = the default `*.safetensors`, `*.json`, `*.jinja`); `resolve(configuration:)` downloads it, with the model subdirectory prefix. It takes part in `==`.
- `GenericModelFactory.load` / `loadContainer` take the patterns of the `modelRegistry` entry for the same repository and model subdirectory when the configuration states none, so a caller that pins a revision (as IntegrationTesting and Extras do) keeps them. Explicit patterns win.
- `MTPDrafterRegistry.inkling2_mini_oq4e_mtp` downloads `mtp.safetensors` + `config.json`. The drafter load reads only `config.json` (the factory) and the weights; `*.json` would add the 26 MB `tokenizer.json` (the drafter borrows the target's tokenizer) and `*.jinja` nothing it reads. With no index match, the weight selection falls back to the files present, which is `mtp.safetensors`. The other drafter entries keep nil (defaults).
- Unit tests: `DownloadPatternsTests` (9: own patterns, subdirectory, defaults, equality, registry adoption at a pinned revision, explicit patterns win, other subdirectory / entry without patterns / unregistered id keep defaults); `Inkling2DraftLoadingTests` registry patterns and a pinned-revision drafter-only load through `MTPDrafterModelFactory.shared` with a stub downloader holding only the two files (records exactly `[mtp.safetensors, config.json]`, loads `Inkling2DraftModel`). The stub, `RecordingDownloader` in MLXLMTestSupport, replaces the private downloaders of ResolveTests and ModelConversionTests.
- Integration: `Inkling2MTPIntegrationTests.drafterLoadIntoAnEmptyCacheFetchesOnlyTheHead` loads the drafter into an empty temporary `HubCache` (not the user's cache): `["config.json", "mtp.safetensors"]`, 0.59 GiB in the cache, the drafter loads. Whole suite (Release, under the GPU lock): 7/7 pass, so the drafter still loads for parity, the verify checks, throughput and chat.
- All 5 bundles green: MLXLMTests 1260 XCTest + 2590 Swift Testing (3 known issues, which were there before), MLXGuidedGenerationTests 86, MLXFoundationModelsTests 583, CXGrammarTests 7, MLXHuggingFaceMacrosTests 11.
- Review of 358e2957: 3 findings on `RecordingDownloader.swift` (reuse with `ConversionMockDownloader`, missing doc on `download`, `@unchecked Sendable`), all fixed in 5581bd20; review of 5581bd20: 0 findings.