---
assignees:
- claude-code
position_column: todo
position_ordinal: 8d80
title: 'MTP drafter download pulls the whole checkpoint (Inkling oQ4e: 15 GB for a 0.59 GB mtp.safetensors)'
---
Reported via FoundationModelsACPAgent from Extras (their card 6a9p51q). `resolve(configuration:...)` in Libraries/MLXLMCommon/ModelFactory.swift downloads with the package-wide `modelDownloadPatterns` (`*.safetensors`, ...) for drafters too. `mlx-community/Inkling-2-Mini-oQ4e-mtp` holds the full oQ4e model (model-0000{1..4}-of-00004.safetensors) beside `mtp.safetensors`, so a fresh machine downloads ~15 GB to load a 0.59 GB drafter.

- [ ] Let a drafter configuration (or the drafter registration / `AdditionalWeightFilesProviding`) state its own download patterns: for Inkling, `mtp.safetensors` + `*.json` + `*.jinja` (check which config/tokenizer files the drafter load actually reads).
- [ ] Keep the default patterns for drafters that ship as their own checkpoint (Qwen MTP, LFM2 DSpark, Gemma assistant).
- [ ] Unit test with a stub downloader that records the requested patterns; integration check that a drafter-only load of the oQ4e repo into an empty cache fetches only those files.

Extras already sizes the drafter from mtp.safetensors alone (Extras 159f080).

#inkling