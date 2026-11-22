---
assignees:
- claude-code
position_column: todo
position_ordinal: 8a80
title: 'Inkling 2 MTP never drafts through MLXLanguageModel (guided path: incompatibleDrafter; tool calls: drafted 0)'
---
Reported by FoundationModelsACPAgent on stable 2dafc7c4 (Router af9b3cbc, Extras 5e43c365). Target mlx-community/Inkling-2-Mini-4bit rev 9856dc53 (VLM factory), drafter mlx-community/Inkling-2-Mini-oQ4e-mtp rev 3eea0427 loaded via `MLXSpeculativeDecoding(drafterConfiguration:from:using:registration: .matchingTarget(.qwen35Text, .qwen35VLM, .inkling2), drafterBytes:, memoryPolicy: .recommendedWorkingSet)`, default block size 4, fp16 KV, ~228k context. Every LanguageModelSession call decodes plain:
- 2 calls without a tool call: `generation.mtp.speculated=false`, `fallback_reason=incompatibleDrafter`, drafted 0.
- 5 calls with a tool call: `speculated=false`, no fallback reason, drafted 0.

Configs match (hidden 2304, vocab 201024/200058, loop 6/18/3), so `Inkling2DraftModel.isCompatible` should hold. Likely cause: `GuidedSpeculation.drafter(of:for:)` (Libraries/MLXGuidedGeneration/GuidedSpeculation.swift) requires `GuidedMTPDrafter` with `!requiresSharedTargetKV`; Inkling2DraftModel is neither, so guided generations report incompatibleDrafter; and the tool-call passes appear to never attach speculation or fall back silently (no reason recorded — that silence is a bug too).

- [ ] Reproduce in IntegrationTesting through MLXLanguageModel + LanguageModelSession with MLXSpeculativeDecoding built exactly as above (registration via matchingTarget, drafterBytes, memory policy), one plain call, one guided/generable call, one tool loop; assert drafted > 0 on each pass.
- [ ] Make the Inkling drafter speculate under guided generation (GuidedMTPDrafter conformance; whatever requiresSharedTargetKV / verifiesInPlace needs for Inkling's caches) — check how Qwen35 MTP does it.
- [ ] Find why tool-call passes draft 0 with no fallback reason; fix, and make any fallback record its reason.
- [ ] MTP greedy == plain greedy still; multi-turn cache reuse still works; numbers for plain vs MTP through MLXLanguageModel.

Also noted (Extras-side): Extras sums every *.safetensors in the drafter snapshot for drafterBytes, which includes the full oQ4e model-0000N files; the memory policy may then refuse the drafter (memoryBudgetExceeded). Consider whether MLXSpeculativeDecoding should size the drafter from its own weight files (additionalWeightFiles) instead.

#inkling