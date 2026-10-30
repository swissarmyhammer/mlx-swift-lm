---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01m66zpp2dtz4jmm6d8pn21x7f
  text: 'Research: MTPChatTokenMatch.init (Libraries/BenchmarkHelpers/MTPChatTokenMatch.swift) calls a divergence a near tie only when MTP''s token == plain.candidates[1] and top-2 gap <= 0.5. In MTPNearTieDivergenceTests the plain candidates come from a replay, not from plain decode, so at an exact tie the replay can rank MTP''s token first (candidates[0]) while plain decode wrote the other: the rule fails and the report says "Bug, no near tie". MTPSWEBenchScaleTests.tokenMatch works around the same asymmetry by trying forward and swapped (candidates on either side). Plan: near tie iff both the plain and MTP token at the step appear in plain''s candidates within nearTieLogitTolerance of candidates[0] (the row max); the case gap becomes the lower token''s distance below that max (nil on diverged when a token is not among the candidates). The rule is then symmetric in mtp/plain, so the sweep''s swapped retry becomes redundant and is removed. Callers checked: MTPChatBenchmark, MTPGenerationBenchmark, GenerationBenchmarkTests (both sites), MTPDrafterWindowSweepTests (read-only, not touched; behavior only widens holds), MTPSWEBenchScaleTests.tokenMatch, MTPNearTieDivergenceTests. Note: the dispatcher named this card ^dvqvn but the card''s short_id is ^xrdeqvn.'
  timestamp: 2026-10-30T18:03:42.029451+00:00
- actor: claude-code
  id: 01m67083yj0bxk005gvztsmx7b
  text: |-
    Implementation landed (TDD). RED: new Tests/MLXLMTests/MTPChatTokenTieTests.swift (5 tests, 2.5 KB) — 3 failed before the fix (exact tie with MTP's token ranked first, near tie with MTP's token ranked first, token outside the candidates expected gap nil); exact tie with plain's token first and the wider-than-tolerance divergence already held. GREEN after the fix; the 6 existing MTPChatTokenMatchTests stay green unchanged.
    Fix: MTPChatTokenMatch.init now finds both turns' tokens at the divergence among plain's reported candidates and computes how far below the best logit the lower lies (private static distanceBelowBest). <= nearTieLogitTolerance -> .nearTie, else .diverged; nil (a token missing from the candidates, or a turn ended) -> .diverged(gap: nil). The case `gap` now means that distance, so a diverged summary no longer prints a top-2 gap of 0.0 for a real divergence. Docs updated.
    Callers: MTPSWEBenchScaleTests.tokenMatch — the forward/swapped retry is redundant now (the rule is symmetric in mtp/plain with one candidate row) and was removed; equivalent logic. MTPNearTieDivergenceTests — failure message now says "Bug: plain's and MTP's tokens do not both lie within 0.5 of the best logit of plain's row", the report's "replay reproduces plain" field became "replay ranks plain's token first", suite doc no longer says "MTP took the second". MTPChatBenchmark, MTPGenerationBenchmark, GenerationBenchmarkTests (2 sites) read plain's own 2 candidates (MTPBlockSweep.reportedCandidateCount = 2), so they behave as before except that an MTP turn taking plain's top token while plain wrote the runner-up now also holds. MTPDrafterWindowSweepTests not touched (per instruction); it only uses .summary/.holds, which still compile and mean the same.
    Not done: subtask "Re-run MTPNearTieDivergenceTests on both paths" — the dispatcher said no real-model run unless a real-model test's logic changed substantially; the changes there are the message, the doc and an equivalent simplification, so it was not run. That box stays open for whoever runs the model.
    Build note: `swift build --build-tests` prints the pre-existing SwiftPM line "warning: missing creator for mutated node: ...mlx-swift_Cmlx.bundle/Contents/MacOS" (recorded on earlier cards; build-system, not a compiler warning); 0 compiler warnings.
  timestamp: 2026-10-30T18:13:13.298168+00:00
- actor: claude-code
  id: 01m67084m441bcnypw21s7z7q3
  text: |-
    ### implement — changed
    - evidence: 4 files — Libraries/BenchmarkHelpers/MTPChatTokenMatch.swift, Tests/MLXLMTests/MTPChatTokenTieTests.swift (new), IntegrationTesting/.../MTPSWEBenchScaleTests.swift, IntegrationTesting/.../MTPNearTieDivergenceTests.swift. swift build --build-tests: 0 compiler warnings. xcrun xctest: CXGrammarTests 7 pass, MLXFoundationModelsTests 457 pass, MLXGuidedGenerationTests 72 pass, MLXHuggingFaceMacrosTests 7 XCTest pass, MLXLMTests 920 XCTest + 2073 Swift Testing pass; 0 fail, 0 skip. IntegrationTesting Release build-for-testing: TEST BUILD SUCCEEDED.
    - next: /review. Open subtask: real-model re-run of MTPNearTieDivergenceTests on both paths (not run per dispatcher scope).
  timestamp: 2026-10-30T18:13:13.988928+00:00
- actor: claude-code
  id: 01m670axbvnt6cnb281smy0mmr
  text: |-
    ### test — green
    - evidence: Release xcodebuild test, -only-testing:IntegrationTestingTests/MTPNearTieDivergenceTests, 1 test with 2 cases (text, router) passed in 35.6s; uptime 13:13 / 13:14 up 13 days 5:01 (load 2.23 -> 2.53).
    - text path: first divergence at generated token 74 (ledger 148): plain wrote 16570 (logprob -1.7059), MTP wrote 58614 (-1.8309); plain-row gap 0.125; verify-row gap 0.125; replay ranks plain's token first true; near-tie at token 148, gap 0.125. Accepted 373 of 400 drafts.
    - router path: first divergence at generated token 74 (ledger 148): plain wrote 58614, MTP wrote 16570; plain-row gap 0.0 (both -1.7447); verify-row gap 0.0 (both -1.7460); replay ranks plain's token first false; near-tie at token 148, gap 0.0. Accepted 374 of 399 drafts.
    - next: close ^xrdeqvn.
  timestamp: 2026-10-30T18:14:44.859336+00:00
position_column: doing
position_ordinal: '8280'
title: 'Benchmarks: MTPChatTokenMatch must accept an exact top-2 tie (gap 0) whichever token argmax picked'
---
MTPNearTieDivergenceTests (^j2tnzpn phase 2) failed on the router path: first difference at generated token 74, plain 58614 vs MTP 16570, top-2 gap 0.0 on both the plain row and the verify row — an exact tie. The replay ranked 16570 first while plain decode wrote 58614 ("replay reproduces plain false"), so the check failed on tie ordering and printed the misleading "Bug, no near tie". By the near-tie rule this is a tie, not an MTP bug.

- [x] MTPChatTokenMatch: treat the divergence as a near tie when the two tokens are both within `nearTieLogitTolerance` of the row's max (not "plain's token is top-1"), so argmax tie-breaking order does not matter.
- [x] Unit test with an exact tie in both orders; keep a real non-tie divergence failing.
- [x] Re-run MTPNearTieDivergenceTests on both paths.

#mtp #benchmark