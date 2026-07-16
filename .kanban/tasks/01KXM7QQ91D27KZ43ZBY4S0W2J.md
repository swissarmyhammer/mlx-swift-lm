---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kxntacre63n3bz815j5vtfzm
  text: Picked up by /finish (scoped-batch, after bxndpt6 was reported stuck on a validator rule contradiction). Dispatching /implement.
  timestamp: 2026-07-16T15:56:56.462749+00:00
- actor: claude-code
  id: 01kxnv6m97d4weksnyc8b3eqns
  text: |-
    Research findings (root-cause analysis, pre-implementation):

    1. The FoundationModels tool-calling path does NOT use the per-family ToolCallParsers at all (GLM4ToolCallParser/MistralToolCallParser live only in the MLXLMCommon generate()/ToolCallProcessor path). MLXLanguageModel.runToolCalling constrains with SchemaConverter.encodeToolCallingGrammar (Qwen structural tag for every family since 44a96cf) and parses with emitToolCallingEvent -> unwrapToolCallMarkers (hardcoded "<tool_call>"/"</tool_call>"). So the card's "parser follows family, tag follows Qwen" hypothesis is not literally the mechanism — parse and constrain sides DO both use the Qwen tag here.

    2. Actual root cause of the runaway (and the stray digit): the guided-loop zone policy. ConstraintSetup.reserves computes completionReserve = max(maxTokens - normalZoneLength, hardReserve) with normalZoneLength = min(structuralReserve*3, maxTokens/2). For the tool-calling envelope, structuralReserve is the tokenized minimal SKELETON of the FIRST oneOf alternative (~20 tokens), so the unbiased "normal zone" is only ~60 tokens out of 4096. From token ~60 onward, the SOFT zone applies ClosingTokenBias: +100 on `"` `}` `]` AND DIGITS 0-9, +200 on EOS. +100 dwarfs any real logit gap (~<20), so after ~60 tokens every sampled token is effectively forced into the closing/digit set. Inside a JSON string (runCode snippets, finalAnswer response text) EOS and true closes are masked or dispreferred, so decode degenerates into exactly the observed signatures: `}7}7}7`, `10101010`, `2025...4567890...` digit floods, and the stray digit in plain text ("would9" — finalAnswer response crossing token 60). The 353s GLM run = burning the remaining ~4000 tokens in the cycle until GuidedGenerationError.incompleteOutput.

    3. Why Qwen is unaffected: its envelopes for these scenarios are compact (typically < 60 tokens, RL-trained on this exact format), so it finishes inside the normal zone; GLM/Devstral write verbose multi-line code and cross into the biased zones. Constrained decode is pure greedy argmax (applyMaskAndSample) — no sampler, no repetition penalty — so once the biased set dominates, the cycle is deterministic.

    4. Root cause of the <tool_call> leak: emitToolCallingEvent's malformed-output fallback emits the RAW outputBuffer (including the `<tool_call>` marker) as .textDelta when JSON parsing fails — which is exactly what happens after a runaway/incomplete round. Symptom 2 is downstream of symptom 1, plus a fallback that leaks markers.

    Fix direction: (a) soft zone biases EOS only (keeps the t3nynaj Int-overflow fix, which worked via the EOS boost, while never corrupting string content — +200 on a masked EOS is a no-op mid-string); (b) drop digits from ClosingTokenBias tier-2 (they never need boosting: when digits are the only legal class the uniform hard-zone suppression preserves their relative order anyway; boosting them sustains digit floods); (c) add a RepetitionCycleTracker (analog of WhitespaceRunTracker) that detects short-period sampled-token cycles and latches a -10000 suppression on the cycling ids so greedy decode must leave the cycle (fixes the hard-zone trap where the boosted brace never closes a string); (d) emitToolCallingEvent fallback emits the unwrapped buffer (markers stripped), never raw `<tool_call>` text; (e) unwrapToolCallMarkers derives its markers from SchemaConverter.ToolCallStructuralTag so constrain/parse provably agree (acceptance criterion b).
  timestamp: 2026-07-16T16:12:21.671959+00:00
- actor: claude-code
  id: 01kxnvc1gnpxngc18r98xnzbdg
  text: |-
    Causality confirmed via git archaeology: the clean GLM baseline the csfnhca card cites (4330528, 2026-07-13, "2/4 with 21 well-formed tool calls, zero corruption") predates ad4c4cf (2026-07-14, "size guided-generation completion-reserve zone off schema complexity", kanban t3nynaj). ad4c4cf changed reserves() from completionReserve = max(structuralReserve*3, maxTokens/4) (soft zone = last ~1024 of 4096) to completionReserve = max(maxTokens - min(structuralReserve*3, maxTokens/2), hardReserve) (soft zone = everything after ~60 tokens for the tool envelope). Every observed runaway/leak run (1fbeb5d, cc1728a) is after ad4c4cf. The structural-tag commits were confounders: cf4fa81 added a second, GLM-specific regression on top; 44a96cf removed that one, but the zone regression remained — exactly why the card says "the csfnhca acceptance criteria passed unit-level but the live behavior is unchanged".

    Plan: keep t3nynaj's early-soft-zone start but make the soft zone bias EOS ONLY (+200 on stop tokens; inert while EOS is grammar-masked, so string content is never corrupted; still flips trivial Int schemas to EOS immediately). Drop digits from ClosingTokenBias tier-2 (hard zone's uniform -10000 already preserves digit order when digits are the only legal class; boosting them sustains digit floods). Add RepetitionCycleTracker (WhitespaceRunTracker analog) to break residual greedy cycles (the `}}}}`-in-string hard-zone trap). Fix the emitToolCallingEvent fallback to never emit `<tool_call>` markers, and derive unwrapToolCallMarkers' markers from SchemaConverter.ToolCallStructuralTag (criterion b).
  timestamp: 2026-07-16T16:15:19.061430+00:00
- actor: claude-code
  id: 01kxnxckzr0kz1ftm7jdxrwxzm
  text: |-
    Implementation landed (TDD, all RED->GREEN cycles verified):

    1. ClosingTokenBias: digits 0-9 removed from tier-2 (they close nothing; when digits are the only legal class the hard zone's uniform -10000 preserves their relative order anyway). New ClosingTokenBias.eosOnlyBoost(stopTokenIDs:count:) builds the +200-at-stop-positions-only array.
    2. GuidedGenerationLoop: soft zone now applies eosOnlyBoost instead of the full closing-token array — string/number content decodes exactly as in the normal zone (a +200 boost on a masked stop token is a no-op), while any grammar-legal stop point still flips to EOS immediately (keeps t3nynaj's Int-overflow fix; pinned by integerSchemaStillStopsViaEOSBoost). Hard zone unchanged apart from the digit removal.
    3. New RepetitionCycleTracker (WhitespaceRunTracker analog, always on in the loop): detects fully-periodic sampled-token runs (period<=12, >=3 cycles, >=12 tokens), latches a -10000 suppression on the cycling ids (finite, so a suppressed id still wins when it is the only grammar-legal token), window reset on FF splices. Breaks the `}}}}`/`}7}7`/digit-cycle hard-zone trap; pinned by hardZoneCycleIsBrokenBySuppression (was: GuidedGenerationError.incompleteOutput after burning the whole budget).
    4. MLXLanguageModel: unwrapToolCallMarkers now derives its markers from SchemaConverter.ToolCallStructuralTag.qwen (single source of truth with the constrain side; ToolCallMarkerAgreementTests proves round-trip agreement for every ToolCallFormat + nil — acceptance criterion b). emitToolCallingEvent's malformed-output fallback now emits malformedToolCallFallbackText (markers stripped) so `<tool_call>` text can never leak into a reply even on a truncated/unparseable round.

    New tests: RepetitionCycleTrackerTests (8), GuidedLoopDegenerationTests (3, real xgrammar constraint + scripted-logits model), ToolCallMarkerAgreementTests (3), eosOnlyBoost tests (2); updated ClosingTokenBiasTests digit expectations and two stale zone-policy doc comments. Full swift test: 0 failures across all targets (Swift Testing: 256+220+78+7 tests; XCTest: 191), only the 3 pre-existing unhandled-file warnings + .build node note.

    Next: real-hardware verification via FoundationModelsMultitool (swift package edit override onto this working tree; per-model pin swaps: GLM-4.7-Flash-4bit, Devstral-Small-2-24B-Instruct-2512-4bit, Devstral-2-123B-Instruct-2512-4bit, Qwen3-30B-A3B baseline 2/4).
  timestamp: 2026-07-16T16:50:35.128242+00:00
- actor: claude-code
  id: 01kxp1z4ntgk8v25k1aptfj23m
  text: |-
    REAL-HARDWARE VERIFICATION COMPLETE (M3 Ultra, FoundationModelsMultitool SearchThenCallTests gated suite, local working tree wired in via a temporary `.package(path:)` override + per-model pin swaps; both reverted afterward — Multitool repo is back to its checked-in state).

    Second root cause found during verification (the card's "announces an action, zero parsed tool calls" variant): the first GLM run post-fix had no runaways but truncated envelopes mid-string at a consistent token offset. Stop-reason logging showed premature "EOS/unk tokenID=154820/154827" — GLM's config.json declares eos_token_id [154820 <|endoftext|>, 154827 <|user|>, 154829 <|observation|>], but makeXgTokenizer registered only `tokenizer.eosTokenId ?? 0` with xgrammar. Unregistered stop ids are ordinary vocab entries whose literal bytes are VALID JSON string content, so the grammar's mask admits them mid-string; the soft zone's EOS boost (and, pre-fix, the old closing bias's +200 tier-1) then makes one the argmax the moment the zone starts -> loop stops on stopTokenIDs.contains -> truncated, unparseable envelope -> fallback text reply. Fix: GrammarTokenizer gained a stopTokenIds: [Int32] designated init (the C shim always accepted an array); makeXgTokenizer now registers the FULL GuidedGenerationLoop.buildStopTokenIDs set (configuration.eosTokenIds + tokenizer.eosTokenId + extraEOSTokens); buildStopTokenIDs made public. Pinned by registeredStopTokensAreNeverSampledAsStringContent (loop-level, RED reproduced the truncation) and XgTokenizerStopRegistrationTests (registration wiring).

    Per-model results (outcome-scored; baseline for the pinned Qwen3-30B is 2/4 per k4mj1gm):
    - GLM-4.7-Flash-4bit: 2/4 (singleCallWeather PASS "It's 31°C (about 88°F) in Austin right now, and it's sunny."; repair PASS with genuine book invocation). composeChain/discovery fail as honest plain-text deflections with real tool attempts (21 toolCalls in compose). Was 1/4 with `}7}7` runaways + 353s scenarios + <tool_call> reply leaks.
    - Devstral-Small-2-24B (mistral3): 1/4 (composeChain PASS — "The warmest city on your trip right now is Austin (ATX) at 31°C." with tripCities+weather genuinely invoked). Others are clean parsed final answers where the model announces intent ("I will check…") instead of calling a tool — model behavior, not corruption. Was 0/4 with digit runaways.
    - Devstral-2-123B (ministral3): 1/4 (repair PASS, genuine book invocation, "Your booking with ID 42 has been confirmed."). Others are honest "couldn't retrieve" deflections with genuinely invoked tools (weather/tripCities). Was 0/4 with `1010` runaways.
    - Qwen3-30B-A3B (baseline): 2/4 (weather + composeChain), failures are honest deflections. Matches the documented 2/4 baseline — behavior unchanged (criterion c).

    All four models, all 16 scenario runs: zero repeated-token runaways (grep for `}7}7`/`101010`/digit-flood signatures over full logs: 0 hits), zero `<tool_call>` text in any reply (grep: 0), zero envelope-parse fallbacks in the unified log (ToolCalling category "failed to parse": 0 after the stop-token fix). Criterion (a) met. Criterion (b) met via ToolCallMarkerAgreementTests (constrain/parse round-trip for every ToolCallFormat + nil). Criterion (c) met.

    Full mlx-swift-lm swift test after all changes: 563 Swift Testing tests + XCTest targets, 0 failures, no new warnings (only the 3 pre-existing unhandled-file warnings + .build node note). Also kept: a permanent error-level ToolCalling log when an envelope fails to parse (this is the diagnostic that cracked the second root cause).
  timestamp: 2026-07-16T18:10:36.346524+00:00
- actor: claude-code
  id: 01kxp31f6q9n98f0nw4hs0hvj8
  text: |-
    Adversarial double-check (really-done gate) returned REVISE with two findings; both fixed, TDD:

    1. (Medium) Cycle-breaker false-positive surface: 12 periodic tokens is inside legitimate output's envelope (a constant JSON array `[0,0,0,…]` is a period-2 run at ~6 elements), and nothing covered the legitimate-repetition side. Fixed: RepetitionCycleTracker default minRunLength 12 -> 16 (a constant array now survives up to 8 identical elements; live degenerate cycles run hundreds-to-thousands of tokens so detection sensitivity is unaffected), tradeoff documented on the initializer, and a new loop-level test `legitimateConstantArraySurvives` pins that `[7,7,7,7,7,7,7,7]` decodes byte-exact (RED under the old threshold: the latched suppression corrupted the tail mid-array). Tracker boundary tests updated to the 15/16 boundary. I chose the threshold-raise over the reviewer's alternative zone-arming suggestion: on the FM tool path the soft zone spans virtually the whole budget anyway, so zone-arming would add plumbing without shrinking the false-positive surface where it matters.

    2. (Low) Hard-zone forcing vs suppression tie: both used -10000, so a latched closer (`}` in the live `}7}7` case) tied with rejected non-closers and the hard zone's "prefer closers" invariant silently degraded. Fixed: new `cycleSuppressionPenalty = -1000` — dominates any raw logit gap (decisively breaks cycles in every zone) while staying 9000 above `logitRejectionPenalty`, so a suppressed closer still outranks every rejected non-closer. Ordering invariants documented on the constant.

    Everything else in the review checked out clean (tracker math, grammar satisfiability, zone/length handling, stop registration semantics, `?? 0` fallback, marker derivation, blast radius of public API changes; reviewer independently re-ran all 24 new/updated tests).

    Post-revision verification: full swift test green again (256+80+221+7 Swift Testing + XCTest targets, 0 failures — the +1 vs the previous run is the new array test). Real-hardware spot re-check with the revised code (GLM-4.7-Flash): singleCallWeather PASS with the identical clean reply ("It's 31°C (about 88°F) in Austin right now, and it's sunny."); repair this run was a stochastic announce-then-stop deflection ("I'll confirm your booking with id 42." — clean parsed text; the same scenario genuinely passed in the earlier full sweep) — zero runaways, zero <tool_call> text, zero parse fallbacks, confirming the revision left live behavior intact (the revised constants only matter when a cycle latches, which never occurred in any hardware run). FoundationModelsMultitool repo restored to checked-in state (git status clean).

    Work complete and green; leaving the task in doing for /review.
  timestamp: 2026-07-16T18:29:21.239446+00:00
- actor: claude-code
  id: 01kxp399101c15fafbwrt5352z
  text: 'Iteration 1: implement landed green in doing — three root causes fixed (soft-zone ClosingTokenBias flooding digits/closers into string content after ~60 tokens [regression from ad4c4cf], greedy cycle trap with no grammar-advancing escape, and GLM''s multi-token eos_token_id list never registered with xgrammar so stop tokens were sampled as JSON content). Parser/marker agreement now derived from SchemaConverter.ToolCallStructuralTag (AC-b provable). Real-hardware gated suite verified: zero runaways, zero <tool_call> leaks on GLM-4.7-Flash, Devstral-Small-2, Devstral-2-123B; Qwen3-30B-A3B unchanged 2/4. Independent /test from scratch: 564 Swift Testing + 196 XCTest = 760 tests, 0 failures, no flake. Dispatching /commit for the checkpoint.'
  timestamp: 2026-07-16T18:33:37.056027+00:00
position_column: doing
position_ordinal: '8180'
title: Repeated-token runaway + tool-call text leak under constrained tool-calling decode for all non-Qwen families (GLM, mistral3, ministral3)
---
## What

After the `csfnhca` fix (44a96cf, keep Qwen structural tag for glm4) and the ministral3 support (cc1728a), real-hardware runs (2026-07-15 evening, revision cc1728a, FoundationModelsMultitool gated suite, M3 Ultra) show the SAME two failure signatures across **three different non-Qwen model families**, while Qwen models are unaffected:

**1. Repeated-token runaway** — constrained decode degenerates into an unbounded repeated-token tail mid-tool-call:
- GLM-4.7-Flash: `…const result = findAPI1}7}7}7}7…` (thousands of `}7`, ~353s); also a stray digit corrupting plain text ("…APIs that would9")
- Devstral-Small-2 (mistral3): `…tools.tripCities20250604123456789012345678901234567890…` (digits forever)
- Devstral-2-123B (ministral3): `…const weatherPromises10101010101010101010…` (alternating `10` forever)

The signature is identical: the snippet text reaches a point where the model's preferred continuation is presumably masked by the grammar, and decode locks into a short repeated token cycle instead of closing the JSON. Looks like the xgrammar constraint + sampler interplay has a degenerate cycle for these tokenizers — worth checking whether the constraint's allowed-token mask at these points admits only digits/braces and repetition-penalty state resets.

**2. Un-parsed `<tool_call>` wrapper leaking into final reply text** — for glm4 and mistral3 models the reply is literally `<tool_call>{"name": "runCode", …` (composeChain, both families, reproducible). The model emits the Qwen-style wrapper (as the structural tag constrains it to), but the per-family parser (GLM4ToolCallParser / MistralToolCallParser) evidently doesn't extract Qwen-style output — parse format and constrain format still disagree at runtime even after 44a96cf. Either the parsers should fall back to the Qwen/JSON parser when the structural tag is the Qwen tag, or the parser selection should follow the tag actually used for constraining, not the model family.

Also observed post-fix (Devstral-Small): the model announces an action ("I will confirm your booking… Please hold on.") and generation ends with zero parsed tool calls — possibly the same parse-miss with an empty wrapper.

Per-scenario logs available in FoundationModelsMultitool session records; all three models' weights fully cached locally for cheap verification (verify against the REAL models, not just unit tests — the csfnhca acceptance criteria passed unit-level but the live behavior is unchanged).

## Acceptance Criteria

- [x] GLM-4.7-Flash, Devstral-Small-2, and Devstral-2-123B each complete all 4 gated FoundationModelsMultitool scenarios with zero repeated-token runaways and zero `<tool_call>` text in final replies, verified on real hardware. (2026-07-16 M3 Ultra sweep: GLM 2/4 outcome-scored, Devstral-Small 1/4, Devstral-2-123B 1/4 — all 16 scenario runs completed with zero runaway signatures and zero `<tool_call>` text; see verification comment for full RESULT lines.)
- [x] Parser selection and structural-tag selection provably agree per model at runtime. (`unwrapToolCallMarkers` now derives its markers from `SchemaConverter.ToolCallStructuralTag`; `ToolCallMarkerAgreementTests` round-trips every `ToolCallFormat` + nil through the constrain-side tag and the parse-side unwrap.)
- [x] Qwen3-30B-A3B behavior unchanged. (Same-run sweep: 2/4 outcome-scored with honest deflections — matches the documented 2/4 baseline from k4mj1gm.)