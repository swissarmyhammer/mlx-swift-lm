---
comments:
- actor: claude-code
  id: 01m23ksk443940vqz12bcysksv
  text: |-
    ### Research 3 — upstream Python mlx-lm (main, read 2026-09-09)

    Files: `mlx_lm/models/cache.py`, `mlx_lm/models/qwen3_5.py`, `mlx_lm/server.py`, `mlx_lm/generate.py`.

    **The recurrent cache has no offset.** Upstream has no `MambaCache` class. The recurrent layers use `ArraysCache` (`cache.py` lines 596-724). It has no `offset` attribute. `advance(N)` (lines 681-685) only decrements `lengths` and `left_padding`. `is_trimmable()` comes from `_BaseCache` (lines 164-165) and returns `False`. `size` returns 0. `can_trim_prompt_cache` (lines 115-119) is `all(c.is_trimmable())`, thus one recurrent layer makes the whole stack non-trimmable, and `trim_prompt_cache` (lines 122-138) returns 0 without change.

    **No upstream recurrent layer advances an offset.** `qwen3_5.py` `GatedDeltaNet.__call__` writes `cache[0]` at lines 160-167 and `cache[1] = state; cache.advance(S)` at lines 197-199. It does not touch `cache.offset`. The same pattern is in `mamba2.py` (lines 120-191), `nemotron_h.py` (156-230) and `falcon_h1.py` (259-329). `TextModel.make_cache` (`qwen3_5.py` lines 345-346) returns `ArraysCache(size=2)` for linear layers and `KVCache()` for attention layers. The attention offset is the only position; the masks read `cache[self.fa_idx]` and `cache[self.ssm_idx]` (lines 287-289).

    Conclusion for defect 1: upstream never needed a Mamba offset because its server never compares per-layer offsets. Our `ExecutorPromptCachePlan.committed` does compare them, thus in our port the Mamba offset must advance, the way `FalconH1.swift` does (`cache.offset += y.dim(1)`).

    **What the server does on a strict extension.** `LRUPromptCache.fetch_nearest_cache` (`cache.py` lines 1652-1672): when the new prompt extends a stored prompt (`result.shorter`), it deep-copies the stored cache and returns `tokens[short_length:]` to feed. No trim is necessary. When the new prompt diverges from a longer stored prompt, it trims only when `can_trim_prompt_cache` is True (line 1661). For a hybrid model that test fails, thus it falls through to a shorter stored prefix (lines 1668-1670) or to a full prefill from zero (line 1672).

    **Where the server stores.** `server.py` lines 814-830 store a copy at the end of the `system` and `user` segments, and lines 861-868 store at the end of generation, keyed on `all_tokens` (prompt + generated). `insert_cache` (`cache.py` lines 1674-1715) keeps every prefix entry for a non-trimmable cache (lines 1697-1703), thus one conversation can hold several prefix snapshots.

    **Generation.** `generate_step` (`generate.py` lines 304-473) prefills all tokens except the last in `prefill_step_size` chunks and has no rewind. The only rewind is `_rewind_cache` in speculative decoding (lines 588-590), and it raises `ValueError` on a non-trimmable cache (lines 533-536).

    **No checkpoint mechanism.** No `checkpoint`, `snapshot` or `rollback` in `cache.py`, `server.py` or `generate.py`. Upstream relies on `copy.deepcopy` of the stored cache at each fetch (lines 1656, 1662, 1670) and on the segment-boundary stores above.

    Conclusion for this card: upstream proves the strict-extension path is the whole answer for a hybrid model. The stored cache is keyed on prompt + generated tokens, and the next prompt must start with that key. That is the design of `ExecutorPromptCache` today.
  timestamp: 2026-09-09T17:35:03.812867+00:00
- actor: claude-code
  id: 01m23kt2egegjs054fz3sj5wp7
  text: |-
    ### Research 4 — llama.cpp server context checkpoints (master, read 2026-09-09)

    Pull requests: #15293 (SWA checkpoints, 2025-08-14), #16382 (hybrid and recurrent checkpoints, 2025-10-03), #16391 (host-memory prompt cache), #20288 (two checkpoints near the end of the prompt, 2026-03-10), #22929 (checkpoints before the last user message), #24411 (skip checkpoints past `pos_next`).

    **What a checkpoint stores.** Only the part of the memory that cannot roll back. Flag `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (= `SWA_ONLY` = 1, `include/llama.h` lines 906-918). `src/llama-memory-hybrid.cpp` lines 190-202: with the flag, `state_write` skips `mem_attn` and writes `mem_recr` only. Thus for a hybrid model the checkpoint is the recurrent state alone; the full-attention KV cache stays live and rolls back with `seq_rm`. The server passes the flag at `tools/server/server-context.cpp` lines 2363-2364.

    **Where.** `server_slot::prompt.checkpoints`, a `std::list<common_prompt_checkpoint>` in host RAM (`tools/server/server-task.h` lines 566-586; `common/common.h` lines 1165-1179). Each entry holds `n_tokens`, `pos_min`, `pos_max`, `id_task`, and byte vectors `data_tgt`, `data_dft`, `data_spec`. The list moves with the prompt into the `--cache-ram` cache (`server-task.cpp` lines 1779-1788).

    **How many.** `n_ctx_checkpoints`, default 32 (`common/common.h` line 629; was 3 in #15293). `checkpoint_min_step`, default 8192 tokens (line 631). Eviction in `create_checkpoint` (`server-context.cpp` lines 2309-2352): when full, cull other tasks' entries within `min_step` of the previous kept entry, then drop the oldest, then replace a same-position duplicate.

    **When.** Before `llama_decode()` of a prompt batch, never during generation (`server-context.cpp` lines 3631-3635). Positions: the start of the last user message, earlier user-message starts more than `min_step` after the last checkpoint, `4 + n_ubatch` and 4 tokens before the end of the prompt, and the final prompt batch (lines 3549-3628). Detection is a runtime probe `common_context_can_seq_rm` (`common/common.cpp` lines 1583-1616) that returns `FULL` or `RS`, not `llama_model_is_recurrent`.

    **Restore.** `server-context.cpp` lines 3349-3385: search from newest to oldest for the first entry with `pos_max <= pos_next` (the common prefix) and `pos_min < pos_next - n_swa` or `pos_min == 0`. Load it with the partial flag, set `n_past` to its token count, `seq_rm` the memory after it, and reprocess from there to the end of the new prompt.

    **No checkpoint.** Lines 3379-3384: `n_past = 0`, full reprocess, and erase every checkpoint past `pos_next` (lines 3388-3399).

    Conclusion for this card: this is the design of our commit `e78994c` (snapshot the recurrent state at a transcript-stable boundary, before generation). The important detail: llama.cpp puts the checkpoint 4 tokens BEFORE the prompt end, on purpose, so a `<think>` priming sequence the template rewrites in history does not sit inside the checkpoint. That is the seam commit `e78994c` solved with `addGenerationPrompt:false`.
  timestamp: 2026-09-09T17:35:19.504301+00:00
- actor: claude-code
  id: 01m23kthmmdbfjwfy4dd8r6ztn
  text: |-
    ### Research 5 — vLLM hybrid prefix caching (main, read 2026-09-09)

    Pull requests: #25752 (Mamba2 automatic prefix caching, "all" mode), #30877 ("align" mode, now the default; covers GDN, LinearAttention, ShortConv), #45845 (retention interval), tracking issue #26201, limits in issues #45238 and #40696.

    **Block.** A Mamba block is one page that holds one full snapshot of the recurrent state of one layer (`vllm/v1/core/kv_cache_utils.py` line ~1710). `vllm/config/cache.py` lines 161-173 define `mamba_block_size` and `mamba_cache_mode` ("all" = state at every `i * block_size`; "align" = only the last block-aligned token of a scheduler step). `vllm/platforms/interface.py` `_align_hybrid_block_size` (lines ~540-630) grows the attention block until one attention page is at least one Mamba page; issue #40696 reports 528 tokens for Qwen3.5.

    **Snapshot positions.** "all" mode: `mamba_mixer2.py` lines 547-625 write every `chunk_stride`-th chunk state of the chunked scan into the block for that boundary. "align" mode (default, and the only mode for GDN: `gdn_attn.py` line 234; PR #36649 for GDN "all" was closed): `vllm/v1/kv_cache_interface.py` lines 1068-1078 `get_mamba_prefill_checkpoint_position` = `floor((P-1)/block_size) * block_size`, plus the running state; `single_type_kv_cache_manager.py` `MambaManager.allocate_new_blocks` (lines ~1395-1517) allocates only the last state block and fills the rest with null blocks.

    **Hit.** `MambaManager.find_longest_cache_hit` (lines ~1097-1172) scans from right to left and takes the first block whose hash is stored; it needs no chain, because a Mamba state at N holds all of 0..N. A hit exists only where a real block was written. `HybridKVCacheCoordinator.find_longest_cache_hit` (`kv_cache_coordinator.py` lines 554-680) truncates the attention hit to the Mamba hit, thus a Mamba miss vetoes the whole hit (issue #45238).

    **Divergent prefix.** `mamba_mixer2.py` lines 533-541 restore the state at `block_idx_last_computed_token` and scan the rest. In "align" mode a shared prefix that ends before the one checkpoint gives zero hits (issue #45238: 52/64 to 0/64 by moving the split 100 tokens).

    **Memory argument.** `MambaSpec.max_memory_usage_bytes` (`kv_cache_interface.py` lines ~1075-1085): "all" = `max_model_len / block_size` pages per request, "align" = 2 pages. PR #45845 measured dense snapshots at block 128: 80% of the KV pool, hit rate 85% -> 75%, throughput -18%, p99 3x.

    **Conclusion for an agent loop.** One snapshot at the end of each round is what vLLM ships as the default. The next round's prompt always contains the whole previous round (prompt + output), thus the hit lands at the newest snapshot and vLLM recomputes only the new tail. More positions help only for history edits, compaction, or shared system prompts across requests. Our snapshot needs no block grid: take it at the exact end of the round (the live caches after the last generated token), and the next round recomputes nothing before its new tokens. That is what `ExecutorPromptCachePlan.committed` already keeps, provided the ledger and the caches agree.
  timestamp: 2026-09-09T17:35:35.060638+00:00
- actor: claude-code
  id: 01m23mead5t8meb7zm5w1becq5
  text: |-
    ### Research 1 — this fork's own history (e78994c, 83c43e8, 5891f01)

    Recovered to the scratchpad with `git show`: `PromptCache.swift` (1371 lines), `PromptCacheChunks.swift` (732), `MLXLanguageModel.swift` at 5891f01 (4839), `TranscriptConverter.swift`, `ReasoningConfig.swift`, `Evaluate.swift` at 83c43e8, and `IntegrationTesting/.../TextGeneration/PromptCacheHybridReuseTests.swift` at 5891f01.

    **The old hybrid checkpoint.** `PromptCacheChunks.swift` lines 292-315: `HybridCheckpoint { tokens: [Int]; layers: [(kind, state: [MLXArray])]; byteSize; lastUsed }`, one owned copy of every layer's raw `state` (`snapshotHybridCheckpoint`, lines 347-376; `ownedCopy` lines 136-140 = `MLXArray(data: asData(.copy))`). Keyed by a content hash of the FULL token prefix (`PromptCache.swift` lines 133-141, 1007). Stored per model in the process-global `PromptCache` actor (`hybridCheckpoints: [String: [ChunkKey: HybridCheckpoint]]`, line 148), not per session, with byte-budget LRU eviction (lines 189-215, 1118-1121, 1222-1227). Resolution (`resolveHybridCheckpoint`, lines 1038-1064) took the LONGEST checkpoint that is a strict prefix of the new prompt and restored it into a fresh `KVCacheSimple`/`MambaCache` stack by copying again (`restoreHybridCheckpoint`, `PromptCacheChunks.swift` lines 407-423), thus the store kept its own copy and generation never wrote into it.

    **The transcript-stable boundary (e78994c).** `transcriptStableLength` (`MLXLanguageModel.swift` at 5891f01, lines 2990-3006) rendered the same messages with `addGenerationPrompt: false` and took the common prefix with the real prompt. `makeSplitPrefillSlot` (lines 2914-2943) prefilled `[matched, stable)` with `prefillPromptCache` (lines 3028-3046, 512-token steps), stored a checkpoint keyed on `promptTokens[0..<stable]` BEFORE generation, then fed the priming suffix `<|im_start|>assistant\n<think>\n` as the generation input. Reason: the Qwen 3.6 template stripped `<think>` from a past turn, thus round N's render was never a prefix of round N+1's.

    **The stop token (83c43e8).** `Evaluate.swift` lines 1887-1906 set `stopTokenFedToCache` when the loop discards a stop token whose forward pass already ran; `GenerateCompletionInfo.stopTokenFedToCache` (line 2035); `PromptCache.planCacheStore` (lines 575-594) answered `.storeExtended(fedStopToken:)` for a non-trimmable cache, thus the key was `prompt + generated + [stop]`, which is the cache's real position.

    **`preserve_thinking` replay (5891f01).** `ReasoningConfig.historyPreservationKey` (lines 100-117) named the template kwarg; the `qwen3_5` row set `"preserve_thinking"` (lines 177-181). `TranscriptConverter.mlxMessages(for:toolCallFormat:replayReasoning:)` (lines 44-83) attached a `.reasoning` entry's text to the NEXT `.response` as `Chat.Message.reasoning` (`reasoning_content`). Think-then-call reasoning before a `.toolCalls` entry was dropped on purpose (lines 55-59). The executor gated replay on `historyPreservationKey != nil` (lines 1496-1499).

    **The old integration test.** `PromptCacheHybridReuseTests` (302 lines): two rounds, no tools, `Qwen3.6-27B-mxfp4`, bound `cachedTokenCount >= promptTokenCount + outputTokenCount - 8` in suppressed and thinking modes (lines 191-193, 285-287).

    **What `ExecutorPromptCache.swift` has today, and what it lacks.**
    Has: (1) a ledger of prompt + every generated token, widened to the real cache position, stop token included, because `RecordingGeneratedTokens` records the token before the stop check (`Evaluate.swift` line 2517) and `committed(generatedTokens:)` (`ExecutorPromptCache.swift` lines 239-251) takes `position - promptTokens.count` of them — this IS the `.storeExtended` idea of 83c43e8, with no plan needed; (2) the strict-extension path through `reusablePromptPrefix` → `ExtendCachedPrefixRule` (`PromptCacheReusePolicy.swift` lines 176-192, 281-308), which needs no trim; (3) one live cache per session, checked out and in (lines 55-111), thus no copy at all — cheaper than the old double copy.
    Lacks: (1) a position on the recurrent caches (defect 1, `Qwen35.swift` lines 340-350, 779-789, 1042-1045; `MLXVLM/Models/Qwen35.swift` lines 697-700 — `cache.advance(S)` only touches `lengths`/`leftPadding`, `ArraysCache.advance`, `KVCache.swift` lines 1424-1431; compare `FalconH1.swift` line 527); (2) reasoning replay in `TranscriptConverter.swift` (lines 61-68 drop `.reasoning`); (3) a protocol rule in the executor path (`reusablePromptPrefix` "consults no protocol rule", lines 267-271); (4) a stable-boundary checkpoint. Item 4 is NOT needed for Qwen 3.5: its template (`chat_template.jinja` line 130) keeps `<think>` in a past turn when `preserve_thinking` is undefined, thus round N's render IS a prefix of round N+1's once the reasoning is replayed. Research 6 measures that with the real tokenizer.
  timestamp: 2026-09-09T17:46:23.013487+00:00
- actor: claude-code
  id: 01m23mf5sqdqkmydnfp82nea4c
  text: |-
    ### Research 2 — the DeepSeek-V4 precedent (12223aa, f661489) and the decision for Qwen

    **`DSMLCommittedTurnRule`** (`Libraries/MLXLMCommon/Tool/Parsers/DSMLCommittedTurnRule.swift`, 159 lines). It claims a turn when (lines 91-103): the new render starts with the render of the previous prefill (`cache.previousRenderTokens`), the ledger is aligned, the render adds exactly ONE end-of-sentence commit after the previous render (`soleCommitIndex`, lines 126-131), and the ledger ends at that commit (`suffixStart`, lines 141-158: after the commit when the ledger holds it, AT the commit when the generation stopped on budget or a speculative iterator returned it). It answers `.appendSuffix(suffixStart:representedTokens: cachedTokens + render[suffixStart...])`, thus the live trajectory stays and only the new tail is fed. `ChatSession.swift` records `previousRenderTokens` for each prefill. `ToolCallFormat.promptCacheReuseRules(tokenizer:)` (`ToolCallFormat.swift` lines 246-258) contributes it for `.dsml`; `.qwen35` contributes nothing.

    **f661489** made `committed(generatedTokens:)` widen the ledger to the real cache position instead of rewinding, and made the three executor paths hand the generated tokens over. That is in place.

    **Does Qwen need a committed-turn rule?** Yes. The Qwen 3.5 template re-renders a past assistant turn from PARSED data, and four things make that render differ from the tokens the model wrote:
    1. `reasoning_content|trim` and `content|trim` (`chat_template.jinja` lines 117, 129): leading and trailing whitespace the model wrote is gone.
    2. Tool-call arguments: `TranscriptConverter.swift` lines 76-92 decode `call.arguments.jsonString` into `[String: JSONValue]`, a Swift dictionary with no order, and the template writes `<parameter=...>` in `tool_call.arguments|items` order (lines 150-155). A call with two or more arguments (every file edit of `acp-agent`) can come back in another order. `ToolCall.Function.argumentsJSON` (`Chat.swift` lines 309-324) keeps the model's bytes only when the parser fills it.
    3. Non-string values go through `tojson` (line 152), thus `42` and `true` come back canonical, and a value the model wrote differently does not.
    4. Non-canonical token splits, which card `^v7z7v99` measured on DeepSeek-V4, can happen on any BPE model.
    None of those can be corrected by a render. The DSML rule's shape corrects all of them, because it never compares the generated region: it splices after the `<|im_end|>` (id 248046, the tokenizer `eos_token`) that closes the assistant turn.

    **Difference from DSML.** A Qwen tool round adds TWO `<|im_end|>` after the previous render: the assistant's own, then the one that closes the `<tool_response>` user turn (template lines 160, 169-171). The Qwen rule therefore takes the FIRST commit after the previous render, not the sole one. That is safe: the model's own turn cannot hold `<|im_end|>` (generation stops on it), and every later one belongs to a later message.

    **Reasoning replay is ALSO required**, not an alternative. Round 1's prompt ends with `<|im_start|>assistant\n<think>\n` (template lines 177-183). The history render of that turn writes `<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}` (line 131) only when `reasoning_content` is present; `TranscriptConverter` drops it today (lines 61-68), thus the render writes `<think>\n\n</think>` and the prefix check `render.starts(with: previousRender)` still holds (the first `<think>\n` is the same), but the cold render of the conversation no longer holds what the model read. The done-when criterion "the generated text of a cached round matches a cold round" needs the cold render to carry the reasoning. The template preserves thinking by default (`preserve_thinking is undefined` at line 130), thus no `historyPreservationKey` kwarg is necessary; only the `reasoning_content` key must be present.

    **Does the executor path stay rule-free?** No. `reusablePromptPrefix` gets `previousRenderTokens` and `protocolRules`, `ExecutorPromptCacheEntry` records the render of the last pass, and `ExecutorPromptCachePlan.make` passes `toolCallFormat.promptCacheReuseRules(tokenizer:)`. The unit tests of `ExecutorPromptCacheTests` keep the attention-only behaviour unchanged (no rule contributed → same decisions).

    **Decision (the shape of the fix):**
    1. Advance `MambaCache.offset` in both `Qwen35.swift` GDN paths (MLXLLM: prefill, `decodeLinearLayer`, the fused decode schedule; MLXVLM: prefill), and keep the speculative checkpoint offset in step. `Qwen35MTP.swift` builds `forceFullAttention: true` layers with `KVCacheSimple` only (lines 28-41), thus it has no recurrent cache to advance.
    2. Keep the rule of `committed(generatedTokens:)` — every cache reports one position — and add the hybrid tests the card names.
    3. Replay `reasoning_content` onto the next assistant entry (`.response` OR `.toolCalls`), gated by a new `ReasoningConfig` flag that Qwen 3.5 sets.
    4. Add `QwenCommittedTurnRule` for `.qwen35`, and let the executor consult protocol rules.
    5. No stable-boundary snapshot: the template needs none, and vLLM's default proves one snapshot at the end of the round is enough.
  timestamp: 2026-09-09T17:46:51.063675+00:00
- actor: claude-code
  id: 01m23pf6rw228vn04twsb4hfj2
  text: |-
    ### Progress — unit level green

    Production code now in the tree:

    - `Libraries/MLXLMCommon/KVCache.swift`: `MambaCache.advancePosition(by:)` moves `offset` with the lengths; `saveSpeculativeCheckpoint(advancedBy:)` records the position after the input.
    - `Libraries/MLXLLM/Models/Qwen35.swift` (GDN prefill, `decodeLinearLayer`, fused `decodeStep`) and `Libraries/MLXVLM/Models/Qwen35.swift` (GDN): every recurrent layer calls `advancePosition(by:)`.
    - `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`: `reconcilePromptCache(promptTokens:cachedTokens:previousRenderTokens:caches:protocolRules:)` returns a `PromptCacheReuse` (suffix start plus the tokens the cache represents after the feed). `reusablePromptPrefix` is a wrapper on it.
    - `Libraries/MLXLMCommon/Tool/Parsers/CommittedTurnSplice.swift` (new, shared body), `DSMLCommittedTurnRule.swift` (uses the shared body, `.sole` commit), `QwenCommittedTurnRule.swift` (new, `<|im_end|>`, `.first` commit after the previous render).
    - `Libraries/MLXLMCommon/Tool/ToolCallFormat.swift`: `.qwen35` contributes `QwenCommittedTurnRule`; `promptCacheReuseRules(tokenizer:)` is `package`.
    - `Libraries/MLXLMCommon/ReasoningConfig.swift`: `replaysReasoningIntoHistory` (default `false`). `QwenReasoningProtocol.qwen35` sets it; both Qwen 3.5 model classes and the VLM class declare it.
    - `Libraries/MLXFoundationModels/TranscriptConverter.swift`: `mlxMessages(for:replayReasoning:)` attaches a `.reasoning` entry to the assistant message after it.
    - `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`: the entry records `renderTokens`; the plan records `representedTokens`; `make` consults the protocol rules; `committed` builds the ledger from `representedTokens`; the store has `peek`.
    - `Libraries/MLXFoundationModels/MLXLanguageModel.swift`: the three plan sites pass `promptCacheReuseRules(of:)`; `runRespond` resolves the configuration before it renders and re-renders with reasoning when the protocol replays it.

    Tests (all written before the code, red first):

    - `Tests/MLXLMTests/Qwen35RecurrentCachePositionTests.swift` (6), `QwenCommittedTurnRuleTests.swift`, `PromptCachePrefixReuseTests.swift` (+3), `ReasoningConfigTests.swift` (+3), `MarkerTokenizer.swift` (shared fixture).
    - `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift` (+4), `ExecutorPromptCacheTests.swift` (hybrid stack, Mamba at offset 0, splice, commit after splice, render on record, `peek`).
    - `ChatConventionsTests.swift` and `Qwen35ContinuationTests.swift` now expect `QwenReasoningProtocol.qwen35`.

    Bundle results (`swift build --build-tests`, then `xcrun xctest`):

    - MLXFoundationModelsTests: 183 tests, 0 issues.
    - MLXLMTests: Swift Testing 1021 tests, 101 issues; XCTest 639 tests, 35 failures. Both numbers are the baseline of this tree before my change (SSMTests, the VL ContinuationTests, NanbeigeTests, Qwen35MTPMetalTests). I confirmed the baseline by a run with my test files stashed.
    - MLXGuidedGenerationTests: 70 tests, 0 issues. CXGrammarTests: 7, 0. MLXHuggingFaceMacrosTests: 5, 0.

    Next: Research 6 (tokenizer seam diff), the real-weights test, the SearchTools session count, the Router test, the SWE-bench run.
  timestamp: 2026-09-09T18:21:49.212926+00:00
- actor: claude-code
  id: 01m23rh9hy2t6tvs6vncy222sh
  text: |-
    ### Research 6 — the real render seam on `mlx-community/Qwen3.8-27B-mxfp4`

    Source: the tokenizer and chat template of the checkpoint (`~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-mxfp4/snapshots/*/chat_template.jinja`), read through the new real-weights test `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift`, which prints the first token where each round's render parts from the ledger the round before it left (`ExecutorPromptCacheStore.peek`).

    1. `preserve_thinking`: line 116 of the template, `{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}`. Nothing in this tree sets the key, thus it is undefined and the template keeps the `<think>` block of EVERY past assistant turn: `<|im_start|>assistant\n<think>\n{reasoning_content|trim}\n</think>\n\n{content}` (line 117), then the tool calls as `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` (lines 121-144), then `<|im_end|>\n` (line 146). A tool result renders as `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n` (lines 147-158). The generation prompt ends with `<|im_start|>assistant\n<think>\n` (lines 163-170).

    2. The seam BEFORE the fix (run 1, 2026-09-09 13:31): round 1 rendered 27434 tokens; round 2 rendered 27500 and parted from the 27581-token ledger at index 27433, the LAST token of round 1's render. The render held `\n\n</think>\n\n<tool_call>\n<function=get_stock_level>` where the ledger held `The user is asking me to look up bays`. The `<think>` block was EMPTY in the render: `<think>` + `\n` + `` + `\n</think>` tokenizes `\n\n` as one token, thus the previous render was no longer a prefix of the new one, `QwenCommittedTurnRule` declined, `ExtendCachedPrefixRule` declined, and a `MambaCache` cannot rewind, thus every round fed the whole prompt (27500, 27566, 27634, 27702 tokens; 60 to 91 s of prefill each).

       Cause: `Qwen3VLMessageGenerator.generate(message:)` (`Libraries/MLXVLM/Models/Qwen3VL.swift`, lines 2034-2049) called `addToolMetadata(to:for:)` and never `addReasoningMetadata(to:for:)`, although the doc of `MessageGenerator.addReasoningMetadata` (`Libraries/MLXLMCommon/Chat.swift`, lines 293-305) says a custom generator must call both. The Qwen 3.5/3.8 VL checkpoints load through `VLMModelFactory` with the Qwen 3 VL processor, thus `reasoning_content` never reached the template. Fixed with one call and a unit test (`Tests/MLXLMTests/UserInputTests.swift`, `testQwen3VLMessageGeneratorReplaysReasoning`, red then green).

    3. The seam AFTER that fix (run 2, 13:42): round 2 reused the cache and the model threw `ContinuationStateError.missingState(model: "Qwen35", key: "qwen35.ropeDeltas")` from `Libraries/MLXVLM/Models/Qwen35.swift` line 1231 (`QwenVL.continuationAnchor`, `Libraries/MLXVLM/Models/QwenVL.swift` lines 25-40). The VL model refuses a warm cache without the M-RoPE anchor that its prefill left in `LMOutput.State`, and the executor carried the caches without that state. `ChatSession` carries it (`Libraries/MLXLMCommon/ChatSession.swift` lines 1213-1217 and 1338-1340: `lmState = iterator.state` after the iterator is built). Fixed: `ExecutorPromptCacheEntry.state`, `ExecutorPromptCachePlan.state`, `committed(generatedTokens:state:)`, `reconcilePromptCache(... carriesModelState:)` (a carried state forbids a rewind, as `RewindToCommonPrefixRule` requires), and `generateTaskRecordingTokens` / `generateProtocolTokensTask` (`Libraries/MLXLMCommon/Evaluate.swift`) hand the post-prefill state back through a `preparedState` callback. Unit tests: `ExecutorPromptCacheTests` (+4), `PromptCachePrefixReuseTests` (+2), red then green.

    4. The seam AFTER both fixes (run 3, 14:02): every round's render extends the ledger WHOLE. Round 2 fed 35 tokens: `\n<|im_start|>user\n<tool_response>\n{"bay":"bay 3","pallets":4172,"status":"sealed"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n`. The template writes the model's own reasoning, its XML tool call (one parameter, thus no argument reorder) and its `<|im_end|>` token for token, thus `QwenCommittedTurnRule` and `ExtendCachedPrefixRule` agree on this model and the splice is not needed for this transcript. The rule stays: a call with two or more arguments, or reasoning the `|trim` changes, parts the two streams inside the generated region, and the rule then splices after the `<|im_end|>` the model wrote.
  timestamp: 2026-09-09T18:57:54.750327+00:00
- actor: claude-code
  id: 01m23vwfmkztb7b0pmma7wk7zs
  text: |-
    ### Measurements — real weights, 2026-09-09

    All runs: `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -destination 'platform=macOS' -parallel-testing-enabled NO -only-testing:IntegrationTestingTests/<Suite>/<test>()` on this tree (branch `stable`, uncommitted), greedy decoding, temperature 0, thinking on, tools enabled on every round.

    **`Qwen35AgenticPromptCacheAssessmentTests/hybridModelCarriesThePromptCacheAcrossToolRounds()` — `mlx-community/Qwen3.8-27B-mxfp4` (run 3, 14:02, PASSED, 162.8 s)**

    | round | rendered prompt tokens | fed prompt tokens | cachedTokenCount | prefill seconds | generated tokens | round seconds | emitted | first divergent token |
    |---|---|---|---|---|---|---|---|---|
    | 1 | 27434 | 27434 | 0 | 68.03 | 146 | 75.58 | tool call get_stock_level {"bay":"bay 3"} | no ledger before this round |
    | 2 | 27616 | 35 | 27581 | 0.275 | 49 | 3.02 | tool call get_stock_level {"bay":"bay 7"} | none, the render extends the 27581-token ledger whole |
    | 3 | 27701 | 35 | 27666 | 0.270 | 49 | 3.10 | tool call get_stock_level {"bay":"bay 11"} | none, the render extends the 27666-token ledger whole |
    | 4 | 27787 | 36 | 27751 | 0.288 | 50 | 3.14 | tool call get_stock_level {"bay":"bay 15"} | none, the render extends the 27751-token ledger whole |
    | 5 | 27874 | 36 | 27838 | 0.294 | 51 | 3.23 | text | none, the render extends the 27838-token ledger whole |
    | cold control of round 5 | 27874 | 27874 | 0 | 68.78 | 51 | 72.11 | text | no ledger |

    The cached round 5 and the cold control share 32 of the first 32 generated tokens. Every "Done when" bound of the card holds: cachedTokenCount of round N+1 ≥ rendered tokens of round N − 16 (it is +147, +50, +50, +51 above the previous render); fed tokens after round 1 = 35 or 36 (< 2000); round 4 prefill 0.288 s < 2 × round 2 prefill 0.275 s; the transcript of round 5 holds 27874 tokens (> 20000); four tool rounds.

    **Same test, `controlModelCarriesThePromptCacheAcrossToolRounds()` — `mlx-community/Qwen3-4B-4bit` (13:57, PASSED, 92.8 s)**

    | round | rendered | fed | cachedTokenCount | prefill s | generated | round s | emitted |
    |---|---|---|---|---|---|---|---|
    | 1 | 26473 | 26473 | 0 | 16.98 | 676 | 30.59 | tool call {"bay":"3"} |
    | 2 | 26526 | 53 | 26473 | 0.154 | 258 | 5.64 | tool call {"bay":"7"} |
    | 3 | 26579 | 53 | 26526 | 0.151 | 328 | 7.06 | tool call {"bay":"11"} |
    | 4 | 26634 | 55 | 26579 | 0.152 | 456 | 9.78 | tool call {"bay":"15"} |
    | 5 | 26689 | 55 | 26634 | 0.140 | 423 | 9.05 | text |
    | cold control | 26689 | 26689 | 0 | 16.13 | 353 | 29.41 | text |

    Cached round 5 and cold control share 32 of 32 generated tokens. The Qwen 3 template drops the `<think>` block of a past turn, thus each render parts from the ledger at `<think>` (the ledger holds `<think>\nOkay, the user wants...`, the render holds `<tool_call>\n{"name": ...`); the pure-attention caches rewind to the common prefix, which is exactly the previous render, and `cachedTokenCount == previous rendered` on every round.

    **`Qwen35SessionPromptCacheTests/aSecondTurnOfAFrameworkSessionReusesTheFirstTurn()` — a real `LanguageModelSession` on the 27B (14:12, PASSED)**: turn 1 rendered 73, generated 27; turn 2 rendered 126, cachedTokenCount 101, the render extends the 101-token ledger whole (73 prompt + 27 generated + the `<|im_end|>` the iterator fed). The framework's transcript order is `instructions prompt response reasoning` — the reasoning entry FOLLOWS the response entry — which the converter now handles (see the next comment).

    **`FoundationModelsRouter` `secondTurnReusesFirstTurnsKVCache` with `sessionBackendModel = "mlx-community/Qwen3.8-27B-mxfp4"`** (`swift package edit mlx-swift-lm --path <this tree>` in `IntegrationTests`, then `swift test --package-path IntegrationTests --filter secondTurnReusesFirstTurnsKVCache`; 14:26, PASSED): `turn1In=73 turn1Out=28 turn2Cached=101`. Before the stop-token fix below it printed `turn1Out=27 turn2Cached=101` and failed its upper bound `cached <= turn1In + turn1Out` by the one stop token. The Router edit and the model constant are reverted; nothing in the Router repository is changed.
  timestamp: 2026-09-09T19:56:27.155617+00:00
- actor: claude-code
  id: 01m23vy8rjxbh7n7we9nsgqxm9
  text: |-
    ### Defects the real-weights runs found after the unit-level work, each fixed red-then-green

    1. **`Qwen3VLMessageGenerator` dropped `reasoning_content`** (`Libraries/MLXVLM/Models/Qwen3VL.swift`, `generate(message:)`). The Qwen 3.5 / 3.8 VL checkpoints render through this generator, thus the replayed reasoning never reached the template and every history `<think>` block was empty. Test: `Tests/MLXLMTests/UserInputTests.swift` `testQwen3VLMessageGeneratorReplaysReasoning`.

    2. **The executor carried no model state with the caches.** The VL model keys its M-RoPE anchor in `LMOutput.State` and throws `ContinuationStateError.missingState(model: "Qwen35", key: "qwen35.ropeDeltas")` on a warm cache without it. `ExecutorPromptCacheEntry.state`, `ExecutorPromptCachePlan.state`, `committed(generatedTokens:state:)`, `ExecutorPromptCacheSlot.commit(_:generatedTokens:state:)`, `reconcilePromptCache(... carriesModelState:)` (a carried state forbids a rewind), and the `preparedState` callback of `generateTaskRecordingTokens` / `generateProtocolTokensTask`. Tests: `ExecutorPromptCacheTests` (+4), `PromptCachePrefixReuseTests` (+2).

    3. **The framework appends the `.reasoning` entry AFTER the `.response` entry** (`instructions prompt response reasoning`, measured with a real `LanguageModelSession`). The converter attached reasoning to the NEXT assistant entry only, thus a plain answer turn lost its reasoning again. `TranscriptConverter.mlxMessages(for:replayReasoning:)` now hands a reasoning entry to the assistant message that follows it before any other entry, or else to the assistant message right before it (`ReasoningReplay`). Tests: `TranscriptConverterTests` (+2). Real-weights test: `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift` (new, drives a `LanguageModelSession`).

    4. **The stop token the iterator fed was reported nowhere.** `generateLoopTask` (`Libraries/MLXLMCommon/Evaluate.swift`) declared `stopTokenFedToCache` and a `handleStopToken` helper that nothing called; the inline stop check never set the field, and `GenerateCompletionInfo.withRejectedToolCallCount` dropped it on the copy. Now the inline check sets it whenever `includeStopToken` is false, the dead helper is gone, and the copy keeps it. The executor's output count adds that token back (`MLXLanguageModel.Executor.generatedTokenCount(of:)`), thus `prompt + output` of a turn equals what the next turn can reuse, which is the bound the Router test holds. Tests: `Tests/MLXLMTests/GenerateLoopStopTokenTests.swift` (new, 2), `Tests/MLXFoundationModelsTests/ExecutorUsageOutputTests.swift` (new, 2).

    5. **`GenerationEvent.completion`** (test mirror only, `MLXLanguageModel.swift`): the channel has no event for prefill seconds, thus the real-weights test reads `GenerateCompletionInfo.promptTime` through the observer.

    **SearchTools sessions and the parent (`ExecutorPromptCacheStore.maximumRetainedSessions = 4`)**: read `FoundationModelsMultitool/Sources/FoundationModelsMultitool/Discovery/SearchToolsTool.swift` (lines 255-310) and `SampleSnippet.swift` (lines 119-137). One `searchTools` call opens ONE selection session (grammar-guided; the guided path calls `carriesNoCache()` and commits no entry, thus it takes no store slot) and at most ONE sample-snippet session (`SampleSnippet.generate` opens one session and reuses it across its repair attempts; a plain `respond`, thus one store slot on the flash model). The parent's entry is checked OUT for the whole of its own pass and checked IN between rounds, most recent first. The parent thus leaves the store only when four or more other sessions check in between two of its rounds, which needs four sample-generating tool calls inside one tool body. One `searchTools` call per round, the common case, cannot evict it.

    **Build note**: the `swiftbuild` incremental build keyed on content, thus a `touch` rebuilds nothing; `rm` of an object or a bundle forces the rebuild. I rebuilt `MLXLMTests.xctest` and `MLXFoundationModelsTests.xctest` from scratch before the final runs.
  timestamp: 2026-09-09T19:57:25.650924+00:00
- actor: claude-code
  id: 01m23x12nmnyv5bw7zrb2k58dt
  text: |-
    ### SWE-bench instance in `FoundationModelsACPAgent` — round times

    Command: `swift package edit mlx-swift-lm --path /Users/wballard/github/swissarmyhammer/mlx-swift-lm`, `swift build -c release`, then `uv run bench/swebench_run.py <scratch>/swebench-preds.jsonl --limit 1 --verbose` (2026-09-09 14:26 to 14:59, instance `astropy__astropy-12907`, models: standard `mlx-community/Qwen3.8-27B-mxfp4`, flash `mlx-community/Qwen3-4B-4bit`). The edit is reverted (`swift package unedit`); the repository's own `preds.jsonl` and `run.log` are untouched.

    The verbose event lines carry no clock, thus the round times come from the ULIDs of the tool-call identifiers (the first ten characters of a ULID hold the millisecond of creation). The agent made 27 `runCode` calls in one turn. Seconds from one tool call to the next — the tool's own run time PLUS the model's next round (prefill, thinking, generation):

    | calls | seconds between consecutive calls |
    |---|---|
    | 1-5 | 4.6, 8.0, 7.8, 8.4 |
    | 6-10 | 246.9, 11.2, 8.9, 19.1, 89.5 |
    | 11-16 | 7.8, 62.2, 7.9, 7.5, 11.8, 7.9 |
    | 17-21 | 564.8, 46.6, 54.9, 73.6, 103.3 |
    | 22-27 | 22.8, 15.1, 14.1, 13.1, 15.7, 12.6 |

    The long gaps (246.9 s, 89.5 s, 62.2 s, 564.8 s) are `runCode` bodies that ran the astropy checkout (the sandbox builds and tests the package); the log cannot split tool time from model time. The rounds with a short tool body stay flat: 7.5 to 8.4 s at calls 2-5 and 12.6 to 15.7 s at calls 23-27, after twenty-seven tool rounds on a transcript that holds the whole session. A cold prefill of a transcript this long costs 60 to 90 s on this model (the cold control above: 68.8 s for 27874 tokens), thus a session that lost its cache would show every late round above one minute. It does not.

    The instance ended with an EMPTY patch (`stop end_turn` after 1992 s; 0 patches, 1 empty). That is the agent's answer quality on this instance, not the cache: the turn ran to its own end, and no round timed out.
  timestamp: 2026-09-09T20:16:26.292818+00:00
- actor: claude-code
  id: 01m23x1sdj3srg0vzjcg2x9qpw
  text: |-
    ### implement — changed

    **What changed** (all in `/Users/wballard/github/swissarmyhammer/mlx-swift-lm`, uncommitted, `swift-format` applied to every changed file):

    - `Libraries/MLXLMCommon/KVCache.swift`: `MambaCache.advancePosition(by:)`; the speculative checkpoint records the position after the input.
    - `Libraries/MLXLLM/Models/Qwen35.swift`, `Libraries/MLXVLM/Models/Qwen35.swift`: every GDN path advances the recurrent cache position; both declare `QwenReasoningProtocol.qwen35`.
    - `Libraries/MLXVLM/Models/Qwen3VL.swift`: `Qwen3VLMessageGenerator` writes `reasoning_content`.
    - `Libraries/MLXLMCommon/ReasoningConfig.swift`, `QwenReasoningProtocol.swift`: `replaysReasoningIntoHistory`, `qwen35`.
    - `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift`: `PromptCacheReuse`, `reconcilePromptCache(promptTokens:cachedTokens:previousRenderTokens:caches:protocolRules:carriesModelState:)`, package types.
    - `Libraries/MLXLMCommon/Tool/Parsers/CommittedTurnSplice.swift` (new), `QwenCommittedTurnRule.swift` (new), `DSMLCommittedTurnRule.swift`; `Tool/ToolCallFormat.swift`: `.qwen35` contributes the rule.
    - `Libraries/MLXLMCommon/Evaluate.swift`: `preparedState` callback on `generateTaskRecordingTokens` and `generateProtocolTokensTask`; the loop sets `stopTokenFedToCache`; the copy keeps it; the dead `handleStopToken` is gone.
    - `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`: `renderTokens`, `state`, `representedTokens`, protocol rules, `peek`, `committed(generatedTokens:state:)`.
    - `Libraries/MLXFoundationModels/MLXLanguageModel.swift`: protocol rules at the three plan sites, model state through the three generation sites, `historyMessages(of:resolved:otherwise:)`, `promptCacheReuseRules(of:)`, `generatedTokenCount(of:)`, `GenerationEvent.completion`.
    - `Libraries/MLXFoundationModels/TranscriptConverter.swift`: `mlxMessages(for:replayReasoning:)` with `ReasoningReplay`.

    **Tests** (red first, then green): `Tests/MLXLMTests/` `Qwen35RecurrentCachePositionTests.swift` (new), `QwenCommittedTurnRuleTests.swift` (new), `GenerateLoopStopTokenTests.swift` (new), `MarkerTokenizer.swift` (new shared fixture), `PromptCachePrefixReuseTests.swift`, `ReasoningConfigTests.swift`, `UserInputTests.swift`, `ChatConventionsTests.swift`, `Qwen35ContinuationTests.swift`, `Qwen35MTPTests.swift`, `DSMLCommittedTurnRuleTests.swift`; `Tests/MLXFoundationModelsTests/` `ExecutorPromptCacheTests.swift`, `TranscriptConverterTests.swift`, `ExecutorUsageOutputTests.swift` (new); `IntegrationTesting/IntegrationTestingTests/` `Qwen35AgenticPromptCacheAssessmentTests.swift` (new, real weights, hybrid and control) and `Qwen35SessionPromptCacheTests.swift` (new, real `LanguageModelSession`).

    **Final runs** (`rm -rf .build/out/Products/Debug/*.xctest`, `swift build --build-tests`, `xcrun xctest` on each bundle, 15:15):
    - MLXLMTests: Swift Testing 1025 tests, 101 issues; XCTest 640 tests, 35 failures. Both are the baseline of this tree before the change (SSMTests, the VL ContinuationTests, NanbeigeTests, Qwen35MTPMetalTests; confirmed by a run with my test files stashed at the start). Every new test passes.
    - MLXFoundationModelsTests: 191 tests, 0 issues. MLXGuidedGenerationTests: 70, 0. CXGrammarTests: 7, 0. MLXHuggingFaceMacrosTests: 5, 0.
    - `xcodebuild build-for-testing` of `IntegrationTesting.xcodeproj`: TEST BUILD SUCCEEDED.
    - Real weights: the hybrid test, the control test, the session test and the Router test all pass; the numbers stand in the two measurement comments above.

    **Done-when check**: research comments 1-6 on the card; round 2+ cachedTokenCount ≥ previous render − 16 (yes, above the previous render); fed tokens after round 1 = 35-36 (< 2000) on a 27874-token transcript; round 4 prefill 0.288 s < 2 × 0.275 s; cached round and cold round share 32 of 32 tokens at temperature 0; `ExecutorPromptCacheTests` covers the hybrid stack and the Mamba-at-offset-0 defect; the Router test passes with the qwen3_5 model; SWE-bench round times stay flat (comment above); the SearchTools sessions cannot evict the parent within one tool call per round (comment above); the stop token the iterator feeds is counted in the ledger and in the output count.

    **Not done / notes**: nothing committed, nothing pushed, task left in `doing`. The pre-existing 101 + 35 baseline failures of MLXLMTests are not part of this card. The SWE-bench instance produced an empty patch; the round times, not the patch, were the evidence the card asked for.
  timestamp: 2026-09-09T20:16:49.586486+00:00
- actor: wballard
  id: 01m23x3y165dvdzhy574fb6n3d
  text: |-
    ### Request from the SWE-bench watch: one log line per model call from the executor cache

    The SWE-bench instance `astropy__astropy-12907` on the fixed build (14:26 to 14:59) made 66 model calls. Most took 4 to 16 seconds. Three took 86, 247 and 469 seconds, and rounds 46 to 53 took 42 to 99 seconds each. The unified log shows that every tool returned in milliseconds, so each long gap is one model call. From the outside there is no way to say whether such a call was a long generation or a cold rebuild of the prompt cache. The agent transcript does not carry `cachedTokenCount`, and the bench runner removed the temporary repository with the transcript in it (the runner now keeps a copy).

    Please add one log line per generation pass to the executor cache, at `info` level in the `com.apple.FoundationModels-MLX` subsystem (or the logger `ExecutorPromptCache.swift` already uses), when `ExecutorPromptCachePlan.make` has decided:

    - the session key (model id and session id)
    - the rendered prompt tokens
    - the tokens reused from the carried cache
    - the tokens fed to the model
    - the rule that decided: extend, splice (committed-turn rule), rewind, rebuild, or cold (no carried entry)
    - when the decision is rebuild or rewind: the index of the first divergent token, and the decoded token on each side, the way `Qwen35AgenticPromptCacheAssessmentTests` prints it

    And one line at `committed(generatedTokens:)`: the ledger length checked in, or the reason nothing was checked in.

    With that line, `log show --predicate 'process == "acp-agent"'` names every rebuild in an agent run, and a SWE-bench round-time table can attribute each slow round. Please add it before the card closes, and paste the lines of the three long calls of a rerun of the same instance here.
  timestamp: 2026-09-09T20:17:59.846128+00:00
- actor: claude-code
  id: 01m23x41a10nq6mmkqwgpz831v
  text: |-
    ### finish iteration 1 — changed
    - implement: changed — 35 files (2940 insertions, 331 deletions), nine research and measurement comments on this card
    - test: the implement step ran all five bundles after a from-scratch rebuild. MLXFoundationModelsTests 191/0, MLXGuidedGenerationTests 70/0, CXGrammarTests 7/0, MLXHuggingFaceMacrosTests 5/0. MLXLMTests 1025 Swift Testing tests with 101 issues and 640 XCTest tests with 35 failures. The orchestrator compared that failure set with the first MLXLMTests run of this session, before the fix: the same suites and the same counts (DeepSeekV4AttentionTests 53, DeepSeekV4MoETests 32, SSMTests 6, DeepSeekV4HyperConnectionTests 6, Qwen35MTPTests 3, MiniMaxM3Tests 1; XCTest Qwen25VLContinuationTests 12, Qwen3VLContinuationTests 8, Qwen35ContinuationTests 8, GlmOcrContinuationTests 6, NanbeigeTests 1). All are numeric tolerance failures. This task added no failure and removed the 11 red issues of its own new test. The baseline of this tree is not green; that is a separate problem.
    - commit: 235dcad4e7ff99bb5f39bcf2deaa6fd4b6f766f2 — pushed to origin/stable at the request of the user
    - review: not run yet in this iteration
  timestamp: 2026-09-09T20:18:03.201797+00:00
- actor: claude-code
  id: 01m23xg4ch4xkvzjtx0kcdqhwc
  text: |-
    ### review — findings
    - scope: `review sha HEAD~1..HEAD` (commit 235dcad on branch `stable`)
    - evidence: 28 findings — IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:77, :474, :475, :476, :477, :478, :479, :480, :483, :537; IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:85, :88, :89, :106, :107, :108, :120; Libraries/MLXLLM/Models/Qwen35.swift:1289, :1294; Libraries/MLXVLM/Models/Qwen35.swift:1509; Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:410, :443, :473, :511, :556, :584; Tests/MLXLMTests/QwenCommittedTurnRuleTests.swift:19; Tests/MLXLMTests/UserInputTests.swift:266
    - counts: findings 28, confirmed 31, refuted 1, review tasks attempted 14, failed 1. The engine reports that the result is incomplete because one review task failed.
    - next: correct each finding in the `## Review Findings (2026-09-09 15:18)` section, remove the same cause from the whole file in each case, then run the review again. The task stays in `review`.
  timestamp: 2026-09-09T20:24:39.569169+00:00
- actor: claude-code
  id: 01m23xhfjay9svr61bh0wr23se
  text: |-
    ### finish iteration 1 — findings (amended)
    - implement: changed — 35 files
    - test: five bundles at the pre-existing baseline of this tree (see the entry above)
    - commit: 235dcad4e7ff99bb5f39bcf2deaa6fd4b6f766f2, pushed to origin/stable
    - review: findings — 28 items in `## Review Findings (2026-09-09 15:18)`: IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:77, :474-:480, :483, :537; IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:85, :88, :89, :106-:108, :120; Libraries/MLXLLM/Models/Qwen35.swift:1289, :1294; Libraries/MLXVLM/Models/Qwen35.swift:1509; Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:410, :443, :473, :511, :556, :584; Tests/MLXLMTests/QwenCommittedTurnRuleTests.swift:19; Tests/MLXLMTests/UserInputTests.swift:266
    - also open: the request of the user (comment of 20:17) for one log line per generation pass in the executor cache and a rerun of the SWE-bench instance with those lines on this card
    - next: iteration 2, implement the findings and the request
  timestamp: 2026-09-09T20:25:23.786619+00:00
- actor: claude-code
  id: 01m23xynt7rnk3xg87zyqhc3hx
  text: |-
    ### Iteration 2 picked up — the 28 findings and the log-line request

    Plan for the log lines (the request of 20:17):

    - `PromptCacheReusePolicy` gets `resolve(turn:cache:)`, which answers the decision AND whether a protocol rule made it. `decide` stays as a wrapper. `PromptCacheReuse` gets a `kind` (`prefill`, `extend`, `splice`, `rewind`), thus the executor knows the rule that decided and does not guess it from the numbers. Reason: on Qwen 3.8 the committed-turn rule and the extension rule agree (research 6), thus an inspection of `suffixStart` and `representedTokens` would name the wrong rule.
    - `ExecutorPromptCachePlan` records an `ExecutorPromptCacheDecision` (`cold`, `extend`, `splice`, `rewind(divergence:)`, `rebuild(divergence:)`). `committed(generatedTokens:state:)` throws an `ExecutorPromptCacheCommitError` that names the reason instead of answering nil.
    - `ExecutorPromptCacheSlot` carries the session key and logs one line at `plan` and one at `commit` through `Logger(subsystem: "com.apple.FoundationModels-MLX", category: "ExecutorPromptCache")` at `info`, every field `.public`. The line text is composed by `ExecutorPromptCacheReport`, a pure function, thus the unit tests read the text. The three plan sites hand the slot `context.tokenizer.decode(tokenIds:)` for the decoded seam tokens.

    Findings: the `print` lines of the two IntegrationTesting files, the magic number, the three doc comments, the test names, the redundant `@Suite`, and the `public func test` access control. Each cause is removed from its whole file.
  timestamp: 2026-09-09T20:32:36.167258+00:00
- actor: claude-code
  id: 01m23yd866e41ww8ysgb3dyayk
  text: |-
    ### finish iteration 2 — interrupted
    - implement: the sub agent stopped on an API session limit (HTTP 429) in the middle of its edits. It had started the log-line work: `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift` now holds `PromptCacheReuseKind` (prefill, extend, splice, rewind), `PromptCacheReuse.kind`, and `PromptCacheReuseVerdict.isProtocolDecision`; `Tests/MLXLMTests/PromptCachePrefixReuseTests.swift` holds new tests for the kind. The tree compiles (`swift build`). The 28 review findings and the executor log lines are not done.
    - test, commit, review: not run
    - next: the user logged in again; the implement step starts again on the same tree.

    Note from the user (mid-turn, 2026-09-09): the SearchTools design in FoundationModelsMultitool may need the tool search pushed down to the ranker; and when the tool surface is short enough, all tools can stand in the prompt and the "search" becomes pure LLM prompting with no search step at all. This is a design idea for the other repository, recorded here because the card measured the SearchTools session count. Not part of this card's work.
  timestamp: 2026-09-09T20:40:33.734798+00:00
- actor: claude-code
  id: 01m2407y1xvq5ryrg5tmknxaz5
  text: |-
    ### Iteration 2 — the 28 findings and the executor log lines (2026-09-09, 15:30 to 16:10)

    **The 28 findings**, each cause removed from its whole file:

    - `print` in the two IntegrationTesting files (17 findings): every line now goes through a `Logger` in the subsystem `com.apple.FoundationModels-MLX`, categories `Qwen35AgenticPromptCacheAssessment` and `Qwen35SessionPromptCache`, at `info` level with every field `.public`. The `QWEN35 CACHE:` / `QWEN35 SESSION:` prefixes stay, thus `log show --info --predicate 'subsystem == "com.apple.FoundationModels-MLX"'` shows the numbers beside the executor's own lines. The file headers name the command.
    - Magic number at `Qwen35AgenticPromptCacheAssessmentTests.swift:77`: `queriedBays` is now derived from `queriedBayCount`, `firstQueriedBay` and `queriedBayStride`. swiftlint `no_magic_numbers` with the shipped options reports nothing in the file now.
    - Missing docs: the `toolCallFormat` and `reasoningConfig` declarations of `Qwen35Model`, `Qwen35TextModel` (LLM) and `Qwen35` (VLM), and `public enum Qwen35Language` (VLM, line 230, which the same swiftlint run reported in that file) each carry a doc comment. swiftlint `missing_docs` with the shipped options reports nothing in the four library files of this change.
    - `TranscriptConverterTests.swift`: the six test functions this task added lost their `test` prefix. The 21 `test`-prefixed tests that existed before this task are unchanged: the review skill drops findings that refactor tests that already existed, and the review reads the diff only.
    - `QwenCommittedTurnRuleTests.swift:19`: the bare `@Suite` is gone.
    - `UserInputTests.swift:266`: `testQwen3VLMessageGeneratorReplaysReasoning` is `internal`. The 21 `public func test...` methods that existed before this task are unchanged, for the same reason.

    I ran the tools of the validators on the changed files with the options the rules ship (swiftlint 0.65.0 `no_magic_numbers`, `missing_docs`, the `no_direct_standard_out_logs` custom rule; swiftformat 0.62.1 with the idioms roster): zero findings on the lines of this change.

    **The executor log lines** (request of 20:17), written test-first in `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift` (+22 tests, red then green; the bundle holds 213 tests, 0 issues):

    - `Libraries/MLXLMCommon/PromptCacheReusePolicy.swift` (the partial edit of the interrupted agent, kept): `PromptCacheReuseKind` (`prefill`, `extend`, `splice`, `rewind`), `PromptCacheReuse.kind`, `PromptCacheReusePolicy.resolve(turn:cache:)` which names whether a protocol rule decided. Reason: on Qwen 3.8 the committed-turn rule and the extend rule agree on the numbers, thus the numbers alone cannot name the rule. The SWE-bench rerun confirms it: every warm round logged `rule=splice`.
    - `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`:
      - `ExecutorPromptCacheDecision` (`cold`, `extend`, `splice`, `rewind(seam)`, `rebuild(seam)`) on `ExecutorPromptCachePlan.decision`. `ExecutorPromptCacheDivergence` holds the index of the first divergent token and at most 12 tokens of each side (`reportedTokenCount`), thus the plan needs no tokenizer; the slot decodes at log time.
      - `commitOutcome(generatedTokens:state:)` returns `ExecutorPromptCacheCommitOutcome`: `.checkedIn(entry)` or `.refused(reason)` with `ExecutorPromptCacheCommitRefusal` (`noPlan`, `noCaches`, `positionsDisagree([Int])`, `behindTheLedger`, `pastTheGeneration`). `committed(generatedTokens:state:)` stays as the wrapper, thus no existing test changed.
      - `ExecutorPromptCacheReport` composes the lines (pure functions, the tests read the exact text). `ExecutorPromptCacheSlot` takes the session `key` and a `report` sink (the unified log by default, `Logger(subsystem: "com.apple.FoundationModels-MLX", category: "ExecutorPromptCache")`, `info`, `.public`). `plan(...)` takes `decodeTokens:`; the three plan sites of `MLXLanguageModel.swift` pass `context.tokenizer.decode(tokenIds:)`.
    - The lines:
      - `prompt cache plan model=<id> session=<id> rendered=<n> reused=<n> fed=<n> rule=<cold|extend|splice|rewind|rebuild>`, plus ` divergence=<index> render=<<<decoded>>> ledger=<<<decoded>>>` for rewind and rebuild.
      - `prompt cache plan model=<id> session=<id> rule=none (the input carries media, a batch or a mask)` when no plan is possible, and `rule=guided (the guided pass owns its cache and carries none)` for a guided pass.
      - `prompt cache commit model=<id> session=<id> ledger=<n>`, or `... checked in nothing: <reason>`.

    **Real-weights check of the lines** (`Qwen35SessionPromptCacheTests`, 15:57, PASSED): turn 1 `rendered=73 reused=0 fed=73 rule=cold`, `ledger=101`; turn 2 `rendered=126 reused=101 fed=25 rule=splice`, `ledger=149`. `log show --info` shows them beside the suite's own lines.

    **A limit of `log show` for `info` lines**: the unified log keeps `info` messages in a memory buffer and drops the oldest ones. After the 8-minute SWE-bench run, `log show --info --start <start> --predicate 'process == "acp-agent" ...'` returned 11 of the 42 plan lines and 10 of the 38 commit lines. A `log stream --info --predicate ... > file` started before the run kept all of them. Use the stream for a run that is longer than a few minutes, or raise the level of the lines to `notice` (which the log persists) if `log show` after the fact must be enough.

    **Bundles** (`rm -rf` of the two changed bundles, `swift build --build-tests`, `xcrun xctest`): MLXFoundationModelsTests 213/0; MLXGuidedGenerationTests 70/0; CXGrammarTests 7/0; MLXHuggingFaceMacrosTests 5/0; MLXLMTests Swift Testing 1029 tests with 101 issues (DeepSeekV4AttentionTests 53, DeepSeekV4MoETests 32, SSMTests 6, DeepSeekV4HyperConnectionTests 6, Qwen35MTPMetalTests 3, MiniMaxM3Tests 1) and XCTest 640 tests with 35 failures (Qwen25VLContinuationTests, Qwen3VLContinuationTests, Qwen35ContinuationTests, GlmOcrContinuationTests, NanbeigeTests) — the same set and the same counts as the baseline of this tree. `xcodebuild build-for-testing` of `IntegrationTesting.xcodeproj`: TEST BUILD SUCCEEDED. `swift-format` ran recursively over the repository; it changed no file outside this change.
  timestamp: 2026-09-09T21:12:36.669155+00:00
- actor: claude-code
  id: 01m2408wrv963whr9zqt7ty4kx
  text: |-
    ### SWE-bench rerun with the executor log lines — `astropy__astropy-12907`, 2026-09-09 15:59 to 16:07

    Command, in `FoundationModelsACPAgent`: `swift package edit mlx-swift-lm --path /Users/wballard/github/swissarmyhammer/mlx-swift-lm`, `swift build -c release` (67 s), `log stream --predicate 'process == "acp-agent" AND subsystem == "com.apple.FoundationModels-MLX"' --info --style compact > <scratch>/acp-agent-log-stream.txt &`, then `uv run bench/swebench_run.py <scratch>/swebench-preds-2.jsonl --limit 1 --verbose`. Models: standard `mlx-community/Qwen3.8-27B-mxfp4`, flash `mlx-community/Qwen3-4B-4bit`. The edit is reverted (`swift package unedit`, the `Packages/` link removed); `git status` of the agent repository is what it was before (the user's own modified `bench/` files and untracked `preds.jsonl`, `run.log`).

    The turn made 15 `runCode` calls, then one `skills` call that failed, and stopped with `stop _error` after 506 s (an EMPTY patch, the agent's own turn end; 0 errors in the runner). The executor logged 42 plan lines and 38 commit lines: 2 `cold` (the first standard round, and the flash model's one round), 4 `guided` (grammar-guided passes, which own their cache), 36 `splice`, 0 `rewind`, 0 `rebuild`, and no `checked in nothing`. Every warm round of the 27B session was decided by `QwenCommittedTurnRule` (`splice`), which agrees with the extend rule on the numbers, as research 6 measured.

    Seconds from the plan line to the commit line of each 27B pass (prefill of the fed tokens plus the whole generation; the tool bodies run between passes and are not in these numbers):

    | pass | plan at | rendered | reused | fed | rule | seconds |
    |---|---|---|---|---|---|---|
    | 1 | 15:59:34 | 2479 | 0 | 2479 | cold | 4.5 |
    | 2 | 15:59:40 | 2569 | 2533 | 36 | splice | 1.7 |
    | 3 | 15:59:42 | 3571 | 2620 | 951 | splice | 5.1 |
    | 4 | 15:59:47 | 3828 | 3700 | 128 | splice | 2.1 |
    | 5 | 15:59:50 | 4345 | 3891 | 454 | splice | 2.9 |
    | 6 | 15:59:53 | 4553 | 4421 | 132 | splice | 2.0 |
    | 7 | 15:59:55 | 5854 | 4611 | 1243 | splice | 4.3 |
    | 8 | 15:59:59 | 6072 | 5944 | 128 | splice | 1.9 |
    | 9 | 16:00:01 | 7307 | 6128 | 1179 | splice | 4.7 |
    | 10 | 16:00:06 | 7533 | 7403 | 130 | splice | 2.1 |
    | 11 | 16:00:08 | 7637 | 7591 | 46 | splice | 17.7 |
    | 12 | 16:00:26 | 8256 | 8210 | 46 | splice | 5.5 |
    | 13 | 16:00:31 | 8475 | 8428 | 47 | splice | 3.7 |
    | 14 | 16:00:35 | 8721 | 8589 | 132 | splice | 2.1 |
    | 15 | 16:00:37 | 9969 | 8779 | 1190 | splice | 6.9 |
    | 16 | 16:00:44 | 10261 | 10127 | 134 | splice | 2.2 |
    | 17 | 16:00:46 | 10441 | 10320 | 121 | splice | 3.5 |
    | 18 | 16:00:50 | 10674 | 10544 | 130 | splice | 2.1 |
    | 19 | 16:00:52 | 11058 | 10731 | 327 | splice | 84.0 |
    | 20 | 16:02:16 | 13118 | 12986 | 132 | splice | 3.6 |
    | 21 | 16:02:20 | 13906 | 13176 | 730 | splice | 7.6 |
    | 22 | 16:02:28 | 14167 | 14035 | 132 | splice | 2.5 |
    | 23 | 16:02:31 | 14596 | 14225 | 371 | splice | 11.1 |
    | 24 | 16:02:42 | 14983 | 14853 | 130 | splice | 2.5 |
    | 25 | 16:02:45 | 16129 | 15040 | 1089 | splice | 16.1 |
    | 26 | 16:03:01 | 16598 | 16464 | 134 | splice | 2.6 |
    | 27 | 16:03:04 | 16866 | 16657 | 209 | splice | 12.3 |
    | 28 | 16:03:16 | 17286 | 17154 | 132 | splice | 3.2 |
    | 29 | 16:03:20 | 17639 | 17344 | 295 | splice | 92.4 |
    | 30 | 16:04:52 | 19965 | 19833 | 132 | splice | 3.1 |
    | 31 | 16:04:56 | 20745 | 20023 | 722 | splice | 7.0 |
    | 32 | 16:05:04 | 20888 | 20852 | 36 | splice | 3.8 |
    | 33 | 16:05:08 | 21114 | 20982 | 132 | splice | 2.7 |
    | 34 | 16:05:11 | 22616 | 21172 | 1444 | splice | 6.9 |
    | 35 | 16:05:18 | 22835 | 22703 | 132 | splice | 3.0 |
    | 36 | 16:05:22 | 23488 | 22893 | 595 | splice | 125.2 |
    | 37 | 16:07:29 | 26112 | 26078 | 34 | splice | 3.3 |
    | 38 (flash) | 16:07:32 | 60 | 0 | 60 | cold | 6.6 |

    The fed tokens never grow with the transcript (34 to 1444 tokens on a transcript that reaches 26112 tokens), and a pass with a short generation stays at 2 to 4 s at pass 37 as at pass 2.

    **The three longest passes**, with their lines as the log shows them (`session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065` is the 27B session):

    1. 125.2 s, plan at 16:05:22.083, commit at 16:07:27.317:
       `prompt cache plan model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 rendered=23488 reused=22893 fed=595 rule=splice`
       `prompt cache commit model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 ledger=26175`
       The ledger grew by 26175 − 23488 = 2687 tokens: the model wrote 2687 tokens in that pass, about 21 tokens/s. A generation, not a rebuild.
    2. 92.4 s, plan at 16:03:20.355, commit at 16:04:52.755:
       `prompt cache plan model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 rendered=17639 reused=17344 fed=295 rule=splice`
       `prompt cache commit model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 ledger=19885`
       2246 generated tokens. A generation.
    3. 84.0 s, plan at 16:00:52.595, commit at 16:02:16.636:
       `prompt cache plan model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 rendered=11058 reused=10731 fed=327 rule=splice`
       `prompt cache commit model=mlx-community/Qwen3.8-27B-mxfp4 session=84E8D37A-6BCE-4BCF-92DD-70B4BB940065 ledger=13014`
       1956 generated tokens. A generation.

    Thus in this run every long model call was a long generation on a warm cache, and no call rebuilt the prompt. The cold prefill of the 2479-token first render took 4.5 s.

    The files of the run stand in the scratchpad of this session: `acp-agent-log-stream.txt` (every line), `acp-agent-log-show.txt` (what `log show` kept after the run: 11 of 42 plan lines), `swebench-run-2.log`, `swebench-preds-2.jsonl` and `swebench-preds-2.transcripts/`.
  timestamp: 2026-09-09T21:13:08.123928+00:00
- actor: claude-code
  id: 01m240bfvzc43wng3wypmzxwqc
  text: |-
    ### implement — changed
    - evidence: 14 files, uncommitted (1103 insertions, 84 deletions over HEAD 235dcad) — Libraries/MLXFoundationModels/ExecutorPromptCache.swift, Libraries/MLXFoundationModels/MLXLanguageModel.swift, Libraries/MLXLMCommon/PromptCacheReusePolicy.swift, Libraries/MLXLLM/Models/Qwen35.swift, Libraries/MLXVLM/Models/Qwen35.swift, Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift (+22 tests), Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift, Tests/MLXLMTests/PromptCachePrefixReuseTests.swift (+4 tests), Tests/MLXLMTests/QwenCommittedTurnRuleTests.swift, Tests/MLXLMTests/UserInputTests.swift, IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift, IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift, and the two `.kanban` files of this card. All 28 findings of `## Review Findings (2026-09-09 15:18)` are checked; the 8 "Done when" items are checked on the evidence of iteration 1 and the rerun above. Bundles: MLXFoundationModelsTests 213/0, MLXGuidedGenerationTests 70/0, CXGrammarTests 7/0, MLXHuggingFaceMacrosTests 5/0, MLXLMTests 1029 Swift Testing tests with 101 issues and 640 XCTest tests with 35 failures (the baseline set of this tree, unchanged). IntegrationTesting: TEST BUILD SUCCEEDED; `Qwen35SessionPromptCacheTests` PASSED on real weights with the new log lines. SWE-bench `astropy__astropy-12907` rerun: 38 passes of the 27B session, 0 rebuild, 0 rewind, the three longest calls named on this card. The agent repository is restored.
    - next: `/review`. The task stays in `doing`. Nothing is committed or pushed.
  timestamp: 2026-09-09T21:14:33.215102+00:00
- actor: claude-code
  id: 01m240pej78p7db31zvvdpz78c
  text: |-
    ### review — clean
    - evidence: `review sha HEAD~1..HEAD` (commit 7b6c5e3, iteration 2). Zero findings. 7 review tasks attempted, 0 failed, 1 candidate refuted. 12 files reviewed, 2 kanban files not reviewed (no validator). This run is complete. The prior run had 1 failed review task of 14; this run has none.
    - prior findings: 28 of 28 checked.
    - next: none. The task moved from review to done.
  timestamp: 2026-09-09T21:20:32.327664+00:00
- actor: claude-code
  id: 01m240qhbykwgdjttffd1ysevg
  text: |-
    ### finish iteration 2 — clean
    - implement: changed — 14 files; 28 of 28 findings checked; executor log lines added with 22 unit tests; SWE-bench rerun of astropy__astropy-12907 logged 38 passes, 0 rebuild, 0 rewind, every warm pass `rule=splice`; the three longest calls (125 s, 92 s, 84 s) were long generations on a warm cache
    - test: MLXFoundationModelsTests 213/0, MLXGuidedGenerationTests 70/0, CXGrammarTests 7/0, MLXHuggingFaceMacrosTests 5/0, MLXLMTests 1029 tests with 101 issues and 640 XCTest with 35 failures (the pre-existing baseline, unchanged)
    - commit: 7b6c5e3
    - review: clean — 0 findings, 7 of 7 review tasks complete; the task moved to done
    - next: push 7b6c5e3 to origin/stable at the standing request of the user
  timestamp: 2026-09-09T21:21:07.966117+00:00
position_column: done
position_ordinal: ff9380
title: Make the prompt cache carry across tool rounds on Qwen 3.5 hybrid models, with real-weights proof
---
## The problem

A SWE-bench run of `acp-agent` on 2026-09-09 used `mlx-community/Qwen3.8-27B-mxfp4` (model type `qwen3_5`, 48 linear-attention layers and 16 full-attention layers). One session did 35 tool rounds in 37 minutes. Each round did a full prefill of the whole transcript. The round time grew with the transcript: 15 s at round 5, 64 s at round 12, 3 min at round 25, 6 min at round 34. The prompt cache carried nothing from one round to the next.

The user wants Qwen, and the user wants the cache to work. This card is the work to make that true, with real numbers.

## Read this before you say "Mamba cannot rewind"

Every session that looks at this problem stops at the same sentence: "a hybrid Mamba/attention stack is not trimmable, thus the rewind rule answers `.rebuild`". That sentence is true, and it is not a reason to stop. The rewind rule is one of two paths. The other path, strict prefix extension, needs no trim at all. A recurrent state at the end of the cached tokens is exactly the state a strict extension needs. The work is to make the extension path fire on every tool round, and to keep it from breaking at the seam between rounds.

This fork made Qwen 3.6 caching work before. The commits are in this repository's history:

- `e78994c` (2026-07-21): split-prefill hybrid checkpoints at the transcript-stable boundary. The real-weights test `PromptCacheHybridReuseTests` passed on `Qwen3.6-27B-mxfp4` with a nonzero `cachedTokenCount`.
- `83c43e8` (2026-07-20): store the extended hybrid checkpoint on EOS-terminated rounds. It found that `TokenIterator` feeds the stop token into the caches before the loop drops it, which moves the Mamba state one token past the ledger.
- `5891f01` (2026-07-22): `preserve_thinking` history replay, thus the history render of a past assistant turn carries `reasoning_content` and the prefix stays stable through responses. The bound "prompt + output - 8 tokens cached" passed on real Qwen 3.6 weights in thinking mode and in suppressed mode.

That design (`PromptCache.swift`, `PromptCacheChunks.swift`, `PromptCacheHybridArchitectureTests.swift`) was lost in the upstream catch-up merge of 2026-08-13. Card `^2ajc82t` tracked the port and was abandoned on 2026-08-14 in favor of DeepSeek-V4. Card `^tbyb0dy` holds the nine tests that specified the port. Read both cards in full. Read the three commits with `git show`. Nothing was proven impossible. The work was dropped.

## What is broken today, found by reading the current code

The executor cache is `Libraries/MLXFoundationModels/ExecutorPromptCache.swift`. Two defects stop it on a `qwen3_5` model:

1. **The recurrent cache never advances its offset.** `Libraries/MLXLLM/Models/Qwen35.swift` writes `cache[0]` and `cache[1]` of the `MambaCache` and never touches `cache.offset`. Compare `Libraries/MLXLLM/Models/FalconH1.swift:527`, which does `cache.offset += y.dim(1)`. After a round, the 48 Mamba caches sit at offset 0 and the 16 attention caches sit at the prompt length.
2. **The commit step then refuses the cache.** `ExecutorPromptCachePlan.committed(generatedTokens:)` requires every cache to hold one offset, at or past the prompt length. With the Mamba caches at 0 it returns nil, the session checks in nothing, and the next round starts cold. The unit test "caches that disagree on their position leave the session cold" in `Tests/MLXFoundationModelsTests/ExecutorPromptCacheTests.swift` codifies this as intended. It is correct for two attention caches. It is what kills every hybrid model.

Even with both fixed, the extension rule needs the new render to start with the ledger (render + generated tokens). Today `TranscriptConverter.swift` drops `.reasoning` entries, so a thinking round leaves `<think>...</think>` in the ledger and the next render does not have it. That breaks the prefix, and the rewind rule cannot save a hybrid model. Commit `5891f01` solved this exact problem before with `reasoning_content` replay. `Chat.Message.reasoning` still exists in `Libraries/MLXLMCommon/Chat.swift`, so the model side is still there.

## Research to do first, and to record on this card

Do not guess. Read each of these and write what you found as a comment on this card before you change code.

1. **This fork's own history.** `git show e78994c`, `git show 83c43e8`, `git show 5891f01`. Recover the old `PromptCache.swift` and `PromptCacheHybridReuseTests.swift` from those commits and read how the stable boundary and the checkpoint worked. List which of those ideas the current `ExecutorPromptCache` design already has, and which it lacks.
2. **The DeepSeek-V4 precedent in this fork.** Commit `12223aa` and `Libraries/MLXLMCommon/Tool/Parsers/DSMLCommittedTurnRule.swift`. That rule keeps the live trajectory across a tool round when a cold render cannot reproduce the tokens the model wrote. Commit `f661489` widened the ledger to the real cache position instead of rewinding. Decide whether Qwen needs a `QwenCommittedTurnRule` of the same shape, for the `<tool_call>` block and the `<think>` block. Note that the executor path "consults no protocol rule" today (`reusablePromptPrefix` in `PromptCacheReusePolicy.swift`). Decide whether that stays true.
3. **Upstream Python mlx-lm.** Read `mlx_lm/models/cache.py` (`ArraysCache`, `MambaCache`, `can_trim_prompt_cache`), `mlx_lm/models/qwen3_5.py`, and `mlx_lm/server.py` (its `PromptCache` and how it handles a prompt that extends the cache when the cache cannot trim). Record whether upstream advances the Mamba offset for this family, and what upstream does for a hybrid model on a strict extension.
4. **llama.cpp.** The server keeps context checkpoints of the recurrent state for hybrid and recurrent models, and it restores the newest checkpoint at or before the common prefix, then reprocesses from there. Read the server code and the pull requests that added it. This is the same design as commit `e78994c`. Record what it stores, where it stores it, and how many checkpoints it keeps.
5. **vLLM.** vLLM added prefix caching for hybrid Mamba/attention models in 2025 with block-aligned snapshots of the recurrent state. Read how it picks the snapshot positions. Record whether a snapshot at the end of the prompt render is enough for an agent loop, or whether more positions help.
6. **The Qwen 3.5 chat template.** Render two consecutive rounds of one tool loop with the real tokenizer and diff the token lists. Name the exact seam tokens, as card `^2ajc82t` did for Qwen 3.6 (`<think>` at the end of round 1 against the assistant text at the start of round 2). Check `preserve_thinking` in the template of `Qwen3.8-27B-mxfp4`.

## The fix, in the shape the research supports

The minimum set, to be confirmed by the research:

- Advance `MambaCache.offset` in the `qwen3_5` linear layers (prefill and decode paths, and the MTP path in `Qwen35MTP.swift`), the way `FalconH1` does. Add a unit test that feeds N tokens and reads offset N from every layer's cache.
- Make `committed(generatedTokens:)` accept a hybrid stack. Decide the rule for the Mamba offset (it must equal the attention offset once defect 1 is fixed) and change the unit test "caches that disagree on their position" so it states the new rule.
- Make the round-to-round render a strict extension of the ledger on Qwen: replay `reasoning_content` for past assistant turns (commit `5891f01`), or claim the turn with a Qwen committed-turn rule (commit `12223aa`), or snapshot the recurrent state at the stable boundary before generation (commit `e78994c`). Pick with the research, and write on this card why.
- Handle the stop token that `TokenIterator` feeds into the caches before the loop drops it (commit `83c43e8`). The ledger must hold what the caches hold.
- The `SearchTools` subagent of `acp-agent` opens its own session in the same process. `ExecutorPromptCacheStore.maximumRetainedSessions` is 4. Check that a parent session and its subagent sessions do not evict each other in one agent turn.

## Test with real weights, and with numbers

Unit tests alone do not close this card. Every previous session that closed a caching card on unit tests alone left the SWE-bench run cold. The test must run the shape that failed: one session, tools enabled, thinking on, at least four tool rounds, on `mlx-community/Qwen3.8-27B-mxfp4` (it is in the local Hugging Face cache).

Write the test in `IntegrationTesting/IntegrationTestingTests/`, next to `DeepseekV4AgenticPromptCacheAssessmentTests.swift`, and reuse its measurement helpers. For each round print, in the same format as that test:

- rendered prompt tokens
- fed prompt tokens
- `cachedTokenCount` from the usage event
- prefill seconds
- the first divergent token between this render and the ledger, decoded, when there is one

Run it with `xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj -scheme IntegrationTesting -only-testing:<Suite>/<test>`, one test at a time. Paste the numbers on this card. Then run the same test on `mlx-community/Qwen3-4B-4bit` (a plain attention model) as a control, so a regression on the non-hybrid path shows.

## Done when

- [x] The research comments above are on this card, one per source, with file names and line numbers.
- [x] Round 2 and every later round of the real-weights test report `cachedTokenCount` at or above the rendered prompt tokens of the previous round minus 16.
- [x] Fed prompt tokens per round stay under 2000 after round 1, on a transcript that grows past 20000 tokens.
- [x] Prefill seconds per round do not grow with the transcript length; the round 4 prefill is under 2 times the round 2 prefill.
- [x] The generated text of a cached round matches a cold round of the same prompt at temperature 0, for at least the first 32 tokens, so a wrong recurrent state shows.
- [x] The unit tests in `ExecutorPromptCacheTests.swift` cover a hybrid stack: one `MambaCache` and one `KVCacheSimple`, both advanced, commit to a ledger; a Mamba cache left at offset 0 is a test of the defect, and it must fail before the fix and pass after.
- [x] `FoundationModelsRouter`'s `secondTurnReusesFirstTurnsKVCache` passes against the fixed revision with a `qwen3_5` model.
- [x] A SWE-bench instance in `FoundationModelsACPAgent` (`uv run bench/swebench_run.py preds.jsonl --limit 1`) shows round times that do not grow with the round number. Paste the round times. #performance #prompt-cache #qwen

## Review Findings (2026-09-09 15:18)

> Scope: `review sha HEAD~1..HEAD` — reviewed the diffs only — lines this change added or modified. 31 file(s) reviewed, 4 not reviewed.

> ⚠️ 1/14 review tasks failed — results are INCOMPLETE.

> 4 file(s) not reviewed — no validator matched:
> - `.kanban/tags/01M23J7ZY0ZZDVNT9EDYQM7RTX.jsonl` — no validator matches this file
> - `.kanban/tags/01M23J7ZY0ZZDVNT9EDYQM7RTX.yaml` — no validator matches this file
> - `.kanban/tasks/01M23J7ZXTX1V5QKQ6GXX5G893.jsonl` — no validator matches this file
> - `.kanban/tasks/01M23J7ZXTX1V5QKQ6GXX5G893.md` — no validator matches this file

- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:77` `code-hygiene/magic-numbers-swift` — Magic numbers should be replaced by named constants.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:474` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:475` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:476` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:477` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:478` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:479` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:480` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:483` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35AgenticPromptCacheAssessmentTests.swift:537` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:85` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:88` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:89` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:106` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:107` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:108` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `IntegrationTesting/IntegrationTestingTests/Qwen35SessionPromptCacheTests.swift:120` `code-hygiene/disallowed-constructs-swift` — no_direct_standard_out_logs: Do not commit print(…), debugPrint(…), dump(…) or _printChanges(), which write to standard out in release. Log to a dedicated logging system, or silence one debug-only line with // swiftlint:disable:next no_direct_standard_out_logs and the reason after it.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:1289` `code-hygiene/missing-docs-swift` — public declarations should be documented.
- [x] `Libraries/MLXLLM/Models/Qwen35.swift:1294` `code-hygiene/missing-docs-swift` — public declarations should be documented.
- [x] `Libraries/MLXVLM/Models/Qwen35.swift:1509` `code-hygiene/missing-docs-swift` — public declarations should be documented.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:410` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:443` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:473` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:511` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:556` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXFoundationModelsTests/TranscriptConverterTests.swift:584` `code-hygiene/idioms-swift` — swiftTestingTestCaseNames: Format Swift Testing @Test and @Suite names.
- [x] `Tests/MLXLMTests/QwenCommittedTurnRuleTests.swift:19` `code-hygiene/idioms-swift` — redundantSwiftTestingSuite: Remove redundant @Suite attribute with no arguments.
- [x] `Tests/MLXLMTests/UserInputTests.swift:266` `code-hygiene/idioms-swift` — testSuiteAccessControl: Test methods should have the configured access control (default internal), and other properties / functions in a test suite should be private.