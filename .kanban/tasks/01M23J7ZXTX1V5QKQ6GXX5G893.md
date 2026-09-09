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
position_column: doing
position_ordinal: '80'
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

- [ ] The research comments above are on this card, one per source, with file names and line numbers.
- [ ] Round 2 and every later round of the real-weights test report `cachedTokenCount` at or above the rendered prompt tokens of the previous round minus 16.
- [ ] Fed prompt tokens per round stay under 2000 after round 1, on a transcript that grows past 20000 tokens.
- [ ] Prefill seconds per round do not grow with the transcript length; the round 4 prefill is under 2 times the round 2 prefill.
- [ ] The generated text of a cached round matches a cold round of the same prompt at temperature 0, for at least the first 32 tokens, so a wrong recurrent state shows.
- [ ] The unit tests in `ExecutorPromptCacheTests.swift` cover a hybrid stack: one `MambaCache` and one `KVCacheSimple`, both advanced, commit to a ledger; a Mamba cache left at offset 0 is a test of the defect, and it must fail before the fix and pass after.
- [ ] `FoundationModelsRouter`'s `secondTurnReusesFirstTurnsKVCache` passes against the fixed revision with a `qwen3_5` model.
- [ ] A SWE-bench instance in `FoundationModelsACPAgent` (`uv run bench/swebench_run.py preds.jsonl --limit 1`) shows round times that do not grow with the round number. Paste the round times. #performance #prompt-cache #qwen