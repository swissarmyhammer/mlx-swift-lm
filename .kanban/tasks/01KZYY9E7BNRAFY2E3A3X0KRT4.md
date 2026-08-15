---
assignees:
- claude-code
comments:
- actor: claude-code
  id: 01kzzyys56qmbdg9bez5d3dec0
  text: |
    ### Cured, and verified again at HEAD on 2026-08-14

    The sparse path of `^ab1eq0r` (commit `1c7bd06`) closed this defect. The
    reproduction now passes in its own process:

    ```
    DSV4 CACHE: recall rendered prompt tokens = 3626
    DSV4 CACHE: recall answer = <<<4172>>>
    ✔ Test longPromptWithoutToolsRecallsAPlantedFact() passed after 35.560 seconds.
    ```

    The three sub-tasks of the work list are answered:

    - The layers with a compress ratio of 0 take the 128-token sliding window; the
      41 layers with a compressor take that window AND the pooled chunks of
      everything before it.
    - `slidingWindow` now reaches the layers that take it.
    - `DeepSeekV4Cache` records the pooled chunks, and both the indexer and the
      sparse path read them.

    A note for the reader: the recall test ran while commit `1c7bd06` was being
    made, thus the 2026-08-13 pass in the log and this 2026-08-14 pass are the same
    code.
  timestamp: 2026-08-14T11:01:46.790639+00:00
position_column: done
position_ordinal: fd80
title: DeepSeek-V4 writes gibberish on a prompt of more than a few hundred tokens
---
Measured on 2026-08-13 against `mlx-community/DeepSeek-V4-Flash-4bit` with the real weights.

A 3,626-token user turn, with NO tool in the prompt, that plants the number `4172` in its own body and asks the model to read it back, answers:

```
 huh? huh? huh? huh  huh  huh  huh  huh  huh  huh  huh  huh
```

The same checkpoint answers a 30-token prompt correctly ("The sea exhales a salty sigh against the shore."), matches the Python greedy fixture token for token, and recalls a number across two short rounds. Thus the weights, the load path and the encoder are good, and the defect belongs to the length of the prompt alone.

A prompt with tools fails the same way, and it fails at the same length. Tools are not the cause.

## The likely cause

`DeepSeekV4Model` runs plain dense attention on every layer.

- `DeepSeekV4Configuration.slidingWindow` decodes from the checkpoint (128) and NOTHING reads it. Every layer whose compress ratio is 0 must attend inside that window, and this port lets it attend over the whole sequence.
- The header of `Libraries/MLXLLM/Models/DeepSeekV4.swift` records the second half: "The compressor and the indexer both load, and neither runs yet... sparse attention needs the pooled cache that DeepSeekV4Compressor.swift records." Thus the global context never comes through the pooled chunks either.

Below 128 tokens a window and no window give the same attention, thus every earlier test passed. Above it the forward pass degenerates.

## Why this blocks the agentic work

An agent's transcript passes 128 tokens at the first turn. Card ^wg4k2x6 could not measure the prompt cache across a tool round, because the model never wrote a tool call: it wrote gibberish instead.

## The work

- [ ] Read the reference attention of DeepSeek-V4 and state which layers take the sliding window and which take the compressed global context
- [ ] Apply `slidingWindow` to the layers that take it
- [ ] Record the pooled cache the compressor needs, and run the indexer and the sparse path
- [ ] Make `longPromptWithoutToolsRecallsAPlantedFact` in `IntegrationTesting/IntegrationTestingTests/DeepseekV4AgenticPromptCacheAssessmentTests.swift` pass; it is the reproduction

The reproduction test is already committed and it fails. #deepseek-v4