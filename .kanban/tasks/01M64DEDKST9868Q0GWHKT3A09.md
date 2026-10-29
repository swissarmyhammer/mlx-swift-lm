---
assignees:
- claude-code
position_column: todo
position_ordinal: '8280'
title: 'MTP: cheaper drafting at long context — window the drafter''s attention (drafter steps cost 3–12 ms at 100k)'
---
Step-4 breakdown (^j2tnzpn): the drafter's cache holds the full context (32768 / 100000 entries) and its attention runs over all of it on every chained step. Chained-step cost is ~1.5 ms at 32k but 3–12 ms at 100k (block-5 proposal 22–24 ms vs 6 ms at 32k).

The drafter only proposes tokens; the target verifies them, so a drafter that attends to a window (e.g. the last N tokens plus the kept prefix) cannot change the output — only acceptance.

- [ ] Add a drafter attention window (rotating drafter cache with a bounded window, reusing the rotating-ring support from ^crdd8sf) that applies only past a context threshold; keep the drafter continuation/prompt-cache save/restore consistent.
- [ ] Measure acceptance and tok/s at 32k/64k/100k for windows 2k/4k/8k/16k vs full; pick the default by measured speed (perf-gains rule), never slower than full-context drafting.
- [ ] Output identity tests (greedy equals plain up to the near-tie rule) and ChatSession/ExecutorPromptCache continuation tests with the window.

#mtp #perf