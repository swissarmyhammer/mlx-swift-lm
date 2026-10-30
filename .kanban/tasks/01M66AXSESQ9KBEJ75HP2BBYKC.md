---
assignees:
- claude-code
position_column: todo
position_ordinal: '8580'
title: 'MTP block sweep guard: shipped bs5/pMin0.75 trails single-draft bs2 on prose at quiet load (noise 1.2%)'
---
`GenerationBenchmarkTests.qwen38FourBitMTPGeneration()` failed at quiet load (2026-10-30 06:41, load 2.5 before / 7.3 after, thermal state 1, Release, HEAD 02226efe; the sweep does not use the adaptive guard) on `sweep.isNoSlowerThanSingleDraft(shipped)`:

- shipped bs5/pMin0.75: math 100.0 / code 86.2 / prose 81.0 tok/s (plain ~59.4), mean 89.0
- single draft bs2/pMin0: math 96.2 / code 86.2 / prose 83.9, mean 88.7
- bs4/pMin0.75 was the fastest: 98.3 / 86.9 / 83.7, mean 89.7
- measured noise 1.2%, so prose 81.0 vs 83.9 (-3.5%) fails the check. The same test passed at load 22–27 (noise 13.8%).

Decide whether the shipped setting should move (bs4/pMin0.75 led every prompt pair here) or the check needs a noise floor that matches repeated quiet runs; measure at least twice at quiet load before changing the default.

#mtp #perf