---
assignees:
- claude-code
position_column: todo
position_ordinal: '80'
title: MTPDrafterRegistry must restate inherited @unchecked Sendable (build warning)
---
`swift build --build-tests` warns: `Libraries/MLXLMCommon/MTPDrafterModelFactory.swift: warning: class 'MTPDrafterRegistry' must restate inherited '@unchecked Sendable' conformance`. Found while building for ^mga47hg, whose diff does not touch this file. Restate the conformance on the class so the build is warning-free.

#bug