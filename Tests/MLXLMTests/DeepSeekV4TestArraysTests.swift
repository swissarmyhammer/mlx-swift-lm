// Copyright © 2026 Apple Inc.
//
// Tests for the shared DeepSeek-V4 test helpers.
//
// The no-gather tests assert that a graph holds 0 gather nodes. A counter that
// cannot see a gather also gives 0, thus these tests hold the counter to a
// positive control: a graph with one known gather.

import Testing

@Suite
struct DeepSeekV4TestArraysTests {

    @Test
    func theGatherCounterFindsTheOneGatherOfTheControlGraph() {
        DeepSeekV4TestArrays.expectCounterSeesTheControlGather()
    }
}
