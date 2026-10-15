// Copyright © 2026 Apple Inc.

/// The axes of a `(batch, heads, rows, width)` tensor.
///
/// The queries and the key and value caches of attention use this layout: one
/// row for each query token or cached key, and one width number for each
/// number of a head. The recurrent state of a gated delta net uses this
/// layout too: one row for each value number of a head, and one width number
/// for each key number of a head.
enum HeadRowsAxis {
    /// The axis that holds the sequences.
    static let batch = 0
    /// The axis that holds the heads.
    static let head = 1
    /// The axis that holds the rows of one head.
    static let row = 2
    /// The axis that holds the numbers of one row.
    static let width = 3
    /// The number of axes.
    static let count = 4
}
