// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//! The mutation table: one broken rule per entry, and the test that must notice.
//!
//! 🔑 **This is a typed array, and that is the point.** A malformed entry does not compile.
//!
//! ⚠️ Patterns are matched **literally**, once, at the first occurrence. Raw strings (`r#"…"#`)
//! mean quotes, backslashes and newlines all appear exactly as they do in the source.
//!
//! Most rules here are operation ORDERS: a mutation that reorders a sum changes the last bit of
//! some factor entry, and the golden stage dumps (`tests/golden.rs`) compare every entry bit for
//! bit, so the case that exercises the kernel is the test that must notice.
//!
//! ⛔ **Removed as EQUIVALENT (2026-10-25), not fixed** — each changes the source and cannot
//! change a result, so a test that catches it cannot be written:
//!
//! * `coletree`'s `row >= col` skip as `row > col`: at `row == col` the set found is the column's
//!   own, whose root is the column, so the branch it would take is a no-op.
//! * `relax_snode` climbing at `descendants == relax_columns`: the factorisation reads only
//!   WHETHER a relaxed supernode starts at a column (to cut panels), and at relax 1 climbing one
//!   step further joins a leaf to a one-child parent — the set of starts (the leaves) is the same.
//! * the fixed-size kernels for segments of 2 and 3: unreachable (a panel's one-column supernode
//!   has a one-row segment; the column update has no fixed sizes), so the code was REMOVED.
//! * the one-column L solve's loop order: each step writes its own row.

use crate::Mutation;

pub const MUTATIONS: &[Mutation] = &[
    // ---------------------------------------------------------------- ordering
    Mutation {
        // The postorder numbers siblings in ascending order: children are linked from the last.
        name: r#"postorder-siblings-descending"#,
        file: r#"src/etree.rs"#,
        find: r#"    for v in (0..n).rev() {
        let dad = parent[v];"#,
        replace: r#"    for v in 0..n {
        let dad = parent[v];"#,
        want: r#"grid12"#,
    },
    Mutation {
        // A dense column is one with MORE than half the rows.
        name: r#"colamd-dense-col-threshold"#,
        file: r#"src/colamd.rs"#,
        find: r#"const DENSE_COL: f64 = 0.5;"#,
        replace: r#"const DENSE_COL: f64 = 0.02;"#,
        want: r#"rand90_dense"#,
    },
    Mutation {
        // A dense row is one with MORE than half the columns; equality is kept.
        name: r#"colamd-dense-row-at-equality"#,
        file: r#"src/colamd.rs"#,
        find: r#"        if deg > dense_row_count || deg == 0 {"#,
        replace: r#"        if deg >= dense_row_count || deg == 0 {"#,
        want: r#"dense_row_edge"#,
    },
    Mutation {
        // A column re-entering the degree lists lowers the minimum score it can be found at.
        name: r#"colamd-min-score-not-lowered"#,
        file: r#"src/colamd.rs"#,
        find: r#"            min_score = min_score.min(cur_score);"#,
        replace: r#"            min_score = min_score.min(cur_score + 1);"#,
        want: r#"grid12"#,
    },
    // ---------------------------------------------------------------- factorisation
    Mutation {
        // Panels are 16 columns wide.
        name: r#"panel-size-8"#,
        file: r#"src/factor.rs"#,
        find: r#"pub const PANEL_SIZE: usize = 16;"#,
        replace: r#"pub const PANEL_SIZE: usize = 8;"#,
        want: r#"grid20_pivot"#,
    },
    Mutation {
        // The panel update's product is coefficient by coefficient below 20 (rows + depth + columns).
        name: r#"lazy-product-threshold"#,
        file: r#"src/factor.rs"#,
        find: r#"if u_rows_n + nrow_n + u_cols < 20 && u_rows_n > 0 {"#,
        replace: r#"if u_rows_n + nrow_n + u_cols < 12 && u_rows_n > 0 {"#,
        want: r#"lazy_threshold"#,
    },
    Mutation {
        // Partial pivoting keeps the diagonal only when it is the largest (threshold 1.0).
        name: r#"pivot-threshold-half"#,
        file: r#"src/factor.rs"#,
        find: r#"pub const DIAG_PIVOT_THRESH: f64 = 1.0;"#,
        replace: r#"pub const DIAG_PIVOT_THRESH: f64 = 0.5;"#,
        want: r#"grid20_pivot"#,
    },
    Mutation {
        // An all-zero pivot column is singular (pivmax 0), not only a missing one (pivmax -1).
        name: r#"singular-only-when-empty"#,
        file: r#"src/factor.rs"#,
        find: r#"    if pivmax <= 0.0 {"#,
        replace: r#"    if pivmax < 0.0 {"#,
        want: r#"network3374_re"#,
    },
    Mutation {
        // A supernode stops growing at 128 columns.
        name: r#"maxsuper-small"#,
        file: r#"src/factor.rs"#,
        find: r#"pub const MAXSUPER: i64 = 128;"#,
        replace: r#"pub const MAXSUPER: i64 = 32;"#,
        want: r#"blocks"#,
    },
    Mutation {
        // Pruning: a supernode is pruned once its prune pointer reaches the next one's start.
        name: r#"prune-strict"#,
        file: r#"src/factor.rs"#,
        find: r#"        if w.xprune[u(irep)] >= glu.xlsub[u(irep1)] {"#,
        replace: r#"        if w.xprune[u(irep)] > glu.xlsub[u(irep1)] {"#,
        want: r#"rand200"#,
    },
    // ---------------------------------------------------------------- kernels
    Mutation {
        // Col-major matrix × vector: each row's sum takes the columns in order.
        name: r#"gemv-columns-reversed"#,
        file: r#"src/dense.rs"#,
        find: r#"            for (j, &xj) in x.iter().enumerate().take(jend).skip(j2) {"#,
        replace: r#"            for (j, &xj) in x.iter().enumerate().take(jend).skip(j2).rev() {"#,
        want: r#"gemv_col_takes_the_columns_in_order"#,
    },
    Mutation {
        // Col-major matrix × vector: below 128 columns, one block.
        name: r#"gemv-block-cols"#,
        file: r#"src/dense.rs"#,
        find: r#"    let block_cols = if cols < 128 {
        cols"#,
        replace: r#"    let block_cols = if cols < 4 {
        cols"#,
        want: r#"blocks"#,
    },
    Mutation {
        // A strided-row dot product is a plain sum in order.
        name: r#"dot-seq-reversed"#,
        file: r#"src/dense.rs"#,
        find: r#"    for k in 1..n {
        s += a(k) * x(k);"#,
        replace: r#"    for k in (1..n).rev() {
        s += a(k) * x(k);"#,
        want: r#"dot_seq_starts_from_the_first_product_in_order"#,
    },
    Mutation {
        // Row-major × vector: two lanes.
        name: r#"two-lanes-collapsed"#,
        file: r#"src/dense.rs"#,
        find: r#"        c1 += a(k + 1) * x[k + 1];"#,
        replace: r#"        c0 += a(k + 1) * x[k + 1];"#,
        want: r#"two_lanes_keep_even_and_odd_apart"#,
    },
    Mutation {
        // The vector triangular solve works in panels of 8.
        name: r#"trsv-panel-4"#,
        file: r#"src/dense.rs"#,
        find: r#"const TRSV_PANEL: usize = 8;"#,
        replace: r#"const TRSV_PANEL: usize = 4;"#,
        want: r#"blocks"#,
    },
    Mutation {
        // The upper solve DIVIDES by the diagonal (not a reciprocal product).
        name: r#"trsv-upper-reciprocal"#,
        file: r#"src/dense.rs"#,
        find: r#"                rhs[i] /= a.at(i, i);"#,
        replace: r#"                rhs[i] *= 1.0 / a.at(i, i);"#,
        want: r#"blocks"#,
    },
    Mutation {
        // The lazy product's vectorised rows start at the destination's parity.
        name: r#"lazy-product-parity-ignored"#,
        file: r#"src/dense.rs"#,
        find: r#"pub fn lazy_product(nrow: usize, cols: usize, depth: usize, b: Mat, u: Mat, parity: usize, l: &mut [f64], ldl: usize) {
    let start = parity.min(nrow);"#,
        replace: r#"pub fn lazy_product(nrow: usize, cols: usize, depth: usize, b: Mat, u: Mat, parity: usize, l: &mut [f64], ldl: usize) {
    let start = (1 - parity).min(nrow);"#,
        want: r#"lazy_product_sums_from_zero_only_in_the_vectorised_rows"#,
    },
    Mutation {
        // gebp: the one-vector band splits the depth between two accumulators.
        name: r#"gebp-no-split-band"#,
        file: r#"src/dense.rs"#,
        find: r#"            let split = (i as i64) >= mc2 && (i as i64) < mc1 && j < cols4;"#,
        replace: r#"            let split = false;"#,
        want: r#"blocks"#,
    },
    Mutation {
        // The register block is 4 rows on a build without fused multiply-add.
        name: r#"gebp-mr-6"#,
        file: r#"src/dense.rs"#,
        find: r#"const MR: i64 = 4;"#,
        replace: r#"const MR: i64 = 6;"#,
        want: r#"matrix_solve_rows_past_the_small_panel_go_through_the_product_kernel"#,
    },
    Mutation {
        // Products whose every dimension is under 48 stay one block.
        name: r#"blocking-small-threshold"#,
        file: r#"src/dense.rs"#,
        find: r#"    if k.max(m).max(n) < 48 {"#,
        replace: r#"    if k.max(m).max(n) < 24 {"#,
        want: r#"blocking_matches_the_reference_sizes"#,
    },
    // ---------------------------------------------------------------- solve
    Mutation {
        // The U solve subtracts a supernode's columns in ascending order.
        name: r#"usolve-columns-reversed"#,
        file: r#"src/lib.rs"#,
        find: r#"            for jcol in fsupc..fsupc + nsupc {
                let xj = x[jcol];"#,
        replace: r#"            for jcol in (fsupc..fsupc + nsupc).rev() {
                let xj = x[jcol];"#,
        want: r#"blocks"#,
    },
    Mutation {
        // The right-hand side is scattered by the row pivoting: b[i] goes to row perm_r[i].
        // (Not the one-column L solve's loop order: each step writes its own row, so that order
        // cannot change a result.)
        name: r#"rows-gathered-not-scattered"#,
        file: r#"src/lib.rs"#,
        find: r#"            x[self.perm_r[i] as usize] = bi;"#,
        replace: r#"            x[i] = b[self.perm_r[i] as usize]; let _ = bi;"#,
        want: r#"grid20_pivot"#,
    },
];
