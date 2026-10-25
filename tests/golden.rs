// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//! Every stage of the factorisation and the solve against the pinned Eigen build (`eigen-pin.yaml`),
//! to the last bit: `tests/fixtures/<case>/` holds a matrix (`a.txt`, `column row bits` per entry),
//! a right-hand side (`b.txt`, `bits` per row) and the stage dump that build wrote for them
//! (`stages.txt`): the column ordering and elimination tree, the row pivoting, the supernodal L and U
//! arrays, then the vector after the L solve, after the U solve, and the solution.
//!
//! The cases are built so that between them every path fires, each rule pinned by a mutation in
//! `teeth/`:
//! - pivoting off the diagonal: `grid20-pivot`, `rand200`;
//! - large supernodes and the blocked product: `blocks`, `rand90-dense`;
//! - the lazy-product threshold, visible in one product only: `lazy-threshold`;
//! - the small-segment kernels and COLAMD's degree lists: `grid12`;
//! - a row at exactly COLAMD's dense-row count: `dense-row-edge`;
//! - the trivial systems: `one`, `diag`;
//! - a singular matrix, small (`singular`) and found deep in a factorisation (`network3374-re`).
//!
//! `network3374-*` are the real and imaginary parts of a 3374-node network matrix from
//! sparse_complex (MIT; THIRD-PARTY-NOTICES), for a realistic sparsity pattern. The others are
//! synthetic and deterministic.

use vyges_sparselu::{Csc, SparseLu, REFERENCE_CACHES};

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex bits"))
}

fn case(name: &str) {
    let dir = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    let read = |f: &str| std::fs::read_to_string(format!("{dir}/{f}")).expect(f);
    let b: Vec<f64> = read("b.txt").lines().map(bits).collect();
    let n = b.len();
    let mut a = Csc { nrows: n, ncols: n, col_ptr: vec![0; n + 1], row_idx: Vec::new(), values: Vec::new() };
    for line in read("a.txt").lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let c: usize = w[0].parse().unwrap();
        a.col_ptr[c + 1] += 1;
        a.row_idx.push(w[1].parse().unwrap());
        a.values.push(bits(w[2]));
    }
    for c in 0..n {
        a.col_ptr[c + 1] += a.col_ptr[c];
    }
    // A singular matrix's dump is the reference's one error line, naming the 1-based column whose
    // pivot search found nothing.
    let ours = match SparseLu::compute(&a, REFERENCE_CACHES) {
        Ok(lu) => lu.stage_lines(&b),
        Err(col) => vec![format!("error THE MATRIX IS STRUCTURALLY SINGULAR ... ZERO COLUMN AT {col}")],
    };
    let want = read("stages.txt");
    let want: Vec<&str> = want.lines().collect();
    for (i, (o, w)) in ours.iter().zip(&want).enumerate() {
        assert_eq!(o, w, "{name}: first difference at line {}", i + 1);
    }
    assert_eq!(ours.len(), want.len(), "{name}: line count");
}

#[test]
fn one() {
    case("one");
}

#[test]
fn diag() {
    case("diag");
}

#[test]
fn grid12() {
    case("grid12");
}

#[test]
fn grid20_pivot() {
    case("grid20-pivot");
}

#[test]
fn rand200() {
    case("rand200");
}

#[test]
fn rand90_dense() {
    case("rand90-dense");
}

#[test]
fn blocks() {
    case("blocks");
}

#[test]
fn singular() {
    case("singular");
}

#[test]
fn network3374_re() {
    case("network3374-re");
}

#[test]
fn network3374_im() {
    case("network3374-im");
}

#[test]
fn dense_row_edge() {
    case("dense-row-edge");
}

// The one place the panel update's lazy-product threshold (20) shows in the bits: a blocked product
// small enough that the coefficient-by-coefficient one would otherwise take it.
#[test]
fn lazy_threshold() {
    case("lazy-threshold");
}
