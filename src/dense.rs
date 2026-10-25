// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/Core/products/GeneralBlockPanelKernel.h
//   Eigen/src/Core/products/GeneralMatrixMatrix.h
//   Eigen/src/Core/products/GeneralMatrixVector.h
//   Eigen/src/Core/products/TriangularSolverMatrix.h
//   Eigen/src/Core/products/TriangularSolverVector.h
//   Eigen/src/Core/SolveTriangular.h
//   Eigen/src/Core/arch/SSE/PacketMath.h
//   Copyright (C) 2008-2016 Gael Guennebaud
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//! The dense kernels the supernodal factorisation and the solves call, each in the operation order
//! of the reference build: double precision, two-lane vectors, no fused multiply-add (an x86-64
//! baseline build).
//!
//! An element-wise kernel rounds the same whatever its lane width, so only the SUMS below carry an
//! order, and each is written in the order its kernel accumulates it:
//! - col-major matrix × vector: per row, from zero over the columns in order;
//! - row-major matrix × vector: two lanes (even and odd columns) from zero, added, then the odd
//!   column left over;
//! - the block-panel kernel: per row band, sequential except in the one-vector band, where the
//!   columns grouped in fours split the depth's blocks of eight between two accumulators (even and
//!   odd steps) before adding them;
//! - a coefficient of the coefficient-by-coefficient product: in order, from zero in the rows of
//!   the destination's vectorised part (which its parity decides) and from the first product
//!   elsewhere — the two differ only in the sign of an all-zero sum.
//!
//! Matrices are column-major views `(data, off, ld)`: element `(i, j)` is `data[off + i + j * ld]`.
// The loops keep the reference's index form (counters, ranges over parallel arrays), so each
// reads against the reference line by line.
#![allow(clippy::needless_range_loop, clippy::explicit_counter_loop, clippy::too_many_arguments)]

/// The cache sizes the reference's product blocking reads at run time, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caches {
    pub l1: i64,
    pub l2: i64,
    pub l3: i64,
}

/// The register block of the product kernel: rows per band step and columns per group. Without
/// fused multiply-add the reference sizes the row step from half its sixteen vector registers over
/// the four columns: (16 / 2 / 4) two-lane vectors, 4 rows. (With fused multiply-add it would be
/// three vectors, 6 rows; the reference build has none.)
const MR: i64 = 4;
const NR: i64 = 4;
/// Rows per two-lane vector: the kernel's row bands are whole multiples of it.
const LHS_PROGRESS: i64 = 2;
/// The triangular solves' panel width (vector solve) and small-panel width (matrix solve: the
/// larger of the product kernel's row step and column group, so its trailing rows go through it).
const TRSV_PANEL: usize = 8;
const TRSM_PANEL: usize = if MR > NR { MR as usize } else { NR as usize };

/// `n` rounded up to a whole number of two-lane vectors.
pub fn first_multiple(n: usize) -> usize {
    n.div_ceil(2) * 2
}

/// A read-only column-major view.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [f64],
    pub off: usize,
    pub ld: usize,
}

impl<'a> Mat<'a> {
    pub fn new(data: &'a [f64], off: usize, ld: usize) -> Self {
        Mat { data, off, ld }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[self.off + i + j * self.ld]
    }

    /// The view starting at `(i, j)`.
    pub fn sub(&self, i: usize, j: usize) -> Mat<'a> {
        Mat { data: self.data, off: self.off + i + j * self.ld, ld: self.ld }
    }
}

/// `res[i] += alpha * Σ_j a(i, j) x[j]` for `i < rows` (col-major matrix × vector): per row the
/// sum starts at zero and takes the columns in order, in blocks of every column below 128 columns
/// (else 16, or 4 when a column is 4000 or more entries apart).
pub fn gemv_col(rows: usize, cols: usize, a: Mat, x: &[f64], res: &mut [f64], alpha: f64) {
    let block_cols = if cols < 128 {
        cols
    } else if a.ld * 8 < 32000 {
        16
    } else {
        4
    };
    let mut j2 = 0;
    while j2 < cols {
        let jend = (j2 + block_cols).min(cols);
        for (i, r) in res.iter_mut().enumerate().take(rows) {
            let mut c = 0.0;
            for (j, &xj) in x.iter().enumerate().take(jend).skip(j2) {
                c += a.at(i, j) * xj;
            }
            *r += c * alpha;
        }
        j2 += block_cols;
    }
}

/// `Σ_k a[k] x[k]` as a plain sum: the first product, then the rest in order (a dot product whose
/// left side is a strided row, so it is never vectorised).
pub fn dot_seq(n: usize, a: impl Fn(usize) -> f64, x: impl Fn(usize) -> f64) -> f64 {
    let mut s = a(0) * x(0);
    for k in 1..n {
        s += a(k) * x(k);
    }
    s
}

/// One row of a row-major matrix × vector, `Σ_k a(k) x[k]`: even and odd `k` in two lanes from
/// zero over the whole pairs, the lanes added, then the odd `k` left over.
pub fn dot_two_lanes(n: usize, a: impl Fn(usize) -> f64, x: &[f64]) -> f64 {
    let full = n / 2 * 2;
    let (mut c0, mut c1) = (0.0, 0.0);
    let mut k = 0;
    while k < full {
        c0 += a(k) * x[k];
        c1 += a(k + 1) * x[k + 1];
        k += 2;
    }
    let mut cc = c0 + c1;
    for (k, &xk) in x.iter().enumerate().take(n).skip(full) {
        cc += a(k) * xk;
    }
    cc
}

/// Solve `L y = rhs` in place, `L` unit lower triangular (`size` × `size`): in panels of eight,
/// each solved column by column (a zero right-hand side skips its column), the rows below the
/// panel then updated by one matrix × vector product.
pub fn trsv_unit_lower(size: usize, a: Mat, rhs: &mut [f64]) {
    let mut pi = 0;
    while pi < size {
        let pw = (size - pi).min(TRSV_PANEL);
        for k in 0..pw {
            let i = pi + k;
            if rhs[i] != 0.0 {
                let r = pw - k - 1;
                let s = i + 1;
                let ri = rhs[i];
                for t in 0..r {
                    rhs[s + t] -= ri * a.at(s + t, i);
                }
            }
        }
        let end = pi + pw;
        if size > end {
            let x = rhs[pi..end].to_vec();
            gemv_col(size - end, pw, a.sub(end, pi), &x, &mut rhs[end..size], -1.0);
        }
        pi += TRSV_PANEL;
    }
}

/// Solve `U y = rhs` in place, `U` upper triangular with its diagonal (`size` × `size`): panels of
/// eight from the bottom, each column divided by its diagonal then eliminated upwards (a zero
/// right-hand side skips its column), the rows above the panel then updated by one product.
pub fn trsv_upper(size: usize, a: Mat, rhs: &mut [f64]) {
    let mut pi = size;
    while pi > 0 {
        let pw = pi.min(TRSV_PANEL);
        let start = pi - pw;
        for k in 0..pw {
            let i = pi - k - 1;
            if rhs[i] != 0.0 {
                rhs[i] /= a.at(i, i);
                let r = pw - k - 1;
                let s = i - r;
                let ri = rhs[i];
                for t in 0..r {
                    rhs[s + t] -= ri * a.at(s + t, i);
                }
            }
        }
        if start > 0 {
            let x = rhs[start..pi].to_vec();
            gemv_col(start, pw, a.sub(0, start), &x, &mut rhs[..start], -1.0);
        }
        pi = start;
    }
}

/// `L = B U` coefficient by coefficient (`nrow` × `cols`, depth `depth` not fixed): each a sum in
/// order, from zero in the destination's vectorised rows (from row `parity`, in whole pairs, every
/// column alike) and from its first product elsewhere.
pub fn lazy_product(nrow: usize, cols: usize, depth: usize, b: Mat, u: Mat, parity: usize, l: &mut [f64], ldl: usize) {
    let start = parity.min(nrow);
    let end = start + (nrow - start) / 2 * 2;
    for c in 0..cols {
        for i in 0..nrow {
            let mut s = if (start..end).contains(&i) { 0.0 + b.at(i, 0) * u.at(0, c) } else { b.at(i, 0) * u.at(0, c) };
            for k in 1..depth {
                s += b.at(i, k) * u.at(k, c);
            }
            l[i + c * ldl] = s;
        }
    }
}

/// The block-panel kernel: `res(i, j) += alpha * Σ_k a(i, k) b(k, j)` over `rows` × `cols` at
/// depth `depth`, each sum from zero in the order of the band its row falls in (bands of 6, then 4,
/// then 2 rows, then single rows; only the 2-row band, on columns grouped in fours, splits).
pub fn gebp(res: &mut [f64], r0: usize, ldr: usize, rows: usize, depth: usize, cols: usize, a: Mat, b: Mat, alpha: f64) {
    let (rows_i, cols_i) = (rows as i64, cols as i64);
    let cols4 = (cols_i / NR * NR) as usize;
    // The row bands of three, two and one vectors; a band wider than the row step does not exist.
    let lp = LHS_PROGRESS;
    let mc3 = if MR >= 3 * lp { rows_i / (3 * lp) * (3 * lp) } else { 0 };
    let mc2 = if MR >= 2 * lp { mc3 + (rows_i - mc3) / (2 * lp) * (2 * lp) } else { 0 };
    let mc1 = if MR >= lp { mc2 + (rows_i - mc2) / lp * lp } else { 0 };
    let peeled = depth & !7;
    for j in 0..cols {
        for i in 0..rows {
            let split = (i as i64) >= mc2 && (i as i64) < mc1 && j < cols4;
            let x = |k: usize| a.at(i, k) * b.at(k, j);
            let acc = if split {
                let (mut c, mut d) = (0.0, 0.0);
                for k in 0..peeled {
                    if k % 2 == 0 {
                        c += x(k);
                    } else {
                        d += x(k);
                    }
                }
                c += d;
                for k in peeled..depth {
                    c += x(k);
                }
                c
            } else {
                let mut c = 0.0;
                for k in 0..depth {
                    c += x(k);
                }
                c
            };
            res[r0 + i + j * ldr] += acc * alpha;
        }
    }
}

/// The product blocking for a single thread: `(kc, mc, nc)` from the depth, rows and columns and
/// the cache sizes (`kc_factor` 1 for a product, 4 for a triangular solve). Problems under 48 in
/// every dimension are one block.
pub fn blocking(kc_factor: i64, k: i64, m: i64, n: i64, caches: Caches) -> (i64, i64, i64) {
    let (mut k, mut m, mut n) = (k, m, n);
    if k.max(m).max(n) < 48 {
        return (k, m, n);
    }
    let k_peeling = 8;
    let k_div = kc_factor * (MR * 8 + NR * 8);
    let k_sub = MR * NR * 8;
    let max_kc = (((caches.l1 - k_sub) / k_div) & !(k_peeling - 1)).max(1);
    let old_k = k;
    if k > max_kc {
        k = if k % max_kc == 0 { max_kc } else { max_kc - k_peeling * ((max_kc - 1 - (k % max_kc)) / (k_peeling * (k / max_kc + 1))) };
    }
    let actual_l2 = 1_572_864;
    let lhs_bytes = m * k * 8;
    let remaining_l1 = caches.l1 - k_sub - lhs_bytes;
    let max_nc = if remaining_l1 >= NR * 8 * k { remaining_l1 / (k * 8) } else { (3 * actual_l2) / (2 * 2 * max_kc * 8) };
    let nc = (actual_l2 / (2 * k * 8)).min(max_nc) & !(NR - 1);
    if n > nc {
        n = if n % nc == 0 { nc } else { nc - NR * ((nc - (n % nc)) / (NR * (n / nc + 1))) };
    } else if old_k == k {
        let problem_size = k * n * 8;
        let mut actual_lm = actual_l2;
        let mut max_mc = m;
        if problem_size <= 1024 {
            actual_lm = caches.l1;
        } else if caches.l3 != 0 && problem_size <= 32768 {
            actual_lm = caches.l2;
            max_mc = max_mc.min(576);
        }
        let mut mc = (actual_lm / (3 * k * 8)).min(max_mc);
        if mc > MR {
            mc -= mc % MR;
        } else if mc == 0 {
            return (k, m, n);
        }
        m = if m % mc == 0 { mc } else { mc - MR * ((mc - (m % mc)) / (MR * (m / mc + 1))) };
    }
    (k, m, n)
}

/// `res = B U` (`rows` × `cols`, depth `depth`) as the general product runs it from a zeroed
/// destination: row blocks of `mc`, depth blocks of `kc`, column blocks of `nc`, the block-panel
/// kernel on each.
pub fn gemm(rows: usize, cols: usize, depth: usize, b: Mat, u: Mat, res: &mut [f64], ldr: usize, caches: Caches) {
    let (kc, mc, nc) = blocking(1, depth as i64, rows as i64, cols as i64, caches);
    let (kc, mc, nc) = (kc as usize, (mc as usize).min(rows), (nc as usize).min(cols));
    let mut i2 = 0;
    while i2 < rows {
        let amc = mc.min(rows - i2);
        let mut k2 = 0;
        while k2 < depth {
            let akc = kc.min(depth - k2);
            let mut j2 = 0;
            while j2 < cols {
                let anc = nc.min(cols - j2);
                gebp(res, i2 + j2 * ldr, ldr, amc, akc, anc, b.sub(i2, k2), u.sub(k2, j2), 1.0);
                j2 += nc;
            }
            k2 += kc;
        }
        i2 += mc;
    }
}

/// Solve `L X = other` in place, `L` unit lower triangular (`size` × `size`), `other` `size` ×
/// `cols` with leading dimension `ldo`: depth blocks of `kc`; in each, small panels of six solved
/// column by column, the block's rows below each panel updated by the block-panel kernel; then
/// the rows below the depth block.
pub fn trsm_unit_lower(size: usize, cols: usize, tri: Mat, other: &mut [f64], ldo: usize, caches: Caches) {
    let (kc, mc, _) = blocking(4, size as i64, size as i64, cols as i64, caches);
    let (kc, mc) = (kc as usize, (mc as usize).min(size));
    let subcols = if cols > 0 { caches.l2 / (4 * 8 * (ldo.max(size) as i64)) } else { 0 };
    let subcols = ((subcols / NR) * NR).max(NR) as usize;
    let mut k2 = 0;
    while k2 < size {
        let akc = (size - k2).min(kc);
        let mut j2 = 0;
        while j2 < cols {
            let acols = (cols - j2).min(subcols);
            let mut k1 = 0;
            while k1 < akc {
                let pw = (akc - k1).min(TRSM_PANEL);
                for k in 0..pw {
                    let i = k2 + k1 + k;
                    let rs = pw - k - 1;
                    let s = i + 1;
                    // The reference scales by the unit diagonal's reciprocal, 1: exact, so omitted.
                    for j in j2..j2 + acols {
                        let b = other[i + j * ldo];
                        for i3 in 0..rs {
                            other[s + i3 + j * ldo] -= b * tri.at(s + i3, i);
                        }
                    }
                }
                let target = akc - k1 - pw;
                let start_block = k2 + k1;
                if target > 0 {
                    let start_target = k2 + k1 + pw;
                    let bpanel = other.to_vec();
                    gebp(
                        other,
                        start_target + j2 * ldo,
                        ldo,
                        target,
                        pw,
                        acols,
                        tri.sub(start_target, start_block),
                        Mat::new(&bpanel, start_block + j2 * ldo, ldo),
                        -1.0,
                    );
                }
                k1 += TRSM_PANEL;
            }
            j2 += subcols;
        }
        let (start, end) = (k2 + kc, size);
        let mut i2 = start;
        while i2 < end {
            let amc = mc.min(end - i2);
            let bblock = other.to_vec();
            gebp(other, i2, ldo, amc, akc, cols, tri.sub(i2, k2), Mat::new(&bblock, k2, ldo), -1.0);
            i2 += mc;
        }
        k2 += kc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOX: Caches = Caches { l1: 32 * 1024, l2: 256 * 1024, l3: 55 * 1024 * 1024 };

    // Values whose sums round differently in each order, so an order mistake shows.
    fn vals(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let m = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                (m - 0.5) * 10f64.powi(((s >> 3) % 7) as i32 - 3)
            })
            .collect()
    }

    // The matrix solve's small panel is max(mr, nr) = 4 rows wide on a build without fused
    // multiply-add, so a fifth row is not eliminated column by column: the block-panel kernel sums
    // its four products from zero and subtracts the sum once. A unit-lower 5×5 block and right-hand
    // side taken from a real factorisation where the two orders differ in the last bit (a 6-row
    // panel gives ...903e).
    #[test]
    fn matrix_solve_rows_past_the_small_panel_go_through_the_product_kernel() {
        let h = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
        let lower = [
            (1, 0, "bfa24d8a746652cf"),
            (2, 0, "bfc6daff50e6d673"),
            (2, 1, "bfb74ae84f8f09a5"),
            (3, 0, "bfcb6dc1c8192281"),
            (3, 1, "bfe935636ea0210c"),
            (3, 2, "bfe000228c334bfc"),
            (4, 0, "bfc6dbb4ccde01e0"),
            (4, 1, "bfb7467598a80545"),
            (4, 2, "bfd7fff8d5f399b5"),
            (4, 3, "bfe7ee24e4ace00f"),
        ];
        let mut tri = vec![0.0; 25];
        for (i, k, v) in lower {
            tri[i + 5 * k] = h(v);
        }
        let b = [h("bf8d1e4d6a314883"), 0.0, 0.0, 0.0, h("bec3b547a2a59690")];
        let mut other = vec![0.0; 6 * 2];
        for j in 0..2 {
            other[j * 6..j * 6 + 5].copy_from_slice(&b);
        }
        trsm_unit_lower(5, 2, Mat::new(&tri, 0, 5), &mut other, 6, BOX);
        let want = ["bf8d1e4d6a314883", "bf40a79a729c0de6", "bf652d17977c8f93", "bf7369e10376a687", "bf7d16529708903f"];
        for j in 0..2 {
            let got: Vec<String> = (0..5).map(|i| format!("{:016x}", other[i + j * 6].to_bits())).collect();
            assert_eq!(got, want, "column {j}");
        }
    }

    // Col-major matrix × vector: per row, from zero, columns in order; then `res + c*alpha`.
    #[test]
    fn gemv_col_sums_each_row_from_zero_in_column_order() {
        let a = vals(5 * 3, 1);
        let x = vals(3, 2);
        let mut res = vals(5, 3);
        let want: Vec<f64> = (0..5)
            .map(|i| {
                let c = ((0.0 + a[i] * x[0]) + a[i + 5] * x[1]) + a[i + 10] * x[2];
                res[i] - c
            })
            .collect();
        gemv_col(5, 3, Mat::new(&a, 0, 5), &x, &mut res, -1.0);
        assert_eq!(res, want);
    }

    // Row-major × vector: even/odd lanes from zero, added, then the odd column left over.
    #[test]
    fn two_lane_dot_adds_the_lanes_before_the_tail() {
        let a = vals(5, 4);
        let x = vals(5, 5);
        let want = ((0.0 + a[0] * x[0] + a[2] * x[2]) + (0.0 + a[1] * x[1] + a[3] * x[3])) + a[4] * x[4];
        assert_eq!(dot_two_lanes(5, |k| a[k], &x).to_bits(), want.to_bits());
    }

    // Blocking (kc factor, k, m, n) -> (kc, mc, nc): each row as the reference's own blocking
    // function returns it with these cache sizes and its register block (mr 4, nr 4).
    #[test]
    fn blocking_matches_the_reference_sizes() {
        assert_eq!(blocking(1, 16, 40, 16, BOX), (16, 40, 16));
        assert_eq!(blocking(1, 16, 100, 16, BOX), (16, 100, 16));
        assert_eq!(blocking(4, 60, 60, 60, BOX), (60, 60, 8));
        assert_eq!(blocking(4, 200, 300, 50, BOX), (104, 300, 50));
        assert_eq!(blocking(1, 300, 600, 40, BOX), (300, 200, 40));
        // Just under the shortcut's 48: kept whole, though the cache arithmetic would split n.
        assert_eq!(blocking(1, 47, 47, 47, BOX), (47, 47, 47));
        assert_eq!(blocking(4, 47, 47, 47, BOX), (47, 47, 47));
        assert_eq!(blocking(1, 30, 40, 45, BOX), (30, 40, 45));
        assert_eq!(blocking(4, 40, 30, 40, BOX), (40, 30, 40));
    }

    // Products whose sum is order-sensitive: 1 + 1e-16 rounds back to 1, so which pair meets first
    // decides the result.
    const ORDERED: [f64; 4] = [1.0, 1e-16, -1.0, 1e-16];

    // Col-major × vector takes each row's columns in order: ((1 + 1e-16) - 1) + 1e-16 = 1e-16.
    #[test]
    fn gemv_col_takes_the_columns_in_order() {
        let x = [1.0; 4];
        let mut res = [0.0];
        gemv_col(1, 4, Mat::new(&ORDERED, 0, 1), &x, &mut res, 1.0);
        assert_eq!(res[0].to_bits(), (((0.0 + 1.0 + 1e-16) - 1.0) + 1e-16f64).to_bits());
        assert_eq!(res[0], 1e-16);
    }

    // A strided dot: from its first product, in order — so a single negative product times zero
    // keeps its sign (-0), where a sum from zero would give +0.
    #[test]
    fn dot_seq_starts_from_the_first_product_in_order() {
        // In order: ((1 + 1e-16) + 1e-16) - 1 = 0; any other order of the last three differs.
        let a = [1.0, 1e-16, 1e-16, -1.0];
        assert_eq!(dot_seq(4, |k| a[k], |_| 1.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(dot_seq(1, |_| -1.0, |_| 0.0).to_bits(), (-0.0f64).to_bits());
    }

    // Two lanes: even k (1 - 1 = 0) and odd k (1e-16 + 1e-16) apart, then added: 2e-16, where one
    // running sum gives 1e-16.
    #[test]
    fn two_lanes_keep_even_and_odd_apart() {
        assert_eq!(dot_two_lanes(4, |k| ORDERED[k], &[1.0; 4]), 2e-16);
    }

    // The coefficient-by-coefficient product sums from zero in the destination's vectorised rows
    // (from its parity, in whole pairs) and from the first product elsewhere: visible only in the
    // sign of an all-zero sum, -1 × 0.
    #[test]
    fn lazy_product_sums_from_zero_only_in_the_vectorised_rows() {
        let b = [-1.0; 3];
        let u = [0.0];
        for parity in 0..2 {
            let mut l = [9.0; 3];
            lazy_product(3, 1, 1, Mat::new(&b, 0, 3), Mat::new(&u, 0, 1), parity, &mut l, 3);
            let signs: Vec<bool> = l.iter().map(|v| v.is_sign_negative()).collect();
            let want = if parity == 0 { [false, false, true] } else { [true, false, false] };
            assert_eq!(signs, want, "parity {parity}");
        }
    }
}
