// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen 3.4.0 (tag 3147391d946bb4b6c68edd901f2add6ac1f31f8c):
//   Eigen/src/SparseLU/SparseLU_gemm_kernel.h
//   Copyright (C) 2012 Gael Guennebaud
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//! `internal::sparselu_gemm`: the product kernel Eigen 3.4.0's supernodal updates call
//! (`panel_bmod`, `LU_kernel_bmod`), `C += A B`, column major. The 3.4 branch after the tag
//! replaced it with the generic product ([`crate::dense::gemm`]); the sums differ in order.
//!
//! Per element of `C` (the packet kernel on two doubles computes each lane as the scalar does,
//! multiply then add, no fused multiply-add — the reference is built `-ffp-contract=off`), the
//! order of the additions depends on the element's ROW only:
//! - the rows before `A`'s first 16-byte-aligned row (`first_default_aligned`: one row when `A`
//!   starts on an odd double, none otherwise) add `b_k a_k` for every `k` in order;
//! - every other row, in chunks of 512 rows from there: for each block of four `k` up to
//!   `d - d % 4` (four registers of the depth, sixteen registers on x86-64), a vectorized row
//!   (the chunk's first `2 ⌊rows / 2⌋`) adds the four products one at a time, the chunk's last
//!   odd row adds their sum `((p0 + p1) + p2) + p3`; then the remaining one to three `k` are added
//!   as one sum `(p0 + p1) + p2`.

/// The rows of `A` before its first aligned one (`first_default_aligned` for doubles in 16-byte
/// packets): `A` starts at double `a_off` of an aligned buffer.
pub fn first_aligned(a_off: usize, m: usize) -> usize {
    (a_off % 2).min(m)
}

/// `C[i + j ldc] += sum_k A[a_off + i + k lda] B[b_off + k + j ldb]`, for `i < m`, `j < n`,
/// `k < d`, in the reference kernel's order (module doc). `a_off` places `A` in its aligned
/// buffer (its alignment decides the order).
#[allow(clippy::too_many_arguments)]
pub fn sparselu_gemm(m: usize, n: usize, d: usize, a: &[f64], a_off: usize, lda: usize, b: &[f64], b_off: usize, ldb: usize, c: &mut [f64], ldc: usize) {
    const RK: usize = 4;
    const BM: usize = 4096 / 8;
    const PACKET: usize = 2;
    let at = |i: usize, k: usize| a[a_off + i + k * lda];
    let bt = |k: usize, j: usize| b[b_off + k + j * ldb];
    let d_end = (d / RK) * RK;
    let i0 = first_aligned(a_off, m);
    for i in 0..i0 {
        for j in 0..n {
            let mut s = c[i + j * ldc];
            for k in 0..d {
                s += bt(k, j) * at(i, k);
            }
            c[i + j * ldc] = s;
        }
    }
    let mut ib = i0;
    while ib < m {
        let actual_b = BM.min(m - ib);
        let vector_end = (actual_b / PACKET) * PACKET;
        for j in 0..n {
            for r in 0..actual_b {
                let i = ib + r;
                let mut s = c[i + j * ldc];
                let mut k = 0;
                while k < d_end {
                    let p = [at(i, k) * bt(k, j), at(i, k + 1) * bt(k + 1, j), at(i, k + 2) * bt(k + 2, j), at(i, k + 3) * bt(k + 3, j)];
                    if r < vector_end {
                        s += p[0];
                        s += p[1];
                        s += p[2];
                        s += p[3];
                    } else {
                        s += ((p[0] + p[1]) + p[2]) + p[3];
                    }
                    k += RK;
                }
                match d - d_end {
                    0 => {}
                    1 => s += bt(d_end, j) * at(i, d_end),
                    2 => s += bt(d_end, j) * at(i, d_end) + bt(d_end + 1, j) * at(i, d_end + 1),
                    _ => s += (bt(d_end, j) * at(i, d_end) + bt(d_end + 1, j) * at(i, d_end + 1)) + bt(d_end + 2, j) * at(i, d_end + 2),
                }
                c[i + j * ldc] = s;
            }
        }
        ib += BM;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Half an ulp of 1: added to 1 one at a time it is lost (a tie, to even); summed first it survives.
    const E: f64 = 1.0 / 9_007_199_254_740_992.0;

    // C starts at 1 and the products are e, e, e, 0.
    // A vectorized row adds each product to the running sum: ((1 + e) + e) + e = 1 (each a tie, to even).
    // The chunk's last odd row adds the four products' sum: 1 + (((e + e) + e) + 0) = 1 + 3e > 1.
    #[test]
    fn the_odd_last_row_adds_the_block_sum() {
        // three rows, A aligned (a_off 0): rows 0 and 1 vectorized, row 2 the scalar tail
        let (m, d) = (3, 4);
        let mut a = vec![0.0; m * d];
        for i in 0..m {
            a[i] = E;
            a[i + m] = E;
            a[i + 2 * m] = E;
        }
        let b = vec![1.0; d];
        let mut c = vec![1.0; m];
        sparselu_gemm(m, 1, d, &a, 0, m, &b, 0, d, &mut c, m);
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 1.0);
        // 1 + 3e is a tie between 1 + 2e and 1 + 4e: to even, 1 + 4e
        assert_eq!(c[2], 1.0 + 4.0 * E);
    }

    // An A that starts on an odd double: its first row is the unaligned head, summed in order;
    // the rest start a chunk of their own (row 1 vectorized with row 2; row 3 the tail).
    #[test]
    fn an_unaligned_first_row_is_summed_in_order() {
        let (m, d) = (4, 4);
        let mut a = vec![0.0; 1 + m * d];
        for i in 0..m {
            a[1 + i] = E;
            a[1 + i + m] = E;
            a[1 + i + 2 * m] = E;
        }
        let b = vec![1.0; d];
        let mut c = vec![1.0; m];
        sparselu_gemm(m, 1, d, &a, 1, m, &b, 0, d, &mut c, m);
        assert_eq!([c[0], c[1], c[2]], [1.0, 1.0, 1.0]);
        assert!(c[3] > 1.0);
        assert_eq!(first_aligned(1, m), 1);
        assert_eq!(first_aligned(2, m), 0);
        assert_eq!(first_aligned(1, 0), 0);
    }

    // The depth past the last block of four is added as one sum: 1 + (e + e) > 1, where adding
    // each in turn leaves 1.
    #[test]
    fn the_remaining_depth_is_one_sum() {
        let (m, d) = (2, 7);
        let mut a = vec![0.0; m * d];
        for i in 0..m {
            a[i] = 1.0;
            a[i + 4 * m] = E;
            a[i + 5 * m] = E;
        }
        let b = vec![1.0; d];
        let mut c = vec![0.0; m];
        sparselu_gemm(m, 1, d, &a, 0, m, &b, 0, d, &mut c, m);
        assert!(c[0] > 1.0 && c[1] > 1.0);
    }
}
