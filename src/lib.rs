// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/SparseLU/SparseLU.h
//   Eigen/src/SparseLU/SparseLU_SupernodalMatrix.h
//   Copyright (C) 2012 Désiré Nuentsa-Wakam
//   Copyright (C) 2012-2014 Gael Guennebaud
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Those Eigen files are modified versions of SuperLU; its notice is in THIRD-PARTY-NOTICES, and
// this code was modified from it (translated into Rust, Vyges, 2026).
//! A supernodal sparse LU for square real systems, bit-for-bit Eigen 3.4's `SparseLU` with its
//! default ordering: column approximate-minimum-degree ordering, column elimination tree
//! postordered, relaxed supernodes, panel by panel, partial pivoting with a diagonal preference of
//! 1.0, then the forward and back substitution. Each stage is one function, in Eigen's order, and
//! each floating-point operation is in the order Eigen's build performs it (`eigen-pin.yaml`: the
//! source commit, the build, and the sizes it yields), so the solution matches to the last bit.

pub mod colamd;
pub mod dense;
pub mod etree;
pub mod factor;
pub mod relax;

pub use dense::Caches;
use dense::Mat;
use factor::Glu;

/// A compressed sparse column matrix: column `c`'s rows (ascending) and values are
/// `row_idx[col_ptr[c]..col_ptr[c + 1]]` and `values[..]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Csc {
    pub nrows: usize,
    pub ncols: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl Csc {
    pub fn rows(&self, c: usize) -> &[usize] {
        &self.row_idx[self.col_ptr[c]..self.col_ptr[c + 1]]
    }

    pub fn vals(&self, c: usize) -> &[f64] {
        &self.values[self.col_ptr[c]..self.col_ptr[c + 1]]
    }

    /// The same matrix with its columns reordered: column `perm[c]` of the result is column `c`.
    pub fn permute_cols(&self, perm: &[usize]) -> Csc {
        let mut inv = vec![0usize; self.ncols];
        for (c, &p) in perm.iter().enumerate() {
            inv[p] = c;
        }
        let mut out = Csc { nrows: self.nrows, ncols: self.ncols, col_ptr: vec![0], row_idx: Vec::new(), values: Vec::new() };
        for &c in &inv {
            out.row_idx.extend_from_slice(self.rows(c));
            out.values.extend_from_slice(self.vals(c));
            out.col_ptr.push(out.row_idx.len());
        }
        out
    }
}

/// The cache sizes of the machine the pinned build's results were taken on (`eigen-pin.yaml`): the
/// product blocking reads them at run time, so they are part of the result.
pub const REFERENCE_CACHES: Caches = Caches { l1: 32_768, l2: 262_144, l3: 57_671_680 };

/// The factored matrix: the column ordering (`perm_c[c]` is original column `c`'s position) and
/// the postordered elimination tree it was taken from, the row pivoting and the factors.
#[derive(Debug, Clone)]
pub struct SparseLu {
    pub n: usize,
    pub perm_c: Vec<i64>,
    pub etree: Vec<i64>,
    pub perm_r: Vec<i64>,
    pub glu: Glu,
}

/// The column ordering and elimination tree (`analyzePattern`): the approximate-minimum-degree
/// order, the columns' elimination tree in that order, then its postorder composed into both.
pub fn analyze_pattern(a: &Csc) -> (Vec<i64>, Vec<i64>) {
    let n = a.ncols;
    let p: Vec<i32> = a.col_ptr.iter().map(|&x| x as i32).collect();
    let rows: Vec<i32> = a.row_idx.iter().map(|&x| x as i32).collect();
    let order = colamd::compute_ordering(a.nrows as i32, n as i32, &p, &rows);
    let mut perm_c = vec![0usize; n];
    for (k, &c) in order.iter().take(n).enumerate() {
        perm_c[c as usize] = k;
    }
    let parent = etree::coletree(&a.permute_cols(&perm_c));
    let post = etree::tree_postorder(n, &parent);
    let mut et = vec![0i64; n];
    for i in 0..n {
        et[post[i]] = post[parent[i]] as i64;
    }
    let perm_c = perm_c.iter().map(|&c| post[c] as i64).collect();
    (perm_c, et)
}

impl SparseLu {
    /// Order, then factor (`compute`). `Err` names the structurally singular column (1-based).
    pub fn compute(a: &Csc, caches: Caches) -> Result<SparseLu, usize> {
        let n = a.ncols;
        let (perm_c, etree) = analyze_pattern(a);
        let perm: Vec<usize> = perm_c.iter().map(|&c| c as usize).collect();
        let mut iperm_c = vec![0i64; n];
        for (c, &p) in perm.iter().enumerate() {
            iperm_c[p] = c as i64;
        }
        let et: Vec<usize> = etree.iter().map(|&p| p as usize).collect();
        let relax_end = relax::relax_snode(n, &et, 1);
        let (glu, perm_r) = factor::factorize(&a.permute_cols(&perm), &iperm_c, &relax_end, caches)?;
        Ok(SparseLu { n, perm_c, etree, perm_r, glu })
    }

    fn nsuper(&self) -> usize {
        self.glu.supno[self.n] as usize
    }

    /// `X ← L⁻¹ X` (the supernodal lower solve): per supernode, a one-column one subtracts its
    /// column below the diagonal; a wider one solves its triangle, then subtracts the product of
    /// the rows below.
    pub fn solve_l(&self, x: &mut [f64]) {
        let g = &self.glu;
        for k in 0..=self.nsuper() {
            let fsupc = g.xsup[k] as usize;
            let istart = g.xlsub[fsupc] as usize;
            let nsupr = g.xlsub[fsupc + 1] as usize - istart;
            let nsupc = g.xsup[k + 1] as usize - fsupc;
            let nrow = nsupr - nsupc;
            let luptr = g.xlusup[fsupc] as usize;
            if nsupc == 1 {
                let xf = x[fsupc];
                for t in 1..nsupr {
                    x[g.lsub[istart + t] as usize] -= xf * g.lusup[luptr + t];
                }
            } else {
                let lda = g.xlusup[fsupc + 1] as usize - luptr;
                let mut uv = x[fsupc..fsupc + nsupc].to_vec();
                dense::trsv_unit_lower(nsupc, Mat::new(&g.lusup, luptr, lda), &mut uv);
                x[fsupc..fsupc + nsupc].copy_from_slice(&uv);
                let a = Mat::new(&g.lusup, luptr + nsupc, lda);
                let mut work = vec![0.0; nrow];
                if nrow == 1 {
                    work[0] += dense::dot_seq(nsupc, |c| a.at(0, c), |c| uv[c]);
                } else {
                    dense::gemv_col(nrow, nsupc, a, &uv, &mut work, 1.0);
                }
                for (i, &wv) in work.iter().enumerate() {
                    x[g.lsub[istart + nsupc + i] as usize] -= wv;
                }
            }
        }
    }

    /// `X ← U⁻¹ X` (the upper solve): supernodes from the last, each one's triangle (its diagonal
    /// included) solved, then each of its columns' U entries subtracted.
    pub fn solve_u(&self, x: &mut [f64]) {
        let g = &self.glu;
        for k in (0..=self.nsuper()).rev() {
            let fsupc = g.xsup[k] as usize;
            let luptr = g.xlusup[fsupc] as usize;
            let lda = g.xlusup[fsupc + 1] as usize - luptr;
            let nsupc = g.xsup[k + 1] as usize - fsupc;
            if nsupc == 1 {
                x[fsupc] /= g.lusup[luptr];
            } else {
                let mut uv = x[fsupc..fsupc + nsupc].to_vec();
                dense::trsv_upper(nsupc, Mat::new(&g.lusup, luptr, lda), &mut uv);
                x[fsupc..fsupc + nsupc].copy_from_slice(&uv);
            }
            for jcol in fsupc..fsupc + nsupc {
                let xj = x[jcol];
                for e in g.xusub[jcol] as usize..g.xusub[jcol + 1] as usize {
                    x[g.usub[e] as usize] -= xj * g.ucol[e];
                }
            }
        }
    }

    /// The solution of `A v = b`, with the vector after each stage: rows pivoted, `L` solved,
    /// `U` solved, columns unpermuted (`(after L, after U, v)`).
    pub fn solve_stages(&self, b: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n;
        let mut x = vec![0.0; n];
        for (i, &bi) in b.iter().enumerate() {
            x[self.perm_r[i] as usize] = bi;
        }
        self.solve_l(&mut x);
        let after_l = x.clone();
        self.solve_u(&mut x);
        let after_u = x.clone();
        let v = (0..n).map(|i| x[self.perm_c[i] as usize]).collect();
        (after_l, after_u, v)
    }

    /// The stage dump, one `name index value` line each (values as IEEE bits in hex), in the
    /// order and form the reference instrument writes it.
    pub fn stage_lines(&self, b: &[f64]) -> Vec<String> {
        let n = self.n;
        let g = &self.glu;
        let mut out = Vec::new();
        let ints = |out: &mut Vec<String>, name: &str, v: &[i64], count: usize| {
            out.extend(v.iter().take(count).enumerate().map(|(i, x)| format!("{name} {i} {x}")));
        };
        let hex = |out: &mut Vec<String>, name: &str, v: &[f64], count: usize| {
            out.extend(v.iter().take(count).enumerate().map(|(i, x)| format!("{name} {i} {:016x}", x.to_bits())));
        };
        ints(&mut out, "perm_c", &self.perm_c, n);
        ints(&mut out, "etree", &self.etree, n);
        ints(&mut out, "perm_r", &self.perm_r, n);
        ints(&mut out, "xsup", &g.xsup, self.nsuper() + 2);
        ints(&mut out, "supno", &g.supno, n + 1);
        ints(&mut out, "xlsub", &g.xlsub, n + 1);
        ints(&mut out, "lsub", &g.lsub, g.xlsub[n] as usize);
        ints(&mut out, "xlusup", &g.xlusup, n + 1);
        hex(&mut out, "lusup", &g.lusup, g.xlusup[n] as usize);
        ints(&mut out, "xusub", &g.xusub, n + 1);
        ints(&mut out, "usub", &g.usub, g.xusub[n] as usize);
        hex(&mut out, "ucol", &g.ucol, g.xusub[n] as usize);
        let (xl, xu, v) = self.solve_stages(b);
        hex(&mut out, "xl", &xl, n);
        hex(&mut out, "xu", &xu, n);
        hex(&mut out, "v", &v, n);
        out
    }
}
