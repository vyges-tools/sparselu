// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/SparseLU/SparseLU.h
//   Eigen/src/SparseLU/SparseLUImpl.h
//   Eigen/src/SparseLU/SparseLU_Memory.h
//   Eigen/src/SparseLU/SparseLU_Utils.h
//   Eigen/src/SparseLU/SparseLU_*_dfs.h
//   Eigen/src/SparseLU/SparseLU_*_bmod.h
//   Eigen/src/SparseLU/SparseLU_copy_to_ucol.h
//   Eigen/src/SparseLU/SparseLU_pivotL.h
//   Eigen/src/SparseLU/SparseLU_pruneL.h
//   Copyright (C) 2012 Désiré Nuentsa-Wakam
//   Copyright (C) 2012-2014 Gael Guennebaud
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Those Eigen files are modified versions of SuperLU; its notice is in THIRD-PARTY-NOTICES, and
// this code was modified from it (translated into Rust, Vyges, 2026).
//! The supernodal factorisation, `factorize`: panel by panel, a depth-first search of each panel's
//! columns (`panel_dfs`) and the panel's update by the supernodes it reaches (`panel_bmod`); then
//! per column its own search (`column_dfs`), its update by the panel's earlier columns
//! (`column_bmod`), its U part copied out (`copy_to_ucol`), its pivot chosen (`pivot_l`) and the
//! supernodes it lets be pruned (`prune_l`); last, L's row indices renumbered by the pivoting
//! (`fixup_l`).
//!
//! Index vectors keep the reference's `-1` for "none" (`EMPTY`), so each test reads as written.
// The loops keep the reference's index form (counters, ranges over parallel arrays), so each
// reads against the reference line by line.
#![allow(clippy::needless_range_loop, clippy::explicit_counter_loop, clippy::too_many_arguments)]

use super::dense::{self, Caches, Mat};
use super::Csc;

pub const EMPTY: i64 = -1;
/// The tuning the reference factors with: panel width, largest supernode.
pub const PANEL_SIZE: usize = 16;
pub const MAXSUPER: i64 = 128;
/// Partial pivoting keeps the diagonal whenever it is the largest (threshold 1.0).
pub const DIAG_PIVOT_THRESH: f64 = 1.0;

/// The factors' storage (`GlobalLU_t`): supernode bounds and membership, L's row structure and
/// values (column-major per supernode, each column padded to a whole number of two-lane
/// vectors), U's entries by column.
#[derive(Debug, Default, Clone)]
pub struct Glu {
    pub xsup: Vec<i64>,
    pub supno: Vec<i64>,
    pub lsub: Vec<i64>,
    pub xlsub: Vec<i64>,
    pub lusup: Vec<f64>,
    pub xlusup: Vec<i64>,
    pub ucol: Vec<f64>,
    pub usub: Vec<i64>,
    pub xusub: Vec<i64>,
}

/// Write `x` at `i`, growing the array as the reference expands its storage.
fn put<T: Copy + Default>(v: &mut Vec<T>, i: usize, x: T) {
    if i >= v.len() {
        v.resize((i + 1).max(v.len() * 3 / 2), T::default());
    }
    v[i] = x;
}

#[inline]
fn u(i: i64) -> usize {
    i as usize
}

/// Where a depth-first search writes the rows it finds, and which representatives it records.
enum Traits<'a> {
    /// The panel's search: rows to the panel's row list; a supernode recorded once per panel
    /// (`marker1` holds the column that recorded it).
    Panel { lsub: &'a mut [i64], marker1: &'a mut [i64], jcol: i64 },
    /// A column's search: rows straight into L's row structure; a row not seen by the previous
    /// column ends the supernode.
    Column { jsuper: &'a mut i64, jcol: i64 },
}

impl Traits<'_> {
    fn push(&mut self, glu: &mut Glu, nextl: &mut usize, row: i64, mark: i64) {
        match self {
            Traits::Panel { lsub, .. } => {
                lsub[*nextl] = row;
                *nextl += 1;
            }
            Traits::Column { jsuper, jcol } => {
                put(&mut glu.lsub, *nextl, row);
                *nextl += 1;
                if mark != *jcol - 1 {
                    **jsuper = EMPTY;
                }
            }
        }
    }

    fn update_segrep(&mut self, krep: i64, jj: i64) -> bool {
        match self {
            Traits::Panel { marker1, jcol, .. } => {
                if marker1[u(krep)] < *jcol {
                    marker1[u(krep)] = jj;
                    true
                } else {
                    false
                }
            }
            Traits::Column { .. } => true,
        }
    }
}

/// The search from one row (`dfs_kernel`): an unpivoted row joins the list; a pivoted one leads to
/// its supernode's representative, whose first nonzero is lowered or whose subtree is walked,
/// each finished representative recorded in postorder.
fn dfs_kernel(
    jj: i64,
    perm_r: &[i64],
    nseg: &mut usize,
    segrep: &mut [i64],
    repfnz: &mut [i64],
    xprune: &[i64],
    marker: &mut [i64],
    parent: &mut [i64],
    xplore: &mut [i64],
    glu: &mut Glu,
    nextl: &mut usize,
    krow: i64,
    traits: &mut Traits,
) {
    let kmark = marker[u(krow)];
    marker[u(krow)] = jj;
    let kperm = perm_r[u(krow)];
    if kperm == EMPTY {
        traits.push(glu, nextl, krow, kmark);
        return;
    }
    let mut krep = glu.xsup[u(glu.supno[u(kperm)] + 1)] - 1;
    let mut myfnz = repfnz[u(krep)];
    if myfnz != EMPTY {
        if myfnz > kperm {
            repfnz[u(krep)] = kperm;
        }
        return;
    }
    let mut oldrep = EMPTY;
    parent[u(krep)] = oldrep;
    repfnz[u(krep)] = kperm;
    let mut xdfs = glu.xlsub[u(krep)];
    let mut maxdfs = xprune[u(krep)];
    loop {
        while xdfs < maxdfs {
            let kchild = glu.lsub[u(xdfs)];
            xdfs += 1;
            let chmark = marker[u(kchild)];
            if chmark != jj {
                marker[u(kchild)] = jj;
                let chperm = perm_r[u(kchild)];
                if chperm == EMPTY {
                    traits.push(glu, nextl, kchild, chmark);
                } else {
                    let chrep = glu.xsup[u(glu.supno[u(chperm)] + 1)] - 1;
                    myfnz = repfnz[u(chrep)];
                    if myfnz != EMPTY {
                        if myfnz > chperm {
                            repfnz[u(chrep)] = chperm;
                        }
                    } else {
                        xplore[u(krep)] = xdfs;
                        oldrep = krep;
                        krep = chrep;
                        parent[u(krep)] = oldrep;
                        repfnz[u(krep)] = chperm;
                        xdfs = glu.xlsub[u(krep)];
                        maxdfs = xprune[u(krep)];
                    }
                }
            }
        }
        if traits.update_segrep(krep, jj) {
            segrep[*nseg] = krep;
            *nseg += 1;
        }
        let kpar = parent[u(krep)];
        if kpar == EMPTY {
            break;
        }
        krep = kpar;
        xdfs = xplore[u(krep)];
        maxdfs = xprune[u(krep)];
    }
}

/// The working arrays of one factorisation.
struct Work {
    m: usize,
    segrep: Vec<i64>,
    parent: Vec<i64>,
    xplore: Vec<i64>,
    repfnz: Vec<i64>,
    panel_lsub: Vec<i64>,
    xprune: Vec<i64>,
    /// Three markers of `m` each: the search's, the panel's recorded supernodes, the column's.
    marker: Vec<i64>,
    dense: Vec<f64>,
}

/// The panel's search (`panel_dfs`): each column's entries scattered into its dense column, a
/// search from each row not yet reached by this column.
fn panel_dfs(w: &mut Work, width: usize, jcol: usize, a: &Csc, perm_r: &[i64], glu: &mut Glu) -> usize {
    let m = w.m;
    let mut nseg = 0;
    let (marker0, rest) = w.marker.split_at_mut(m);
    let marker1 = &mut rest[..m];
    let mut traits = Traits::Panel { lsub: &mut w.panel_lsub, marker1, jcol: jcol as i64 };
    for jj in jcol..jcol + width {
        let mut nextl_col = (jj - jcol) * m;
        let col = nextl_col;
        for (&krow, &v) in a.rows(jj).iter().zip(a.vals(jj)) {
            w.dense[col + krow] = v;
            if marker0[krow] == jj as i64 {
                continue;
            }
            dfs_kernel(
                jj as i64,
                perm_r,
                &mut nseg,
                &mut w.segrep,
                &mut w.repfnz[col..col + m],
                &w.xprune,
                marker0,
                &mut w.parent,
                &mut w.xplore,
                glu,
                &mut nextl_col,
                krow as i64,
                &mut traits,
            );
        }
    }
    nseg
}

/// The panel's update by each supernode it reaches (`panel_bmod`), representatives taken from
/// the last recorded: a supernode of two or more columns updates the whole panel at once
/// (triangular solve, then product); a single column updates each panel column in turn.
fn panel_bmod(w: &mut Work, width: usize, jcol: usize, nseg: usize, glu: &mut Glu, caches: Caches) {
    let m = w.m;
    for ksub in 0..nseg {
        let krep = w.segrep[nseg - 1 - ksub];
        let fsupc = glu.xsup[u(glu.supno[u(krep)])];
        let nsupc = krep - fsupc + 1;
        let nsupr = glu.xlsub[u(fsupc + 1)] - glu.xlsub[u(fsupc)];
        let nrow = nsupr - nsupc;
        let lptr = glu.xlsub[u(fsupc)];
        let (mut u_rows, mut u_cols) = (0i64, 0usize);
        for jj in jcol..jcol + width {
            let kfnz = w.repfnz[(jj - jcol) * m + u(krep)];
            if kfnz == EMPTY {
                continue;
            }
            u_cols += 1;
            u_rows = u_rows.max(krep - kfnz + 1);
        }
        if nsupc >= 2 {
            let (u_rows_n, nrow_n) = (u(u_rows), u(nrow));
            let ldu = dense::first_multiple(u_rows_n);
            let mut um = vec![0.0; ldu * u_cols.max(1)];
            let mut u_col = 0;
            for jj in jcol..jcol + width {
                let col = (jj - jcol) * m;
                let kfnz = w.repfnz[col + u(krep)];
                if kfnz == EMPTY {
                    continue;
                }
                let segsize = u(krep - kfnz + 1);
                let mut isub = u(lptr + kfnz - fsupc);
                let off = u_rows_n - segsize;
                for i in 0..segsize {
                    um[i + off + u_col * ldu] = w.dense[col + u(glu.lsub[isub])];
                    isub += 1;
                }
                u_col += 1;
            }
            let lda = u(glu.xlusup[u(fsupc + 1)] - glu.xlusup[u(fsupc)]);
            let no_zeros = u((krep - u_rows + 1) - fsupc);
            let mut luptr = u(glu.xlusup[u(fsupc)]) + lda * no_zeros + no_zeros;
            dense::trsm_unit_lower(u_rows_n, u_cols, Mat::new(&glu.lusup, luptr, lda), &mut um, ldu, caches);
            luptr += u_rows_n;
            let b = Mat::new(&glu.lusup, luptr, lda);
            let ldl = dense::first_multiple(nrow_n);
            let mut l = vec![0.0; ldl * u_cols.max(1)];
            let parity = luptr % 2;
            if u_rows_n + nrow_n + u_cols < 20 && u_rows_n > 0 {
                dense::lazy_product(nrow_n, u_cols, u_rows_n, b, Mat::new(&um, 0, ldu), parity, &mut l, ldl);
            } else if nrow_n > 0 && u_cols > 0 && u_rows_n > 0 {
                if u_cols == 1 {
                    if nrow_n == 1 {
                        l[0] += dense::dot_seq(u_rows_n, |k| b.at(0, k), |k| um[k]);
                    } else {
                        dense::gemv_col(nrow_n, u_rows_n, b, &um[..u_rows_n], &mut l[..nrow_n], 1.0);
                    }
                } else if nrow_n == 1 {
                    let x: Vec<f64> = (0..u_rows_n).map(|k| b.at(0, k)).collect();
                    for c in 0..u_cols {
                        let cc = dense::dot_two_lanes(u_rows_n, |k| um[k + c * ldu], &x);
                        l[c * ldl] += cc;
                    }
                } else {
                    dense::gemm(nrow_n, u_cols, u_rows_n, b, Mat::new(&um, 0, ldu), &mut l, ldl, caches);
                }
            }
            let mut u_col = 0;
            for jj in jcol..jcol + width {
                let col = (jj - jcol) * m;
                let kfnz = w.repfnz[col + u(krep)];
                if kfnz == EMPTY {
                    continue;
                }
                let segsize = u(krep - kfnz + 1);
                let mut isub = u(lptr + kfnz - fsupc);
                let off = u_rows_n - segsize;
                for i in 0..segsize {
                    let irow = u(glu.lsub[isub]);
                    isub += 1;
                    w.dense[col + irow] = um[i + off + u_col * ldu];
                }
                for i in 0..nrow_n {
                    let irow = u(glu.lsub[isub]);
                    isub += 1;
                    w.dense[col + irow] -= l[i + u_col * ldl];
                }
                u_col += 1;
            }
        } else {
            for jj in jcol..jcol + width {
                let col = (jj - jcol) * m;
                let kfnz = w.repfnz[col + u(krep)];
                if kfnz == EMPTY {
                    continue;
                }
                let segsize = u(krep - kfnz + 1);
                let luptr = u(glu.xlusup[u(fsupc)]);
                let lda = u(glu.xlusup[u(fsupc + 1)] - glu.xlusup[u(fsupc)]);
                let no_zeros = u(kfnz - fsupc);
                let dense_col = &mut w.dense[col..col + m];
                // A one-column supernode: its segment is one row, so the reference's fixed-size
                // instantiations for segments of 2 and 3 cannot be reached from here (nor from
                // column_bmod, which has none) and only the one-column and general kernels exist.
                kernel_bmod(segsize, dense_col, glu, luptr, lda, u(nrow), u(lptr), no_zeros);
            }
        }
    }
}

/// One supernode's update of one dense column (`LU_kernel_bmod`): one column is a scaled
/// subtraction in pairs; more, the segment solved against the supernode's triangle and the rows
/// below less the product.
fn kernel_bmod(segsize: usize, dense: &mut [f64], glu: &Glu, luptr: usize, lda: usize, nrow: usize, lptr: usize, no_zeros: usize) {
    let lsub = &glu.lsub;
    if segsize == 1 {
        let f = dense[u(lsub[lptr + no_zeros])];
        let start = luptr + lda * no_zeros + no_zeros + 1;
        let a = &glu.lusup[start..start + nrow];
        let rows = &lsub[lptr + no_zeros + 1..lptr + no_zeros + 1 + nrow];
        let mut i = 0;
        while i + 1 < nrow {
            let (i0, i1) = (u(rows[i]), u(rows[i + 1]));
            let (d0, d1) = (dense[i0] - f * a[i], dense[i1] - f * a[i + 1]);
            dense[i0] = d0;
            dense[i1] = d1;
            i += 2;
        }
        if i < nrow {
            dense[u(rows[i])] -= f * a[i];
        }
        return;
    }
    let isub = lptr + no_zeros;
    let mut tempv: Vec<f64> = (0..segsize).map(|i| dense[u(lsub[isub + i])]).collect();
    let mut luptr = luptr + lda * no_zeros + no_zeros;
    let a = Mat::new(&glu.lusup, luptr, lda);
    dense::trsv_unit_lower(segsize, a, &mut tempv);
    luptr += segsize;
    let b = Mat::new(&glu.lusup, luptr, lda);
    let mut l = vec![0.0; nrow];
    if nrow == 1 {
        l[0] += dense::dot_seq(segsize, |k| b.at(0, k), |k| tempv[k]);
    } else {
        dense::gemv_col(nrow, segsize, b, &tempv, &mut l, 1.0);
    }
    for (i, &t) in tempv.iter().enumerate() {
        dense[u(lsub[isub + i])] = t;
    }
    for (i, &li) in l.iter().enumerate() {
        dense[u(lsub[isub + segsize + i])] -= li;
    }
}

/// A column's own search (`column_dfs`): from each row the panel search listed for it, then the
/// supernode decision: the column joins the current supernode when its rows are the previous
/// column's less one and the supernode is below the size limit; else a new supernode starts
/// (and a finished one of three or more columns keeps only its first column's row list).
fn column_dfs(w: &mut Work, jcol: usize, perm_r: &[i64], nseg: &mut usize, k: usize, glu: &mut Glu) {
    let m = w.m;
    let jcol_i = jcol as i64;
    let mut jsuper = glu.supno[jcol];
    let mut nextl = u(glu.xlsub[jcol]);
    {
        let marker2 = &mut w.marker[2 * m..3 * m];
        let mut traits = Traits::Column { jsuper: &mut jsuper, jcol: jcol_i };
        let mut i = 0;
        while i < m && w.panel_lsub[k + i] != EMPTY {
            let krow = w.panel_lsub[k + i];
            w.panel_lsub[k + i] = EMPTY;
            if marker2[u(krow)] != jcol_i {
                dfs_kernel(
                    jcol_i,
                    perm_r,
                    nseg,
                    &mut w.segrep,
                    &mut w.repfnz[k..k + m],
                    &w.xprune,
                    marker2,
                    &mut w.parent,
                    &mut w.xplore,
                    glu,
                    &mut nextl,
                    krow,
                    &mut traits,
                );
            }
            i += 1;
        }
    }
    let mut nsuper = glu.supno[jcol];
    if jcol == 0 {
        nsuper = 0;
        glu.supno[0] = 0;
    } else {
        let jcolm1 = jcol - 1;
        let fsupc = glu.xsup[u(nsuper)];
        let jptr = glu.xlsub[jcol];
        let jm1ptr = glu.xlsub[jcolm1];
        if nextl as i64 - jptr != jptr - jm1ptr - 1 {
            jsuper = EMPTY;
        }
        if jcol_i - fsupc >= MAXSUPER {
            jsuper = EMPTY;
        }
        if jsuper == EMPTY {
            if fsupc < jcolm1 as i64 - 1 {
                let mut ito = glu.xlsub[u(fsupc + 1)];
                glu.xlsub[jcolm1] = ito;
                let istop = ito + jptr - jm1ptr;
                w.xprune[jcolm1] = istop;
                glu.xlsub[jcol] = istop;
                for ifrom in jm1ptr..nextl as i64 {
                    let v = glu.lsub[u(ifrom)];
                    put(&mut glu.lsub, u(ito), v);
                    ito += 1;
                }
                nextl = u(ito);
            }
            nsuper += 1;
            glu.supno[jcol] = nsuper;
        }
    }
    glu.xsup[u(nsuper + 1)] = jcol_i + 1;
    glu.supno[jcol + 1] = nsuper;
    w.xprune[jcol] = nextl as i64;
    glu.xlsub[jcol + 1] = nextl as i64;
}

/// A column's update by the panel's earlier supernodes and columns (`column_bmod`): each
/// supernode outside its own, last recorded first; then the column copied into L's storage
/// (padded to a whole number of two-lane vectors) and, inside its own supernode, the panel's
/// earlier columns of it applied as one triangular solve and one product.
fn column_bmod(w: &mut Work, jcol: usize, nseg1: usize, nseg: usize, fpanelc: usize, k: usize, glu: &mut Glu) {
    let m = w.m;
    let jsupno = glu.supno[jcol];
    let fpanelc = fpanelc as i64;
    for ksub in 0..nseg - nseg1 {
        let krep = w.segrep[nseg - 1 - ksub];
        let ksupno = glu.supno[u(krep)];
        if jsupno == ksupno {
            continue;
        }
        let fsupc = glu.xsup[u(ksupno)];
        let fst_col = fsupc.max(fpanelc);
        let d_fsupc = fst_col - fsupc;
        let luptr = glu.xlusup[u(fst_col)] + d_fsupc;
        let lptr = glu.xlsub[u(fsupc)] + d_fsupc;
        let kfnz = w.repfnz[k + u(krep)].max(fpanelc);
        let segsize = krep - kfnz + 1;
        let nsupc = krep - fst_col + 1;
        let nsupr = glu.xlsub[u(fsupc + 1)] - glu.xlsub[u(fsupc)];
        let nrow = nsupr - d_fsupc - nsupc;
        let lda = glu.xlusup[u(fst_col + 1)] - glu.xlusup[u(fst_col)];
        let no_zeros = kfnz - fst_col;
        let dense_col = &mut w.dense[k..k + m];
        kernel_bmod(u(segsize), dense_col, glu, u(luptr), u(lda), u(nrow), u(lptr), u(no_zeros));
    }
    let mut nextlu = u(glu.xlusup[jcol]);
    let fsupc = glu.xsup[u(jsupno)];
    let new_next = nextlu + u(glu.xlsub[u(fsupc + 1)] - glu.xlsub[u(fsupc)]);
    let offset = dense::first_multiple(new_next) - new_next;
    for isub in glu.xlsub[u(fsupc)]..glu.xlsub[u(fsupc + 1)] {
        let irow = u(glu.lsub[u(isub)]);
        let v = w.dense[k + irow];
        put(&mut glu.lusup, nextlu, v);
        w.dense[k + irow] = 0.0;
        nextlu += 1;
    }
    for _ in 0..offset {
        put(&mut glu.lusup, nextlu, 0.0);
        nextlu += 1;
    }
    glu.xlusup[jcol + 1] = nextlu as i64;
    let fst_col = fsupc.max(fpanelc);
    if fst_col < jcol as i64 {
        let d_fsupc = fst_col - fsupc;
        let luptr = u(glu.xlusup[u(fst_col)] + d_fsupc);
        let nsupr = glu.xlsub[u(fsupc + 1)] - glu.xlsub[u(fsupc)];
        let nsupc = u(jcol as i64 - fst_col);
        let nrow = u(nsupr - d_fsupc) - nsupc;
        let ufirst = u(glu.xlusup[jcol] + d_fsupc);
        let lda = u(glu.xlusup[jcol + 1] - glu.xlusup[jcol]);
        let mut uv = glu.lusup[ufirst..ufirst + nsupc].to_vec();
        dense::trsv_unit_lower(nsupc, Mat::new(&glu.lusup, luptr, lda), &mut uv);
        glu.lusup[ufirst..ufirst + nsupc].copy_from_slice(&uv);
        let a = Mat::new(&glu.lusup, luptr + nsupc, lda);
        let mut l = glu.lusup[ufirst + nsupc..ufirst + nsupc + nrow].to_vec();
        if nrow == 1 {
            l[0] -= dense::dot_seq(nsupc, |k| a.at(0, k), |k| uv[k]);
        } else {
            dense::gemv_col(nrow, nsupc, a, &uv, &mut l, -1.0);
        }
        glu.lusup[ufirst + nsupc..ufirst + nsupc + nrow].copy_from_slice(&l);
    }
}

/// The column's U part out of the dense column (`copy_to_ucol`): per supernode outside its own,
/// last recorded first, the segment from its first nonzero, rows renumbered by the pivoting.
fn copy_to_ucol(w: &mut Work, jcol: usize, nseg: usize, k: usize, perm_r: &[i64], glu: &mut Glu) {
    let jsupno = glu.supno[jcol];
    let mut nextu = u(glu.xusub[jcol]);
    for ksub in 0..nseg {
        let krep = w.segrep[nseg - 1 - ksub];
        let ksupno = glu.supno[u(krep)];
        if jsupno == ksupno {
            continue;
        }
        let kfnz = w.repfnz[k + u(krep)];
        if kfnz == EMPTY {
            continue;
        }
        let fsupc = glu.xsup[u(ksupno)];
        let mut isub = u(glu.xlsub[u(fsupc)] + kfnz - fsupc);
        let segsize = u(krep - kfnz + 1);
        for _ in 0..segsize {
            let irow = u(glu.lsub[isub]);
            put(&mut glu.usub, nextu, perm_r[irow]);
            put(&mut glu.ucol, nextu, w.dense[k + irow]);
            w.dense[k + irow] = 0.0;
            nextu += 1;
            isub += 1;
        }
    }
    glu.xusub[jcol + 1] = nextu as i64;
}

/// The column's pivot (`pivotL`): the largest magnitude below the supernode's columns, the
/// diagonal kept when it reaches the threshold; that row swapped to the top, the column below it
/// scaled by its reciprocal. `Err` names a structurally singular column.
fn pivot_l(jcol: usize, perm_r: &mut [i64], iperm_c: &[i64], glu: &mut Glu) -> Result<i64, usize> {
    let fsupc = u(glu.xsup[u(glu.supno[jcol])]);
    let nsupc = jcol - fsupc;
    let lptr = u(glu.xlsub[fsupc]);
    let nsupr = u(glu.xlsub[fsupc + 1]) - lptr;
    let lda = u(glu.xlusup[fsupc + 1] - glu.xlusup[fsupc]);
    let sup = u(glu.xlusup[fsupc]);
    let colp = u(glu.xlusup[jcol]);
    let diagind = iperm_c[jcol];
    let mut pivmax = -1.0f64;
    let mut pivptr = nsupc;
    let mut diag = EMPTY;
    for isub in nsupc..nsupr {
        let rtemp = glu.lusup[colp + isub].abs();
        if rtemp > pivmax {
            pivmax = rtemp;
            pivptr = isub;
        }
        if glu.lsub[lptr + isub] == diagind {
            diag = isub as i64;
        }
    }
    if pivmax <= 0.0 {
        let pivrow = if pivmax < 0.0 { diagind } else { glu.lsub[lptr + pivptr] };
        perm_r[u(pivrow)] = jcol as i64;
        return Err(jcol + 1);
    }
    let thresh = DIAG_PIVOT_THRESH * pivmax;
    if diag >= 0 {
        let rtemp = glu.lusup[colp + u(diag)].abs();
        if rtemp != 0.0 && rtemp >= thresh {
            pivptr = u(diag);
        }
    }
    let pivrow = glu.lsub[lptr + pivptr];
    perm_r[u(pivrow)] = jcol as i64;
    if pivptr != nsupc {
        glu.lsub.swap(lptr + pivptr, lptr + nsupc);
        for icol in 0..=nsupc {
            glu.lusup.swap(sup + pivptr + icol * lda, sup + nsupc + icol * lda);
        }
    }
    let temp = 1.0 / glu.lusup[colp + nsupc];
    for k in nsupc + 1..nsupr {
        glu.lusup[colp + k] *= temp;
    }
    Ok(pivrow)
}

/// Supernodes the new pivot row lets be pruned (`pruneL`): in each recorded supernode that ends at
/// its representative and lists the pivot row, the pivoted rows are moved to the front (values
/// too for a one-column supernode) and its search stops after them.
fn prune_l(w: &mut Work, jcol: usize, perm_r: &[i64], pivrow: i64, nseg: usize, k: usize, glu: &mut Glu) {
    let jsupno = glu.supno[jcol];
    for i in 0..nseg {
        let irep = w.segrep[i];
        let irep1 = irep + 1;
        if w.repfnz[k + u(irep)] == EMPTY {
            continue;
        }
        if glu.supno[u(irep)] == glu.supno[u(irep1)] {
            continue;
        }
        if glu.supno[u(irep)] == jsupno {
            continue;
        }
        let mut do_prune = false;
        let (mut kmin, mut kmax) = (0i64, 0i64);
        if w.xprune[u(irep)] >= glu.xlsub[u(irep1)] {
            kmin = glu.xlsub[u(irep)];
            kmax = glu.xlsub[u(irep1)] - 1;
            for krow in kmin..=kmax {
                if glu.lsub[u(krow)] == pivrow {
                    do_prune = true;
                    break;
                }
            }
        }
        if do_prune {
            let movnum = irep == glu.xsup[u(glu.supno[u(irep)])];
            while kmin <= kmax {
                if perm_r[u(glu.lsub[u(kmax)])] == EMPTY {
                    kmax -= 1;
                } else if perm_r[u(glu.lsub[u(kmin)])] != EMPTY {
                    kmin += 1;
                } else {
                    glu.lsub.swap(u(kmin), u(kmax));
                    if movnum {
                        let minloc = glu.xlusup[u(irep)] + (kmin - glu.xlsub[u(irep)]);
                        let maxloc = glu.xlusup[u(irep)] + (kmax - glu.xlsub[u(irep)]);
                        glu.lusup.swap(u(minloc), u(maxloc));
                    }
                    kmin += 1;
                    kmax -= 1;
                }
            }
            w.xprune[u(irep)] = kmin;
        }
    }
}

/// L's row indices renumbered by the row pivoting, each supernode's list kept once (`fixupL`).
fn fixup_l(n: usize, perm_r: &[i64], glu: &mut Glu) {
    let mut nextl = 0i64;
    let nsuper = glu.supno[n];
    for i in 0..=u(nsuper) {
        let fsupc = u(glu.xsup[i]);
        let jstart = glu.xlsub[fsupc];
        glu.xlsub[fsupc] = nextl;
        for j in jstart..glu.xlsub[fsupc + 1] {
            glu.lsub[u(nextl)] = perm_r[u(glu.lsub[u(j)])];
            nextl += 1;
        }
        for k in fsupc + 1..u(glu.xsup[i + 1]) {
            glu.xlsub[k] = nextl;
        }
    }
    glu.xlsub[n] = nextl;
}

/// The factorisation of `a` with its columns already permuted (column `j` of `a` is the matrix's
/// `j`-th in elimination order), `iperm_c` the inverse column permutation, `relax_end` the
/// relaxed supernodes: the factors and the row permutation, or the singular column (1-based).
pub fn factorize(a: &Csc, iperm_c: &[i64], relax_end: &[Option<usize>], caches: Caches) -> Result<(Glu, Vec<i64>), usize> {
    let (m, n) = (a.nrows, a.ncols);
    let maxpanel = PANEL_SIZE * m;
    let mut w = Work {
        m,
        segrep: vec![0; m],
        parent: vec![0; m],
        xplore: vec![0; m],
        repfnz: vec![EMPTY; maxpanel],
        panel_lsub: vec![EMPTY; maxpanel],
        xprune: vec![0; n],
        marker: vec![EMPTY; 3 * m],
        dense: vec![0.0; maxpanel],
    };
    let nnz = a.row_idx.len();
    let mut glu = Glu {
        xsup: vec![0; n + 1],
        supno: vec![0; n + 1],
        lsub: vec![0; nnz.max(1) * 5],
        xlsub: vec![0; n + 1],
        lusup: vec![0.0; nnz.max(1) * 20],
        xlusup: vec![0; n + 1],
        ucol: vec![0.0; nnz.max(1) * 20],
        usub: vec![0; nnz.max(1) * 20],
        xusub: vec![0; n + 1],
    };
    glu.supno[0] = EMPTY;
    let mut perm_r = vec![EMPTY; m];
    let mut jcol = 0;
    while jcol < n {
        let mut panel_size = PANEL_SIZE;
        let mut k = jcol + 1;
        while k < (jcol + PANEL_SIZE).min(n) {
            if relax_end[k].is_some() {
                panel_size = k - jcol;
                break;
            }
            k += 1;
        }
        if k == n {
            panel_size = n - jcol;
        }
        let nseg1 = panel_dfs(&mut w, panel_size, jcol, a, &perm_r, &mut glu);
        panel_bmod(&mut w, panel_size, jcol, nseg1, &mut glu, caches);
        for jj in jcol..jcol + panel_size {
            let k = (jj - jcol) * m;
            let mut nseg = nseg1;
            column_dfs(&mut w, jj, &perm_r, &mut nseg, k, &mut glu);
            column_bmod(&mut w, jj, nseg1, nseg, jcol, k, &mut glu);
            copy_to_ucol(&mut w, jj, nseg, k, &perm_r, &mut glu);
            let pivrow = pivot_l(jj, &mut perm_r, iperm_c, &mut glu)?;
            prune_l(&mut w, jj, &perm_r, pivrow, nseg, k, &mut glu);
            for i in 0..nseg {
                let irep = u(w.segrep[i]);
                w.repfnz[k + irep] = EMPTY;
            }
        }
        jcol += panel_size;
    }
    fixup_l(n, &perm_r, &mut glu);
    Ok((glu, perm_r))
}
