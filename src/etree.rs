// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/SparseCore/SparseColEtree.h
//   Copyright (C) 2012 Désiré Nuentsa-Wakam
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Those Eigen files are modified versions of SuperLU; its notice is in THIRD-PARTY-NOTICES, and
// this code was modified from it (translated into Rust, Vyges, 2026).
//! The column elimination tree of the column-permuted matrix, and its postorder: the order the
//! factorisation visits columns in.

use super::Csc;

/// The root of `i`'s set, halving the path on the way (`etree_find`).
fn etree_find(mut i: usize, pp: &mut [usize]) -> usize {
    let mut p = pp[i];
    let mut gp = pp[p];
    while gp != p {
        pp[i] = gp;
        i = gp;
        p = pp[i];
        gp = pp[p];
    }
    p
}

/// The elimination tree of `AᵀA` from `A`'s columns (`coletree`): `parent[c]` is `c`'s parent, or
/// `ncols` for a root. Each row stands for its first column; a column's missing diagonal is
/// visited after its entries, as if it were there.
pub fn coletree(a: &Csc) -> Vec<usize> {
    let (nc, m) = (a.ncols, a.nrows);
    let diag = nc.min(m);
    let mut root = vec![0usize; nc];
    let mut pp = vec![0usize; nc];
    let mut parent = vec![nc; nc];
    let mut first_row_elt: Vec<usize> = (0..m).map(|r| if r < diag { r } else { nc }).collect();
    for col in 0..nc {
        for &row in a.rows(col) {
            first_row_elt[row] = first_row_elt[row].min(col);
        }
    }
    for col in 0..nc {
        let mut found_diag = col >= m;
        pp[col] = col;
        let mut cset = col;
        root[cset] = col;
        parent[col] = nc;
        let rows = a.rows(col);
        let mut k = 0;
        while k < rows.len() || !found_diag {
            let i = if k < rows.len() { rows[k] } else { col };
            k += 1;
            if i == col {
                found_diag = true;
            }
            let row = first_row_elt[i];
            if row >= col {
                continue;
            }
            let rset = etree_find(row, &mut pp);
            let rroot = root[rset];
            if rroot != col {
                parent[rroot] = col;
                pp[cset] = rset;
                cset = rset;
                root[cset] = col;
            }
        }
    }
    parent
}

/// The tree's postorder (`treePostorder` + `nr_etdfs`): `post[v]` is `v`'s number, children
/// numbered before their parent, siblings in ascending order; the virtual root `n` is numbered `n`.
pub fn tree_postorder(n: usize, parent: &[usize]) -> Vec<usize> {
    const NONE: usize = usize::MAX;
    let mut first_kid = vec![NONE; n + 1];
    let mut next_kid = vec![0usize; n + 1];
    let mut post = vec![0usize; n + 1];
    for v in (0..n).rev() {
        let dad = parent[v];
        next_kid[v] = first_kid[dad];
        first_kid[dad] = v;
    }
    let mut postnum = 0usize;
    let mut current = n;
    while postnum != n {
        let first = first_kid[current];
        if first == NONE {
            post[current] = postnum;
            postnum += 1;
            let mut next = next_kid[current];
            while next == NONE {
                current = parent[current];
                post[current] = postnum;
                postnum += 1;
                next = next_kid[current];
            }
            if postnum == n + 1 {
                return post;
            }
            current = next;
        } else {
            current = first;
        }
    }
    post
}
