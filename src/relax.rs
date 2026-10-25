// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/SparseLU/SparseLU_relax_snode.h
//   Copyright (C) 2012 Désiré Nuentsa-Wakam
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Those Eigen files are modified versions of SuperLU; its notice is in THIRD-PARTY-NOTICES, and
// this code was modified from it (translated into Rust, Vyges, 2026).
//! Relaxed supernodes from the postordered elimination tree (`relax_snode`).

/// `relax_end[j]` is the last column of the relaxed supernode starting at `j`, or `None` where no
/// supernode starts: from each start, climb while the parent has fewer than `relax_columns`
/// descendants; the next start is the next column with no descendants.
pub fn relax_snode(n: usize, et: &[usize], relax_columns: usize) -> Vec<Option<usize>> {
    let mut relax_end = vec![None; n];
    let mut descendants = vec![0usize; n + 1];
    for j in 0..n {
        let parent = et[j];
        if parent != n {
            descendants[parent] += descendants[j] + 1;
        }
    }
    let mut j = 0;
    while j < n {
        let mut parent = et[j];
        let snode_start = j;
        while parent != n && descendants[parent] < relax_columns {
            j = parent;
            parent = et[j];
        }
        relax_end[snode_start] = Some(j);
        j += 1;
        while j < n && descendants[j] != 0 {
            j += 1;
        }
    }
    relax_end
}
