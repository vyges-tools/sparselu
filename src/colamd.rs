// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//
// A translation into Rust of Eigen (https://gitlab.com/libeigen/eigen, commit in eigen-pin.yaml):
//   Eigen/src/OrderingMethods/Eigen_Colamd.h
//   Eigen/src/OrderingMethods/Ordering.h
//   Copyright (C) 2012 Désiré Nuentsa-Wakam
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Those Eigen files are modified versions of COLAMD; its notice is in THIRD-PARTY-NOTICES, and
// this code was modified from it (translated into Rust, Vyges, 2026).
//! The column approximate-minimum-degree ordering the sparse LU runs before it factors.
//!
//! One function per stage, in the order `compute_ordering` runs them: `init_rows_cols`,
//! `init_scoring`, `find_ordering` (with `garbage_collection`, `clear_mark` and
//! `detect_super_cols` as it calls them), `order_children`. Integer only, so the order is exact
//! when every tie is taken the same way: columns leave the degree lists from the head, enter
//! at the head, and the hash buckets chain in the order the pivot row lists its columns.
//!
//! ⚠️ The row and column records overlay several fields on one slot (a column's thickness is
//! later its parent; a row's mark is its first column during a garbage collection). They are
//! kept as the overlaid slots, `s1`..`s4`, so a write through one name is seen through the other
//! exactly where it is in the reference.
// The loops keep the reference's index form (counters, ranges over parallel arrays), so each
// reads against the reference line by line.
#![allow(clippy::needless_range_loop, clippy::explicit_counter_loop, clippy::too_many_arguments)]

/// No column / no row.
const EMPTY: i32 = -1;
/// A killed principal column's `start`.
const DEAD_PRINCIPAL: i32 = -1;
/// A column absorbed into a supercolumn: its `start`.
const DEAD_NON_PRINCIPAL: i32 = -2;
/// The knobs' dense-row and dense-column fractions.
const DENSE_ROW: f64 = 0.5;
const DENSE_COL: f64 = 0.5;

/// A column: `s1` thickness | parent, `s2` score | order, `s3` headhash | hash | prev,
/// `s4` degree_next | hash_next.
#[derive(Debug, Clone, Copy, Default)]
struct Col {
    start: i32,
    length: i32,
    s1: i32,
    s2: i32,
    s3: i32,
    s4: i32,
}

impl Col {
    fn is_dead(&self) -> bool {
        self.start < 0
    }
    fn is_alive(&self) -> bool {
        self.start >= 0
    }
    fn is_dead_principal(&self) -> bool {
        self.start == DEAD_PRINCIPAL
    }
}

/// A row: `s1` degree | p, `s2` mark | first_column.
#[derive(Debug, Clone, Copy, Default)]
struct Row {
    start: i32,
    length: i32,
    s1: i32,
    s2: i32,
}

impl Row {
    fn is_dead(&self) -> bool {
        self.s2 < 0
    }
    fn is_alive(&self) -> bool {
        self.s2 >= 0
    }
    fn kill(&mut self) {
        self.s2 = -1;
    }
}

/// The ordering of a compressed column-major pattern: `perm[p[i]] = i` as the caller builds it,
/// returned here as `p` (the column eliminated `k`-th is `p[k]`).
///
/// `p` is the column pointer array (`n_col + 1` entries), `rows` the row indices in column order.
pub fn compute_ordering(n_row: i32, n_col: i32, p: &[i32], rows: &[i32]) -> Vec<i32> {
    let nnz = p[n_col as usize];
    // The work array the caller sizes (`recommended`), less the column and row records it
    // carves off its end: twice the entries, a slot per column, a fifth of the entries spare.
    let alen = 2 * nnz + n_col + nnz / 5;
    let mut a = vec![0i32; alen as usize];
    a[..nnz as usize].copy_from_slice(&rows[..nnz as usize]);
    let mut p = p.to_vec();
    let mut col = vec![Col::default(); n_col as usize + 1];
    let mut row = vec![Row::default(); n_row as usize + 1];
    init_rows_cols(n_row, n_col, &mut row, &mut col, &mut a, &mut p);
    let (n_col2, max_deg) = init_scoring(n_row, n_col, &mut row, &mut col, &mut a, &mut p);
    find_ordering(n_row, n_col, alen, &mut row, &mut col, &mut a, &mut p, n_col2, max_deg, 2 * nnz);
    order_children(n_col, &mut col, &mut p);
    p.truncate(n_col as usize);
    p
}

/// Column records from `p`, row lengths and the row form; a jumbled column (unsorted or
/// duplicated rows) has its duplicates dropped and the column form rebuilt from the rows.
fn init_rows_cols(n_row: i32, n_col: i32, row: &mut [Row], col: &mut [Col], a: &mut [i32], p: &mut [i32]) {
    for c in 0..n_col as usize {
        col[c].start = p[c];
        col[c].length = p[c + 1] - p[c];
        col[c].s1 = 1;
        col[c].s2 = 0;
        col[c].s3 = EMPTY;
        col[c].s4 = EMPTY;
    }
    for r in row.iter_mut().take(n_row as usize) {
        r.length = 0;
        r.s2 = -1;
    }
    let mut jumbled = false;
    for c in 0..n_col {
        let mut last_row = -1;
        for i in p[c as usize]..p[c as usize + 1] {
            let r = a[i as usize] as usize;
            if r as i32 <= last_row || row[r].s2 == c {
                jumbled = true;
            }
            if row[r].s2 != c {
                row[r].length += 1;
            } else {
                col[c as usize].length -= 1;
            }
            row[r].s2 = c;
            last_row = r as i32;
        }
    }
    row[0].start = p[n_col as usize];
    row[0].s1 = row[0].start;
    row[0].s2 = -1;
    for r in 1..n_row as usize {
        row[r].start = row[r - 1].start + row[r - 1].length;
        row[r].s1 = row[r].start;
        row[r].s2 = -1;
    }
    for c in 0..n_col {
        for i in p[c as usize]..p[c as usize + 1] {
            let r = a[i as usize] as usize;
            if jumbled {
                if row[r].s2 != c {
                    a[row[r].s1 as usize] = c;
                    row[r].s1 += 1;
                    row[r].s2 = c;
                }
            } else {
                a[row[r].s1 as usize] = c;
                row[r].s1 += 1;
            }
        }
    }
    for r in row.iter_mut().take(n_row as usize) {
        r.s2 = 0;
        r.s1 = r.length;
    }
    if jumbled {
        col[0].start = 0;
        p[0] = 0;
        for c in 1..n_col as usize {
            col[c].start = col[c - 1].start + col[c - 1].length;
            p[c] = col[c].start;
        }
        for r in 0..n_row as usize {
            for i in row[r].start..row[r].start + row[r].length {
                let c = a[i as usize] as usize;
                a[p[c] as usize] = r as i32;
                p[c] += 1;
            }
        }
    }
}

/// Kills empty and dense columns (each taking the next order from the END), dense and empty
/// rows, scores the rest, and threads the live columns onto the degree lists, last column
/// first so each list reads in ascending column order. Returns the live column count and the
/// largest live row degree.
fn init_scoring(n_row: i32, n_col: i32, row: &mut [Row], col: &mut [Col], a: &mut [i32], head: &mut [i32]) -> (i32, i32) {
    let dense_row_count = 0.max(((DENSE_ROW * f64::from(n_col)) as i32).min(n_col));
    let dense_col_count = 0.max(((DENSE_COL * f64::from(n_row)) as i32).min(n_row));
    let mut max_deg = 0;
    let mut n_col2 = n_col;
    for c in (0..n_col as usize).rev() {
        if col[c].length == 0 {
            n_col2 -= 1;
            col[c].s2 = n_col2;
            col[c].start = DEAD_PRINCIPAL;
        }
    }
    for c in (0..n_col as usize).rev() {
        if col[c].is_dead() {
            continue;
        }
        if col[c].length > dense_col_count {
            n_col2 -= 1;
            col[c].s2 = n_col2;
            for i in col[c].start..col[c].start + col[c].length {
                row[a[i as usize] as usize].s1 -= 1;
            }
            col[c].start = DEAD_PRINCIPAL;
        }
    }
    for r in row.iter_mut().take(n_row as usize) {
        let deg = r.s1;
        if deg > dense_row_count || deg == 0 {
            r.kill();
        } else {
            max_deg = max_deg.max(deg);
        }
    }
    for c in (0..n_col as usize).rev() {
        if col[c].is_dead() {
            continue;
        }
        let mut score = 0;
        let start = col[c].start;
        let mut new_cp = start;
        for i in start..start + col[c].length {
            let r = a[i as usize];
            if row[r as usize].is_dead() {
                continue;
            }
            a[new_cp as usize] = r;
            new_cp += 1;
            score += row[r as usize].s1 - 1;
            score = score.min(n_col);
        }
        let col_length = new_cp - start;
        if col_length == 0 {
            n_col2 -= 1;
            col[c].s2 = n_col2;
            col[c].start = DEAD_PRINCIPAL;
        } else {
            col[c].length = col_length;
            col[c].s2 = score;
        }
    }
    for h in head.iter_mut().take(n_col as usize + 1) {
        *h = EMPTY;
    }
    for c in (0..n_col as usize).rev() {
        if col[c].is_alive() {
            let score = col[c].s2 as usize;
            let next_col = head[score];
            col[c].s3 = EMPTY;
            col[c].s4 = next_col;
            if next_col != EMPTY {
                col[next_col as usize].s3 = c as i32;
            }
            head[score] = c as i32;
        }
    }
    (n_col2, max_deg)
}

/// The elimination: per step the head of the lowest non-empty degree list is the pivot; its
/// rows' live columns form the new pivot row (collected at `pfree`, collecting the garbage
/// first when the space left is short); set differences, mass elimination, supercolumns, and
/// the new scores back onto the degree lists in pivot-row order.
fn find_ordering(
    n_row: i32,
    n_col: i32,
    alen: i32,
    row: &mut [Row],
    col: &mut [Col],
    a: &mut [i32],
    head: &mut [i32],
    n_col2: i32,
    mut max_deg: i32,
    mut pfree: i32,
) -> i32 {
    let max_mark = i32::MAX - n_col;
    let mut tag_mark = clear_mark(n_row, row);
    let mut min_score = 0i32;
    let mut ngarbage = 0;
    let mut k = 0i32;
    while k < n_col2 {
        while min_score < n_col && head[min_score as usize] == EMPTY {
            min_score += 1;
        }
        let pivot_col = head[min_score as usize] as usize;
        let next_col = col[pivot_col].s4;
        head[min_score as usize] = next_col;
        if next_col != EMPTY {
            col[next_col as usize].s3 = EMPTY;
        }
        let pivot_col_score = col[pivot_col].s2;
        col[pivot_col].s2 = k;
        let pivot_col_thickness = col[pivot_col].s1;
        k += pivot_col_thickness;
        let needed_memory = pivot_col_score.min(n_col - k);
        if pfree + needed_memory >= alen {
            pfree = garbage_collection(n_row, n_col, row, col, a, pfree);
            ngarbage += 1;
            tag_mark = clear_mark(n_row, row);
        }
        let pivot_row_start = pfree;
        let mut pivot_row_degree = 0;
        col[pivot_col].s1 = -pivot_col_thickness;
        for i in col[pivot_col].start..col[pivot_col].start + col[pivot_col].length {
            let r = a[i as usize] as usize;
            if row[r].is_dead() {
                continue;
            }
            for j in row[r].start..row[r].start + row[r].length {
                let c = a[j as usize] as usize;
                let col_thickness = col[c].s1;
                if col_thickness > 0 && col[c].is_alive() {
                    col[c].s1 = -col_thickness;
                    a[pfree as usize] = c as i32;
                    pfree += 1;
                    pivot_row_degree += col_thickness;
                }
            }
        }
        col[pivot_col].s1 = pivot_col_thickness;
        max_deg = max_deg.max(pivot_row_degree);
        for i in col[pivot_col].start..col[pivot_col].start + col[pivot_col].length {
            row[a[i as usize] as usize].kill();
        }
        let pivot_row_length = pfree - pivot_row_start;
        let pivot_row = if pivot_row_length > 0 { a[col[pivot_col].start as usize] } else { EMPTY };

        // Set differences.
        for rp in pivot_row_start..pivot_row_start + pivot_row_length {
            let c = a[rp as usize] as usize;
            let col_thickness = -col[c].s1;
            col[c].s1 = col_thickness;
            let cur_score = col[c].s2;
            let prev_col = col[c].s3;
            let next_col = col[c].s4;
            if prev_col == EMPTY {
                head[cur_score as usize] = next_col;
            } else {
                col[prev_col as usize].s4 = next_col;
            }
            if next_col != EMPTY {
                col[next_col as usize].s3 = prev_col;
            }
            for i in col[c].start..col[c].start + col[c].length {
                let r = a[i as usize] as usize;
                if row[r].is_dead() {
                    continue;
                }
                let mut set_difference = row[r].s2 - tag_mark;
                if set_difference < 0 {
                    set_difference = row[r].s1;
                }
                set_difference -= col_thickness;
                if set_difference == 0 {
                    row[r].kill();
                } else {
                    row[r].s2 = set_difference + tag_mark;
                }
            }
        }

        // Adding set differences: scores, mass elimination, hash buckets.
        for rp in pivot_row_start..pivot_row_start + pivot_row_length {
            let c = a[rp as usize] as usize;
            let mut hash: u32 = 0;
            let mut cur_score = 0i32;
            let start = col[c].start;
            let mut new_cp = start;
            for i in start..start + col[c].length {
                let r = a[i as usize];
                if row[r as usize].is_dead() {
                    continue;
                }
                let row_mark = row[r as usize].s2;
                a[new_cp as usize] = r;
                new_cp += 1;
                hash = hash.wrapping_add(r as u32);
                cur_score += row_mark - tag_mark;
                cur_score = cur_score.min(n_col);
            }
            col[c].length = new_cp - start;
            if col[c].length == 0 {
                col[c].start = DEAD_PRINCIPAL;
                pivot_row_degree -= col[c].s1;
                col[c].s2 = k;
                k += col[c].s1;
            } else {
                col[c].s2 = cur_score;
                hash %= (n_col + 1) as u32;
                let head_column = head[hash as usize];
                let first_col;
                if head_column > EMPTY {
                    first_col = col[head_column as usize].s3;
                    col[head_column as usize].s3 = c as i32;
                } else {
                    first_col = -(head_column + 2);
                    head[hash as usize] = -(c as i32 + 2);
                }
                col[c].s4 = first_col;
                col[c].s3 = hash as i32;
            }
        }

        detect_super_cols(col, a, head, pivot_row_start, pivot_row_length);
        col[pivot_col].start = DEAD_PRINCIPAL;
        tag_mark += max_deg + 1;
        if tag_mark >= max_mark {
            tag_mark = clear_mark(n_row, row);
        }

        // Finalize scores.
        let mut new_rp = pivot_row_start;
        for rp in pivot_row_start..pivot_row_start + pivot_row_length {
            let c = a[rp as usize];
            let cu = c as usize;
            if col[cu].is_dead() {
                continue;
            }
            a[new_rp as usize] = c;
            new_rp += 1;
            a[(col[cu].start + col[cu].length) as usize] = pivot_row;
            col[cu].length += 1;
            let mut cur_score = col[cu].s2 + pivot_row_degree;
            let max_score = n_col - k - col[cu].s1;
            cur_score -= col[cu].s1;
            cur_score = cur_score.min(max_score);
            col[cu].s2 = cur_score;
            let next_col = head[cur_score as usize];
            col[cu].s4 = next_col;
            col[cu].s3 = EMPTY;
            if next_col != EMPTY {
                col[next_col as usize].s3 = c;
            }
            head[cur_score as usize] = c;
            min_score = min_score.min(cur_score);
        }
        if pivot_row_degree > 0 {
            let pr = pivot_row as usize;
            row[pr].start = pivot_row_start;
            row[pr].length = new_rp - pivot_row_start;
            row[pr].s1 = pivot_row_degree;
            row[pr].s2 = 0;
        }
    }
    ngarbage
}

/// Orders each absorbed column right after its supercolumn's principal, in column order, then
/// inverts the order into `p`.
fn order_children(n_col: i32, col: &mut [Col], p: &mut [i32]) {
    for i in 0..n_col as usize {
        if !col[i].is_dead_principal() && col[i].s2 == EMPTY {
            let mut parent = i;
            loop {
                parent = col[parent].s1 as usize;
                if col[parent].is_dead_principal() {
                    break;
                }
            }
            let mut c = i;
            let mut order = col[parent].s2;
            loop {
                col[c].s2 = order;
                order += 1;
                col[c].s1 = parent as i32;
                c = col[c].s1 as usize;
                if col[c].s2 != EMPTY {
                    break;
                }
            }
            col[parent].s2 = order;
        }
    }
    for c in 0..n_col as usize {
        p[col[c].s2 as usize] = c as i32;
    }
}

/// Merges the pivot row's columns that hash alike and have the same length, score and rows
/// (in the same order) into the first of them, bucket by bucket in pivot-row order.
fn detect_super_cols(col: &mut [Col], a: &[i32], head: &mut [i32], row_start: i32, row_length: i32) {
    for rp in row_start..row_start + row_length {
        let c0 = a[rp as usize] as usize;
        if col[c0].is_dead() {
            continue;
        }
        let hash = col[c0].s3 as usize;
        let head_column = head[hash];
        let first_col = if head_column > EMPTY { col[head_column as usize].s3 } else { -(head_column + 2) };
        let mut super_c = first_col;
        while super_c != EMPTY {
            let sc = super_c as usize;
            let length = col[sc].length;
            let mut prev_c = sc;
            let mut c = col[sc].s4;
            while c != EMPTY {
                let cu = c as usize;
                let same = col[cu].length == length
                    && col[cu].s2 == col[sc].s2
                    && (0..length).all(|i| a[(col[sc].start + i) as usize] == a[(col[cu].start + i) as usize]);
                if !same {
                    prev_c = cu;
                    c = col[cu].s4;
                    continue;
                }
                col[sc].s1 += col[cu].s1;
                col[cu].s1 = super_c;
                col[cu].start = DEAD_NON_PRINCIPAL;
                col[cu].s2 = EMPTY;
                col[prev_c].s4 = col[cu].s4;
                c = col[cu].s4;
            }
            super_c = col[sc].s4;
        }
        if head_column > EMPTY {
            col[head_column as usize].s3 = EMPTY;
        } else {
            head[hash] = EMPTY;
        }
    }
}

/// Compacts the live columns' live rows, then the live rows' live columns, to the front of the
/// work array; a row's start slot carries its one's complement meanwhile, its first column held
/// in its mark. Returns the new free position.
fn garbage_collection(n_row: i32, n_col: i32, row: &mut [Row], col: &mut [Col], a: &mut [i32], pfree: i32) -> i32 {
    let mut pdest = 0i32;
    for c in 0..n_col as usize {
        if col[c].is_alive() {
            let mut psrc = col[c].start;
            col[c].start = pdest;
            let length = col[c].length;
            for _ in 0..length {
                let r = a[psrc as usize];
                psrc += 1;
                if row[r as usize].is_alive() {
                    a[pdest as usize] = r;
                    pdest += 1;
                }
            }
            col[c].length = pdest - col[c].start;
        }
    }
    for r in 0..n_row as usize {
        if row[r].is_alive() {
            if row[r].length == 0 {
                row[r].kill();
            } else {
                let psrc = row[r].start as usize;
                row[r].s2 = a[psrc];
                a[psrc] = -(r as i32) - 1;
            }
        }
    }
    let mut psrc = pdest;
    while psrc < pfree {
        let v = a[psrc as usize];
        psrc += 1;
        if v < 0 {
            psrc -= 1;
            let r = (-v - 1) as usize;
            a[psrc as usize] = row[r].s2;
            row[r].start = pdest;
            let length = row[r].length;
            for _ in 0..length {
                let c = a[psrc as usize];
                psrc += 1;
                if col[c as usize].is_alive() {
                    a[pdest as usize] = c;
                    pdest += 1;
                }
            }
            row[r].length = pdest - row[r].start;
        }
    }
    pdest
}

/// Every live row's mark back to zero; the tag restarts at one.
fn clear_mark(n_row: i32, row: &mut [Row]) -> i32 {
    for r in row.iter_mut().take(n_row as usize) {
        if r.is_alive() {
            r.s2 = 0;
        }
    }
    1
}
