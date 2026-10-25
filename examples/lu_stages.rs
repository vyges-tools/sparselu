// SPDX-License-Identifier: MPL-2.0
// Copyright 2026 Vyges.
//! The sparse LU's stage dump straight from a conductance matrix and a right-hand side written as
//! IEEE bits (`column row bits` per entry, column-major; one `bits` line per row), so the
//! factorisation can be reproduced and compared without the design database.
//!
//! Usage: `cargo run --release --example lu_stages -- <G bits> <J bits> <out>`

use vyges_sparselu::{Csc, SparseLu, REFERENCE_CACHES};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: lu_stages <G bits> <J bits> <out>");
        std::process::exit(2);
    }
    let bits = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).expect("hex bits"));
    let j: Vec<f64> = std::fs::read_to_string(&args[2]).expect("J bits").lines().map(bits).collect();
    let n = j.len();
    let mut a = Csc { nrows: n, ncols: n, col_ptr: vec![0; n + 1], row_idx: Vec::new(), values: Vec::new() };
    for line in std::fs::read_to_string(&args[1]).expect("G bits").lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        let (c, r): (usize, usize) = (w[0].parse().expect("column"), w[1].parse().expect("row"));
        a.col_ptr[c + 1] += 1;
        a.row_idx.push(r);
        a.values.push(bits(w[2]));
    }
    for c in 0..n {
        a.col_ptr[c + 1] += a.col_ptr[c];
    }
    let Ok(lu) = SparseLu::compute(&a, REFERENCE_CACHES) else {
        eprintln!("singular");
        std::process::exit(1);
    };
    std::fs::write(&args[3], lu.stage_lines(&j).join("\n") + "\n").expect("write");
}
