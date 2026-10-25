# vyges-sparselu

A supernodal sparse LU factorisation and solve for square, real, double-precision systems, written
so that its results are bit-for-bit those of Eigen 3.4's `SparseLU` (with its default COLAMD
ordering) on an x86-64 baseline build: SSE2 two-lane vectors, no fused multiply-add.

- **Ordering:** column approximate minimum degree (COLAMD), then the column elimination tree's
  postorder.
- **Factorisation:** relaxed supernodes, panel by panel, with partial pivoting that prefers the
  diagonal.
- **Dense kernels:** every floating-point sum is taken in the order the reference kernels take it.
  That covers matrix × vector, the block-panel product, and the triangular solves on vectors and
  on matrices.
- **Cache sizes:** the product blocking reads them, and they are an input (`Caches`), so a result
  can be reproduced for the machine it came from.
- **Solve:** forward and back substitution with the row and column permutations, plus a per-stage
  dump (`stage_lines`) of every factor array for comparison against another run.

Std-only: no dependencies.

```rust
use vyges_sparselu::{Csc, SparseLu, REFERENCE_CACHES};

let lu = SparseLu::compute(&a, REFERENCE_CACHES).expect("non-singular");
let (_, _, x) = lu.solve_stages(&b);
```

`examples/lu_stages.rs` runs the factorisation on a matrix and right-hand side given as IEEE bits
and writes the stage dump.

## Pin

`eigen-pin.yaml` records what the code is transcribed from: the Eigen commit, the source files each
module translates, the build the results match (compiler, flags, instruction set, and the
register-block sizes that build yields) and the cache sizes. Bump it deliberately: a different
Eigen or build changes the last bits.

## Licence

MPL-2.0 (see `LICENSE`). Each source file is a translation of Eigen files that are themselves
MPL-2.0. Those Eigen files carry SuperLU and COLAMD notices, which are reproduced in
`THIRD-PARTY-NOTICES` and must accompany any distribution of this crate, in source or compiled
form. See `NOTICE`.
