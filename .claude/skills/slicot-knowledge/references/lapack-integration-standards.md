# LAPACK Integration Standards for slicot-rs

## Core Principle

**ALWAYS use optimized LAPACK routines via FFI bindings for linear algebra operations. NEVER implement manual algorithms when LAPACK equivalents exist.**

## Why LAPACK Integration is Mandatory

1. **Performance**: LAPACK is 10-50% faster than manual implementations
2. **Numerical Stability**: Decades of refinement for edge cases
3. **Platform Optimization**: Leverages hardware-specific optimizations (SIMD, threading)
4. **Maintainability**: Industry-standard algorithms with well-documented behavior
5. **Correctness**: Battle-tested against numerical pathologies

## Strict Guidelines

### ❌ NEVER Do This

```rust
// WRONG: Manual Householder reflections
for col in 0..n {
    // Compute reflector manually
    // Apply from left and right
    // ... 30+ lines of manual linear algebra
}
```

### ✅ ALWAYS Do This

```rust
// CORRECT: Use LAPACK DGEHRD via FFI
unsafe {
    lapack_sys::dgehrd_(/* proper parameters */);
}
```

## Required LAPACK Operations

When translating SLICOT routines, use these LAPACK routines:

### Matrix Decompositions

| Operation | LAPACK Routine | When to Use |
|-----------|----------------|-------------|
| **QR with pivoting** | DGEQP3 | Rank-revealing decomposition |
| **QR without pivoting** | DGEQRF | Basic QR factorization |
| **Hessenberg reduction** | DGEHRD | Eigenvalue prep, controllability |
| **Schur decomposition** | DGEES | Eigenvalue problems, pole placement |
| **SVD** | DGESDD, DGESVD | Minimum-norm solutions, rank |
| **Eigenvalue decomposition** | DGEEV | General eigenvalue problems |
| **Cholesky** | DPOTRF | Positive definite systems |
| **LU** | DGETRF | General linear systems |

### Matrix Operations

| Operation | LAPACK Routine | When to Use |
|-----------|----------------|-------------|
| **2×2 SVD** | DLASV2 | Givens rotation parameters |
| **Reorder Schur** | DTREXC | Eigenvalue reordering |
| **Apply orthogonal Q** | DORMQR | QR transformation application |
| **Matrix scaling** | DLASCL | Overflow/underflow prevention |

## Implementation Pattern

### 1. Declare FFI Binding

```rust
#[link(name = "lapack")]
extern "C" {
    fn dgehrd_(
        n: *const c_int,
        ilo: *const c_int,
        ihi: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const c_int,
        info: *mut c_int,
    );
}
```

### 2. Create Safe Wrapper

```rust
/// Safe wrapper for LAPACK DGEHRD (Hessenberg reduction)
///
/// # Safety
/// Uses unsafe FFI but validates all inputs before calling LAPACK.
fn call_dgehrd(a: &mut Array2<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();

    // Validate inputs
    if a.ncols() != n {
        return Err("Matrix must be square".to_string());
    }

    // Workspace query
    let mut work_query = vec![0.0; 1];
    let lwork_query = -1;
    let mut info = 0;

    unsafe {
        let mut a_col_major = a.clone().reversed_axes();
        let mut tau = vec![0.0; n.saturating_sub(1)];

        // Query optimal workspace
        dgehrd_(
            &(n as c_int),
            &1,
            &(n as c_int),
            a_col_major.as_mut_ptr(),
            &(n as c_int),
            tau.as_mut_ptr(),
            work_query.as_mut_ptr(),
            &lwork_query,
            &mut info,
        );

        // Allocate optimal workspace
        let lwork = work_query[0] as usize;
        let mut work = vec![0.0; lwork];

        // Actual computation
        dgehrd_(
            &(n as c_int),
            &1,
            &(n as c_int),
            a_col_major.as_mut_ptr(),
            &(n as c_int),
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &(lwork as c_int),
            &mut info,
        );

        // Convert back to row-major
        a.assign(&a_col_major.reversed_axes());

        if info != 0 {
            return Err(format!("DGEHRD failed with INFO={}", info));
        }

        Ok(Array1::from(tau))
    }
}
```

### 3. Use in High-Level Function

```rust
pub fn ab01md(
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    tol: f64,
) -> Result<Ab01mdResult, String> {
    // Use LAPACK for Hessenberg reduction
    let tau = call_dgehrd(a)?;

    // Continue with algorithm...
    Ok(result)
}
```

## Memory Layout Handling

LAPACK uses **column-major** (Fortran) layout; Rust/ndarray defaults to **row-major**.

### Always Handle Conversion

```rust
// Convert to column-major before LAPACK call
let mut a_col_major = a.clone().reversed_axes();

// Call LAPACK
unsafe {
    lapack_routine(a_col_major.as_mut_ptr(), /* ... */);
}

// Convert back to row-major
a.assign(&a_col_major.reversed_axes());
```

### Performance Note

For performance-critical code with many LAPACK calls:
- Consider keeping matrices in column-major layout throughout
- Use `Array2::from_shape_vec` with column-major strides

## Workspace Management

### Always Use Workspace Query

```rust
// Step 1: Query optimal workspace size
let mut work_query = vec![0.0; 1];
let lwork_query = -1;

unsafe {
    lapack_routine(/* params */, &lwork_query, work_query.as_mut_ptr());
}

// Step 2: Allocate optimal workspace
let lwork = work_query[0] as usize;
let mut work = vec![0.0; lwork];

// Step 3: Actual computation
unsafe {
    lapack_routine(/* params */, &(lwork as c_int), work.as_mut_ptr());
}
```

**Why**: LAPACK computes optimal workspace based on algorithm, block size, and platform.

## Error Handling

### Standard Pattern

```rust
let mut info: c_int = 0;

unsafe {
    lapack_routine(/* params */, &mut info);
}

match info {
    0 => Ok(result),
    i if i < 0 => Err(format!("LAPACK parameter {} invalid", -i)),
    i => Err(format!("LAPACK convergence failure, INFO={}", i)),
}
```

## Common Mistakes to Avoid

### ❌ Mistake 1: Manual Loop-Based Algorithms

```rust
// WRONG: Manual QR factorization
for j in 0..n {
    for i in (j+1)..m {
        // Givens rotation computation...
    }
}
```

**Fix**: Use `DGEQRF` or `DGEQP3`

### ❌ Mistake 2: Simplified Algorithms

```rust
// WRONG: Identity matrices as placeholder
let u = Array2::eye(2);
let v = Array2::eye(2);
```

**Fix**: Use proper LAPACK routine (e.g., `DLASV2` for 2×2 SVD)

### ❌ Mistake 3: Reimplementing BLAS

```rust
// WRONG: Manual matrix multiply
for i in 0..m {
    for j in 0..n {
        for k in 0..p {
            c[[i,j]] += a[[i,k]] * b[[k,j]];
        }
    }
}
```

**Fix**: Use ndarray's `.dot()` which calls BLAS DGEMM

### ❌ Mistake 4: Ignoring Workspace Query

```rust
// WRONG: Guessing workspace size
let work = vec![0.0; 10 * n];  // Arbitrary size
```

**Fix**: Always query optimal workspace first

## Testing LAPACK Integration

### Verification Checklist

When integrating LAPACK, verify:

1. ✅ **Correctness**: Results match Fortran SLICOT on identical inputs
2. ✅ **Memory layout**: No corruption from row/column-major mixing
3. ✅ **Performance**: Measure speedup vs manual implementation (should be 10%+)
4. ✅ **Edge cases**: Zero dimensions, singular matrices, nearly-degenerate cases
5. ✅ **Platform**: Test on both macOS (Accelerate) and Linux (OpenBLAS)

### Benchmark Template

```rust
#[test]
fn benchmark_lapack_vs_manual() {
    let n = 100;
    let a = Array2::from_shape_fn((n, n), |(i, j)| {
        ((i + j) as f64).sin()
    });

    // Time LAPACK version
    let start = std::time::Instant::now();
    let result_lapack = function_with_lapack(&a.clone());
    let time_lapack = start.elapsed();

    // Time manual version (if exists for comparison)
    let start = std::time::Instant::now();
    let result_manual = function_manual(&a.clone());
    let time_manual = start.elapsed();

    // Verify results match
    assert!((result_lapack - result_manual).abs() < 1e-10);

    // Verify performance improvement
    println!("LAPACK: {:?}, Manual: {:?}, Speedup: {:.1}x",
             time_lapack, time_manual,
             time_manual.as_secs_f64() / time_lapack.as_secs_f64());
}
```

## Dependency Setup

### Cargo.toml

```toml
[dependencies]
ndarray = "0.15"
ndarray-linalg = "0.16"
lapack-sys = "0.14"  # For raw FFI when needed
num-complex = "0.4"

# Platform-specific BLAS/LAPACK backends
[target.'cfg(target_os = "macos")'.dependencies]
accelerate-src = "0.3"

[target.'cfg(not(target_os = "macos"))'.dependencies]
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

### build.rs

```rust
fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
```

## When to Use ndarray-linalg vs Raw FFI

### Use ndarray-linalg (Preferred)

When high-level traits exist:
- Eigenvalue decomposition: `.eig()`
- SVD: `.svd()`
- Linear solve: `.solve()`
- QR: `.qr()`

**Benefits**: Safe, idiomatic, automatic memory management

### Use Raw FFI

When specific LAPACK routine needed:
- DGEHRD (Hessenberg reduction)
- DGEES (Schur decomposition)
- DTREXC (Schur reordering)
- DLASV2 (2×2 SVD parameters)

**Benefits**: Access to full LAPACK catalog

## Performance Expectations

With proper LAPACK integration:

| Matrix Size | Expected Speedup |
|-------------|------------------|
| N < 10 | 1.0-1.2× (overhead) |
| 10 ≤ N < 50 | 1.2-1.5× |
| 50 ≤ N < 200 | 1.5-2.0× |
| N ≥ 200 | 2.0-5.0× |

## Documentation Requirements

When integrating LAPACK, document:

```rust
/// Computes the controllability staircase form using LAPACK DGEHRD.
///
/// # LAPACK Integration
///
/// Uses `DGEHRD` (Hessenberg reduction) for optimal performance:
/// - Platform: macOS uses Accelerate, Linux uses OpenBLAS
/// - Performance: 10-15% faster than manual Householder for N>20
/// - Workspace: Automatically queries optimal size
///
/// # Algorithm
///
/// Follows SLICOT AB01MD algorithm (Paige, 1981)...
pub fn ab01md(/* ... */) -> Result</* ... */> {
    // Implementation using call_dgehrd()
}
```

## Compliance Verification

Before merging code, verify:

- [ ] No manual matrix decompositions (QR, SVD, Schur, eigen, etc.)
- [ ] All BLAS operations use ndarray `.dot()` or LAPACK
- [ ] Memory layout conversions present for all LAPACK calls
- [ ] Workspace queries used (no hardcoded sizes)
- [ ] Error handling covers all INFO codes
- [ ] Documentation mentions LAPACK routine used
- [ ] Performance measured and documented
- [ ] Tests include reference data from Fortran SLICOT

## Summary

**Golden Rule**: If SLICOT Fortran calls a LAPACK routine, the Rust translation MUST call the same LAPACK routine via FFI. No shortcuts, no simplified algorithms, no manual implementations.

This ensures:
- Algorithmic fidelity to SLICOT
- Optimal performance
- Numerical stability
- Maintainability
