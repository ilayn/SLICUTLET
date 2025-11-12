---
name: slicot-knowledge
description: This skill should be used when working with SLICOT (Subroutine Library In Control Theory) routines, translating Fortran 77 to C, parsing SLICOT HTML documentation, creating test cases from SLICOT examples, understanding SLICOT data formats, or planning translation priorities using dependency analysis tools.
---

# SLICOT Knowledge Skill

Comprehensive knowledge base for working with the SLICOT library, including Fortran-to-C translation, documentation parsing, test data extraction, and dependency analysis.

## When to Use This Skill

Activate this skill when:

1. **Translating SLICOT routines** from Fortran 77 to C
2. **Parsing SLICOT HTML documentation** to understand routine specifications
3. **Creating test cases** from SLICOT example programs and data
4. **Understanding SLICOT data formats** (matrix storage, polynomials, etc.)
5. **Navigating SLICOT library organization** (chapters, naming conventions)
6. **Interpreting Fortran READ statements** for test data
7. **Planning translation order** using the dependency tree (translate leaves first)
8. **Understanding strategic implementation priorities** and time estimates for routine translation
9. **Using dependency analysis tools** to identify SLICOT and LAPACK/BLAS dependencies

## SLICOT Overview

SLICOT is a numerical library for control theoretical computations implemented in Fortran 77. It provides:

- State-space system analysis and synthesis
- Model reduction and realization
- Controller design (pole placement, LQR, H₂, H∞)
- Filtering and identification
- Numerical linear algebra for control theory

### Library Organization

SLICOT routines are organized into functional chapters:

- **A** - Analysis Routines
- **B** - Benchmark and Test Problems
- **D** - Data Analysis
- **F** - Filtering
- **I** - Identification
- **M** - Mathematical Routines
- **N** - Nonlinear Systems
- **S** - Synthesis Routines
- **T** - Transformation Routines
- **U** - Utility Routines

### Naming Convention

Routine names follow the pattern: `XXYYZZ`

- `XX` - Chapter (e.g., AB, SB, MA)
- `YY` - Section number (01, 02, etc.)
- `ZZ` - Variant/subsection (MD, ND, etc.)

**Example**: `AB01MD` = Analysis/Benchmark chapter, section 01, variant MD

## Working with SLICOT HTML Documentation

SLICOT routines have comprehensive HTML documentation in `SLICOT-Reference/doc/*.html`.

### Standard Documentation Structure

Each HTML file contains these sections in order:

1. **Header** - Routine name and brief description
2. **Purpose** - Detailed problem description and mathematical formulation
3. **Specification** - Fortran subroutine signature with parameter types
4. **Arguments** - Detailed parameter descriptions by category:
   - Mode Parameters (JOBZ, UPLO, TRANS, etc.)
   - Input/Output Parameters
   - Tolerances
   - Workspace (DWORK, IWORK, LDWORK, LIWORK)
   - Error Indicator (INFO)
5. **Method** - Algorithm description and numerical techniques
6. **References** - Academic papers and textbooks
7. **Numerical Aspects** - Complexity and stability analysis
8. **Example** - Complete program with:
   - **Program Text** - Fortran example code
   - **Program Data** - Test inputs
   - **Program Results** - Expected outputs

### Parsing Workflow

#### For Implementation

1. Read **Purpose** to understand the mathematical problem
2. Read **Method** to understand the algorithm
3. Read **Specification** to design the function signature
4. Read **Arguments** to implement parameter handling and validation
5. Read **Numerical Aspects** to set performance expectations

#### For Testing

1. Read **Example Program Text** to find READ statements showing data format
2. Read **Program Data** to extract test inputs
3. Read **Program Results** to extract expected outputs
4. Parse data according to Fortran column-major convention
5. Write Python test with appropriate numerical tolerances (rtol=1e-14 for tight validation, or 1e-3 to 5e-3 for looser comparison)

### Key Information Locations

| Need | Look Here |
|------|-----------|
| What does this routine do? | **Purpose** section |
| What algorithm is used? | **Method** section |
| Function signature | **Specification** section |
| Parameter details | **Arguments** section |
| Test inputs | **Example → Program Data** |
| Expected results | **Example → Program Results** |
| Data input format | **Example → Program Text** (READ statements) |

## Parsing Test Data (CRITICAL)

SLICOT uses Fortran column-major storage, but HTML examples may present data differently.

### The Golden Rule

**ALWAYS examine the Fortran READ statements in "Program Text" to determine the true data format.**

### Common Fortran READ Patterns

#### Pattern 1: Column-wise Matrix Read

```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), I = 1,M ), J = 1,N )
```

Reads M×N matrix **column-by-column**:
- First M values → Column 1
- Next M values → Column 2
- Continue for N columns

#### Pattern 2: Row-wise Matrix Read

```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,M )
```

Reads M×N matrix **row-by-row**:
- First N values → Row 1
- Next N values → Row 2
- Continue for M rows

#### Pattern 3: Vector Read

```fortran
READ ( NIN, FMT = * ) ( B(I), I = 1,N )
```

Reads vector sequentially.

### Implied DO Loop Interpretation

For nested loops: `( ( ARRAY(I,J), <inner> ), <outer> )`

- **Inner loop** varies fastest (reads contiguously)
- **Outer loop** varies slowest (advances to next block)

Example: `((A(I,J), I=1,3), J=1,2)` reads:
```
A(1,1), A(2,1), A(3,1),  ← Column 1
A(1,2), A(2,2), A(3,2)   ← Column 2
```

### Practical Example

**HTML Data**:
```
  0.0000  2.0000  1.0000
 -1.0000 -0.1000  1.0000
```

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M )
```

With N=3, M=2, this reads **column-wise**:

Reading HTML sequentially: `0.0, 2.0, 1.0, -1.0, -0.1, 1.0`

With READ `((B(I,J), I=1,3), J=1,2)`:
- Column 1: B(1,1)=0.0, B(2,1)=2.0, B(3,1)=1.0
- Column 2: B(1,2)=-1.0, B(2,2)=-0.1, B(3,2)=1.0

**Python NumPy representation** (row-major, C order):
```python
B = np.array([
    [  0.0,  -1.0 ],
    [  2.0,  -0.1 ],
    [  1.0,   1.0 ]
])
```

### Common Pitfalls

1. **Assuming HTML presentation = data order**
   - Solution: Check Fortran READ loops

2. **Mismatched dimensions**
   - Solution: Count total numbers (should = M×N), verify with dimensions

3. **Ignoring workspace arrays**
   - Solution: Only parse problem data (A, B, C, D, etc.), skip DWORK/IWORK

4. **Wrong tolerance in tests**
   - HTML shows ~4-5 decimal places
   - Use tolerance rtol=1e-14 for exact comparisons (default for most tests)
   - Use 1e-3 to 5e-3 for algorithms with natural variations

## SLICOT Data Formats

### Matrix Storage Schemes

#### 1. Conventional (Full) Storage
- **Column-major** ordering (Fortran standard)
- Element A(i,j) at index `i-1 + (j-1)*LDA` (C 0-based)
- Leading dimension `LDA` ≥ number of rows

#### 2. Packed Storage
- For symmetric, Hermitian, or triangular matrices
- Only relevant triangle stored in 1D array
- Array names end with 'P'
- **Upper triangle** (UPLO='U'): A(i,j) → AP[i + j(j-1)/2] for i ≤ j
- **Lower triangle** (UPLO='L'): A(i,j) → AP[i + (2n-j)(j-1)/2] for i ≥ j

#### 3. Band Storage
- For banded matrices (kl subdiagonals, ku superdiagonals)
- Stored in (kl+ku+1) × n array
- Array names end with 'B'

#### 4. Special Formats
- **Tridiagonal**: 3 separate 1D arrays (diagonal, super-diagonal, sub-diagonal)
- **Bidiagonal**: 2 separate 1D arrays
- **Householder reflectors**: Factored form for orthogonal transformations

### Polynomial Storage

#### Scalar Polynomials
- 1D array P(M) where P(i+1) = coefficient of z^i
- Degree n requires M ≥ n+1

#### Vector Polynomials
- 2D array P(K,M): K components, M coefficients
- Optional degree array DEGP(K)

#### Matrix Polynomials
- 3D array P(K,L,M): K×L matrices, M coefficients
- Optional degree array DEGP(K,L)

## Fortran to C Type Mapping

| Fortran | C (SLICUTLET) | Notes |
|---------|---------------|-------|
| `CHARACTER*1` | `const char*` | Mode parameters as strings |
| `INTEGER` | `i32` | typedef for int32_t |
| `INTEGER` (ILP64) | `i64` | typedef for int64_t with -Dilp64 |
| `DOUBLE PRECISION` | `f64` | typedef for double |
| `COMPLEX*16` | `c128` | typedef for complex double |
| `DOUBLE PRECISION array(N)` | `f64*` | Pointer to array |
| `DOUBLE PRECISION array(LDA,*)` | `f64*` | Pointer to 2D array (column-major) |
| `INFO` parameter | `i32* info` | Last parameter, by pointer |

### Common Mode Parameters

Mode parameters in Fortran (CHARACTER*1) are passed as string literals in C:

```c
// Fortran: JOBZ = 'N' / 'F' / 'I'
// C: Pass as string literals
const char* jobz = "N";  // Do not compute
const char* jobz = "F";  // Factored form
const char* jobz = "I";  // Initialize and compute

// Fortran: UPLO = 'U' / 'L'
// C: Pass as string literals
const char* uplo = "U";  // Upper triangle
const char* uplo = "L";  // Lower triangle

// Fortran: TRANS = 'N' / 'T' / 'C'
// C: Pass as string literals
const char* trans = "N";  // No transpose
const char* trans = "T";  // Transpose
const char* trans = "C";  // Conjugate transpose
```

## BLAS/LAPACK Dependencies

SLICOT routines extensively use BLAS and LAPACK. When translating to C:

### BLAS/LAPACK Call Pattern

All BLAS/LAPACK calls use `SLC_*` prefixed macros defined in `src/include/slc_blaslapack.h`:

```c
// Fortran: CALL DGEMM('N', 'N', M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
// C translation:
SLC_DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
//         ^strings   ^scalar pointers  ^arrays (already pointers)

// Fortran: ANORM = DLANGE('M', N, N, A, LDA, WORK)
// C translation:
anorm = SLC_DLANGE("M", &n, &n, a, &lda, work);

// Fortran: CALL DCOPY(N, X, INCX, Y, INCY)
// C translation:
SLC_DCOPY(&n, x, &incx, y, &incy);

// Fortran: CALL DGETRF(M, N, A, LDA, IPIV, INFO)
// C translation:
SLC_DGETRF(&m, &n, a, &lda, ipiv, &info);
```

**Key Rules:**
1. All BLAS/LAPACK calls use `SLC_*` prefixed macros (e.g., `SLC_DGEMM`, `SLC_DLANGE`)
2. These macros are defined in `src/include/slc_blaslapack.h`
3. Macros resolve to correct symbol mangling via `SLC_F77_FUNC` based on build-time detection
4. All arguments passed by pointer (Fortran calling convention): use `&variable` for scalars
5. Character arguments passed as string literals: `"N"`, `"T"`, `"F"`, etc.

### Build System Auto-Detection

- Detects symbol mangling at configure time: `lowercase_`, `lowercase`, or `UPPERCASE`
- Sets one of: `SLC_FC_LOWER_US`, `SLC_FC_LOWER`, or `SLC_FC_UPPER`
- `SLC_F77_FUNC(dgemm, DGEMM)` expands to correct symbol

### Common BLAS/LAPACK Routines

| Fortran Routine | Purpose | C Macro | Example Usage |
|----------------|---------|---------|---------------|
| DGEQP3 | QR with pivoting | `SLC_DGEQP3` | `SLC_DGEQP3(&m, &n, a, &lda, jpvt, tau, work, &lwork, &info);` |
| DGEMM | Matrix multiply | `SLC_DGEMM` | `SLC_DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);` |
| DGEMV | Matrix-vector multiply | `SLC_DGEMV` | `SLC_DGEMV("N", &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);` |
| DLASET | Matrix initialization | `SLC_DLASET` | `SLC_DLASET("F", &n, &n, &dbl0, &dbl1, z, &ldz);` |
| DLANGE | Matrix norm | `SLC_DLANGE` | `anorm = SLC_DLANGE("M", &n, &n, a, &lda, dwork);` |
| DLAMCH | Machine parameters | `SLC_DLAMCH` | `smlnum = SLC_DLAMCH("S");` |
| DSWAP | Vector swap | `SLC_DSWAP` | `SLC_DSWAP(&n, x, &incx, y, &incy);` |
| DCOPY | Vector copy | `SLC_DCOPY` | `SLC_DCOPY(&n, x, &incx, y, &incy);` |

### Adding New BLAS/LAPACK Routine

If the dependency tool finds a LAPACK routine not yet in `slc_blaslapack.h`:

1. Add declaration in `slc_blaslapack.h` (with `#define int sl_int` in effect):
```c
void   SLC_F77_FUNC(dgesv, DGESV)(const int* n, const int* nrhs, f64* a,
                                   const int* lda, int* ipiv, f64* b,
                                   const int* ldb, int* info);
```

2. Add macro alias below `#undef int`:
```c
#define SLC_DGESV   SLC_F77_FUNC(dgesv, DGESV)
```

3. Use in code:
```c
SLC_DGESV(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
```

## Common SLICOT Patterns

### Pattern 1: Workspace Allocation

Fortran routines require workspace arrays (DWORK, IWORK).

**In C**: Handle workspace internally or require caller to provide it.

```c
// Option 1: Internal allocation (if workspace size is fixed)
f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
// ... use dwork ...
free(dwork);

// Option 2: Caller-provided workspace (common pattern)
// Function signature includes dwork parameter
void routine_name(
    /* ... other params ... */
    f64* dwork,      // Workspace array
    const i32 ldwork // Workspace size
);
```

### Pattern 2: INFO Error Codes

```fortran
INFO = 0:  Success
INFO < 0:  Parameter -INFO is invalid
INFO > 0:  Algorithm-specific error
```

**In C**: Use INFO parameter as last function parameter:

```c
void routine_name(
    const i32 param1,
    f64* param2,
    i32* info        // Last parameter
)
{
    *info = 0;  // Initialize to success

    // Validate parameters
    if (n < 0) {
        *info = -1;  // First parameter invalid
        return;
    }

    // Quick return for zero dimensions
    if (n == 0) {
        return;  // Success, nothing to do
    }

    // Compute

    // Check algorithm-specific errors
    if (!converged) {
        *info = 1;  // Convergence failure
        return;
    }
}
```

### Pattern 3: Input/Output Arrays

Many SLICOT routines modify arrays in-place.

```c
void routine_name(
    const i32 n,
    f64* a,          // Modified in-place
    const i32 lda,
    const f64* b,    // Read-only input (const)
    f64* c,          // Output array
    i32* info
);
```

### Pattern 4: Zero-Dimension Edge Cases

Many routines accept N=0 or M=0 as valid (quick return).

```c
// Check Fortran source for quick return conditions
if (n == 0) {
    return;  // Quick return, *info already 0
}
```

Don't assume zero dimensions are errors unless Fortran code validates them.

## C Translation File Structure

### Standard C File Template

```c
#include "slicutlet.h"              // Public API header
#include "../include/slc_blaslapack.h"  // BLAS/LAPACK wrapper macros
#include <math.h>                   // If needed for fabs, sqrt, etc.

void routine_name(
    const i32 param1,    // Input parameters with const
    f64* param2,         // Output/inout parameters as pointers
    i32* info            // Info parameter last
)
{
    // Local variables
    i32 i, j, k;
    i32 int1 = 1, int0 = 0;
    f64 dbl1 = 1.0, dbl0 = 0.0;
    f64 temp, alpha, beta;

    // Initialize info
    *info = 0;

    // Parameter validation
    if (param1 < 0) {
        *info = -1;
        return;
    }

    // Quick return
    if (param1 == 0) {
        return;
    }

    // Algorithm implementation
    // ...
}
```

### Array Indexing Translation

```c
// Fortran (1-based):  A(I,J) where I=1..M, J=1..N
// C (0-based):        a[(j-1)*lda + (i-1)]  or  a[j*lda + i] with adjusted indices

// Common pattern for Fortran column-major access:
for (i32 j = 0; j < n; j++) {
    for (i32 i = 0; i < m; i++) {
        f64 value = a[j * lda + i];  // Column j, row i
    }
}

// Fortran: A(I,J) = B(I) * C(J)
// C (with i, j adjusted from Fortran loops starting at 1):
for (i32 j = 0; j < n; j++) {
    for (i32 i = 0; i < m; i++) {
        a[j * lda + i] = b[i] * c[j];
    }
}
```

### Internal SLICOT Call Pattern

```c
// Fortran: CALL MB01PD(SCUN, TYPE, M, N, KL, KU, ANRM, NBL, NROWS, A, LDA, INFO)
// C translation:
mb01pd(scun, type, m, n, kl, ku, anrm, nbl, nrows, a, lda, info);
// Direct function call - lowercase name, pass variables directly (already correct types)
```

## Dependency Analysis Tools

### extract_dependencies.py

**Purpose**: Analyze SLICOT Fortran source files to build dependency trees and identify translation order.

**Location**: `tools/extract_dependencies.py`

#### Features

1. **Dependency Classification**:
   - Identifies SLICOT-to-SLICOT dependencies (routine calls within library)
   - Identifies LAPACK/BLAS dependencies (for external call planning)
   - Computes dependency levels (leaf routines → higher-level routines)

2. **Level-Based Analysis**:
   - **Level 0 (Leaves)**: No SLICOT dependencies, only BLAS/LAPACK
     - Can be translated **in parallel**
     - Foundation for higher-level routines
   - **Level 1+**: Depend on routines from lower levels
     - Must be translated **after** dependencies

3. **Translation Planning**:
   - Shows which routines can be implemented immediately (Level 0)
   - Shows which routines block higher-level implementations
   - Identifies LAPACK/BLAS call requirements

#### Usage

**Show Overall Summary:**
```bash
python3 tools/extract_dependencies.py SLICOT-Reference/src/
```

Output:
- Total routines and dependency levels
- Count of routines per level
- First 50 Level 0 routines (translation priorities)
- Sample higher-level routines with dependencies

**Analyze Specific Routine:**
```bash
python3 tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD
```

Output:
- Dependency level of the routine
- Direct SLICOT dependencies (what it calls)
- LAPACK/BLAS dependencies (for C implementation)
- Reverse dependencies (what calls this routine)

**Analyze Single File:**
```bash
python3 tools/extract_dependencies.py SLICOT-Reference/src/MB01PD.f
```

#### Example: Understanding AB01MD Dependencies

```bash
$ python3 tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD

================================================================================
ROUTINE: AB01MD
================================================================================
Dependency Level: 2

SLICOT Dependencies (1):
  ✓ MB01PD   (Level 1)

LAPACK/BLAS Dependencies (5):
  DGEHRD
  DLARF
  DLARFG
  DLASET
  DORGQR
```

**Interpretation**:
- AB01MD is Level 2 (requires MB01PD first)
- MB01PD must be implemented before AB01MD (or use existing translation)
- Need to use these LAPACK macros in C translation:
  - `SLC_DGEHRD(...)` - Hessenberg reduction
  - `SLC_DLARF(...)` - Apply elementary reflector
  - `SLC_DLARFG(...)` - Generate elementary reflector
  - `SLC_DLASET(...)` - Set matrix to constant
  - `SLC_DORGQR(...)` - Generate Q from QR

#### Output Symbols

- `✓` - Routine found in source files (exists)
- `✗` - Routine not found (external dependency or missing)

#### Dependency Levels

- **Level 0**: Immediate translation priority (leaves)
- **Level 1**: Second priority (depend only on Level 0)
- **Level 2+**: Higher-level routines (depend on multiple levels)

## Complete Translation Workflow

### Step 1: Check Dependencies

```bash
python3 tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD
```

Ensure SLICOT dependencies are translated and identify required LAPACK/BLAS routines.

### Step 2: Find Source Files

```bash
# Original Fortran in SLICOT-Reference/src/
ls SLICOT-Reference/src/AB*.f    # AB family
ls SLICOT-Reference/src/MA*.f    # MA family
```

### Step 3: Read Fortran Source

```bash
# Read the Fortran source
cat SLICOT-Reference/src/AB01MD.f
```

### Step 4: Read HTML Documentation

Read HTML documentation to understand:
- **Purpose**: What the routine does
- **Method**: Algorithm description
- **Arguments**: Parameter specifications
- **Example**: Test data and expected results

### Step 5: Create C File

```bash
# Create C file in appropriate directory
# Fortran: AB01MD.f → C: src/AB/ab01md.c (lowercase)
touch src/AB/ab01md.c
```

### Step 6: Translate to C

Follow these patterns:

1. **Include headers**:
   ```c
   #include "slicutlet.h"
   #include "../include/slc_blaslapack.h"
   ```

2. **Map types**:
   - `INTEGER` → `i32`
   - `DOUBLE PRECISION` → `f64`
   - `COMPLEX*16` → `c128`

3. **BLAS/LAPACK calls**: Use `SLC_*` macros with `&` for scalars
   ```c
   SLC_DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
   ```

4. **Internal SLICOT calls**: Direct function calls (lowercase names)
   ```c
   mb01pd(scun, type, m, n, kl, ku, anrm, nbl, nrows, a, lda, info);
   ```

5. **Adjust array indices**: Fortran 1-based → C 0-based

### Step 7: Add to meson.build

```python
# Add to lib_srcs list in meson.build
lib_srcs = files(
  # AB family
  'src/AB/ab01md.c',  # ← Add here
  'src/AB/ab01nd.c',
  ...
)
```

### Step 8: Rebuild

```bash
meson compile -C build
```

### Step 9: Write Python Test

Parse test data from HTML documentation and write Python test in `python/tests/test_abXXxx.py`:

```python
def test_ab01md():
    # Parse test data from HTML documentation
    # Follow Fortran READ statements for correct data format
    result = slicutlet.ab01md(...)
    assert_allclose(result, expected, rtol=1e-14)
```

### Step 10: Run Tests

```bash
uv run pytest python/tests/test_ab01xx.py -v
```

## Common Translation Challenges

### Challenge 1: Column-Major Storage

**Problem**: Fortran uses column-major, C typically uses row-major.

**Solution**: SLICUTLET maintains Fortran column-major ordering for BLAS/LAPACK compatibility.

```c
// Element A(i,j) in Fortran (1-based)
// = a[(j-1)*lda + (i-1)] in C (0-based)
// = a[j*lda + i] when i,j are already 0-based loop variables
```

### Challenge 2: 1-Based vs 0-Based Indexing

**Problem**: Fortran uses 1-based indexing, C uses 0-based. This is subtle beyond simple loop conversion.

#### Simple Case: Loop Variables

```fortran
DO I = 1, N
   A(I,J) = ...
END DO
```

```c
for (i32 i = 0; i < n; i++) {
    a[j * lda + i] = ...;
}
```

#### Complex Case: Variables Representing Fortran Array Indices ⚠️

When Fortran code uses variables like `K` to represent array indices (not loop counters), careful mapping is required:

**Fortran Code:**
```fortran
K = DP          ! K is a Fortran array index (1-based)
ALPHA = DWORK(K+1) / DWORK(K)  ! Access DWORK(K) and DWORK(K+1)
DWORK(I) = DWORK(I) - ALPHA * DWORK(I-1)
```

**Common Mistake:**
```c
// WRONG: Direct translation loses the 1-based conceptual model
i32 k = dp;
f64 alpha = dwork[k] / dwork[k - 1];  // ❌ This is incorrect!
```

**Correct Mapping:**
```c
// RIGHT: Treat k as a Fortran 1-based index, convert each access
i32 k = dp;  // k represents Fortran array position (1-based concept)
// DWORK(K) in Fortran = dwork[k-1] in C (0-based)
// DWORK(K+1) in Fortran = dwork[k] in C
f64 alpha = dwork[k] / dwork[k - 1];  // ✓ Correct

// Update loop: Fortran DO I = K, 2, -2: DWORK(I) = DWORK(I) - ALPHA*DWORK(I-1)
i32 i = k;
while (i >= 2) {
    dwork[i - 1] -= alpha * dwork[i - 2];  // Convert both indices
    i -= 2;
}
```

#### Workspace Offset Conversion ⚠️

When Fortran calculates workspace offsets using 1-based indices:

**Fortran Code:**
```fortran
K2 = DP + 2  ! Workspace offset calculated as Fortran 1-based index
CALL DCOPY(K1, DWORK(K), 1, DWORK(K2), 1)
```

**Correct C Conversion:**
```c
// K2 = DP + 2 in Fortran (1-based) becomes DP + 1 in C (0-based)
i32 k2 = dp + 1;  // NOT dp + 2!

// DWORK(K) in Fortran = dwork[k-1] in C
// DWORK(K2) in Fortran = dwork[k2-1] in C
SLC_DCOPY(&k1, &dwork[k - 1], &int1, &dwork[k2 - 1], &int1);
```

#### Key Principle

When translating Fortran algorithms with variable-based indices:
1. **Identify conceptually which indices are "Fortran 1-based"** (not 0-based loop variables)
2. **Add comments mapping Fortran to C** (e.g., "DWORK(K) in Fortran = dwork[k-1] in C")
3. **Check all array accesses** using that variable for correct -1 offset
4. **Watch for offset calculations** like `DP + 2` which also need adjustment

This pattern caught the MC01TD implementation (polynomial stability checker) - see git commit c0276fc for the fix.

### Challenge 3: Workspace Management

**Problem**: Fortran requires explicit workspace arrays.

**Solution**: Allocate workspace internally in C.

```c
// Calculate required workspace
i32 ldwork = 3 * n + m;
f64* dwork = (f64*)malloc(ldwork * sizeof(f64));
// ... use dwork ...
free(dwork);
```

### Challenge 4: Error Handling

**Problem**: Fortran uses INFO parameter for error codes.

**Solution**: Map to C INFO parameter as last function parameter.

```c
void routine_name(
    const i32 n,
    f64* a,
    i32* info
)
{
    *info = 0;  // Initialize to success

    if (n < 0) {
        *info = -1;  // Invalid parameter
        return;
    }

    // Compute...

    if (!converged) {
        *info = 1;  // Algorithm-specific error
        return;
    }
}
```

## Python Testing

### Test Strategy

Python tests validate:
- Numerical correctness against known results
- Edge cases (zero matrices, singular cases)
- Error handling (via pytest.raises)
- Uses `numpy.testing.assert_allclose` with tight tolerances (rtol=1e-14)

### Test Structure

- One file per function family (test_ma01xx.py, test_mb01xx.py, etc.)
- Classes group related tests (TestMA01AD, TestMA01BD, etc.)
- Descriptive method names indicating test scenario

```python
import numpy as np
from numpy.testing import assert_allclose
import pytest
import slicutlet

class TestAB01MD:
    def test_basic_case(self):
        # Parse test data from HTML documentation
        n = 3
        m = 2
        a = np.array([...], order='F')  # Fortran order (column-major)
        b = np.array([...], order='F')

        # Call routine
        result = slicutlet.ab01md(n, m, a, b)

        # Verify against expected results
        expected = np.array([...], order='F')
        assert_allclose(result, expected, rtol=1e-14)

    def test_zero_dimensions(self):
        # Test edge case
        n = 0
        result = slicutlet.ab01md(n, ...)
        # Verify quick return behavior
```

### Running Tests

```bash
# All tests
uv run pytest python/tests/

# Single file
uv run pytest -v python/tests/test_mb01xx.py

# Specific test
uv run pytest -v python/tests/test_ma01xx.py::TestMA01AD::test_positive_real_positive_imag
```

## Quick Reference

### Finding Information

| Question | Location |
|----------|----------|
| What does routine XYZ do? | HTML: **Purpose** section |
| How does the algorithm work? | HTML: **Method** section |
| What are the parameters? | HTML: **Arguments** section |
| What's the function signature? | HTML: **Specification** section |
| How to parse test data? | HTML: **Example → Program Text** (READ statements) |
| What are test inputs? | HTML: **Example → Program Data** |
| What are expected results? | HTML: **Example → Program Results** |
| What's the complexity? | HTML: **Numerical Aspects** section |
| Where's the Fortran source? | `SLICOT-Reference/src/XYZ.f` |
| Where are Fortran examples? | `SLICOT-Reference/examples/TXYZ.f` |
| What are the dependencies? | `python3 tools/extract_dependencies.py SLICOT-Reference/src/ XYZ` |
| What LAPACK calls are needed? | Dependency tool output: **LAPACK/BLAS Dependencies** |

### Essential Reminders

1. **Always check Fortran READ statements** for data format
2. **Use appropriate tolerances** (rtol=1e-14 for most tests)
3. **Check for zero-dimension edge cases** in Fortran source
4. **Use SLC_* macros** for all BLAS/LAPACK operations
5. **Use dependency tool** before translating to identify prerequisites
6. **Validate parameters** but don't be stricter than Fortran
7. **Trust Fortran source** over HTML if they conflict
8. **Pass scalars by pointer** to BLAS/LAPACK (use `&variable`)
9. **Pass strings as literals** to BLAS/LAPACK (use `"N"`, `"T"`, etc.)

### File Locations

- **Fortran source**: `SLICOT-Reference/src/XXYYZZ.f`
- **HTML docs**: `SLICOT-Reference/doc/XXYYZZ.html`
- **C translation**: `src/XX/xxyyzz.c` (lowercase)
- **Python tests**: `python/tests/test_xxyyxx.py`
- **Type definitions**: `src/include/types.h`
- **BLAS/LAPACK headers**: `src/include/slc_blaslapack.h`
- **Public API**: `include/SLICUTLET/slicutlet.h`
- **Dependency tool**: `tools/extract_dependencies.py`

### Code Structure

- `src/AB/` - State-space transformations
- `src/MA/` - Matrix operations (ma01xx, ma02xx families)
- `src/MB/` - Matrix operations (mb01xx, mb03xx families)
- `src/MC/` - Polynomial operations (mc01xx family)
- `src/B/` - Test utilities

## Getting Help

If stuck:

1. **Use dependency tool** to identify prerequisites
2. **Read HTML documentation** for algorithm understanding
3. **Examine Fortran source** in `SLICOT-Reference/src/`
4. **Look at Fortran examples** in `SLICOT-Reference/examples/`
5. **Check tools/README.md** for complete C translation workflow
6. **Review existing C implementations** in `src/` for patterns
7. **Check SLICOT papers** referenced in HTML documentation
8. **Review LAPACK/BLAS documentation** for specific operations
