# SLICOT Dependency Analysis Tools

This directory contains tools for analyzing dependencies in SLICOT Fortran 77 source code to aid in systematic C translation.

## extract_dependencies.py

**Purpose**: Analyze SLICOT Fortran source files to build dependency trees and identify translation order.

### Features

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

### Usage

#### Show Overall Summary

```bash
uv run tools/extract_dependencies.py SLICOT-Reference/src/
```

Output:
- Total routines and dependency levels
- Count of routines per level
- First 50 Level 0 routines (translation priorities)
- Sample higher-level routines with dependencies

#### Analyze Specific Routine

```bash
uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD
```

Output:
- Dependency level of the routine
- Direct SLICOT dependencies (what it calls)
- LAPACK/BLAS dependencies (for C implementation)
- Reverse dependencies (what calls this routine)

#### Analyze Single File

```bash
uv run tools/extract_dependencies.py SLICOT-Reference/src/MB01PD.f
```

### Examples

**Example 1: Finding leaf routines to translate first**

```bash
$ uv run tools/extract_dependencies.py SLICOT-Reference/src/ | grep "Level 0"
Level 0: 297 routines
  (Leaf routines - no SLICOT dependencies, only BLAS/LAPACK)
```

**Result**: 297 routines can be translated immediately and in parallel.

**Example 2: Understanding AB01MD dependencies**

```bash
$ uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD

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
- Need to call LAPACK routines via macros:
  ```c
  SLC_DGEHRD(...);   // Hessenberg reduction
  SLC_DLARF(...);    // Apply elementary reflector
  SLC_DLARFG(...);   // Generate elementary reflector
  SLC_DLASET(...);   // Set matrix to constant
  SLC_DORGQR(...);   // Generate Q from QR
  ```

**Example 3: Finding what depends on MB03QY**

```bash
$ uv run tools/extract_dependencies.py SLICOT-Reference/src/ MB03QY

Used By (25 routines):
  MB03QD
  SB01BD
  SB02MD
  ...
```

**Interpretation**: Implementing MB03QY will unblock 25 other routines.

### Output Interpretation

#### Symbols

- `✓` - Routine found in source files (exists)
- `✗` - Routine not found (external dependency or missing)

#### Dependency Levels

- **Level 0**: Immediate translation priority (leaves)
- **Level 1**: Second priority (depend only on Level 0)
- **Level 2+**: Higher-level routines (depend on multiple levels)

#### LAPACK Dependencies

Common LAPACK/BLAS routines and how to call them in C:

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

**Translation Pattern:**
1. All BLAS/LAPACK calls use `SLC_*` prefixed macros (e.g., `SLC_DGEMM`, `SLC_DLANGE`)
2. These macros are defined in `src/include/slc_blaslapack.h`
3. Macros resolve to correct symbol mangling via `SLC_F77_FUNC` based on build-time detection
4. All arguments passed by pointer (Fortran calling convention): use `&variable` for scalars
5. Character arguments passed as string literals: `"N"`, `"T"`, `"F"`, etc.

**Build System Auto-Detection:**
- Detects symbol mangling at configure time: `lowercase_`, `lowercase`, or `UPPERCASE`
- Sets one of: `SLC_FC_LOWER_US`, `SLC_FC_LOWER`, or `SLC_FC_UPPER`
- `SLC_F77_FUNC(dgemm, DGEMM)` expands to correct symbol

### C Translation Code Patterns

When translating SLICOT Fortran routines to C, follow these established patterns:

#### File Structure
```c
#include "slicutlet.h"              // Public API header
#include "../include/slc_blaslapack.h"  // BLAS/LAPACK wrapper macros
#include <math.h>                   // If needed

void routine_name(
    const i32 param1,    // Input parameters with const
    f64* param2,         // Output/inout parameters as pointers
    i32* info            // Info parameter last
)
{
    // Local variables
    i32 int1 = 1, int0 = 0;
    f64 dbl1 = 1.0, dbl0 = 0.0;

    // Function body
}
```

#### Type Mappings
```c
// Fortran → C (defined in src/include/types.h)
INTEGER         → i32 (typedef int32_t)
DOUBLE PRECISION → f64 (typedef double)
COMPLEX*16      → c128 (typedef complex double)
```

#### BLAS/LAPACK Call Pattern
```c
// Fortran:     CALL DGEMM('N', 'N', M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
// C translation:
SLC_DGEMM("N", "N", &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
//         ^strings   ^scalar pointers  ^arrays (already pointers)

// Fortran:     ANORM = DLANGE('M', N, N, A, LDA, WORK)
// C translation:
anorm = SLC_DLANGE("M", &n, &n, a, &lda, work);
```

#### Internal SLICOT Call Pattern
```c
// Fortran:     CALL MB01PD(SCUN, TYPE, M, N, KL, KU, ANRM, NBL, NROWS, A, LDA, INFO)
// C translation:
mb01pd(scun, type, m, n, kl, ku, anrm, nbl, nrows, a, lda, info);
// Direct function call - lowercase name, pass variables directly (already correct types)
```

#### Array Indexing Translation
```c
// Fortran (1-based):  A(I,J) where I=1..M, J=1..N
// C (0-based):        a[(j-1)*lda + (i-1)]  or  a[j*lda + i] with adjusted indices

// Common pattern for Fortran column-major access:
for (i32 j = 0; j < n; j++) {
    for (i32 i = 0; i < m; i++) {
        f64 value = a[j * lda + i];  // Column j, row i
    }
}
```

#### Adding New BLAS/LAPACK Routine
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

### Complete Translation Workflow

**Step-by-step process for translating a SLICOT routine:**

1. **Check dependencies**:
   ```bash
   uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB01MD
   ```

2. **Ensure SLICOT dependencies are translated**:
   - If routine depends on other SLICOT routines, translate those first (or verify they exist)
   - Example: AB01MD depends on MB01PD → check `src/MB/mb01pd.c` exists

3. **Read Fortran source**:
   ```bash
   # Fortran source location
   cat SLICOT-Reference/src/AB01MD.f
   ```

4. **Create C file** in appropriate directory:
   ```bash
   # Lowercase name: AB01MD.f → src/AB/ab01md.c
   touch src/AB/ab01md.c
   ```

5. **Translate to C** following patterns above:
   - Include headers: `slicutlet.h`, `slc_blaslapack.h`
   - Map types: `INTEGER → i32`, `DOUBLE PRECISION → f64`
   - BLAS/LAPACK calls: Use `SLC_*` macros with `&` for scalars
   - Internal calls: Direct function calls (lowercase names)
   - Adjust array indices: Fortran 1-based → C 0-based

6. **Add to meson.build**:
   ```python
   # Add to lib_srcs list
   lib_srcs = files(
     # AB family
     'src/AB/ab01md.c',  # ← Add here
     'src/AB/ab01nd.c',
     ...
   )
   ```

7. **Rebuild**:
   ```bash
   meson compile -C build
   ```

8. **Write Python test** in `python/tests/test_abXXxx.py`:
   ```python
   def test_ab01md():
       # Test with known inputs/outputs
       result = slicutlet.ab01md(...)
       assert_allclose(result, expected, rtol=1e-14)
   ```

9. **Run tests**:
   ```bash
   uv run pytest python/tests/test_ab01xx.py -v
   ```

### Workflow Integration

#### Step 1: Identify Translation Targets

```bash
# Find all leaf routines
uv run tools/extract_dependencies.py SLICOT-Reference/src/ > dependency_report.txt

# Look for Level 0 routines in your chapter of interest (e.g., AB)
grep "AB.*Level 0" dependency_report.txt
```

#### Step 2: Check Specific Routine

```bash
# Before translating AB05ND, check its dependencies
uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB05ND
```

#### Step 3: Plan LAPACK/BLAS Calls

```bash
# Collect all LAPACK dependencies for a routine
uv run tools/extract_dependencies.py SLICOT-Reference/src/ AB05ND | grep "LAPACK"
```

**Output**:
```
LAPACK/BLAS Dependencies (8):
  DCOPY
  DGEMM
  DGEMV
  DGETRF
  DGETRS
  DLACPY
  DLASET
```

**Action**:
1. Check if each routine has a macro in `src/include/slc_blaslapack.h`:
   - `SLC_DCOPY` ✓
   - `SLC_DGEMM` ✓
   - `SLC_DGEMV` ✓
   - `SLC_DGETRF` ✓
   - `SLC_DGETRS` ✓
   - `SLC_DLACPY` ✓
   - `SLC_DLASET` ✓

2. If missing, add declaration and macro (see "Adding New BLAS/LAPACK Routine" above)

3. Use in translation:
```c
// Fortran: CALL DCOPY(N, X, INCX, Y, INCY)
SLC_DCOPY(&n, x, &incx, y, &incy);

// Fortran: CALL DGETRF(M, N, A, LDA, IPIV, INFO)
SLC_DGETRF(&m, &n, a, &lda, ipiv, &info);
```

### Limitations

1. **Fixed-Format Fortran Only**: Designed for Fortran 77 fixed-format source
2. **SLICOT-Specific Patterns**: Assumes 6-character routine names (XXYYZZ)
3. **Static Analysis Only**: Doesn't account for conditional calls or preprocessor directives
4. **External Dependencies**: Routines called but not found in source are marked as external

### Using the Tool

**extract_dependencies.py** is the primary tool for dependency analysis:

**Use this tool for**:
- Real-time dependency analysis of any routine
- Getting complete dependency data for all 609 routines
- Identifying LAPACK/BLAS requirements
- Finding leaf routines (Level 0) for immediate translation
- Checking reverse dependencies (what depends on this routine)
- Planning translation order and priorities
- Automating dependency checks in scripts

### Technical Details

#### Dependency Level Algorithm

1. **Level 0**: Routines with no SLICOT dependencies
2. **Level N**: Routines whose maximum dependency level is N-1

**Example**:
```
MB01QD (Level 0) → No SLICOT deps
MB01PD (Level 1) → Depends on MB01QD (Level 0)
AB01MD (Level 2) → Depends on MB01PD (Level 1)
```

#### Pattern Matching

The tool uses these regex patterns:

- **SLICOT calls**: `^\s+CALL\s+([A-Z]{2}[0-9]{2}[A-Z]{2})`
- **LAPACK calls**: `^\s+CALL\s+(D[A-Z]{3,5}|XERBLA|ILAENV)`
- **Routine name**: `^\s+SUBROUTINE\s+([A-Z]{2}[0-9]{2}[A-Z]{2})`

### Future Enhancements

Potential improvements:

1. **GraphViz output**: Generate visual dependency graphs
2. **C implementation tracking**: Cross-reference with `src/` to show translation progress (% complete)
3. **Parallel translation groups**: Suggest optimal parallelization strategies for Level 0 routines
4. **LAPACK/BLAS macro check**: Verify if `SLC_*` macros exist in `slc_blaslapack.h` for dependencies
5. **Circular dependency detection**: Identify and warn about circular dependencies
6. **Meson.build sync**: Auto-check if translated C files are added to `lib_srcs` in `meson.build`

### Contributing

To improve the tool:

1. Test with different SLICOT source file formats
2. Add support for other Fortran 77 libraries
3. Implement GraphViz visualization
4. Add caching for faster repeated analysis

## Other Fortran Dependency Tools

### Research Summary

The following tools were evaluated but found **incompatible with Fortran 77**:

- **fortdepend**: Requires modern Fortran with labeled `end` statements
- **FORD**: Fortran documentation generator for Fortran 90+
- **FortranCallGraph**: Requires GCC assembler output, Fortran 90+ focus

**Conclusion**: Custom `extract_dependencies.py` is the best solution for SLICOT's Fortran 77 codebase.

### Commercial Alternatives

- **Forcheck**: Comprehensive Fortran verifier with call-tree generation (paid)
- **Understand for Fortran**: Static analysis with call graphs (paid)

## License

This tool is part of the SLICUTLET project and follows the same license (BSD-3-Clause).
