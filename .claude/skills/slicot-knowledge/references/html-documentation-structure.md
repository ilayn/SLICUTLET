# SLICOT HTML Documentation Structure

## Overview

Each SLICOT routine has comprehensive HTML documentation following a standardized format. Understanding this structure is essential for translating routines and creating accurate test cases.

## Standard Document Sections

SLICOT HTML documentation files (e.g., `AB01MD.html`, `SB01BD.html`) typically contain these sections in order:

### 1. Header Section

```html
<HTML>
<HEAD><TITLE>AB01MD - SLICOT Library Routine Documentation</TITLE></HEAD>
<BODY>
<H2><A Name="AB01MD">AB01MD</A></H2>
<H3>Brief one-line description of functionality</H3>
```

**Contains**:
- Routine name (6 characters, e.g., AB01MD)
- Short functional description

**Usage**: Understand what the routine does at a high level.

### 2. Navigation Links

```html
<A HREF ="#Specification"><B>[Specification]</B></A>
<A HREF ="#Arguments"><B>[Arguments]</B></A>
<A HREF ="#Method"><B>[Method]</B></A>
<A HREF ="#References"><B>[References]</B></A>
<A HREF ="#Comments"><B>[Comments]</B></A>
<A HREF ="#Example"><B>[Example]</B></A>
```

**Contains**: Quick navigation to major sections.

**Usage**: Jump directly to relevant sections (Specification, Arguments, Example).

### 3. Purpose

```html
<B><FONT SIZE="+1">Purpose</FONT></B>
<PRE>
  To find a controllable realization for the linear time-invariant
  single-input system

          dX/dt = A * X + B * U,

  where A is an N-by-N matrix and B is an N element vector...
</PRE>
```

**Contains**:
- Detailed problem description
- Mathematical formulation
- Context and application

**Usage**: Understand the mathematical problem being solved. Essential for writing documentation and choosing appropriate algorithms.

### 4. Specification

```html
<A name="Specification"><B><FONT SIZE="+1">Specification</FONT></B></A>
<PRE>
      SUBROUTINE AB01MD( JOBZ, N, A, LDA, B, NCONT, Z, LDZ, TAU, TOL,
     $                   DWORK, LDWORK, INFO )
C     .. Scalar Arguments ..
      CHARACTER         JOBZ
      INTEGER           INFO, LDA, LDZ, LDWORK, N, NCONT
      DOUBLE PRECISION  TOL
C     .. Array Arguments ..
      DOUBLE PRECISION  A(LDA,*), B(*), DWORK(*), TAU(*), Z(LDZ,*)
</PRE>
```

**Contains**:
- Fortran subroutine signature
- Parameter types
- Array dimension specifications

**Usage**: Critical for creating Rust/C function signatures. Note:
- Character parameters (mode flags)
- Integer dimensions
- Double precision arrays
- Leading dimension parameters (LDA, LDZ)
- Workspace arrays (DWORK, IWORK)
- INFO parameter for error codes

### 5. Arguments

```html
<A name="Arguments"><B><FONT SIZE="+1">Arguments</FONT></B></A>

<B>Mode Parameters</B>
<PRE>
  JOBZ    CHARACTER*1
          Indicates whether the user wishes to accumulate...
          = 'N':  Do not form Z...
          = 'F':  Do not form Z, but store...
          = 'I':  Z is initialized to the unit matrix...
</PRE>

<B>Input/Output Parameters</B>
<PRE>
  N       (input) INTEGER
          The order of the original state-space representation...

  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
          On entry, the leading N-by-N part contains...
          On exit, the leading NCONT-by-NCONT part contains...
</PRE>

<B>Tolerances</B>
<PRE>
  TOL     DOUBLE PRECISION
          The tolerance to be used in determining...
</PRE>

<B>Workspace</B>
<PRE>
  DWORK   DOUBLE PRECISION array, dimension (LDWORK)
          On exit, if INFO = 0, DWORK(1) returns the optimal value...

  LDWORK  INTEGER
          The length of the array DWORK. LDWORK >= MAX(1,N).
</PRE>

<B>Error Indicator</B>
<PRE>
  INFO    INTEGER
          = 0:  successful exit;
          < 0:  if INFO = -i, the i-th argument had an illegal value.
</PRE>
```

**Contains**:
- Detailed parameter descriptions organized by category:
  - **Mode Parameters**: Control routine behavior (e.g., JOBZ, UPLO, TRANS)
  - **Input/Output Parameters**: Problem data
  - **Tolerances**: Numerical thresholds
  - **Workspace**: Temporary storage requirements
  - **Error Indicator**: Return codes

**Usage**:
- Understand parameter semantics (input, output, input/output)
- Determine minimum workspace sizes
- Map Fortran types to Rust types:
  - `CHARACTER*1` → `char` or `enum`
  - `INTEGER` → `i32` or `usize`
  - `DOUBLE PRECISION` → `f64`
  - `DOUBLE PRECISION array` → `&[f64]`, `&mut [f64]`, or `Array2<f64>`
- Understand error codes for error handling

### 6. Method

```html
<A name="Method"><B><FONT SIZE="+1">Method</FONT></B></A>
<PRE>
  The Householder matrix which reduces all but the first element
  of vector B to zero is found and this orthogonal similarity
  transformation is applied to the matrix A. The resulting A is then
  reduced to upper Hessenberg form by a sequence of Householder
  transformations...
</PRE>
```

**Contains**:
- Algorithm description
- Mathematical techniques used
- Step-by-step procedure
- Special cases and edge conditions

**Usage**:
- Understand the algorithm for implementation
- Identify BLAS/LAPACK operations needed
- Recognize numerical techniques (QR, SVD, eigendecomposition)
- Plan for edge cases

### 7. References

```html
<A name="References"><B><FONT SIZE="+1">References</FONT></B></A>
<PRE>
  [1] Konstantinov, M.M., Petkov, P.Hr. and Christov, N.D.
      Orthogonal Invariants and Canonical Forms for Linear
      Controllable Systems.
      Proc. 8th IFAC World Congress, Kyoto, 1, pp. 49-54, 1981.

  [2] Hammarling, S.J.
      Notes on the use of orthogonal similarity transformations in
      control.
      NPL Report DITC 8/82, August 1982.
</PRE>
```

**Contains**:
- Academic papers
- Textbook references
- Technical reports

**Usage**:
- Deep understanding of algorithm theory
- Verify correctness
- Understand numerical stability properties

### 8. Numerical Aspects

```html
<A name="Numerical Aspects"><B><FONT SIZE="+1">Numerical Aspects</FONT></B></A>
<PRE>
  The algorithm requires 0(N³) operations and is backward stable.
</PRE>
```

**Contains**:
- Computational complexity (O(n), O(n²), O(n³))
- Numerical stability characteristics
- Conditioning information

**Usage**:
- Set performance expectations
- Understand when algorithms may be numerically unstable
- Plan benchmarking

### 9. Further Comments

```html
<A name="Comments"><B><FONT SIZE="+1">Further Comments</FONT></B></A>
<PRE>
  None
</PRE>
```

**Contains**:
- Additional implementation notes
- Special considerations
- Usage warnings

**Usage**: Check for important caveats or special cases.

### 10. Example (Most Important for Testing)

#### Program Text

```html
<A name="Example"><B><FONT SIZE="+1">Example</FONT></B></A>

<B>Program Text</B>
<PRE>
*     AB01MD EXAMPLE PROGRAM TEXT
*     .. Parameters ..
      INTEGER          NIN, NOUT
      PARAMETER        ( NIN = 5, NOUT = 6 )
      ...
*     Skip the heading in the data file and read in the data.
      READ ( NIN, FMT = '()' )
      READ ( NIN, FMT = * ) N, TOL, JOBZ
      READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )
      READ ( NIN, FMT = * ) ( B(I), I = 1,N )
*     Find a controllable realization for the given system.
      CALL AB01MD( JOBZ, N, A, LDA, B, NCONT, Z, LDZ, TAU, TOL,
     $             DWORK, LDWORK, INFO )
      ...
</PRE>
```

**Contains**:
- Complete Fortran example program
- READ statements showing data input format
- Routine invocation with actual parameters
- Output formatting

**Usage**:
- **CRITICAL**: READ statements show exact data input format
- Understand parameter passing
- See workspace allocation
- Verify calling conventions

#### Program Data

```html
<B>Program Data</B>
<PRE>
 AB01MD EXAMPLE PROGRAM DATA
   3     0.0     I
   1.0   2.0   0.0
   4.0  -1.0   0.0
   0.0   0.0   1.0
   1.0   0.0   1.0
</PRE>
```

**Contains**:
- Input dimensions (N=3)
- Control parameters (TOL=0.0, JOBZ='I')
- Matrix/vector data

**Usage**:
- **CRITICAL FOR TESTING**: Use as test case inputs
- Cross-reference with READ statements in Program Text
- Parse according to Fortran column-major convention (see test-data-parsing.md)
- Note the first line often contains dimension parameters

#### Program Results

```html
<B>Program Results</B>
<PRE>
 AB01MD EXAMPLE PROGRAM RESULTS

 The order of the controllable state-space representation =  3

 The state dynamics matrix A of a controllable realization is
   1.0000   1.4142   0.0000
   2.8284  -1.0000   2.8284
   0.0000   1.4142   1.0000

 The input/state vector B of a controllable realization is
  -1.4142
   0.0000
   0.0000

 The similarity transformation matrix Z is
  -0.7071   0.0000  -0.7071
   0.0000  -1.0000   0.0000
  -0.7071   0.0000   0.7071
</PRE>
```

**Contains**:
- Expected output values
- Formatted results
- Scalar outputs (NCONT = 3)
- Matrix/vector outputs

**Usage**:
- **CRITICAL FOR TESTING**: Use as expected results in assertions
- Typical precision: 4 decimal places shown
- Account for numerical precision in comparisons (use tolerances like 1e-3 or 5e-3)
- Results presented in row-major format

### 11. Footer

```html
<HR>
<p>
<A HREF=..\libindex.html><B>Return to index</B></A>
</BODY>
</HTML>
```

**Contains**: Navigation back to library index.

## Document Parsing Workflow

### For Implementation

1. **Read Purpose** → Understand the problem
2. **Read Method** → Understand the algorithm
3. **Read Specification** → Design function signature
4. **Read Arguments** → Implement parameter handling
5. **Read Numerical Aspects** → Set performance expectations

### For Testing

1. **Read Example Program Text** → Understand data format via READ statements
2. **Read Program Data** → Extract test inputs
3. **Read Program Results** → Extract expected outputs
4. **Parse data correctly** → Account for column-major storage (see test-data-parsing.md)
5. **Write test case** → Compare computed results with expected results

## Common Patterns to Recognize

### Pattern 1: Mode Parameters

Many routines have CHARACTER mode parameters:

```fortran
JOBZ = 'N': No computation
JOBZ = 'F': Factored form
JOBZ = 'I': Initialize and compute

UPLO = 'U': Upper triangle
UPLO = 'L': Lower triangle

TRANS = 'N': No transpose
TRANS = 'T': Transpose
TRANS = 'C': Conjugate transpose
```

### Pattern 2: Workspace Query

```fortran
LDWORK >= MAX(1, N)    ! Minimum workspace
DWORK(1) returns optimal LDWORK on exit
```

To query optimal workspace:
1. Call with LDWORK = -1
2. Read optimal size from DWORK(1)
3. Allocate and call again

### Pattern 3: Leading Dimensions

```fortran
A       DOUBLE PRECISION array, dimension (LDA,N)
LDA     INTEGER
        The leading dimension of array A. LDA >= MAX(1,N).
```

LDA allows for matrices stored with extra padding rows.

### Pattern 4: INFO Error Codes

```fortran
INFO = 0:  Success
INFO < 0:  INFO = -i means i-th parameter is invalid
INFO > 0:  Algorithm-specific error (see documentation)
```

### Pattern 5: Input/Output Arrays

```fortran
A       (input/output) DOUBLE PRECISION array
        On entry, contains...
        On exit, contains...
```

Arrays modified in-place (important for memory management).

## Tips for Efficient Navigation

1. **Start with Purpose and Example** for quick understanding
2. **Use browser search** (Ctrl+F) for specific parameters
3. **Check Program Text** for exact READ format before parsing data
4. **Compare Method with Fortran source code** in `src/` for implementation details
5. **Cross-reference Arguments** when implementing parameter validation
6. **Use References** only for deep algorithm understanding

## File Naming Convention

HTML documentation files follow routine names:

```
doc/AB01MD.html  → Routine AB01MD
doc/SB01BD.html  → Routine SB01BD
doc/TF01MD.html  → Routine TF01MD
```

Located in: `reference/doc/*.html`

Fortran source: `reference/src/*.f`

Example programs: `reference/examples/T*.f`

## Special Notes

### Zero Dimensions

Some routines accept N=0, M=0 as valid edge cases (quick return with INFO=0). Check:
- Method section for "quick return" mentions
- Fortran source for early exit conditions
- Don't assume zero dimensions are always errors

### Numerical Precision

Results in HTML typically show 4-5 significant digits. When testing:
- Use tolerance ~1e-3 to 5e-3 for comparisons
- Account for algorithm variations (different LAPACK implementations)
- Trust consistently computed results over documentation if slightly different

### Column-Major vs Row-Major

- **Fortran code and storage**: Column-major
- **HTML presentation**: Often row-major for readability
- **Critical**: Always verify with READ statements in Program Text

## Quick Reference

| Section | Use For |
|---------|---------|
| Purpose | Understanding the mathematical problem |
| Specification | Function signature design |
| Arguments | Parameter types and semantics |
| Method | Algorithm implementation strategy |
| References | Deep theoretical understanding |
| Numerical Aspects | Performance and stability expectations |
| Example (Program Text) | **Data format via READ statements** |
| Example (Program Data) | **Test inputs** |
| Example (Program Results) | **Expected test outputs** |

## Example Usage in Rust Translation

```rust
// 1. Read Purpose → Understand it solves controllability problem
// 2. Read Specification → Design signature:
pub fn ab01md(
    jobz: char,              // CHARACTER
    n: usize,                // INTEGER
    a: &mut Array2<f64>,     // A(LDA,N)
    b: &mut Array1<f64>,     // B(N)
    tol: f64,                // TOL
) -> Result<(usize, Option<Array2<f64>>), String>

// 3. Read Arguments → Implement validation:
if n == 0 {
    return Ok((0, None));  // Quick return for zero dimension
}

// 4. Read Method → Use ndarray-linalg for Householder transformations
// 5. Read Program Data → Create test case
// 6. Read Program Results → Add assertions
```
