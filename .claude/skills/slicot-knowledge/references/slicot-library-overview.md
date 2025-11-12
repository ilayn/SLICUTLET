# SLICOT Library Overview

## What is SLICOT?

**SLICOT (Subroutine Library In COntrol Theory)** is a comprehensive numerical library for control theoretical computations, originally implemented in Fortran 77. It provides reliable, numerically robust algorithms for:

- System analysis and synthesis
- Model reduction and realization
- State-space transformations
- Filtering and identification
- Benchmarking and test problems
- Mathematical and utility routines

## Library Organization

SLICOT routines are organized into chapters based on functionality:

### Core Chapters

- **A** - Analysis Routines
  - System controllability and observability
  - Canonical forms and transformations
  - Model reduction techniques

- **B** - Benchmark and Test Problems
  - Standard test cases for validation
  - Performance benchmarking datasets

- **D** - Data Analysis
  - Time-series analysis
  - System identification preprocessing

- **F** - Filtering
  - Kalman filtering
  - State estimation algorithms

- **I** - Identification
  - System identification from I/O data
  - Parameter estimation

- **M** - Mathematical Routines
  - Basic mathematical operations
  - Linear algebra utilities
  - Specialized matrix computations

- **N** - Nonlinear Systems
  - Nonlinear control algorithms
  - Linearization routines

- **S** - Synthesis Routines
  - Controller design
  - Pole placement
  - Riccati equation solvers

- **T** - Transformation Routines
  - State-space transformations
  - Descriptor systems

- **U** - Utility Routines
  - Helper functions
  - I/O and formatting utilities

### Naming Convention

SLICOT routine names follow a consistent 6-character pattern:

```
AB01MD
││││└─ Variant/subsection identifier
│││└── Section number (second digit)
││└─── Section number (first digit)
│└──── Chapter identifier
└───── Chapter group
```

**Example**: `AB01MD`
- `A` = Analysis chapter
- `B` = Benchmark/special variant
- `01` = Section 1
- `MD` = Variant MD within that section

## Data Storage Formats

SLICOT follows strict data storage standards for compatibility with LAPACK and BLAS:

### Matrix Storage Schemes

#### 1. Conventional Storage
- **Column-major ordering** (Fortran standard)
- General m×n matrices stored in 2D arrays
- Leading dimension parameter `LDA` specifies allocated rows
- Element A(i,j) accessed at index `i-1 + (j-1)*LDA` (0-based)

#### 2. Triangular and Symmetric Matrices
- Only relevant triangle stored (upper or lower)
- `UPLO` parameter specifies which triangle
- Diagonal assumed unit for unit triangular matrices
- Unused elements need not be initialized

#### 3. Packed Storage
- Memory-efficient for symmetric/triangular matrices
- Relevant triangle packed by columns into 1D array
- Array names conventionally end with 'P'
- **Indexing formulas**:
  - Upper: `A(i,j)` → `AP[i + j(j-1)/2]` for i ≤ j
  - Lower: `A(i,j)` → `AP[i + (2n-j)(j-1)/2]` for i ≥ j

#### 4. Band Storage
- For matrices with limited bandwidth
- kl subdiagonals and ku superdiagonals
- Stored in (kl+ku+1) × n array
- Array names conventionally end with 'B'
- Element A(i,j) → AB[ku+1+i-j, j]

#### 5. Special Formats
- **Tridiagonal**: Three 1D arrays (diagonal, super, sub)
- **Bidiagonal**: Two 1D arrays (diagonal, off-diagonal)
- **Householder reflectors**: Product of elementary reflectors

### Polynomial Storage

SLICOT supports polynomials of increasing complexity:

#### Scalar Polynomials
- 1D array `P(M)` where `P(i+1)` = coefficient of z^i
- Degree n requires array size M ≥ n+1

#### Vector Polynomials
- 2D array `P(K,M)` for k-dimensional coefficient vectors
- Columns = polynomial coefficients
- Rows = vector components
- Optional `DEGP(K)` for individual degrees

#### Matrix Polynomials
- 3D array `P(K,L,M)` for k×l coefficient matrices
- First two indices = matrix element
- Third index = polynomial coefficient
- Optional `DEGP(K,L)` for element-wise degrees

## Numerical Precision

- SLICOT uses **double precision** (DOUBLE PRECISION in Fortran, `f64` in Rust)
- Tolerances typically specified via `TOL` parameters
- Default tolerances: `TOL = N*EPS*MAX(NORM(A), NORM(B))`
- EPS = machine precision (~2.22e-16 for double precision)

## Dependencies

SLICOT routines extensively use:

### BLAS (Basic Linear Algebra Subprograms)
- **Level 1**: Vector operations (DAXPY, DCOPY, DDOT, DSCAL)
- **Level 2**: Matrix-vector operations (DGEMV, DGER, DTRSV)
- **Level 3**: Matrix-matrix operations (DGEMM, DTRSM)

### LAPACK (Linear Algebra PACKage)
- Eigenvalue decomposition (DGEEV, DSYEV)
- Singular value decomposition (DGESVD, DGESDD)
- Linear system solving (DGESV, DGELS)
- QR decomposition (DGEQRF, DORGQR)
- Cholesky decomposition (DPOTRF)
- Schur decomposition (DGEES, DHSEQR)
- Generalized eigenvalue problems (DGGES, DGGEV)

## Error Handling

SLICOT routines follow LAPACK conventions:

- **INFO parameter**: Integer return code
  - `INFO = 0`: Successful exit
  - `INFO < 0`: Invalid parameter (INFO = -i means i-th parameter)
  - `INFO > 0`: Algorithm-specific error or warning

## Workspace Arrays

Many SLICOT routines require workspace:

- **DWORK**: Double precision workspace array
- **IWORK**: Integer workspace array
- **LDWORK**: Size of DWORK
- **LIWORK**: Size of IWORK

Workspace size formulas documented in each routine's specification.

## Key Implementation Principles

1. **Numerical Stability**: Algorithms use orthogonal transformations when possible
2. **Backward Stability**: Error bounds relative to problem conditioning
3. **Efficiency**: Optimized for cache performance via BLAS Level 3
4. **Modularity**: Layered design with low-level computational kernels
5. **Compatibility**: Strict adherence to Fortran 77 standard
6. **Portability**: Platform-independent with standard BLAS/LAPACK

## Reference Documentation

For each SLICOT routine, documentation includes:

1. **Purpose**: High-level description of functionality
2. **Specification**: Fortran interface declaration
3. **Arguments**: Detailed parameter descriptions
4. **Method**: Algorithm description and mathematical background
5. **References**: Academic papers and textbooks
6. **Numerical Aspects**: Complexity and stability analysis
7. **Example**: Complete program with test data and results

## Performance Considerations

- Most algorithms have O(n³) complexity for n×n matrices
- Exploit special structure (symmetry, sparsity) when available
- Use blocked algorithms for Level 3 BLAS efficiency
- Workspace size impacts cache performance
- Optimal LDWORK often > minimum required

## Control Theory Applications

SLICOT provides complete toolset for:

- **Linear Systems**: State-space models, transfer functions
- **Descriptor Systems**: DAE (Differential-Algebraic Equations)
- **Continuous/Discrete Time**: Both representations supported
- **MIMO Systems**: Multi-Input Multi-Output systems
- **Model Reduction**: Balanced truncation, Hankel norm
- **Controller Design**: LQR, LQG, H₂, H∞
- **Stability Analysis**: Lyapunov equations, eigenvalue analysis
