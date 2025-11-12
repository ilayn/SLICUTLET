# SLICOT Routine Dependencies and Translation Order

## Overview

This document provides a comprehensive dependency tree analysis of SLICOT routines to guide bottom-up translation from Fortran to Rust. The dependency tree ensures that leaf routines (with no SLICOT dependencies) are translated first, enabling systematic implementation of higher-level routines.

**Last Updated**: Based on analysis of slicot-rs repository with 621 Fortran routines

---

**⚠️ IMPORTANT: This is a curated strategic planning guide.**

For **real-time dependency analysis of any routine**, use the automated tool:

```bash
python3 tools/extract_dependencies.py reference/src/ ROUTINE_NAME
```

See `tools/README.md` for comprehensive tool documentation and workflow examples.

**This document focuses on:**
- Strategic implementation priorities
- Critical path analysis (what to implement first)
- Time estimates for key routines
- "Quick win" opportunities

**The `extract_dependencies.py` tool provides:**
- Automated dependency extraction (all 609 routines)
- Real-time analysis without manual updates
- LAPACK/BLAS dependency identification
- Reverse dependency tracking

**Workflow**: Use the tool first for dependency analysis, then consult this document for strategic context.

---

## Current Implementation Status

### Statistics
- **Total routines in reference**: 621 Fortran routines
- **Implemented in Rust**: 6 routines (0.97%)
- **Dependency levels identified**: 3+ levels (leaf → intermediate → high-level)
- **Circular dependencies**: None (clean acyclic structure)

### Currently Implemented Routines

| Routine | Level | Status | Missing Dependencies |
|---------|-------|--------|---------------------|
| **MB03MY** | 0 (Leaf) | ✓ Complete | None |
| **MB04TU** | 0 (Leaf) | ✓ Complete | None |
| **MA02FD** | 0 (Leaf) | ✓ Complete | None |
| **MA01AD** | 0 (Leaf) | ✓ Complete | None |
| **AB01MD** | 1 | ⚠️ Partial | MB01PD (→ MB01QD) |
| **SB01BD** | 1 | ⚠️ Partial | MB03QD (→ MB03QY), MB03QY, SB01BX, SB01BY |

**Key Finding**: AB01MD and SB01BD are implemented but not fully functional due to missing dependencies.

## Dependency Tree Structure

### Level 0: Leaf Routines (No SLICOT Dependencies)

Leaf routines only call BLAS/LAPACK or perform basic operations. These should be translated first and can be implemented in parallel.

**Implemented (4 routines)**:
- **MB03MY** - Find minimum absolute value in array
  - File: `src/mb/mod.rs`
  - Dependencies: None (pure Rust)

- **MB04TU** - Apply row-permuted Givens rotation
  - File: `src/mb/mod.rs`
  - Dependencies: None (pure Rust)

- **MA02FD** - Compute hyperbolic plane rotation
  - File: `src/ma/mod.rs`
  - Dependencies: None (pure Rust)

- **MA01AD** - Compute complex square root safely
  - File: `src/ma/mod.rs`
  - Dependencies: None (pure Rust)

**High Priority Leaves (2 routines)** - Block current implementations:
- **MB01QD** - QR factorization with column pivoting
  - Estimate: 1-2 hours
  - Blocks: MB01PD → AB01MD
  - BLAS/LAPACK: DGEQP3 (QR with pivoting)

- **MB03QY** - Reorder eigenvalues in Schur form
  - Estimate: 1-2 hours
  - Blocks: MB03QD → SB01BD
  - BLAS/LAPACK: DTREXC (reorder Schur form)

**Other Level 0 Routines**: For a complete list of all 297 leaf routines, use:
```bash
python3 tools/extract_dependencies.py reference/src/ | grep "Level 0"
```

Key examples include: MA02ED, MA01BD, AB04MD, AB05MD, AB05ND, AB05PD, AB05QD, AB07MD, AB07ND, AB09DD, AB09JX, SB01BX, SB01BY, TB01ID, TB01UD, and many more.

### Level 1: Single-Level Dependencies

These routines depend only on Level 0 routines.

**Implemented (2 routines)**:
- **AB01MD** - Staircase form for controllability
  - File: `src/ab/mod.rs`
  - Direct dependency: MB01PD (not implemented)
  - Transitive: MB01PD → MB01QD
  - Status: ⚠️ Missing MB01PD

- **SB01BD** - Pole placement for single-input systems
  - File: `src/sb/mod.rs`
  - Direct dependencies: MB03QD, MB03QY, SB01BX, SB01BY (not implemented)
  - Transitive: MB03QD → MB03QY
  - Status: ⚠️ Missing 4 dependencies

**Critical Level 1 Dependencies** (block existing implementations):
- **MB01PD** - QR factorization with pivoting (wrapper)
  - Estimate: 2-3 hours
  - Depends on: MB01QD
  - Blocks: AB01MD

- **MB03QD** - Reorder Schur form (wrapper)
  - Estimate: 1-2 hours
  - Depends on: MB03QY
  - Blocks: SB01BD

**Other Level 1 Routines**: For the complete list of 126 Level 1 routines, use:
```bash
python3 tools/extract_dependencies.py reference/src/ | grep -A 10 "Level 1"
```

Key examples: AB01ND, AB01OD, SB02MD, SB03MD, SB04MD, and many more.

### Level 2+: Multi-Level Dependencies

Complex, high-level routines that depend on Level 1+ routines.

**Examples**: SB02OD (Riccati solver), SB03OD (Lyapunov solver), AB13BD/AB13DD (H∞ norm), FB01VD (Kalman filter)

**For complete dependency analysis**, use:
```bash
python3 tools/extract_dependencies.py reference/src/ ROUTINE_NAME
```

## Critical Path Analysis

### Quick Win: Enable Existing Implementations

To make AB01MD and SB01BD fully functional, implement these **6 routines**:

**Phase 1: Leaf routines (parallel implementation possible)**
1. **MB01QD** (1-2 hrs) - QR with pivoting
2. **MB03QY** (1-2 hrs) - Reorder Schur form
3. **SB01BX** (1 hr) - Pole placement helper
4. **SB01BY** (1 hr) - Pole placement helper

**Phase 2: Level 1 routines (sequential after Phase 1)**
5. **MB01PD** (2-3 hrs) - Depends on MB01QD
6. **MB03QD** (1-2 hrs) - Depends on MB03QY

**Total effort**: 7-11 hours
**Result**: 6 fully functional control theory routines

### Recommended Translation Order

**Stage 1: Foundation (Priority 1)**
- Implement missing leaves that block current code (MB01QD, MB03QY, SB01BX, SB01BY)
- Implement MB01PD and MB03QD to complete AB01MD and SB01BD
- **Deliverable**: 2 working control algorithms

**Stage 2: Expand Foundation (Priority 2)**
- Implement remaining ~30 Level 0 leaves in parallel
- Focus on commonly-used primitives (AB05MD, AB07MD, TB01ID, etc.)
- **Deliverable**: Strong foundation for higher-level routines

**Stage 3: Build Upward (Priority 3)**
- Implement Level 1 routines (AB01ND, SB02MD, SB03MD, etc.)
- **Deliverable**: Core analysis and synthesis capabilities

**Stage 4: Advanced Features (Priority 4)**
- Implement Level 2+ routines (H∞ control, Kalman filtering, etc.)
- **Deliverable**: Complete control theory library

## Detailed Dependency Mappings

### AB01MD: Staircase Form for Controllability

**Purpose**: Reduce a system pair (A,B) to staircase form to determine controllability

**Dependencies**:
```
AB01MD
└── MB01PD (QR factorization with pivoting)
    └── MB01QD (LAPACK DGEQP3 wrapper)
```

**Current Status**: Implemented but non-functional
**Action Required**: Implement MB01QD (1-2 hrs), then MB01PD (2-3 hrs)

**BLAS/LAPACK Used**:
- DGEQP3 (QR with column pivoting)
- DORMQR (Apply orthogonal Q matrix)
- DLACPY (Copy matrix)

### SB01BD: Pole Placement for Single-Input Systems

**Purpose**: Compute state feedback matrix for pole placement

**Dependencies**:
```
SB01BD
├── MB03QD (Reorder eigenvalues in Schur form)
│   └── MB03QY (LAPACK DTREXC wrapper)
├── MB03QY (Direct call as well)
├── SB01BX (Pole placement helper - split/merge)
└── SB01BY (Pole placement helper - solve equations)
```

**Current Status**: Implemented but non-functional
**Action Required**:
- Implement MB03QY (1-2 hrs)
- Implement SB01BX (1 hr)
- Implement SB01BY (1 hr)
- Implement MB03QD (1-2 hrs)

**BLAS/LAPACK Used**:
- DGEES (Schur decomposition)
- DTREXC (Reorder Schur form)
- DTRSYL (Solve Sylvester equation)

### MB01PD: QR Factorization with Pivoting

**Purpose**: Compute QR factorization with column pivoting for rank determination

**Dependencies**:
```
MB01PD
└── MB01QD (Direct wrapper for DGEQP3)
```

**Current Status**: Not implemented
**BLAS/LAPACK Used**: DGEQP3, DORMQR

### MB03QD: Reorder Eigenvalues in Schur Form

**Purpose**: Reorder eigenvalues in real Schur form by orthogonal similarity transformation

**Dependencies**:
```
MB03QD
└── MB03QY (Direct wrapper for DTREXC)
```

**Current Status**: Not implemented
**BLAS/LAPACK Used**: DTREXC

## Translation Guidelines

### Identifying Dependencies in Fortran Source

When examining a Fortran routine in `reference/src/ROUTINE.f`, look for:

1. **SLICOT calls**: `CALL XX####(...)`
   - Example: `CALL MB01PD(...)` indicates dependency on MB01PD

2. **BLAS calls**: `CALL D[LETTER][ABBREV](...)`
   - Examples: DGEMM, DGEMV, DAXPY, DCOPY
   - Map to ndarray operations (`.dot()`, `+`, `*`, `.assign()`)

3. **LAPACK calls**: `CALL D[OPERATION](...)`
   - Examples: DGEQP3, DGEEV, DGESV, DGEES, DTREXC
   - Map to ndarray-linalg traits (`.qr()`, `.eig()`, `.solve()`)

### Bottom-Up Translation Strategy

1. **Check dependencies first**:
   ```bash
   grep -E "CALL [A-Z]{2}[0-9]{2}[A-Z]{2}" reference/src/ROUTINE.f
   ```

2. **Classify the routine**:
   - No SLICOT calls → Level 0 (translate immediately)
   - Calls only Level 0 → Level 1 (translate after leaves)
   - Calls Level 1+ → Higher level (translate after dependencies)

3. **Verify dependencies are implemented**:
   - Check `src/` for Rust implementations
   - If missing, add to backlog in dependency order

4. **Implement and test**:
   - Translate the algorithm
   - Create tests from `reference/doc/ROUTINE.html`
   - Verify against Fortran output

### Parallel Translation Opportunities

Since there are no circular dependencies, these groups can be translated in parallel:

**Group 1: MA/MB leaves** (5-10 routines)
- MA01BD, MA02ED, MB01RU, MB01SD, MB01TD, etc.

**Group 2: AB leaves** (5-10 routines)
- AB04MD, AB05MD, AB05ND, AB05OD, AB07MD, etc.

**Group 3: SB leaves** (3-5 routines)
- SB01MD, SB01BX, SB01BY, SB10PD, etc.

**Group 4: TB leaves** (5-10 routines)
- TB01ID, TB01UD, TB01WD, TB01XD, etc.

Each group can be assigned to different developers or worked on in different branches.

## Common Dependency Patterns

### Pattern 1: QR Factorization Chain
```
High-level routine
└── MB01PD (QR with pivoting)
    └── MB01QD (LAPACK wrapper)
```
**Examples**: AB01MD, AB01ND, AB01OD

### Pattern 2: Schur Form Chain
```
High-level routine
└── MB03QD (Reorder Schur)
    └── MB03QY (LAPACK wrapper)
```
**Examples**: SB01BD, SB02OD, MB03RD

### Pattern 3: Direct LAPACK (No SLICOT dependencies)
```
Routine
└── LAPACK only (DGEEV, DGESV, etc.)
```
**Examples**: MA01AD, MA02FD, MB03MY, MB04TU

### Pattern 4: Equation Solver Chain
```
High-level solver (Riccati, Lyapunov)
├── Schur decomposition routine
├── Transformation routines
└── Basic equation solver
```
**Examples**: SB02OD, SB03OD

## Verification Strategy

### Testing Dependency Implementations

1. **Unit tests**: Test leaf routines against SLICOT HTML examples
2. **Integration tests**: Test Level 1+ routines to verify dependencies work together
3. **Regression tests**: Run full test suite after each implementation

### Detecting Missing Dependencies

If a Rust implementation fails at runtime with "undefined function" or similar:

1. Check Fortran source for CALL statements
2. Search `src/` for the called routine
3. If missing, add to dependency tree and implement dependencies first
4. Re-run tests

## Reference Files

### Fortran Source Locations

- **All routines**: `reference/src/*.f` (621 files)
- **Example programs**: `reference/examples/T*.f`
- **Documentation**: `reference/doc/*.html`

### Rust Implementation Locations

- **MA routines**: `src/ma/mod.rs`
- **MB routines**: `src/mb/mod.rs`
- **AB routines**: `src/ab/mod.rs`
- **SB routines**: `src/sb/mod.rs`
- **TB routines**: `src/tb/mod.rs` (to be created)

## Next Steps

### Immediate Actions (7-11 hours)

1. Implement **MB01QD** (1-2 hrs)
   - Wrapper for LAPACK DGEQP3
   - Test with simple QR factorization cases

2. Implement **MB03QY** (1-2 hrs)
   - Wrapper for LAPACK DTREXC
   - Test with Schur form reordering

3. Implement **SB01BX** (1 hr)
   - Helper for pole placement
   - Test standalone and with SB01BD

4. Implement **SB01BY** (1 hr)
   - Helper for pole placement
   - Test standalone and with SB01BD

5. Implement **MB01PD** (2-3 hrs)
   - Uses MB01QD
   - Test with AB01MD integration

6. Implement **MB03QD** (1-2 hrs)
   - Uses MB03QY
   - Test with SB01BD integration

**Result**: AB01MD and SB01BD become fully functional

### Medium-Term Goals (Phase 2)

- Implement remaining ~30 Level 0 leaves
- Build out Level 1 routines
- Expand test coverage
- Create benchmarks against Fortran SLICOT

### Long-Term Vision

- Complete translation of all 621 routines
- Match or exceed Fortran performance
- Provide idiomatic Rust API
- Comprehensive documentation and examples

## Summary

The SLICOT dependency tree analysis reveals:

1. **Clean structure**: No circular dependencies, enabling systematic bottom-up translation
2. **Quick wins available**: 7-11 hours to enable existing implementations
3. **Parallel opportunities**: ~30 leaf routines can be translated simultaneously
4. **Clear roadmap**: Level 0 → Level 1 → Level 2+ provides natural progression

**Recommended approach**: Start with the 6 critical routines blocking AB01MD and SB01BD, then expand foundation with remaining leaves, followed by systematic upward progression through dependency levels.
