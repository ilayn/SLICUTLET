# SLICUTLET - Claude Code Instructions

## Project Overview
C11 translation of SLICOT (Systems and Control library) from Fortran77.

**Reference:** Fortran77 source in `SLICOT-Reference/src/`

## Build & Test

**Setup (first time):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,dev]"
pip install meson ninja meson-python
```

**Activate venv (each session):**
```bash
source .venv/bin/activate
```

**Build (first time):**
```bash
meson setup build -Dpython=true
meson compile -C build
meson install -C build --destdir=$PWD/build-install
```

**Rebuild (iterative development - auto-detects changes):**
```bash
meson compile -C build
meson install -C build --destdir=$PWD/build-install
```

**Clean rebuild (when needed):**
```bash
meson compile -C build --clean
meson install -C build --destdir=$PWD/build-install
```

**Test:**
```bash
DYLD_LIBRARY_PATH=build-install/usr/local/lib PYTHONPATH=build-install/usr/local/lib/python3.11/site-packages pytest python/tests/
```

## Code Structure
- `src/AB/` - State-space transformations (ab01nd, ab04md, ab05md, ab05nd, ab07nd)
- `src/MA/` - Matrix operations (ma01xx, ma02xx families)
- `src/MB/` - Matrix operations continued (mb01xx, mb03xx families)
- `src/MC/` - Polynomial operations (mc01xx family)
- `src/include/types.h` - Type aliases (i32, i64, f64, c128)
- `src/include/slc_blaslapack.h` - BLAS/LAPACK declarations
- `python/slicutletmodule.c` - Cython extension
- `python/tests/` - pytest test suite

## Translation Workflow

### TDD Cycle: RED → GREEN → REFACTOR → VERIFY

**RED:** Write test first
- Extract from `SLICOT-Reference/examples/` or docs
- Write in `python/tests/test_xxnnxx.py`
- Verify test FAILS
- Commit: `RED: Add tests for xxnncc`

**GREEN:** Implement to pass
- Read Fortran: `SLICOT-Reference/src/XXNNCC.f`
- Translate to C: `src/XX/xxnncc.c` (lowercase)
- Add to `meson.build` lib_srcs
- Update `python/slicutletmodule.c` exports
- Run tests
- Commit: `GREEN: Implement xxnncc`

**REFACTOR:** Clean up
- Improve code quality + numerical correctness
- Comment complex logic only
- Commit: `REFACTOR: Clean up xxnncc`

**VERIFY:** Final validation
- Build and test all
- Confirm no regressions

### Dependency Analysis
```bash
# With venv activated
python tools/extract_dependencies.py SLICOT-Reference/src/ AB01ND    # Specific
python tools/extract_dependencies.py SLICOT-Reference/src/           # Full
python tools/extract_dependencies.py SLICOT-Reference/src/ | grep "Level 0"  # Leaves
```

**Planning:** Bottom-to-top (Level 0 → Level N)

## Translation Conventions

**Types:** `i32`→INTEGER, `f64`→DOUBLE PRECISION, `c128`→COMPLEX*16

**Mode Parameters (CHARACTER*1 → i32):**
- Fortran `CHARACTER*1` parameters → `const i32` in C
- Use integer values: 0, 1, 2, etc.
- Examples: `UPLO`: 0=Upper, 1=Lower; `TRANS`: 0=NoTranspose, 1=Transpose
- **Do NOT convert to `const char*`**

**Arrays:** 0-based C indexing with Fortran logic adjustments. Preserve lda/ldb/ldz.

## Code Quality Rules

**CRITICAL:** Do NOT run ruff checks on existing files. Follow current coding style strictly.
- Ruff checks make code review via diff impractical
- Match existing file's style, indentation, naming conventions
- Only add comments for: bug fixes, TODO, Known Issues
- Do NOT explain code with inline comments

## Test Strategy
- Min 3 tests per routine
- Test data from SLICOT reference docs
- Edge cases (N=0, singular matrices)
- `numpy.testing.assert_allclose` (rtol=1e-14)
- Structure: File per family (`test_ma01xx.py`), classes per routine

## Quality Checklist
- [ ] Tests pass
- [ ] Builds successfully
- [ ] Min 3 tests per routine
- [ ] Edge cases tested
- [ ] TDD commits (RED→GREEN→REFACTOR)
- [ ] Python exports updated
- [ ] Follows existing code style (no ruff on existing files)
