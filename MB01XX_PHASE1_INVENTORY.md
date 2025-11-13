# MB01XX Test Inventory - Phase 1 Audit

**Date:** 2025-11-12
**Branch:** feat/phase-1-audit-cleanup
**Status:** Foundation for multi-phase test fixes

---

## Summary

**Total Tests:** 43
**All Tests:** Failing with signature mismatches
**Non-existent Functions:** 0 (already removed)
**Unwrapped Functions:** 0 in tests (MB01KD, MB01MD, MB01ND not wrapped, not tested)

---

## Test Inventory by Function

### Phase 2: Simple Functions (No Workspace) - 6 tests

| Function | Tests | Signature | Status |
|----------|-------|-----------|--------|
| **MB01SD** | 3 | `mb01sd(jobs, m, n, a_obj, r_obj, c_obj) → a_obj` | Needs 6 args, tests use 5 |
| **MB01SS** | 2 | `mb01ss(jobs, uplo, n, a_obj, d_obj) → a_obj` | Needs check |
| **MB01XD** | 2 | `mb01xd(m, n, a_obj, b_obj) → (b_obj, info)` | Needs check |
| **MB01XY** | 2 | `mb01xy(uplo, n, a_obj) → (a_obj, info)` | Needs check |

### Phase 3: Scaling/Normalization - 0 tests
| Function | Tests | Wrapped? | Notes |
|----------|-------|----------|-------|
| **MB01PD** | 0 | ✓ | Missing tests |
| **MB01QD** | 0 | ✓ | Missing tests |

### Phase 4: Symmetric/Skew-symmetric Transformations - 8 tests

| Function | Tests | Signature | Status |
|----------|-------|-----------|--------|
| **MB01LD** | 2 | `mb01ld(uplo, trans, m, n, k, alpha, beta, r_obj, a_obj, x_obj, dwork_obj) → (r_obj, x_obj, info)` | Needs 11 args |
| **MB01RB** | 2 | `mb01rb(uplo, trans, n, k, alpha, beta, a_obj, b_obj, c_obj) → (c_obj, info)` | Needs check |
| **MB01RD** | 1 | `mb01rd(uplo, trans, n, k, alpha, beta, b_obj, a_obj) → (a_obj, info)` | Needs check |
| **MB01RH** | 1 | Needs check | Wrapped, untested properly |
| **MB01RT** | 1 | Needs check | Wrapped, untested properly |
| **MB01RU** | 1 | Needs check | Wrapped, untested properly |
| **MB01RW** | 1 | Needs check | Wrapped, untested properly |
| **MB01RX** | 1 | Needs check | Wrapped, untested properly |
| **MB01RY** | 1 | Needs check | Wrapped, untested properly |

### Phase 5: Hessenberg Operations - 14 tests

| Function | Tests | Wrapped? | Status |
|----------|-------|----------|--------|
| **MB01OC** | 1 | ✓ | Signature mismatch |
| **MB01OD** | 1 | ✓ | Signature mismatch |
| **MB01OE** | 1 | ✓ | Signature mismatch |
| **MB01OH** | 1 | ✓ | Signature mismatch |
| **MB01OO** | 1 | ✓ | Signature mismatch |
| **MB01OS** | 1 | ✓ | Signature mismatch |
| **MB01OT** | 1 | ✓ | Signature mismatch |
| **MB01TD** | 1 | ✓ | Signature mismatch |
| **MB01UD** | 1 | ✓ | Signature mismatch |
| **MB01UW** | 1 | ✓ | Signature mismatch |
| **MB01UX** | 2 | ✓ | Signature mismatch |
| **MB01UY** | 2 | ✓ | Signature mismatch |

### Phase 6: Special Operations - 4 tests

| Function | Tests | Wrapped? | Status |
|----------|-------|----------|--------|
| **MB01VD** | 1 | ✓ | Signature mismatch |
| **MB01WD** | 2 | ✓ | Signature mismatch |
| **MB01YD** | 2 | ✓ | Signature mismatch |
| **MB01ZD** | 2 | ✓ | Signature mismatch |

### Phase 7: Complex Function - 2 tests

| Function | Tests | Wrapped? | Status |
|----------|-------|----------|--------|
| **MB01UZ** | 2 | ✓ | Signature mismatch (complex) |

---

## Functions NOT Tested (but wrapped)

Per MB01XX_SIGNATURE_REFERENCE.md, these exist in extension_mb.c but have no tests:

1. **MB01PD** - Safe range scaling (wrapped)
2. **MB01QD** - Scale by CTO/CFROM (wrapped)

---

## Functions NOT Wrapped

Per reference, these have NO Python wrappers:

1. **MB01KD** - Skew-symmetric rank 2k
2. **MB01MD** - Skew-symmetric matrix-vector
3. **MB01ND** - Skew-symmetric rank 2

---

## Phase 1 Deliverables

✓ No non-existent function tests to remove
✓ No unwrapped function tests to skip
✓ Test inventory created
✓ Documented 43 failing tests
✓ Identified 2 wrapped but untested functions (MB01PD, MB01QD)

**Next:** Phase 2 - Fix simple functions (MB01SD, MB01SS, MB01XD, MB01XY)
