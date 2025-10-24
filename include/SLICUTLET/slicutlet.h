#ifndef SLICUTLET_H
#define SLICUTLET_H

// Public API header for SLICUTLET: exposes selected translated SLICOT routines.

#include <stddef.h>
#include <float.h>

// Basic numeric typedefs used by the API
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ab07nd: Elementary operations on a 2x2 partitioned matrix [A B; C D]
// Computes transformed blocks and condition number of D.
void ab07nd(
	const i32 n,
	const i32 m,
	f64 *a,
	const i32 lda,
	f64 *b,
	const i32 ldb,
	f64 *c,
	const i32 ldc,
	f64 *d,
	const i32 ldd,
	f64* rcond,
	i32 *iwork,
	f64 *dwork,
	const i32 ldwork,
	i32* info
);

// mb03oy: Rank-revealing QR with incremental condition estimation
void mb03oy(
	const i32 m,
	const i32 n,
	f64* a,
	const i32 lda,
	const f64 rcond,
	const f64 svlmax,
	i32* rank,
	f64* sval,
	i32* jpvt,
	f64* tau,
	f64* dwork,
	i32* info
);

// mc01td: Stability test (continuous/discrete) of a real polynomial
void mc01td(
	const i32 dico, // 0 discrete, 1 continuous
	i32* dp,
	f64* p,
	i32* stable,
	i32* nz,
	f64* dwork,
	i32* iwarn,
	i32* info
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
