#ifndef _SLICUTLET_TYPES_H
#define _SLICUTLET_TYPES_H

#include <float.h>
#include <stdint.h>
#include <complex.h>

typedef int32_t i32;
typedef int64_t i64;
typedef float   f32;
typedef double  f64;
typedef complex float c64;
typedef complex double c128;

// C11 complex construction macros (fallback for MinGW and other platforms)
#ifndef CMPLX
#define CMPLX(x, y) ((double complex)((double)(x) + _Complex_I * (double)(y)))
#endif
#ifndef CMPLXF
#define CMPLXF(x, y) ((float complex)((float)(x) + _Complex_I * (float)(y)))
#endif

// Unfortunately, we still need to define these ourselves in C.
static inline int MIN(const int a, const int b) { return a < b ? a : b; }
static inline int MAX(const int a, const int b) { return a > b ? a : b; }


#endif
