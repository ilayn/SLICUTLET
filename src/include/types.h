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

// Unfortunately, we still need to define these ourselves in C.
static inline int MIN(const int a, const int b) { return a < b ? a : b; }
static inline int MAX(const int a, const int b) { return a > b ? a : b; }


#endif
