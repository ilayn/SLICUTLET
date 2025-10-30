#include "slicutlet.h"
#include "../include/slc_blaslapack.h"
#include <math.h>

void
ma02rd(
    const i32 id, // 0: increasing, 1: decreasing
    const i32 n,
    f64* d,
    f64* e,
    i32* info
)
{
    // Local constants
    const i32 select = 20;

    // Local variables
    i32 stack[2][32];
    i32 stkpnt, start, endd, i, j;
    f64 d1, d2, d3, dmnmx, tmp;

    // Test the input parameters
    *info = 0;
    if ((id != 0) && (id != 1)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        return;
    }

    // Quick return if possible
    if (n <= 1) {
        return;
    }

    // Initialize stack with full range (using 1-based indexing)
    stkpnt = 1;
    stack[0][0] = 1;      // stack(1,1) = 1
    stack[1][0] = n;      // stack(2,1) = n

    // Main loop: process stack until empty
    while (stkpnt > 0) {
        start = stack[0][stkpnt-1];  // stack(1,stkpnt)
        endd = stack[1][stkpnt-1];   // stack(2,stkpnt)
        stkpnt--;

        if (endd - start <= select && endd - start > 0) {
            // Do Insertion sort on D(START:ENDD)

            if (id == 1) {
                // Sort into decreasing order
                for (i = start + 1; i <= endd; i++) {
                    for (j = i; j >= start + 1; j--) {
                        if (d[j-1] > d[j-2]) {
                            dmnmx = d[j-1];
                            d[j-1] = d[j-2];
                            d[j-2] = dmnmx;
                            dmnmx = e[j-1];
                            e[j-1] = e[j-2];
                            e[j-2] = dmnmx;
                        } else {
                            break;
                        }
                    }
                }
            } else {
                // Sort into increasing order
                for (i = start + 1; i <= endd; i++) {
                    for (j = i; j >= start + 1; j--) {
                        if (d[j-1] < d[j-2]) {
                            dmnmx = d[j-1];
                            d[j-1] = d[j-2];
                            d[j-2] = dmnmx;
                            dmnmx = e[j-1];
                            e[j-1] = e[j-2];
                            e[j-2] = dmnmx;
                        } else {
                            break;
                        }
                    }
                }
            }

        } else if (endd - start > select) {
            // Partition D(START:ENDD) and stack parts, largest one first
            // Choose partition entry as median of 3

            d1 = d[start-1];     // D(START)
            d2 = d[endd-1];      // D(ENDD)
            i = (start + endd) / 2;
            d3 = d[i-1];         // D(I)

            if (d1 < d2) {
                if (d3 < d1) {
                    dmnmx = d1;
                } else if (d3 < d2) {
                    dmnmx = d3;
                } else {
                    dmnmx = d2;
                }
            } else {
                if (d3 < d2) {
                    dmnmx = d2;
                } else if (d3 < d1) {
                    dmnmx = d3;
                } else {
                    dmnmx = d1;
                }
            }

            if (id == 1) {
                // Sort into decreasing order
                i = start - 1;
                j = endd + 1;

                while (1) {
                    // Find element from right that is >= dmnmx
                    do {
                        j--;
                    } while (d[j-1] < dmnmx);  // D(J) < DMNMX

                    // Find element from left that is <= dmnmx
                    do {
                        i++;
                    } while (d[i-1] > dmnmx);  // D(I) > DMNMX

                    if (i < j) {
                        // Swap D(I) and D(J)
                        tmp = d[i-1];
                        d[i-1] = d[j-1];
                        d[j-1] = tmp;
                        // Swap E(I) and E(J)
                        tmp = e[i-1];
                        e[i-1] = e[j-1];
                        e[j-1] = tmp;
                    } else {
                        break;
                    }
                }

                // Stack the larger partition first
                if (j - start > endd - j - 1) {
                    stkpnt++;
                    stack[0][stkpnt-1] = start;
                    stack[1][stkpnt-1] = j;
                    stkpnt++;
                    stack[0][stkpnt-1] = j + 1;
                    stack[1][stkpnt-1] = endd;
                } else {
                    stkpnt++;
                    stack[0][stkpnt-1] = j + 1;
                    stack[1][stkpnt-1] = endd;
                    stkpnt++;
                    stack[0][stkpnt-1] = start;
                    stack[1][stkpnt-1] = j;
                }

            } else {
                // Sort into increasing order
                i = start - 1;
                j = endd + 1;

                while (1) {
                    // Find element from right that is <= dmnmx
                    do {
                        j--;
                    } while (d[j-1] > dmnmx);  // D(J) > DMNMX

                    // Find element from left that is >= dmnmx
                    do {
                        i++;
                    } while (d[i-1] < dmnmx);  // D(I) < DMNMX

                    if (i < j) {
                        // Swap D(I) and D(J)
                        tmp = d[i-1];
                        d[i-1] = d[j-1];
                        d[j-1] = tmp;
                        // Swap E(I) and E(J)
                        tmp = e[i-1];
                        e[i-1] = e[j-1];
                        e[j-1] = tmp;
                    } else {
                        break;
                    }
                }

                // Stack the larger partition first
                if (j - start > endd - j - 1) {
                    stkpnt++;
                    stack[0][stkpnt-1] = start;
                    stack[1][stkpnt-1] = j;
                    stkpnt++;
                    stack[0][stkpnt-1] = j + 1;
                    stack[1][stkpnt-1] = endd;
                } else {
                    stkpnt++;
                    stack[0][stkpnt-1] = j + 1;
                    stack[1][stkpnt-1] = endd;
                    stkpnt++;
                    stack[0][stkpnt-1] = start;
                    stack[1][stkpnt-1] = j;
                }
            }
        }
    }
}
