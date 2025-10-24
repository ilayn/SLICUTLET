#include "slicutlet.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check_eq_int(const char* name, int got, int expect) {
    if (got != expect) {
        fprintf(stderr, "FAIL %s: got=%d expect=%d\n", name, got, expect);
        failures++;
    }
}

static void run_mc01_case(const char* label, int dico, int dp_init, double* p_init,
                          int expect_info, int expect_stable, int expect_nz, int expect_iwarn)
{
    int stable = -1, nz = -1, iwarn = 0, info = -1; // sentinel values
    int dp = dp_init;
    double dwork[256];
    memset(dwork, 0, sizeof(dwork));
    double p[16];
    memset(p, 0, sizeof(p));
    memcpy(p, p_init, (dp_init + 1) * sizeof(double));

    mc01td(dico, &dp, p, &stable, &nz, dwork, &iwarn, &info);

    if (info != expect_info) {
        fprintf(stderr, "FAIL %s: info got=%d expect=%d\n", label, info, expect_info);
        failures++;
        return;
    }

    if (expect_info == 0) {
        check_eq_int("stable", stable, expect_stable);
        check_eq_int("nz", nz, expect_nz);
        check_eq_int("iwarn", iwarn, expect_iwarn);
    }
}

int main(void) {
    // 1) Continuous stable: (s+1)(s+2) = s^2 + 3 s + 2
    {
        double p[] = {2.0, 3.0, 1.0};
        run_mc01_case("cont stable", 1, 2, p, /*info*/0, /*stable*/1, /*nz*/0, /*iwarn*/0);
    }

    // 2) Continuous with trailing zeros trimming (degree lowered, iwarn>0)
    {
        // p(x) = s^2 + 3 s + 2, but declared dp=4 with two trailing zeros
        double p[] = {2.0, 3.0, 1.0, 0.0, 0.0};
        run_mc01_case("cont stable + trailing zeros", 1, 4, p, 0, 1, 0, 2);
    }

    // 3) Zero polynomial (all coefficients zero) -> info=1
    {
        double p[] = {0.0, 0.0, 0.0};
        run_mc01_case("zero polynomial", 1, 2, p, 1, /*stable*/0, /*nz*/0, /*iwarn*/0);
    }

    // 4) Continuous unstable: (s-1)(s-2) = s^2 - 3 s + 2 -> two unstable roots
    {
        double p[] = {2.0, -3.0, 1.0};
        // Expect at least one unstable; nz is the count of unstable zeros; here likely 2
        run_mc01_case("cont unstable", 1, 2, p, 0, /*stable*/0, /*nz*/2, /*iwarn*/0);
    }

    // 5) Discrete stable: (z-0.5)(z-0.25) = z^2 - 0.75 z + 0.125
    {
        double p[] = {0.125, -0.75, 1.0};
        run_mc01_case("disc stable", 0, 2, p, 0, /*stable*/1, /*nz*/0, /*iwarn*/0);
    }

    // 6) Discrete unstable: (z-1.5)(z-0.5) = z^2 - 2 z + 0.75 (one root outside unit circle)
    {
        double p[] = {0.75, -2.0, 1.0};
        run_mc01_case("disc unstable", 0, 2, p, 0, /*stable*/0, /*nz*/1, /*iwarn*/0);
    }

    // 7) Discrete with trailing zero(s)
    {
        double p[] = {0.75, -2.0, 1.0, 0.0};
        run_mc01_case("disc unstable + trailing zero", 0, 3, p, 0, /*stable*/0, /*nz*/1, /*iwarn*/1);
    }

    // Also keep the quick ab07nd check for the library surface
    {
        int info = 0;
        int n = 0, m = 0; // zero sizes trigger quick return
        double rcond = -1.0; int iwork[4] = {0}; double w[4] = {0};
        double dummy = 0.0; // placeholders; not referenced when m=0
        ab07nd(n, m, &dummy, 1, &dummy, 1, &dummy, 1, &dummy, 1, &rcond, iwork, w, 4, &info);
        if (info != 0 || rcond != 1.0) {
            fprintf(stderr, "FAIL ab07nd quick return: info=%d rcond=%g\n", info, rcond);
            failures++;
        }
    }

    if (failures == 0) {
        printf("OK\n");
        return 0;
    } else {
        fprintf(stderr, "Total failures: %d\n", failures);
        return 1;
    }
}
