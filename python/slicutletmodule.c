#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "slicutlet.h"

static PyObject* py_ab04md(PyObject* Py_UNUSED(self), PyObject* args) {
    int typ, n, m, p;
    double alpha, beta;
    PyArrayObject *A_obj, *B_obj, *C_obj, *D_obj, *iwork_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiddO!O!O!O!O!O!",
                          &typ, &n, &m, &p, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&A_obj,
                          &PyArray_Type, (PyObject **)&B_obj,
                          &PyArray_Type, (PyObject **)&C_obj,
                          &PyArray_Type, (PyObject **)&D_obj,
                          &PyArray_Type, (PyObject **)&iwork_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    i32 *iwork = (i32*)PyArray_DATA(iwork_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);
    f64 *A = (f64*)PyArray_DATA(A_obj);
    f64 *B = (f64*)PyArray_DATA(B_obj);
    f64 *C = (f64*)PyArray_DATA(C_obj);
    f64 *D = (f64*)PyArray_DATA(D_obj);

    npy_intp *A_dims = PyArray_DIMS(A_obj);
    npy_intp *B_dims = PyArray_DIMS(B_obj);
    npy_intp *C_dims = PyArray_DIMS(C_obj);
    npy_intp *D_dims = PyArray_DIMS(D_obj);

    i32 lda = (i32)(A_dims[0] > 0 ? A_dims[0] : 1);
    i32 ldb = (i32)(B_dims[0] > 0 ? B_dims[0] : 1);
    i32 ldc = (i32)(C_dims[0] > 0 ? C_dims[0] : 1);
    i32 ldd = (i32)(D_dims[0] > 0 ? D_dims[0] : 1);
    i32 ldwork = (i32)PyArray_DIM(dwork_obj, 0);

    i32 info;
    ab04md(typ, n, m, p, alpha, beta, A, lda, B, ldb, C, ldc, D, ldd,
           iwork, dwork, ldwork, &info);

    return Py_BuildValue("OOOOi", A_obj, B_obj, C_obj, D_obj, info);
}

static PyObject* py_ab01nd(PyObject* Py_UNUSED(self), PyObject* args) {
    int jobz, n, m;
    double tol;
    PyArrayObject *A_obj, *B_obj, *Z_obj, *tau_obj, *iwork_obj, *dwork_obj, *nblk_obj;

    if (!PyArg_ParseTuple(args, "iiidO!O!O!O!O!O!O!",
                          &jobz, &n, &m, &tol,
                          &PyArray_Type, (PyObject **)&A_obj,
                          &PyArray_Type, (PyObject **)&B_obj,
                          &PyArray_Type, (PyObject **)&Z_obj,
                          &PyArray_Type, (PyObject **)&tau_obj,
                          &PyArray_Type, (PyObject **)&nblk_obj,
                          &PyArray_Type, (PyObject **)&iwork_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *A = (f64*)PyArray_DATA(A_obj);
    f64 *B = (f64*)PyArray_DATA(B_obj);
    f64 *Z = (f64*)PyArray_DATA(Z_obj);
    f64 *tau = (f64*)PyArray_DATA(tau_obj);
    i32 *nblk = (i32*)PyArray_DATA(nblk_obj);
    i32 *iwork = (i32*)PyArray_DATA(iwork_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *A_dims = PyArray_DIMS(A_obj);
    npy_intp *B_dims = PyArray_DIMS(B_obj);
    npy_intp *Z_dims = PyArray_DIMS(Z_obj);

    i32 lda = (i32)(A_dims[0] > 0 ? A_dims[0] : 1);
    i32 ldb = (i32)(B_dims[0] > 0 ? B_dims[0] : 1);
    i32 ldz = (i32)(Z_dims[0] > 0 ? Z_dims[0] : 1);
    i32 ldwork = (i32)PyArray_DIM(dwork_obj, 0);

    i32 ncont, indcon, info;
    ab01nd(jobz, n, m, A, lda, B, ldb, &ncont, &indcon, nblk, Z, ldz, tau,
           tol, iwork, dwork, ldwork, &info);

    return Py_BuildValue("OOOOOiii", A_obj, B_obj, Z_obj, tau_obj, nblk_obj, ncont, indcon, info);
}

static PyObject* py_ab05md(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, over, n1, m1, p1, n2, p2;
    PyArrayObject *A1_obj, *B1_obj, *C1_obj, *D1_obj;
    PyArrayObject *A2_obj, *B2_obj, *C2_obj, *D2_obj;
    PyArrayObject *A_obj, *B_obj, *C_obj, *D_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiiiiO!O!O!O!O!O!O!O!O!O!O!O!O!",
                          &uplo, &over, &n1, &m1, &p1, &n2, &p2,
                          &PyArray_Type, (PyObject **)&A1_obj,
                          &PyArray_Type, (PyObject **)&B1_obj,
                          &PyArray_Type, (PyObject **)&C1_obj,
                          &PyArray_Type, (PyObject **)&D1_obj,
                          &PyArray_Type, (PyObject **)&A2_obj,
                          &PyArray_Type, (PyObject **)&B2_obj,
                          &PyArray_Type, (PyObject **)&C2_obj,
                          &PyArray_Type, (PyObject **)&D2_obj,
                          &PyArray_Type, (PyObject **)&A_obj,
                          &PyArray_Type, (PyObject **)&B_obj,
                          &PyArray_Type, (PyObject **)&C_obj,
                          &PyArray_Type, (PyObject **)&D_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *A1 = (f64*)PyArray_DATA(A1_obj);
    f64 *B1 = (f64*)PyArray_DATA(B1_obj);
    f64 *C1 = (f64*)PyArray_DATA(C1_obj);
    f64 *D1 = (f64*)PyArray_DATA(D1_obj);
    f64 *A2 = (f64*)PyArray_DATA(A2_obj);
    f64 *B2 = (f64*)PyArray_DATA(B2_obj);
    f64 *C2 = (f64*)PyArray_DATA(C2_obj);
    f64 *D2 = (f64*)PyArray_DATA(D2_obj);
    f64 *A = (f64*)PyArray_DATA(A_obj);
    f64 *B = (f64*)PyArray_DATA(B_obj);
    f64 *C = (f64*)PyArray_DATA(C_obj);
    f64 *D = (f64*)PyArray_DATA(D_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *A1_dims = PyArray_DIMS(A1_obj);
    npy_intp *B1_dims = PyArray_DIMS(B1_obj);
    npy_intp *C1_dims = PyArray_DIMS(C1_obj);
    npy_intp *D1_dims = PyArray_DIMS(D1_obj);
    npy_intp *A2_dims = PyArray_DIMS(A2_obj);
    npy_intp *B2_dims = PyArray_DIMS(B2_obj);
    npy_intp *C2_dims = PyArray_DIMS(C2_obj);
    npy_intp *D2_dims = PyArray_DIMS(D2_obj);
    npy_intp *A_dims = PyArray_DIMS(A_obj);
    npy_intp *B_dims = PyArray_DIMS(B_obj);
    npy_intp *C_dims = PyArray_DIMS(C_obj);
    npy_intp *D_dims = PyArray_DIMS(D_obj);

    i32 lda1 = (i32)(A1_dims[0] > 0 ? A1_dims[0] : 1);
    i32 ldb1 = (i32)(B1_dims[0] > 0 ? B1_dims[0] : 1);
    i32 ldc1 = (i32)(C1_dims[0] > 0 ? C1_dims[0] : 1);
    i32 ldd1 = (i32)(D1_dims[0] > 0 ? D1_dims[0] : 1);
    i32 lda2 = (i32)(A2_dims[0] > 0 ? A2_dims[0] : 1);
    i32 ldb2 = (i32)(B2_dims[0] > 0 ? B2_dims[0] : 1);
    i32 ldc2 = (i32)(C2_dims[0] > 0 ? C2_dims[0] : 1);
    i32 ldd2 = (i32)(D2_dims[0] > 0 ? D2_dims[0] : 1);
    i32 lda = (i32)(A_dims[0] > 0 ? A_dims[0] : 1);
    i32 ldb = (i32)(B_dims[0] > 0 ? B_dims[0] : 1);
    i32 ldc = (i32)(C_dims[0] > 0 ? C_dims[0] : 1);
    i32 ldd = (i32)(D_dims[0] > 0 ? D_dims[0] : 1);
    i32 ldwork = (i32)PyArray_DIM(dwork_obj, 0);

    i32 n, info;
    ab05md(uplo, over, n1, m1, p1, n2, p2,
           A1, lda1, B1, ldb1, C1, ldc1, D1, ldd1,
           A2, lda2, B2, ldb2, C2, ldc2, D2, ldd2,
           &n, A, lda, B, ldb, C, ldc, D, ldd,
           dwork, ldwork, &info);

    return Py_BuildValue("iOOOOi", n, A_obj, B_obj, C_obj, D_obj, info);
}

static PyObject* py_ab05nd(PyObject* Py_UNUSED(self), PyObject* args) {
    int over, n1, m1, p1, n2;
    double alpha;
    PyArrayObject *A1_obj, *B1_obj, *C1_obj, *D1_obj;
    PyArrayObject *A2_obj, *B2_obj, *C2_obj, *D2_obj;
    PyArrayObject *A_obj, *B_obj, *C_obj, *D_obj, *iwork_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiidO!O!O!O!O!O!O!O!O!O!O!O!O!O!",
                          &over, &n1, &m1, &p1, &n2, &alpha,
                          &PyArray_Type, (PyObject **)&A1_obj,
                          &PyArray_Type, (PyObject **)&B1_obj,
                          &PyArray_Type, (PyObject **)&C1_obj,
                          &PyArray_Type, (PyObject **)&D1_obj,
                          &PyArray_Type, (PyObject **)&A2_obj,
                          &PyArray_Type, (PyObject **)&B2_obj,
                          &PyArray_Type, (PyObject **)&C2_obj,
                          &PyArray_Type, (PyObject **)&D2_obj,
                          &PyArray_Type, (PyObject **)&A_obj,
                          &PyArray_Type, (PyObject **)&B_obj,
                          &PyArray_Type, (PyObject **)&C_obj,
                          &PyArray_Type, (PyObject **)&D_obj,
                          &PyArray_Type, (PyObject **)&iwork_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *A1 = (f64*)PyArray_DATA(A1_obj);
    f64 *B1 = (f64*)PyArray_DATA(B1_obj);
    f64 *C1 = (f64*)PyArray_DATA(C1_obj);
    f64 *D1 = (f64*)PyArray_DATA(D1_obj);
    f64 *A2 = (f64*)PyArray_DATA(A2_obj);
    f64 *B2 = (f64*)PyArray_DATA(B2_obj);
    f64 *C2 = (f64*)PyArray_DATA(C2_obj);
    f64 *D2 = (f64*)PyArray_DATA(D2_obj);
    f64 *A = (f64*)PyArray_DATA(A_obj);
    f64 *B = (f64*)PyArray_DATA(B_obj);
    f64 *C = (f64*)PyArray_DATA(C_obj);
    f64 *D = (f64*)PyArray_DATA(D_obj);
    i32 *iwork = (i32*)PyArray_DATA(iwork_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *A1_dims = PyArray_DIMS(A1_obj);
    npy_intp *B1_dims = PyArray_DIMS(B1_obj);
    npy_intp *C1_dims = PyArray_DIMS(C1_obj);
    npy_intp *D1_dims = PyArray_DIMS(D1_obj);
    npy_intp *A2_dims = PyArray_DIMS(A2_obj);
    npy_intp *B2_dims = PyArray_DIMS(B2_obj);
    npy_intp *C2_dims = PyArray_DIMS(C2_obj);
    npy_intp *D2_dims = PyArray_DIMS(D2_obj);
    npy_intp *A_dims = PyArray_DIMS(A_obj);
    npy_intp *B_dims = PyArray_DIMS(B_obj);
    npy_intp *C_dims = PyArray_DIMS(C_obj);
    npy_intp *D_dims = PyArray_DIMS(D_obj);

    i32 lda1 = (i32)(A1_dims[0] > 0 ? A1_dims[0] : 1);
    i32 ldb1 = (i32)(B1_dims[0] > 0 ? B1_dims[0] : 1);
    i32 ldc1 = (i32)(C1_dims[0] > 0 ? C1_dims[0] : 1);
    i32 ldd1 = (i32)(D1_dims[0] > 0 ? D1_dims[0] : 1);
    i32 lda2 = (i32)(A2_dims[0] > 0 ? A2_dims[0] : 1);
    i32 ldb2 = (i32)(B2_dims[0] > 0 ? B2_dims[0] : 1);
    i32 ldc2 = (i32)(C2_dims[0] > 0 ? C2_dims[0] : 1);
    i32 ldd2 = (i32)(D2_dims[0] > 0 ? D2_dims[0] : 1);
    i32 lda = (i32)(A_dims[0] > 0 ? A_dims[0] : 1);
    i32 ldb = (i32)(B_dims[0] > 0 ? B_dims[0] : 1);
    i32 ldc = (i32)(C_dims[0] > 0 ? C_dims[0] : 1);
    i32 ldd = (i32)(D_dims[0] > 0 ? D_dims[0] : 1);
    i32 ldwork = (i32)PyArray_DIM(dwork_obj, 0);

    i32 n, info;
    ab05nd(over, n1, m1, p1, n2, alpha,
               A1, lda1, B1, ldb1, C1, ldc1, D1, ldd1,
               A2, lda2, B2, ldb2, C2, ldc2, D2, ldd2,
               &n, A, lda, B, ldb, C, ldc, D, ldd,
               iwork, dwork, ldwork, &info);

    return Py_BuildValue("iOOOOi", n, A_obj, B_obj, C_obj, D_obj, info);
}

static PyObject* py_ab07nd(PyObject* Py_UNUSED(self), PyObject* args) {
    int n, m;
    PyArrayObject *A_obj, *B_obj, *C_obj, *D_obj, *iwork_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!O!O!",
                          &n, &m,
                          &PyArray_Type, (PyObject **)&A_obj,
                          &PyArray_Type, (PyObject **)&B_obj,
                          &PyArray_Type, (PyObject **)&C_obj,
                          &PyArray_Type, (PyObject **)&D_obj,
                          &PyArray_Type, (PyObject **)&iwork_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    i32 *iwork = (i32*)PyArray_DATA(iwork_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);
    f64 *A = (f64*)PyArray_DATA(A_obj);
    f64 *B = (f64*)PyArray_DATA(B_obj);
    f64 *C = (f64*)PyArray_DATA(C_obj);
    f64 *D = (f64*)PyArray_DATA(D_obj);

    npy_intp *A_dims = PyArray_DIMS(A_obj);
    npy_intp *B_dims = PyArray_DIMS(B_obj);
    npy_intp *C_dims = PyArray_DIMS(C_obj);
    npy_intp *D_dims = PyArray_DIMS(D_obj);

    i32 lda = (i32)(A_dims[0] > 0 ? A_dims[0] : 1);
    i32 ldb = (i32)(B_dims[0] > 0 ? B_dims[0] : 1);
    i32 ldc = (i32)(C_dims[0] > 0 ? C_dims[0] : 1);
    i32 ldd = (i32)(D_dims[0] > 0 ? D_dims[0] : 1);
    i32 ldwork = (i32)PyArray_DIM(dwork_obj, 0);

    f64 rcond;
    i32 info;
    ab07nd(n, m, A, lda, B, ldb, C, ldc, D, ldd, &rcond,
           iwork, dwork, ldwork, &info);

    return Py_BuildValue("OOOOdi", A_obj, B_obj, C_obj, D_obj, rcond, info);
}

static PyObject* py_mb03oy(PyObject* Py_UNUSED(self), PyObject* args) {
    int m, n;
    double rcond, svlmax;
    PyArrayObject *a_obj, *sval_obj, *jpvt_obj, *tau_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiddO!O!O!O!O!",
                          &m, &n, &rcond, &svlmax,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&sval_obj,
                          &PyArray_Type, (PyObject **)&jpvt_obj,
                          &PyArray_Type, (PyObject **)&tau_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *sval = (f64*)PyArray_DATA(sval_obj);
    i32 *jpvt = (i32*)PyArray_DATA(jpvt_obj);
    f64 *tau = (f64*)PyArray_DATA(tau_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 rank;
    i32 info;
    mb03oy(m, n, a, lda, rcond, svlmax, &rank, sval, jpvt, tau, dwork, &info);

    return Py_BuildValue("OiOOOi", a_obj, rank, sval_obj, jpvt_obj, tau_obj, info);
}

static PyObject* py_mc01td(PyObject* Py_UNUSED(self), PyObject* args) {
    int dico;
    PyArrayObject *dp_obj, *p_obj, *stable_obj, *nz_obj, *dwork_obj, *iwarn_obj, *info_obj;

    if (!PyArg_ParseTuple(args, "iO!O!O!O!O!O!O!",
                          &dico,
                          &PyArray_Type, (PyObject **)&dp_obj,
                          &PyArray_Type, (PyObject **)&p_obj,
                          &PyArray_Type, (PyObject **)&stable_obj,
                          &PyArray_Type, (PyObject **)&nz_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj,
                          &PyArray_Type, (PyObject **)&iwarn_obj,
                          &PyArray_Type, (PyObject **)&info_obj)) {
        return NULL;
    }

    i32 *dp = (i32*)PyArray_DATA(dp_obj);
    f64 *p = (f64*)PyArray_DATA(p_obj);
    i32 *stable = (i32*)PyArray_DATA(stable_obj);
    i32 *nz = (i32*)PyArray_DATA(nz_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);
    i32 *iwarn = (i32*)PyArray_DATA(iwarn_obj);
    i32 *info = (i32*)PyArray_DATA(info_obj);

    mc01td(dico, dp, p, stable, nz, dwork, iwarn, info);

    return Py_BuildValue("Oiiii",
                         stable[0] ? Py_True : Py_False,
                         nz[0], dp[0], iwarn[0], info[0]);
}

static PyObject* py_ma01ad(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 xr, xi, yr, yi;
    if (!PyArg_ParseTuple(args, "dd", &xr, &xi)) { return NULL; }

    ma01ad(xr, xi, &yr, &yi);

    return Py_BuildValue("dd", yr, yi);
}

static PyObject* py_ma01bd(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 base, logbase, alpha, beta;
    i32 k, inca, scale;
    PyArrayObject *a_obj, *s_obj;

    if (!PyArg_ParseTuple(args, "ddiiO!O!",
                          &base, &logbase, &k, &inca,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&s_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    i32 *s = (i32*)PyArray_DATA(s_obj);

    ma01bd(base, logbase, k, s, a, inca, &alpha, &beta, &scale);

    return Py_BuildValue("ddi", alpha, beta, scale);
}

static PyObject* py_ma01bz(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 base;
    c128 alpha, beta;
    i32 k, inca, scale;
    PyArrayObject *a_obj, *s_obj;

    if (!PyArg_ParseTuple(args, "diiO!O!",
                          &base, &k, &inca,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&s_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);
    i32 *s = (i32*)PyArray_DATA(s_obj);

    ma01bz(base, k, s, a, inca, &alpha, &beta, &scale);

    // Convert c128 to Py_complex
    Py_complex py_alpha = {creal(alpha), cimag(alpha)};
    Py_complex py_beta = {creal(beta), cimag(beta)};

    return Py_BuildValue("DDi", &py_alpha, &py_beta, scale);
}

static PyObject* py_ma01cd(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 a, b;
    i32 ia, ib, result;
    if (!PyArg_ParseTuple(args, "didi",
                          &a, &ia, &b, &ib)) {
        return NULL;
    }

    result = ma01cd(a, ia, b, ib);

    return Py_BuildValue("i", result);
}

static PyObject* py_ma01dd(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 ar1, ai1, ar2, ai2, eps, safemin, d;
    if (!PyArg_ParseTuple(args, "dddddd",
                          &ar1, &ai1, &ar2, &ai2, &eps, &safemin)) {
        return NULL;
    }

    ma01dd(ar1, ai1, ar2, ai2, eps, safemin, &d);

    return Py_BuildValue("d", d);
}

static PyObject* py_ma01dz(PyObject* Py_UNUSED(self), PyObject* args) {
    f64 ar1, ar2, ai1, ai2, b1, b2, eps, safemin, d1, d2;
    i32 iwarn;

    if (!PyArg_ParseTuple(args, "dddddddd",
                          &ar1, &ai1, &ar2, &ai2, &b1, &b2, &eps, &safemin)) {
        return NULL;
    }

    ma01dz(ar1, ai1, ar2, ai2, b1, b2, eps, safemin, &d1, &d2, &iwarn);

    return Py_BuildValue("ddi", d1, d2, iwarn);
}

static PyObject* py_ma02bd(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, m, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iO!",
                          &side,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    m = (i32)(a_dims[0]);
    n = (i32)(a_dims[1]);

    ma02bd(side, m, n, a, lda);

    return Py_BuildValue("O", a_obj);
}

static PyObject* py_mb01pd(PyObject* Py_UNUSED(self), PyObject* args) {
    int scun, type, m, n, kl, ku, nbl;
    double anrm;
    PyArrayObject *nrows_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "iiiiiidiO!O!",
                          &scun, &type, &m, &n, &kl, &ku, &anrm, &nbl,
                          &PyArray_Type, (PyObject **)&nrows_obj,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    i32 *nrows = (i32*)PyArray_DATA(nrows_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 info;
    mb01pd(scun, type, m, n, kl, ku, anrm, nbl, nrows, a, lda, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01qd(PyObject* Py_UNUSED(self), PyObject* args) {
    int type, m, n, kl, ku, nbl;
    double cfrom, cto;
    PyArrayObject *nrows_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddiO!O!",
                          &type, &m, &n, &kl, &ku, &cfrom, &cto, &nbl,
                          &PyArray_Type, (PyObject **)&nrows_obj,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    i32 *nrows = (i32*)PyArray_DATA(nrows_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 info;
    mb01qd(type, m, n, kl, ku, cfrom, cto, nbl, nrows, a, lda, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyMethodDef module_methods[] = {
    {"py_ab01nd", py_ab01nd, METH_VARARGS, "Wrapper for ab01nd"},
    {"py_ab04md", py_ab04md, METH_VARARGS, "Wrapper for ab04md"},
    {"py_ab05md", py_ab05md, METH_VARARGS, "Wrapper for ab05md"},
    {"py_ab05nd", py_ab05nd, METH_VARARGS, "Wrapper for ab05nd"},
    {"py_ab07nd", py_ab07nd, METH_VARARGS, "Wrapper for ab07nd"},
    {"py_ma01ad", py_ma01ad, METH_VARARGS, "Wrapper for ma01ad"},
    {"py_ma01bd", py_ma01bd, METH_VARARGS, "Wrapper for ma01bd"},
    {"py_ma01bz", py_ma01bz, METH_VARARGS, "Wrapper for ma01bz"},
    {"py_ma01cd", py_ma01cd, METH_VARARGS, "Wrapper for ma01cd"},
    {"py_ma01dd", py_ma01dd, METH_VARARGS, "Wrapper for ma01dd"},
    {"py_ma01dz", py_ma01dz, METH_VARARGS, "Wrapper for ma01dz"},
    {"py_ma02bd", py_ma02bd, METH_VARARGS, "Wrapper for ma02bd"},
    {"py_mb01pd", py_mb01pd, METH_VARARGS, "Wrapper for mb01pd"},
    {"py_mb01qd", py_mb01qd, METH_VARARGS, "Wrapper for mb01qd"},
    {"py_mb03oy", py_mb03oy, METH_VARARGS, "Wrapper for mb03oy"},
    {"py_mc01td", py_mc01td, METH_VARARGS, "Wrapper for mc01td"},
    {NULL, NULL, 0, NULL}
};

static int pyslicutlet_exec(PyObject *module) {
    (void)module;
    if (PyArray_API == NULL) {
        if (_import_array() < 0) {
            return -1;
        }
    }
    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, pyslicutlet_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    {0, NULL}
};

static struct PyModuleDef pyslicutlet_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pyslicutlet",
    .m_doc = "Python bindings for SLICUTLET",
    .m_size = 0,
    .m_methods = module_methods,
    .m_slots = module_slots,
};

PyMODINIT_FUNC PyInit_pyslicutlet(void) {
    return PyModuleDef_Init(&pyslicutlet_module);
}
