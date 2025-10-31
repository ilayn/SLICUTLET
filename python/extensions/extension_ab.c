// AB family Python extension functions
// This file is included by slicutletmodule.c - do not compile separately

#ifndef SLICUTLET_EXTENSION_INCLUDED
// Headers to silence IDEs - These are included already in slicutletmodule.c during compilation
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slicutlet.h"
#endif

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
