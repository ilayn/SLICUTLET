// MA family Python extension functions
// This file is included by slicutletmodule.c - do not compile separately

#ifndef SLICUTLET_EXTENSION_INCLUDED
// Headers to silence IDEs - These are included already in slicutletmodule.c during compilation
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slicutlet.h"
#endif

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

static PyObject* py_ma02ad(PyObject* Py_UNUSED(self), PyObject* args) {
    int job, m, n;
    PyArrayObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiiO!",
                          &job, &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    // Create output array b with transposed dimensions (n x m) in Fortran order
    npy_intp b_dims[2] = {n, m};
    b_obj = (PyArrayObject*)PyArray_ZEROS(2, b_dims, NPY_FLOAT64, 1);  // 1 = Fortran order
    if (b_obj == NULL) {
        return NULL;
    }

    f64 *b = (f64*)PyArray_DATA(b_obj);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    ma02ad(job, m, n, a, lda, b, ldb);

    return Py_BuildValue("N", PyArray_Return(b_obj));
}

static PyObject* py_ma02az(PyObject* Py_UNUSED(self), PyObject* args) {
    int trans, job, m, n;
    PyArrayObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiiiO!",
                          &trans, &job, &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    // Create output array b with transposed dimensions (n x m) in Fortran order
    npy_intp b_dims[2] = {n, m};
    b_obj = (PyArrayObject*)PyArray_ZEROS(2, b_dims, NPY_COMPLEX128, 1);  // 1 = Fortran order
    if (b_obj == NULL) {
        return NULL;
    }

    c128 *b = (c128*)PyArray_DATA(b_obj);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    ma02az(trans, job, m, n, a, lda, b, ldb);

    return Py_BuildValue("N", PyArray_Return(b_obj));
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

static PyObject* py_ma02bz(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, m, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iO!",
                          &side,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    m = (i32)(a_dims[0]);
    n = (i32)(a_dims[1]);

    ma02bz(side, m, n, a, lda);

    return Py_BuildValue("O", a_obj);
}

static PyObject* py_ma02cd(PyObject* Py_UNUSED(self), PyObject* args) {
    int n, kl, ku;
    PyArrayObject *a_obj, *a_copy;

    if (!PyArg_ParseTuple(args, "iiiO!",
                          &n, &kl, &ku,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_copy);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02cd(n, kl, ku, a, lda);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02cz(PyObject* Py_UNUSED(self), PyObject* args) {
    int n, kl, ku;
    PyArrayObject *a_obj, *a_copy;

    if (!PyArg_ParseTuple(args, "iiiO!",
                          &n, &kl, &ku,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_copy);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02cz(n, kl, ku, a, lda);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02dd(PyObject* Py_UNUSED(self), PyObject* args) {
    int job, uplo, n;
    PyArrayObject *input_obj, *output_obj;

    if (!PyArg_ParseTuple(args, "iiiO!",
                          &job, &uplo, &n,
                          &PyArray_Type, (PyObject **)&input_obj)) {
        return NULL;
    }

    if (job == 0) {
        // Pack: input is full matrix a, output is packed ap
        f64 *a = (f64*)PyArray_DATA(input_obj);
        npy_intp *a_dims = PyArray_DIMS(input_obj);
        i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

        // Create output packed array with size n*(n+1)/2
        npy_intp ap_size = (npy_intp)(n * (n + 1) / 2);
        output_obj = (PyArrayObject*)PyArray_ZEROS(1, &ap_size, NPY_FLOAT64, 0);
        if (output_obj == NULL) {
            return NULL;
        }
        f64 *ap = (f64*)PyArray_DATA(output_obj);

        ma02dd(job, uplo, n, a, lda, ap);
    } else {
        // Unpack: input is packed ap, output is full matrix a
        f64 *ap = (f64*)PyArray_DATA(input_obj);

        // Create output full matrix with dimensions (n, n) in Fortran order
        npy_intp a_dims[2] = {n, n};
        output_obj = (PyArrayObject*)PyArray_ZEROS(2, a_dims, NPY_FLOAT64, 1);  // 1 = Fortran order
        if (output_obj == NULL) {
            return NULL;
        }
        f64 *a = (f64*)PyArray_DATA(output_obj);
        i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

        ma02dd(job, uplo, n, a, lda, ap);
    }

    return Py_BuildValue("N", PyArray_Return(output_obj));
}

static PyObject* py_ma02ed(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, n;
    PyArrayObject *a_obj, *a_copy;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_copy);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02ed(uplo, n, a, lda);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02es(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, n;
    PyArrayObject *a_obj, *a_copy;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_copy);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02es(uplo, n, a, lda);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02ez(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, skew, n;
    PyArrayObject *a_obj, *a_copy;

    if (!PyArg_ParseTuple(args, "iiiiO!",
                          &uplo, &trans, &skew, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_copy);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02ez(uplo, trans, skew, n, a, lda);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02fd(PyObject* Py_UNUSED(self), PyObject* args) {
    double x1, x2, c = 0.0, s = 0.0;  // Initialize c and s
    i32 info;

    if (!PyArg_ParseTuple(args, "dd", &x1, &x2)) {
        return NULL;
    }

    ma02fd(&x1, &x2, &c, &s, &info);

    return Py_BuildValue("dddi", x1, c, s, info);
}

static PyObject* py_ma02gd(PyObject* Py_UNUSED(self), PyObject* args) {
    int n, k1, k2, incx;
    PyArrayObject *a_obj, *a_copy, *ipiv_obj;

    if (!PyArg_ParseTuple(args, "iO!iiO!i",
                          &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &k1, &k2,
                          &PyArray_Type, (PyObject **)&ipiv_obj,
                          &incx)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_copy);
    i32 *ipiv = (i32*)PyArray_DATA(ipiv_obj);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02gd(n, a, lda, k1, k2, ipiv, incx);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02gz(PyObject* Py_UNUSED(self), PyObject* args) {
    int n, k1, k2, incx;
    PyArrayObject *a_obj, *a_copy, *ipiv_obj;

    if (!PyArg_ParseTuple(args, "iO!iiO!i",
                          &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &k1, &k2,
                          &PyArray_Type, (PyObject **)&ipiv_obj,
                          &incx)) {
        return NULL;
    }

    // Create a copy of the input array to modify
    a_copy = (PyArrayObject*)PyArray_NewCopy(a_obj, NPY_FORTRANORDER);
    if (a_copy == NULL) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_copy);
    i32 *ipiv = (i32*)PyArray_DATA(ipiv_obj);

    npy_intp *a_dims = PyArray_DIMS(a_copy);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02gz(n, a, lda, k1, k2, ipiv, incx);

    return Py_BuildValue("N", PyArray_Return(a_copy));
}

static PyObject* py_ma02hd(PyObject* Py_UNUSED(self), PyObject* args) {
    int job, m, n;
    double diag;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiidO!",
                          &job, &m, &n, &diag,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 result = ma02hd(job, m, n, diag, a, lda);

    return Py_BuildValue("i", result);
}

static PyObject* py_ma02hz(PyObject* Py_UNUSED(self), PyObject* args) {
    int job, m, n;
    Py_complex py_diag;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiiDO!",
                          &job, &m, &n, &py_diag,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);
    c128 diag = py_diag.real + py_diag.imag * I;

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 result = ma02hz(job, m, n, diag, a, lda);

    return Py_BuildValue("i", result);
}

static PyObject* py_ma02id(PyObject* Py_UNUSED(self), PyObject* args) {
    int typ, norm, n;
    PyArrayObject *a_obj, *qg_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &typ, &norm, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&qg_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *qg = (f64*)PyArray_DATA(qg_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *qg_dims = PyArray_DIMS(qg_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldqg = (i32)(qg_dims[0] > 0 ? qg_dims[0] : 1);

    f64 result = ma02id(typ, norm, n, a, lda, qg, ldqg, dwork);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02iz(PyObject* Py_UNUSED(self), PyObject* args) {
    int typ, norm, n;
    PyArrayObject *a_obj, *qg_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &typ, &norm, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&qg_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);
    c128 *qg = (c128*)PyArray_DATA(qg_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *qg_dims = PyArray_DIMS(qg_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldqg = (i32)(qg_dims[0] > 0 ? qg_dims[0] : 1);

    f64 result = ma02iz(typ, norm, n, a, lda, qg, ldqg, dwork);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02jd(PyObject* Py_UNUSED(self), PyObject* args) {
    int ltran1, ltran2, n;
    PyArrayObject *q1_obj, *q2_obj, *res_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &ltran1, &ltran2, &n,
                          &PyArray_Type, (PyObject **)&q1_obj,
                          &PyArray_Type, (PyObject **)&q2_obj,
                          &PyArray_Type, (PyObject **)&res_obj)) {
        return NULL;
    }

    f64 *q1 = (f64*)PyArray_DATA(q1_obj);
    f64 *q2 = (f64*)PyArray_DATA(q2_obj);
    f64 *res = (f64*)PyArray_DATA(res_obj);

    npy_intp *q1_dims = PyArray_DIMS(q1_obj);
    npy_intp *q2_dims = PyArray_DIMS(q2_obj);
    npy_intp *res_dims = PyArray_DIMS(res_obj);
    i32 ldq1 = (i32)(q1_dims[0] > 0 ? q1_dims[0] : 1);
    i32 ldq2 = (i32)(q2_dims[0] > 0 ? q2_dims[0] : 1);
    i32 ldres = (i32)(res_dims[0] > 0 ? res_dims[0] : 1);

    f64 result = ma02jd(ltran1, ltran2, n, q1, ldq1, q2, ldq2, res, ldres);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02jz(PyObject* Py_UNUSED(self), PyObject* args) {
    int ltran1, ltran2, n;
    PyArrayObject *q1_obj, *q2_obj, *res_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &ltran1, &ltran2, &n,
                          &PyArray_Type, (PyObject **)&q1_obj,
                          &PyArray_Type, (PyObject **)&q2_obj,
                          &PyArray_Type, (PyObject **)&res_obj)) {
        return NULL;
    }

    c128 *q1 = (c128*)PyArray_DATA(q1_obj);
    c128 *q2 = (c128*)PyArray_DATA(q2_obj);
    c128 *res = (c128*)PyArray_DATA(res_obj);

    npy_intp *q1_dims = PyArray_DIMS(q1_obj);
    npy_intp *q2_dims = PyArray_DIMS(q2_obj);
    npy_intp *res_dims = PyArray_DIMS(res_obj);
    i32 ldq1 = (i32)(q1_dims[0] > 0 ? q1_dims[0] : 1);
    i32 ldq2 = (i32)(q2_dims[0] > 0 ? q2_dims[0] : 1);
    i32 ldres = (i32)(res_dims[0] > 0 ? res_dims[0] : 1);

    f64 result = ma02jz(ltran1, ltran2, n, q1, ldq1, q2, ldq2, res, ldres);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02md(PyObject* Py_UNUSED(self), PyObject* args) {
    int norm, uplo, n;
    PyArrayObject *a_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!",
                          &norm, &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    f64 result = ma02md(norm, uplo, n, a, lda, dwork);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02mz(PyObject* Py_UNUSED(self), PyObject* args) {
    int norm, uplo, n;
    PyArrayObject *a_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!",
                          &norm, &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    f64 result = ma02mz(norm, uplo, n, a, lda, dwork);

    return Py_BuildValue("d", result);
}

static PyObject* py_ma02nz(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, skew, n, k, l;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiiiiiO!",
                          &uplo, &trans, &skew, &n, &k, &l,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    ma02nz(uplo, trans, skew, n, k, l, a, lda);

    return Py_BuildValue("O", a_obj);
}

static PyObject* py_ma02od(PyObject* Py_UNUSED(self), PyObject* args) {
    int skew, m;
    PyArrayObject *a_obj, *de_obj;

    if (!PyArg_ParseTuple(args, "iiO!O!",
                          &skew, &m,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&de_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *de = (f64*)PyArray_DATA(de_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *de_dims = PyArray_DIMS(de_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldde = (i32)(de_dims[0] > 0 ? de_dims[0] : 1);

    i32 result = ma02od(skew, m, a, lda, de, ldde);

    return Py_BuildValue("i", result);
}

static PyObject* py_ma02oz(PyObject* Py_UNUSED(self), PyObject* args) {
    int skew, m;
    PyArrayObject *a_obj, *de_obj;

    if (!PyArg_ParseTuple(args, "iiO!O!",
                          &skew, &m,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&de_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);
    c128 *de = (c128*)PyArray_DATA(de_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *de_dims = PyArray_DIMS(de_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldde = (i32)(de_dims[0] > 0 ? de_dims[0] : 1);

    i32 result = ma02oz(skew, m, a, lda, de, ldde);

    return Py_BuildValue("i", result);
}

static PyObject* py_ma02pd(PyObject* Py_UNUSED(self), PyObject* args) {
    int m, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 nzr, nzc;
    ma02pd(m, n, a, lda, &nzr, &nzc);

    return Py_BuildValue("ii", nzr, nzc);
}

static PyObject* py_ma02pz(PyObject* Py_UNUSED(self), PyObject* args) {
    int m, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    c128 *a = (c128*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 nzr, nzc;
    ma02pz(m, n, a, lda, &nzr, &nzc);

    return Py_BuildValue("ii", nzr, nzc);
}

static PyObject* py_ma02rd(PyObject* Py_UNUSED(self), PyObject* args) {
    int id, n;
    PyArrayObject *d_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "iiO!O!",
                          &id, &n,
                          &PyArray_Type, (PyObject **)&d_obj,
                          &PyArray_Type, (PyObject **)&e_obj)) {
        return NULL;
    }

    f64 *d = (f64*)PyArray_DATA(d_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);

    i32 info;
    ma02rd(id, n, d, e, &info);

    return Py_BuildValue("OOi", d_obj, e_obj, info);
}

static PyObject* py_ma02sd(PyObject* Py_UNUSED(self), PyObject* args) {
    int m, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    f64 result = ma02sd(m, n, a, lda);

    return Py_BuildValue("d", result);
}
