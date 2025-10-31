// MB family Python extension functions
// This file is included by slicutletmodule.c - do not compile separately

#ifndef SLICUTLET_EXTENSION_INCLUDED
// Headers to silence IDEs - These are included already in slicutletmodule.c during compilation
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slicutlet.h"
#endif

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
