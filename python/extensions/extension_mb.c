// MB family Python extension functions
// This file is included by slicutletmodule.c - do not compile separately

#ifndef SLICUTLET_EXTENSION_INCLUDED
    /**
     * Headers to silence IDEs.
     * These are included already in slicutletmodule.c during compilation
     */
    #include <Python.h>
    #include <numpy/arrayobject.h>
    #include "slicutlet.h"
#endif

// This is out of alphabetical order on purpose. Add new functions at the end.
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

// MB01XX family wrappers (alphabetical order, kd-zd)

static PyObject* py_mb01oc(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *x_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&x_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);

    i32 info;
    mb01oc(uplo, trans, n, alpha, beta, r, ldr, h, ldh, x, ldx, &info);

    return Py_BuildValue("OOi", r_obj, x_obj, info);
}

static PyObject* py_mb01oe(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *e_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&e_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *e_dims = PyArray_DIMS(e_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 lde = (i32)(e_dims[0] > 0 ? e_dims[0] : 1);

    mb01oe(uplo, trans, n, alpha, beta, r, ldr, h, ldh, e, lde);

    return Py_BuildValue("OO", r_obj, e_obj);
}

static PyObject* py_mb01oh(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *a_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    mb01oh(uplo, trans, n, alpha, beta, r, ldr, h, ldh, a, lda);

    return Py_BuildValue("OO", r_obj, a_obj);
}

static PyObject* py_mb01oo(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    PyArrayObject *h_obj, *x_obj, *e_obj, *p_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!O!",
                          &uplo, &trans, &n,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&e_obj,
                          &PyArray_Type, (PyObject **)&p_obj)) {
        return NULL;
    }

    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);
    f64 *p = (f64*)PyArray_DATA(p_obj);

    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *e_dims = PyArray_DIMS(e_obj);
    npy_intp *p_dims = PyArray_DIMS(p_obj);

    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 lde = (i32)(e_dims[0] > 0 ? e_dims[0] : 1);
    i32 ldp = (i32)(p_dims[0] > 0 ? p_dims[0] : 1);

    i32 info;
    mb01oo(uplo, trans, n, h, ldh, x, ldx, e, lde, p, ldp, &info);

    return Py_BuildValue("Oi", p_obj, info);
}

static PyObject* py_mb01os(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    PyArrayObject *h_obj, *x_obj, *p_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &uplo, &trans, &n,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&p_obj)) {
        return NULL;
    }

    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *p = (f64*)PyArray_DATA(p_obj);

    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *p_dims = PyArray_DIMS(p_obj);

    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldp = (i32)(p_dims[0] > 0 ? p_dims[0] : 1);

    i32 info;
    mb01os(uplo, trans, n, h, ldh, x, ldx, p, ldp, &info);

    return Py_BuildValue("Oi", p_obj, info);
}

static PyObject* py_mb01ot(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *e_obj, *t_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&e_obj,
                          &PyArray_Type, (PyObject **)&t_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);
    f64 *t = (f64*)PyArray_DATA(t_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *e_dims = PyArray_DIMS(e_obj);
    npy_intp *t_dims = PyArray_DIMS(t_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lde = (i32)(e_dims[0] > 0 ? e_dims[0] : 1);
    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);

    i32 info;
    mb01ot(uplo, trans, n, alpha, beta, r, ldr, e, lde, t, ldt, &info);

    return Py_BuildValue("Oi", r_obj, info);
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

static PyObject* py_mb01rb(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m, n;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddO!O!O!",
                          &side, &uplo, &trans, &m, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&b_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    i32 info;
    mb01rb(side, uplo, trans, m, n, alpha, beta, r, ldr, a, lda, b, ldb, &info);

    return Py_BuildValue("Oi", r_obj, info);
}

static PyObject* py_mb01rx(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m, n;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddO!O!O!",
                          &side, &uplo, &trans, &m, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&b_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    i32 info;
    mb01rx(side, uplo, trans, m, n, alpha, beta, r, ldr, a, lda, b, ldb, &info);

    return Py_BuildValue("Oi", r_obj, info);
}

static PyObject* py_mb01sd(PyObject* Py_UNUSED(self), PyObject* args) {
    int jobs, m, n;
    PyArrayObject *a_obj, *r_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!O!",
                          &jobs, &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&c_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *c = (f64*)PyArray_DATA(c_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    mb01sd(jobs, m, n, a, lda, r, c);

    return Py_BuildValue("O", a_obj);
}

static PyObject* py_mb01ss(PyObject* Py_UNUSED(self), PyObject* args) {
    int jobs, uplo, n;
    PyArrayObject *a_obj, *d_obj;

    if (!PyArg_ParseTuple(args, "iiiO!O!",
                          &jobs, &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&d_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *d = (f64*)PyArray_DATA(d_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    mb01ss(jobs, uplo, n, a, lda, d);

    return Py_BuildValue("O", a_obj);
}

static PyObject* py_mb01td(PyObject* Py_UNUSED(self), PyObject* args) {
    int n;
    PyArrayObject *a_obj, *b_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iO!O!O!",
                          &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&b_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);

    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    i32 info;
    mb01td(n, a, lda, b, ldb, dwork, &info);

    return Py_BuildValue("Oi", b_obj, info);
}

static PyObject* py_mb01ud(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, trans, m, n;
    double alpha;
    PyArrayObject *h_obj, *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "iiiidO!O!O!",
                          &side, &trans, &m, &n, &alpha,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&b_obj)) {
        return NULL;
    }

    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);

    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);

    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    i32 info;
    mb01ud(side, trans, m, n, alpha, h, ldh, a, lda, b, ldb, &info);

    return Py_BuildValue("Oi", b_obj, info);
}

static PyObject* py_mb01uw(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, trans, m, n;
    double alpha;
    PyArrayObject *h_obj, *a_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiidO!O!O!",
                          &side, &trans, &m, &n, &alpha,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01uw(side, trans, m, n, alpha, h, ldh, a, lda, dwork, ldwork, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01ux(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m, n;
    double alpha;
    PyArrayObject *t_obj, *a_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiidO!O!O!",
                          &side, &uplo, &trans, &m, &n, &alpha,
                          &PyArray_Type, (PyObject **)&t_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *t = (f64*)PyArray_DATA(t_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *t_dims = PyArray_DIMS(t_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01ux(side, uplo, trans, m, n, alpha, t, ldt, a, lda, dwork, ldwork, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01uy(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m, n;
    double alpha;
    PyArrayObject *t_obj, *a_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiidO!O!O!",
                          &side, &uplo, &trans, &m, &n, &alpha,
                          &PyArray_Type, (PyObject **)&t_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *t = (f64*)PyArray_DATA(t_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *t_dims = PyArray_DIMS(t_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01uy(side, uplo, trans, m, n, alpha, t, ldt, a, lda, dwork, ldwork, &info);

    return Py_BuildValue("Oi", t_obj, info);
}

static PyObject* py_mb01uz(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m, n;
    Py_complex alpha_py;
    PyArrayObject *t_obj, *a_obj, *zwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiiDO!O!O!",
                          &side, &uplo, &trans, &m, &n, &alpha_py,
                          &PyArray_Type, (PyObject **)&t_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&zwork_obj)) {
        return NULL;
    }

    c128 alpha = alpha_py.real + alpha_py.imag * I;
    c128 *t = (c128*)PyArray_DATA(t_obj);
    c128 *a = (c128*)PyArray_DATA(a_obj);
    c128 *zwork = (c128*)PyArray_DATA(zwork_obj);

    npy_intp *t_dims = PyArray_DIMS(t_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *zwork_dims = PyArray_DIMS(zwork_obj);

    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 lzwork = (i32)(zwork_dims[0] > 0 ? zwork_dims[0] : 1);

    i32 info;
    mb01uz(side, uplo, trans, m, n, alpha, t, ldt, a, lda, zwork, lzwork, &info);

    return Py_BuildValue("Oi", t_obj, info);
}

static PyObject* py_mb01xy(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 info;
    mb01xy(uplo, n, a, lda, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01ld(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, m, n, k;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *x_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddO!O!O!O!",
                          &uplo, &trans, &m, &n, &k, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01ld(uplo, trans, m, n, k, alpha, beta, r, ldr, a, lda, x, ldx, dwork, ldwork, &info);

    return Py_BuildValue("OOi", r_obj, x_obj, info);
}

static PyObject* py_mb01od(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *x_obj, *e_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&e_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *e_dims = PyArray_DIMS(e_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 lde = (i32)(e_dims[0] > 0 ? e_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01od(uplo, trans, n, alpha, beta, r, ldr, h, ldh, x, ldx, e, lde, dwork, ldwork, &info);

    return Py_BuildValue("OOOi", r_obj, h_obj, x_obj, info);
}

static PyObject* py_mb01rd(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, m, n;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *x_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiddO!O!O!O!",
                          &uplo, &trans, &m, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01rd(uplo, trans, m, n, alpha, beta, r, ldr, a, lda, x, ldx, dwork, ldwork, &info);

    return Py_BuildValue("OOi", r_obj, x_obj, info);
}

static PyObject* py_mb01rh(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *x_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01rh(uplo, trans, n, alpha, beta, r, ldr, h, ldh, x, ldx, dwork, ldwork, &info);

    return Py_BuildValue("OOOi", r_obj, h_obj, x_obj, info);
}

static PyObject* py_mb01rt(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n;
    double alpha, beta;
    PyArrayObject *r_obj, *e_obj, *x_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiddO!O!O!O!",
                          &uplo, &trans, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&e_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *e = (f64*)PyArray_DATA(e_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *e_dims = PyArray_DIMS(e_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lde = (i32)(e_dims[0] > 0 ? e_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01rt(uplo, trans, n, alpha, beta, r, ldr, e, lde, x, ldx, dwork, ldwork, &info);

    return Py_BuildValue("OOi", r_obj, x_obj, info);
}

static PyObject* py_mb01ru(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, m, n;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *x_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiddO!O!O!O!",
                          &uplo, &trans, &m, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&x_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *x = (f64*)PyArray_DATA(x_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *x_dims = PyArray_DIMS(x_obj);
    npy_intp *dwork_dims = PyArray_DIMS(dwork_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldx = (i32)(x_dims[0] > 0 ? x_dims[0] : 1);
    i32 ldwork = (i32)(dwork_dims[0] > 0 ? dwork_dims[0] : 1);

    i32 info;
    mb01ru(uplo, trans, m, n, alpha, beta, r, ldr, a, lda, x, ldx, dwork, ldwork, &info);

    return Py_BuildValue("OOi", r_obj, x_obj, info);
}

static PyObject* py_mb01rw(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, m, n;
    PyArrayObject *a_obj, *z_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiO!O!O!",
                          &uplo, &trans, &m, &n,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&z_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *z = (f64*)PyArray_DATA(z_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *z_dims = PyArray_DIMS(z_obj);

    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldz = (i32)(z_dims[0] > 0 ? z_dims[0] : 1);

    i32 info;
    mb01rw(uplo, trans, m, n, a, lda, z, ldz, dwork, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01ry(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, trans, m;
    double alpha, beta;
    PyArrayObject *r_obj, *h_obj, *b_obj, *dwork_obj;

    if (!PyArg_ParseTuple(args, "iiiiddO!O!O!O!",
                          &side, &uplo, &trans, &m, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&h_obj,
                          &PyArray_Type, (PyObject **)&b_obj,
                          &PyArray_Type, (PyObject **)&dwork_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);
    f64 *dwork = (f64*)PyArray_DATA(dwork_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);

    i32 info;
    mb01ry(side, uplo, trans, m, alpha, beta, r, ldr, h, ldh, b, ldb, dwork, &info);

    return Py_BuildValue("OOi", r_obj, h_obj, info);
}

static PyObject* py_mb01vd(PyObject* Py_UNUSED(self), PyObject* args) {
    int trana, tranb, ma, na, mb, nb;
    double alpha, beta;
    PyArrayObject *a_obj, *b_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "iiiiiiddO!O!O!",
                          &trana, &tranb, &ma, &na, &mb, &nb, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&b_obj,
                          &PyArray_Type, (PyObject **)&c_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *b = (f64*)PyArray_DATA(b_obj);
    f64 *c = (f64*)PyArray_DATA(c_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *b_dims = PyArray_DIMS(b_obj);
    npy_intp *c_dims = PyArray_DIMS(c_obj);

    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldb = (i32)(b_dims[0] > 0 ? b_dims[0] : 1);
    i32 ldc = (i32)(c_dims[0] > 0 ? c_dims[0] : 1);

    i32 mc, nc, info;
    mb01vd(trana, tranb, ma, na, mb, nb, alpha, beta, a, lda, b, ldb, c, ldc, &mc, &nc, &info);

    return Py_BuildValue("Oiii", c_obj, mc, nc, info);
}

static PyObject* py_mb01wd(PyObject* Py_UNUSED(self), PyObject* args) {
    int dico, uplo, trans, hess, n;
    double alpha, beta;
    PyArrayObject *r_obj, *a_obj, *t_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddO!O!O!",
                          &dico, &uplo, &trans, &hess, &n, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&r_obj,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&t_obj)) {
        return NULL;
    }

    f64 *r = (f64*)PyArray_DATA(r_obj);
    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *t = (f64*)PyArray_DATA(t_obj);

    npy_intp *r_dims = PyArray_DIMS(r_obj);
    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *t_dims = PyArray_DIMS(t_obj);

    i32 ldr = (i32)(r_dims[0] > 0 ? r_dims[0] : 1);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);

    i32 info;
    mb01wd(dico, uplo, trans, hess, n, alpha, beta, r, ldr, a, lda, t, ldt, &info);

    return Py_BuildValue("OOi", r_obj, a_obj, info);
}

static PyObject* py_mb01xd(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, n;
    PyArrayObject *a_obj;

    if (!PyArg_ParseTuple(args, "iiO!",
                          &uplo, &n,
                          &PyArray_Type, (PyObject **)&a_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);

    i32 info;
    mb01xd(uplo, n, a, lda, &info);

    return Py_BuildValue("Oi", a_obj, info);
}

static PyObject* py_mb01yd(PyObject* Py_UNUSED(self), PyObject* args) {
    int uplo, trans, n, k, l;
    double alpha, beta;
    PyArrayObject *a_obj, *c_obj;

    if (!PyArg_ParseTuple(args, "iiiiiddO!O!",
                          &uplo, &trans, &n, &k, &l, &alpha, &beta,
                          &PyArray_Type, (PyObject **)&a_obj,
                          &PyArray_Type, (PyObject **)&c_obj)) {
        return NULL;
    }

    f64 *a = (f64*)PyArray_DATA(a_obj);
    f64 *c = (f64*)PyArray_DATA(c_obj);

    npy_intp *a_dims = PyArray_DIMS(a_obj);
    npy_intp *c_dims = PyArray_DIMS(c_obj);

    i32 lda = (i32)(a_dims[0] > 0 ? a_dims[0] : 1);
    i32 ldc = (i32)(c_dims[0] > 0 ? c_dims[0] : 1);

    i32 info;
    mb01yd(uplo, trans, n, k, l, alpha, beta, a, lda, c, ldc, &info);

    return Py_BuildValue("Oi", c_obj, info);
}

static PyObject* py_mb01zd(PyObject* Py_UNUSED(self), PyObject* args) {
    int side, uplo, transt, diag, m, n, l;
    double alpha;
    PyArrayObject *t_obj, *h_obj;

    if (!PyArg_ParseTuple(args, "iiiiiidO!O!",
                          &side, &uplo, &transt, &diag, &m, &n, &l, &alpha,
                          &PyArray_Type, (PyObject **)&t_obj,
                          &PyArray_Type, (PyObject **)&h_obj)) {
        return NULL;
    }

    f64 *t = (f64*)PyArray_DATA(t_obj);
    f64 *h = (f64*)PyArray_DATA(h_obj);

    npy_intp *t_dims = PyArray_DIMS(t_obj);
    npy_intp *h_dims = PyArray_DIMS(h_obj);

    i32 ldt = (i32)(t_dims[0] > 0 ? t_dims[0] : 1);
    i32 ldh = (i32)(h_dims[0] > 0 ? h_dims[0] : 1);

    i32 info;
    mb01zd(side, uplo, transt, diag, m, n, l, alpha, t, ldt, h, ldh, &info);

    return Py_BuildValue("Oi", h_obj, info);
}
