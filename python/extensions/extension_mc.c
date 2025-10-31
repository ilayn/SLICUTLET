// MC family Python extension functions
// This file is included by slicutletmodule.c - do not compile separately

#ifndef SLICUTLET_EXTENSION_INCLUDED
// Headers to silence IDEs - These are included already in slicutletmodule.c during compilation
#include <Python.h>
#include <numpy/arrayobject.h>
#include "slicutlet.h"
#endif

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
