#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "slicutlet.h"

// Define guard to prevent extension files from re-including headers
#define SLICUTLET_EXTENSION_INCLUDED
#include "extensions/extension_ab.c"
#include "extensions/extension_ma.c"
#include "extensions/extension_mb.c"
#include "extensions/extension_mc.c"
#undef SLICUTLET_EXTENSION_INCLUDED


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
    {"py_ma02ad", py_ma02ad, METH_VARARGS, "Wrapper for ma02ad"},
    {"py_ma02az", py_ma02az, METH_VARARGS, "Wrapper for ma02az"},
    {"py_ma02bd", py_ma02bd, METH_VARARGS, "Wrapper for ma02bd"},
    {"py_ma02bz", py_ma02bz, METH_VARARGS, "Wrapper for ma02bz"},
    {"py_ma02cd", py_ma02cd, METH_VARARGS, "Wrapper for ma02cd"},
    {"py_ma02cz", py_ma02cz, METH_VARARGS, "Wrapper for ma02cz"},
    {"py_ma02dd", py_ma02dd, METH_VARARGS, "Wrapper for ma02dd"},
    {"py_ma02ed", py_ma02ed, METH_VARARGS, "Wrapper for ma02ed"},
    {"py_ma02es", py_ma02es, METH_VARARGS, "Wrapper for ma02es"},
    {"py_ma02ez", py_ma02ez, METH_VARARGS, "Wrapper for ma02ez"},
    {"py_ma02fd", py_ma02fd, METH_VARARGS, "Wrapper for ma02fd"},
    {"py_ma02gd", py_ma02gd, METH_VARARGS, "Wrapper for ma02gd"},
    {"py_ma02gz", py_ma02gz, METH_VARARGS, "Wrapper for ma02gz"},
    {"py_ma02hd", py_ma02hd, METH_VARARGS, "Wrapper for ma02hd"},
    {"py_ma02hz", py_ma02hz, METH_VARARGS, "Wrapper for ma02hz"},
    {"py_ma02id", py_ma02id, METH_VARARGS, "Wrapper for ma02id"},
    {"py_ma02iz", py_ma02iz, METH_VARARGS, "Wrapper for ma02iz"},
    {"py_ma02jd", py_ma02jd, METH_VARARGS, "Wrapper for ma02jd"},
    {"py_ma02jz", py_ma02jz, METH_VARARGS, "Wrapper for ma02jz"},
    {"py_ma02md", py_ma02md, METH_VARARGS, "Wrapper for ma02md"},
    {"py_ma02mz", py_ma02mz, METH_VARARGS, "Wrapper for ma02mz"},
    {"py_ma02nz", py_ma02nz, METH_VARARGS, "Wrapper for ma02nz"},
    {"py_ma02od", py_ma02od, METH_VARARGS, "Wrapper for ma02od"},
    {"py_ma02oz", py_ma02oz, METH_VARARGS, "Wrapper for ma02oz"},
    {"py_ma02pd", py_ma02pd, METH_VARARGS, "Wrapper for ma02pd"},
    {"py_ma02pz", py_ma02pz, METH_VARARGS, "Wrapper for ma02pz"},
    {"py_ma02rd", py_ma02rd, METH_VARARGS, "Wrapper for ma02rd"},
    {"py_ma02sd", py_ma02sd, METH_VARARGS, "Wrapper for ma02sd"},
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
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    // signal that this module can be imported in isolated subinterpreters
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
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

PyMODINIT_FUNC PyInit_slicutletlib(void) {
    return PyModuleDef_Init(&pyslicutlet_module);
}
