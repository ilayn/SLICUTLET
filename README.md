# SLICUTLET
A contemporary C11 translation of the excellent but venerable Systems and Control library SLICOT written in Fortran77. The original Fortran code can be found in https://github.com/SLICOT/SLICOT-Reference.

The main goal of this project is to ease the pain of including SLICOT into projects using other programming languages due to difficulties of binding to legacy Fortran ABIs, in particular, the current state of BLAS/LAPACK libraries and their wildly differing conventions and symbol definitions.

The code requires a C-standard compliant C11 compiler. Hence MSVC is not supported due to lack of conformance to C standard regarding `complex.h`. Its verbose workarounds are not feasible to provide maintainable code. Windows users can use Clang or any other conforming compilers instead.

## Building & Installing

In the root directory of the folder, run

```
meson setup build -Dpython=true
meson compile -C build
meson install -C build --destdir="$(pwd)/build-install"
```

or whichever destination folder you would like to install it to. Then meson will take care of the rest or complain about what issues it has encountered. If you don't need the Python extension module, you can omit `-Dpython` flag or set it to `false` explicitly.

Python extension module also has a test suite that is used to validate the translations and the module itself. Notice we installed it in a local folder in the example above and that's why we need to add it in the Python path for the module to be found. Then pytest works as usual, say, for a single test file, run at the project root

```
PYTHONPATH=build-install/usr/local/lib/python3.13/site-packages pytest -v python/tests/test_mb01xx.py
```

If you install it somewhere that is already on the path, then obviously you don't need the path setting.
