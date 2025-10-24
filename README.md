# SLICUTLET
A contemporary C11 translation of the excellent but venerable Systems and Control library SLICOT written in Fortran77. The original Fortran code can be found in https://github.com/SLICOT/SLICOT-Reference.

The main goal of this project is to ease the pain of including SLICOT into projects using other programming languages due to difficulties of binding to legacy Fortran ABIs, in particular, the current state of BLAS/LAPACK libraries and their wildly differing conventions and symbol definitions.

The code requires a C-standard compliant C11 compiler. Hence MSVC is not supported due to lack of conformance to C standard regarding `complex.h`. Its verbose workarounds are not feasible to provide maintainable code. Windows users can use Clang or any other conforming compilers instead.
