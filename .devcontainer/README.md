# SLICUTLET Dev Container

Pre-configured development environment with all dependencies.

## Features

- Python 3.11 with numpy, Cython, pytest
- Meson + Ninja build system
- OpenBLAS + LAPACK
- VS Code extensions (Python, C/C++, Meson, Ruff)

## Usage

1. Open in VS Code: "Reopen in Container"
2. Build: `meson compile -C build`
3. Test: `PYTHONPATH=build/python pytest -v python/tests/`

Container auto-configures on first launch.
