# Meson build file for Python extension module

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Check for NumPy
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# Cythonize the extension
cython = find_program('cython', required: true)

slicutlet_pyx = custom_target('_slicutlet_pyx',
  input: 'slicutlet/_slicutlet.pyx',
  output: '_slicutlet.c',
  command: [cython, '@INPUT@', '-o', '@OUTPUT@', '-3'],
)

# Build the extension module
py.extension_module('_slicutlet',
  slicutlet_pyx,
  include_directories: [inc_np, inc_src, inc_pub, inc_pub_root],
  link_with: slicutlet_lib,
  dependencies: [py_dep, deps_common],
  install: true,
  subdir: 'slicutlet',
)

# Install pure Python files
py.install_sources(
  'slicutlet/__init__.py',
  subdir: 'slicutlet',
)
