py = import('python').find_installation()

# Get numpy include path as relative path (works with in-tree venv)
numpy_include_result = run_command(
    py,
    ['-c', 'import os; import numpy; print(os.path.relpath(numpy.get_include()))'],
    check: true
)
incdir_numpy = numpy_include_result.stdout().strip()

slicutletlib_module = py.extension_module(
    'slicutletlib',
    'slicutletmodule.c',
    include_directories: [inc_pub, inc_root, inc_src, include_directories(incdir_numpy)],
    link_with: slicutlet_lib,
    dependencies: deps_common,
    install: true,
    subdir: 'slicutlet',
    install_rpath: '$ORIGIN/../../../../lib64:$ORIGIN/../../../../lib:$ORIGIN/../..',
    build_rpath: meson.current_build_dir() / '..',
)

py.install_sources(
    'slicutlet/__init__.py',
    subdir: 'slicutlet'
)
