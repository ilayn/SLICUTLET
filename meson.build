project('SLICUTLET', 'c',
  version: '0.0.1',
  license : 'BSD-3',
  default_options: ['c_std=c11', 'warning_level=2']
)

# User-selectable option is declared in meson_options.txt: 'ilp64'

cc = meson.get_compiler('c')
inc_src = include_directories('src/include')
inc_pub = include_directories('include/SLICUTLET')
inc_pub_root = include_directories('include')
dep_m = cc.find_library('m', required: false)

# Verify complex ABI early: c128 must be two f64s (COMPLEX*16 layout)
code_c128_check = '''
  #include "types.h"
  _Static_assert(sizeof(c128) == 2*sizeof(f64), "c128 must be two f64s");
  int main(void){return 0;}
'''
if not cc.compiles(code_c128_check, name: 'Check complex layout: c128 == 2*f64', include_directories: inc_src)
  error('C complex ABI mismatch: c128 must be two f64s (Fortran COMPLEX*16). Try a different compiler or rebuild LAPACK/BLAS toolchain.')
endif

# Try to discover BLAS and LAPACK via pkg-config; fall back to CMake packages.
dep_blas = dependency('blas', required: false)
if not dep_blas.found()
  dep_blas = dependency('openblas', required: false)
endif

dep_lapack = dependency('lapack', required: false)

if (not dep_blas.found()) and (not dep_lapack.found())
  dep_blas = dependency('BLAS', method: 'cmake', required: false)
  dep_lapack = dependency('LAPACK', method: 'cmake', required: false)
endif

if (not dep_blas.found()) and (not dep_lapack.found())
  error('Neither BLAS nor LAPACK was found. Provide them via pkg-config or CMake, or set up custom link args.')
endif

# Detect Fortran symbol name mangling style by attempting to link against a known routine.
conf = configuration_data()
conf.set10('SLC_ILP64', get_option('ilp64'))

mangle_detected = false

if dep_blas.found()
  # Prefer detecting against BLAS symbol dgemm
  code_lower_us = '''
    extern void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
    void test(void){ (void)dgemm_; }
    int main(void){ return 0; }
  '''
  code_lower = '''
    extern void dgemm(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
    void test(void){ (void)dgemm; }
    int main(void){ return 0; }
  '''
  code_upper = '''
    extern void DGEMM(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
    void test(void){ (void)DGEMM; }
    int main(void){ return 0; }
  '''

  if cc.links(code_lower_us, dependencies: [dep_blas], name: 'Check BLAS symbol: dgemm_')
    conf.set10('SLC_FC_LOWER_US', true)
    mangle_detected = true
  elif cc.links(code_lower, dependencies: [dep_blas], name: 'Check BLAS symbol: dgemm')
    conf.set10('SLC_FC_LOWER', true)
    mangle_detected = true
  elif cc.links(code_upper, dependencies: [dep_blas], name: 'Check BLAS symbol: DGEMM')
    conf.set10('SLC_FC_UPPER', true)
    mangle_detected = true
  endif
else
  # Fall back to detecting against LAPACK symbol dgetrf
  code_lower_us = '''
    extern void dgetrf_(int*,int*,double*,int*,int*,int*);
    void test(void){ (void)dgetrf_; }
    int main(void){ return 0; }
  '''
  code_lower = '''
    extern void dgetrf(int*,int*,double*,int*,int*,int*);
    void test(void){ (void)dgetrf; }
    int main(void){ return 0; }
  '''
  code_upper = '''
    extern void DGETRF(int*,int*,double*,int*,int*,int*);
    void test(void){ (void)DGETRF; }
    int main(void){ return 0; }
  '''

  if cc.links(code_lower_us, dependencies: [dep_lapack], name: 'Check LAPACK symbol: dgetrf_')
    conf.set10('SLC_FC_LOWER_US', true)
    mangle_detected = true
  elif cc.links(code_lower, dependencies: [dep_lapack], name: 'Check LAPACK symbol: dgetrf')
    conf.set10('SLC_FC_LOWER', true)
    mangle_detected = true
  elif cc.links(code_upper, dependencies: [dep_lapack], name: 'Check LAPACK symbol: DGETRF')
    conf.set10('SLC_FC_UPPER', true)
    mangle_detected = true
  endif
endif

if not mangle_detected
  warning('Could not detect Fortran symbol naming; defaulting to lower+underscore (e.g., dgemm_).')
  conf.set10('SLC_FC_LOWER_US', true)
endif

# Determine if LAPACK is provided via the BLAS dependency (e.g., OpenBLAS ships LAPACK)
lapack_via_blas = false
have_lapack = dep_lapack.found()
if (not have_lapack) and dep_blas.found()
  code_dgetrf_us = '''
    extern void dgetrf_(int*,int*,double*,int*,int*,int*);
    void test(void){ (void)dgetrf_; }
    int main(void){ return 0; }
  '''
  if cc.links(code_dgetrf_us, dependencies: [dep_blas], name: 'Probe LAPACK via BLAS dep (dgetrf_)')
    have_lapack = true
    lapack_via_blas = true
  endif
endif

# Generate a private config header with detected options. This header lives in the build dir.
# Generate a private config header with detected options. This lands in the build root.
cfg_h = configure_file(
  input: 'src/include/slc_config.h.in',
  output: 'slc_config.h',
  configuration: conf
)

# Stash found dependencies for later subdir use (not creating targets yet).
deps_blaslapack = []
if dep_blas.found()
  deps_blaslapack += dep_blas
endif
if dep_lapack.found()
  deps_blaslapack += dep_lapack
endif

message('SLICUTLET configuration:')
message('  ILP64:        ' + (get_option('ilp64') ? 'true' : 'false'))
message('  BLAS found:   ' + (dep_blas.found() ? 'true' : 'false'))
lapack_msg = 'false'
if have_lapack
  lapack_msg = 'true' + (lapack_via_blas and dep_blas.found() and (not dep_lapack.found()) ? ' (provided by BLAS dep)' : '')
endif
message('  LAPACK found: ' + lapack_msg)

# Build the core library from the translated C sources
lib_srcs = files(
  'src/A/ab04md.c',
  'src/A/ab05md.c',
  'src/A/ab05nd.c',
  'src/A/ab07nd.c',
  'src/M/mb03oy.c',
  'src/M/mc01td.c',
)

inc_root = include_directories('.')

deps_common = deps_blaslapack
if dep_m.found()
  deps_common += dep_m
endif

slicutlet_lib = library(
  'slicutlet',
  lib_srcs,
  include_directories: [inc_src, inc_pub, inc_root],
  dependencies: deps_common,
  version: meson.project_version(),
  install: true,
)

# Install public headers
install_headers('include/SLICUTLET/slicutlet.h', subdir: 'SLICUTLET')
install_headers('src/include/types.h',            subdir: 'SLICUTLET')

# Generate a pkg-config file for downstreams
pkgc = import('pkgconfig')
pkgc.generate(
  libraries: slicutlet_lib,
  version: meson.project_version(),
  name: 'slicutlet',
  description: 'SLICOT C translations (early preview)',
  filebase: 'slicutlet',
  subdirs: ['SLICUTLET'],
  requires: deps_blaslapack,
  libraries_private: dep_m.found() ? [dep_m] : [],
)

# Minimal smoke test executable
test_exe = executable(
  'test_slicutlet',
  'src/B/test_slicutlet.c',
  include_directories: [inc_pub, inc_src, inc_root],
  link_with: slicutlet_lib,
  dependencies: deps_common,
)

test('smoke', test_exe)

# Python bindings (optional)
if get_option('python')
  subdir('python')
  run_target(
    'pytest-dev',
    command: [find_program('python3'), meson.current_source_dir() / 'python' / 'tools' / 'run_pytests.py'],
    depends: slicutlet_lib
  )
endif
