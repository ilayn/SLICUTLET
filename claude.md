# Quick Start

## Build
```bash
meson setup build -Dpython=true
meson compile -C build
meson install -C build --destdir="$(pwd)/build-install"
```

## Test
```bash
PYTHONPATH=build-install/usr/local/lib/python3.11/site-packages pytest python/tests/
```
