#!/bin/bash
set -e

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    ninja-build \
    pkg-config

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install meson ninja numpy Cython pytest pytest-cov ruff

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Cleaning any existing build directory..."
rm -rf build

echo "Setting up build directory..."
meson setup build -Dpython=true

echo "Building project..."
meson compile -C build

echo "Installing to local directory..."
meson install -C build --destdir="$(pwd)/build-install"

echo "Dev container ready!"
echo "PYTHONPATH is set to: /workspace/build-install/usr/local/lib/python3.11/site-packages"
