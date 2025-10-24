# SLICUTLET Python Bindings

Python interface to the SLICUTLET library (SLICOT C translations).

## Installation

### Build from source

```bash
# Activate your conda environment
conda activate scipy-dev

# Configure with Python bindings enabled
meson setup build -Dpython=true

# Build
meson compile -C build

# Install (optional)
pip install -e .
```

### Development installation

For development with editable install:

```bash
pip install -e . --no-build-isolation
```

## Usage

```python
import numpy as np
from slicutlet import <function_name>

# Example usage - refer to function docstrings for details
result = <function_name>(*args)
```

## Testing

Run tests with pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov hypothesis

# Run tests
pytest python/tests -v

# With coverage
pytest python/tests --cov=slicutlet --cov-report=html
```

## Documentation

Function documentation is available via docstrings:

```python
import slicutlet
help(slicutlet.<function_name>)
```
