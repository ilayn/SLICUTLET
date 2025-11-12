# Parsing SLICOT Test Data from HTML Documentation

## Overview

SLICOT HTML documentation includes example programs with test data in the **"Program Data"** and **"Program Results"** sections. Understanding how to parse this data correctly is crucial for creating accurate test cases.

## Key Challenge: Fortran vs. Mathematical Notation

- **Fortran storage**: Column-major order (columns contiguous in memory)
- **Mathematical notation**: Row-major order (rows written sequentially)
- **HTML presentation**: Often appears row-wise for human readability
- **Fortran READ statements**: Always read column-wise

## Critical Rule

**ALWAYS check the Fortran example code's READ statement** to understand the true data order. The HTML presentation may be misleading.

## Common Fortran READ Pattern

```fortran
! Reading matrix A (M rows, N columns) COLUMN-WISE
READ ( NIN, FMT = * ) ( ( A(I,J), I = 1,M ), J = 1,N )
```

This reads:
1. First M values → Column 1 of A
2. Next M values → Column 2 of A
3. Continue for all N columns

## Parsing Strategies

### Strategy 1: Row-Wise Presentation (Common)

When data appears as complete rows in HTML:

```
Program Data:
  1.0   2.0   3.0  --> Row 1
  4.0   5.0   6.0  --> Row 2
```

**Action**: Read sequentially, fill row-by-row (row-major). This is the straightforward mathematical interpretation.

### Strategy 2: Column-Wise Data (Fortran Native)

When Fortran reads column-wise but HTML shows numbers sequentially:

```fortran
READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M )
```

For N=3, M=2, sequence `1.0, 2.0, 3.0, 4.0, 5.0, 6.0` means:
- Column 1: [1.0, 2.0, 3.0]
- Column 2: [4.0, 5.0, 6.0]

**Row-major equivalent** (after transpose):
```
B = [ 1.0   4.0 ]
    [ 2.0   5.0 ]
    [ 3.0   6.0 ]
```

## Detailed Examples

### Example 1: AB01MD

**HTML Data**:
```
AB01MD EXAMPLE PROGRAM DATA
  3     0.0     I
  1.0   2.0   0.0
  4.0  -1.0   0.0
  0.0   0.0   1.0
  1.0   0.0   1.0
```

**Parameters**: N=3

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )
READ ( NIN, FMT = * ) ( B(I), I = 1,N )
```

**Parsing A (3×3)**:
- Read row-wise (Fortran reads with J outer loop, I inner = rows)
```
A = [  1.0   2.0   0.0 ]
    [  4.0  -1.0   0.0 ]
    [  0.0   0.0   1.0 ]
```

**Parsing B (3×1 vector)**:
```
B = [ 1.0 ]
    [ 0.0 ]
    [ 1.0 ]
```

**Program Results**:
```
The state dynamics matrix A of a controllable realization is
  1.0000   1.4142   0.0000
  2.8284  -1.0000   2.8284
  0.0000   1.4142   1.0000
```

Parse row-by-row:
```
A_result = [  1.0000   1.4142   0.0000 ]
           [  2.8284  -1.0000   2.8284 ]
           [  0.0000   1.4142   1.0000 ]
```

### Example 2: AB01ND

**HTML Data**:
```
AB01ND EXAMPLE PROGRAM DATA
  3     2     0.0     I
  -1.0   0.0   0.0
  -2.0  -2.0  -2.0
  -1.0   0.0  -3.0
   1.0   0.0   0.0
   0.0   2.0   1.0
```

**Parameters**: N=3, M=2

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )
READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M )
```

**Parsing A (3×3)**: Row-wise as shown
```
A = [ -1.0   0.0   0.0 ]
    [ -2.0  -2.0  -2.0 ]
    [ -1.0   0.0  -3.0 ]
```

**Parsing B (3×2)**: Column-wise read!
- Sequence: `1.0, 0.0, 0.0, 0.0, 2.0, 1.0`
- Column 1: [1.0, 0.0, 0.0]
- Column 2: [0.0, 2.0, 1.0]

```
B = [ 1.0   0.0 ]
    [ 0.0   2.0 ]
    [ 0.0   1.0 ]
```

### Example 3: TF01MD (Complex Case)

**HTML Data**:
```
TF01MD EXAMPLE PROGRAM DATA
  3     2     2     10
  0.0000 -0.0700  0.0150  --> A, Row 1
  1.0000  0.8000 -0.1500  --> A, Row 2
  0.0000  0.0000  0.5000  --> A, Row 3
  0.0000  2.0000  1.0000  --> B data (see note)
 -1.0000 -0.1000  1.0000
  0.0000  1.0000           --> C data (see note)
  1.0000  0.0000
  1.0000  0.5000           --> D, Row 1
  0.0000  0.5000           --> D, Row 2
  1.0000  1.0000  1.0000  --> X vector
```

**Parameters**: N=3, M=2, P=2, NY=10

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,N )
READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M )
READ ( NIN, FMT = * ) ( ( C(I,J), I = 1,P ), J = 1,N )
READ ( NIN, FMT = * ) ( ( D(I,J), I = 1,P ), J = 1,M )
```

**Parsing B (3×2)**: Column-wise!
- Sequence from HTML: `0.0000, -1.0000, 0.0000, 2.0000, -0.1000, 1.0000`
- Column 1: [0.0000, -1.0000, 0.0000]
- Column 2: [2.0000, -0.1000, 1.0000]

```
B = [  0.0000   2.0000 ]
    [ -1.0000  -0.1000 ]
    [  0.0000   1.0000 ]
```

**Parsing C (2×3)**: Column-wise!
- Sequence from HTML: `0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000`
- Column 1: [0.0000, 1.0000]
- Column 2: [0.0000, 0.0000]
- Column 3: [1.0000, 0.0000]

```
C = [ 0.0000   0.0000   1.0000 ]
    [ 1.0000   0.0000   0.0000 ]
```

**Parsing D (2×2)**: Column-wise!
- Sequence: `1.0000, 0.0000, 0.5000, 0.5000`
- Column 1: [1.0000, 0.0000]
- Column 2: [0.5000, 0.5000]

```
D = [ 1.0000   0.5000 ]
    [ 0.0000   0.5000 ]
```

### Example 4: Input/Output Time Series Data

**HTML Data**:
```
 -0.6922 -1.4934  0.3081 -2.7726  2.0039  --> U(1,1) to U(1,5)
  0.2614 -0.9160 -0.6030  1.2556  0.2951  --> U(2,1) to U(2,5)
 -1.5734  1.5639 -0.9942  1.8957  0.8988  --> U(1,6) to U(1,10)
  0.4118 -1.4893 -0.9344  1.2506 -0.0701  --> U(2,6) to U(2,10)
```

This shows M=2 inputs over NY=10 samples, presented row-wise.

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( U(I,J), J = 1,NY ), I = 1,M )
```

Reads:
- All 10 samples of input 1
- All 10 samples of input 2

**Parsing**: Read as shown (row = input channel, columns = time samples)

```
U = [ -0.6922  -1.4934   0.3081  -2.7726   2.0039  -1.5734   1.5639  -0.9942   1.8957   0.8988 ]
    [  0.2614  -0.9160  -0.6030   1.2556   0.2951   0.4118  -1.4893  -0.9344   1.2506  -0.0701 ]
```

## Best Practices

### 1. Always Check Fortran Source

The Fortran example code (`*.f` files in `examples/`) contains the definitive READ statements.

### 2. Understand the Implied DO Loop

```fortran
( ( ARRAY(I,J), I = 1,M ), J = 1,N )
```
- Inner loop (I): Varies fastest → reads down columns
- Outer loop (J): Varies slowest → advances across columns

### 3. Cross-Reference with Test Results

If your parsing produces incorrect results, verify against "Program Results" section.

### 4. Account for Presentation Truncation

HTML may show incomplete rows if data is wide. Check dimensions and Fortran code.

### 5. Use CSV for Large Datasets

For tests with many data points, create CSV files in `tests/data/` and use:

```cpp
load_test_data_from_csv(
    filepath,
    input_column_names,
    output_column_names,
    u_vector,
    y_vector,
    num_samples);
```

## Common Pitfalls

### Pitfall 1: Assuming Row-Major Everywhere

**Problem**: HTML presentation looks row-major, but Fortran reads column-major.

**Solution**: Check the Fortran READ loops carefully.

### Pitfall 2: Mismatched Dimensions

**Problem**: HTML shows 3 numbers on a line for a 3×2 matrix → confusion.

**Solution**: Count total numbers (should equal M×N) and use Fortran loop order.

### Pitfall 3: Ignoring Workspace Arrays

**Problem**: HTML may show DWORK or IWORK which aren't test inputs.

**Solution**: Only parse problem data (A, B, C, D, U, Y, etc.), skip workspace.

### Pitfall 4: Trusting HTML Formatting

**Problem**: HTML whitespace doesn't always indicate array structure.

**Solution**: Use dimension parameters (N, M, P) and Fortran code as ground truth.

## Summary Table

| Matrix | Fortran READ Pattern | Interpretation |
|--------|---------------------|----------------|
| `((A(I,J), J=1,N), I=1,M)` | Inner=J, Outer=I | Read row-wise (rare) |
| `((A(I,J), I=1,M), J=1,N)` | Inner=I, Outer=J | Read column-wise (common) |
| `(B(I), I=1,N)` | Single loop | Read vector sequentially |

## Quick Reference: Column-Major to Row-Major

If Fortran reads `B(3,2)` column-wise with sequence `[b11, b21, b31, b12, b22, b32]`:

**Column-major** (Fortran storage):
```
B_fortran = column1: [b11, b21, b31]
            column2: [b12, b22, b32]
```

**Row-major** (Mathematical representation):
```
B = [ b11  b12 ]
    [ b21  b22 ]
    [ b31  b32 ]
```

**Verification**: Element B(2,1) = b21 in both representations.
