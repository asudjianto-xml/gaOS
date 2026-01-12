# Track I: Foundations

Part of the [Geometric Algebra for Data Science](https://agussudjianto.substack.com/p/geometric-algebra-for-data-science) series.

## Chapters

### Chapter 1: The Great Embedding - Escaping the Scalar Trap ✅
**Article**: [The Great Embedding](https://agussudjianto.substack.com/p/the-great-embedding-escaping-the)

Implements a dimensionally-aware vector system using PyTorch with GPU acceleration.
Solves the "Scalar Trap" problem where multi-dimensional data is incorrectly treated as
raw scalar arrays.

### Chapter 2: Beyond the Arrow - The Wedge Product ✅
**Article**: Coming January 14, 2026

Implements the Wedge Product (∧) - a superior alternative to the cross product that works
in ANY dimension. Measures linear independence and creates bivectors (oriented areas).

## The Problem

Traditional approach (THE SCALAR TRAP):
```python
house = [2500, 3, 20]  # sqft, bedrooms, age
sum(house)  # = 2523 - MEANINGLESS!
```

Our solution (GEOMETRIC VECTORS):
```python
house = GeometricVector([2500, 3, 20], ['sqft', 'bedrooms', 'age'])
# Dimensions stay separate and meaningful!
```

## Files

### Chapter 1 Files
- **geometric_vector.py** - Main implementation
  - `GeometricVector` class: Dimensionally-aware vectors
  - `WedgeProduct` class: Captures vector relationships
  - `GeometricVectorBatch` class: GPU-accelerated batch processing

- **example_usage.py** - Quick start examples
  - Scalar trap problem demonstration
  - Law of Non-Interaction
  - All vector operations

- **geometric_vectors_demo.ipynb** - Interactive Jupyter notebook

### Chapter 2 Files
- **chapter_2_wedge_product.py** - Wedge product implementation
  - `wedge_product_tensor()`: Compute u ∧ v as antisymmetric outer product
  - `wedge_magnitude()`: Area of parallelogram
  - `independence_strength()`: Measure of linear independence
  - `BatchWedgeProduct`: GPU-accelerated batch processing

- **chapter_2_examples.py** - Practical demonstrations
  - 3D and 4D wedge products
  - Cross product limitations
  - Linear independence detection
  - Multicollinearity in features
  - High-dimensional embeddings (100D)

- **chapter_2_wedge_product_demo.ipynb** - Interactive Jupyter notebook
  - Cross product vs wedge product comparison
  - Visualizations of bivectors
  - Independence testing examples
  - GPU batch processing demo
  - Hands-on exercises

## Setup

Environment is pre-configured at:
```bash
~/jupyterlab/ga_verify/venv/bin
```

Activate the environment:
```bash
source ~/jupyterlab/ga_verify/venv/bin/activate
```

## Quick Start

### Run the Quick Example
```bash
python example_usage.py
```

### Run Full Demonstrations
```bash
python geometric_vector.py
```

### Use in Your Code
```python
from geometric_vector import GeometricVector, create_housing_vector

# Create vectors
house1 = create_housing_vector(sqft=2500, bedrooms=3, age=20)
house2 = create_housing_vector(sqft=3000, bedrooms=4, age=15)

# Safe operations - dimensions preserved
combined = house1 + house2

# Get specific components
total_sqft = combined.get_component('sqft')

# Geometric operations
magnitude = house1.magnitude()
similarity = house1.dot(house2)
relationship = house1 ^ house2  # Wedge product
```

## Key Features

### 1. Dimensional Safety
Prevents meaningless operations between incompatible spaces:
```python
house = GeometricVector([2500, 3, 20], ['sqft', 'bedrooms', 'age'])
car = GeometricVector([150, 4, 8], ['horsepower', 'wheels', 'cylinders'])

# This will raise ValueError - protected by Law of Non-Interaction
try:
    invalid = house + car
except ValueError:
    print("Cannot add vectors from different spaces!")
```

### 2. GPU Acceleration
All operations automatically use CUDA when available:
```python
# Automatically detects and uses GPU
device = get_device()

# Creates vector on GPU
vector = GeometricVector([1, 2, 3], ['x', 'y', 'z'], device)

# Batch processing on GPU
batch = GeometricVectorBatch([vector1, vector2, vector3, ...])
distances = batch.pairwise_distances()  # Computed in parallel on GPU
```

### 3. Rich Geometric Operations

- **Addition/Subtraction**: Preserves dimensional structure
- **Magnitude**: Euclidean norm
- **Normalization**: Unit vectors
- **Dot Product**: Similarity measure
- **Wedge Product**: Captures relationships without collapsing dimensions
- **Projection**: Project onto other vectors
- **Angle**: Compute angles between vectors

### 4. Wedge Product (Key Innovation)

The wedge product captures vector relationships while preserving dimensional information:

```python
v1 = GeometricVector([2500, 3, 20], ['sqft', 'bedrooms', 'age'])
v2 = GeometricVector([3000, 4, 15], ['sqft', 'bedrooms', 'age'])

wedge = v1 ^ v2  # Wedge product operator

# Measure correlation
correlation = wedge.correlation_strength()
# 0 = parallel (highly correlated)
# 1 = orthogonal (uncorrelated)

# Get magnitude (area of parallelogram)
area = wedge.magnitude()
```

Applications:
- Correlation analysis
- Multicollinearity detection
- Feature relationship understanding

### 5. Batch Processing

Efficient parallel operations on GPU:

```python
from geometric_vector import GeometricVectorBatch

vectors = [vector1, vector2, vector3, ...]
batch = GeometricVectorBatch(vectors)

# All computed in parallel on GPU
magnitudes = batch.magnitudes()
mean_vector = batch.mean()
distances = batch.pairwise_distances()
similarities = batch.gram_matrix()
```

## Core Concepts

### The Law of Non-Interaction

**Incompatible dimensions cannot be accidentally combined.**

Mathematical representation:
```
x = v1*e1 + v2*e2 + v3*e3

where:
  ei = orthogonal basis vectors (sqft, bedrooms, age)
  vi = scaling coefficients

(2500*e1) + (3*e2) != 2503
Components remain structurally separate!
```

### Basis Vectors

Each `GeometricVector` has a basis defining its dimensional space:
```python
house.basis  # ('sqft', 'bedrooms', 'age')
car.basis    # ('horsepower', 'wheels', 'cylinders')
```

Operations only work between vectors sharing the same basis.

## API Reference

### GeometricVector Class

**Constructor**:
```python
GeometricVector(values, basis_names, device=None)
```

**Properties**:
- `.values` - Tensor of coefficients
- `.basis` - Tuple of basis names
- `.dim` - Dimensionality
- `.device` - Computing device

**Methods**:
- `.magnitude()` - Euclidean norm
- `.normalize()` - Return unit vector
- `.add(other)` / `+` - Vector addition
- `.subtract(other)` / `-` - Vector subtraction
- `.scale(scalar)` / `*` - Scalar multiplication
- `.dot(other)` - Dot product
- `.wedge(other)` / `^` - Wedge product
- `.project_onto(other)` - Vector projection
- `.angle_with(other)` - Angle in radians
- `.get_component(name)` - Get specific coefficient
- `.to_numpy()` - Convert to numpy array

### WedgeProduct Class

**Methods**:
- `.magnitude()` - Magnitude of bivector
- `.correlation_strength()` - Measure of correlation
- `.as_matrix()` - Antisymmetric matrix representation

### GeometricVectorBatch Class

**Constructor**:
```python
GeometricVectorBatch(vectors)
```

**Methods**:
- `.magnitudes()` - All magnitudes
- `.normalize()` - Normalize all vectors
- `.mean()` - Mean vector
- `.pairwise_distances()` - Distance matrix
- `.gram_matrix()` - Similarity matrix (dot products)

## Performance

GPU acceleration provides significant speedups for:
- Batch magnitude computations
- Pairwise distance calculations
- Gram matrix construction
- Large-scale vector operations

Example benchmark results (3000x3000 matrices):
- CPU: 0.1234 seconds
- GPU: 0.0056 seconds
- Speedup: 22x

## Applications

1. **Feature Engineering**: Maintain semantic meaning through transformations
2. **Embedding Spaces**: Ensure embeddings respect dimensional boundaries
3. **Loss Functions**: Design losses that respect geometric structure
4. **Correlation Analysis**: Use wedge products to understand feature relationships
5. **Data Validation**: Catch dimensional mismatches at runtime

## Windows Compatibility

All code avoids Unicode characters to ensure Windows compatibility.
No special characters in variable names or string literals.

## Further Reading

- **Original Article**: [The Great Embedding](https://agussudjianto.substack.com/p/the-great-embedding-escaping-the)
- **Series Overview**: [Geometric Algebra for Data Science](https://agussudjianto.substack.com/p/geometric-algebra-for-data-science)
- **Main Documentation**: [../CLAUDE.md](../CLAUDE.md)
- **Main README**: [../README.md](../README.md)

## License

Implementation of concepts from "The Great Embedding" article by Agus Sudjianto.
