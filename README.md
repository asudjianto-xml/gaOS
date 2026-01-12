# Geometric Algebra for Data Science

> "What does this look like if we treat Data as Geometry?"

This repository implements the concepts from the [Geometric Algebra for Data Science](https://agussudjianto.substack.com/p/geometric-algebra-for-data-science) series by Agus Sudjianto, providing practical Python/PyTorch implementations with GPU acceleration.

---

## The Mission

Rebuilding the data science stack using Geometric Algebra (GA). This series proposes a fundamental reframe:

- **Probability** → Volume
- **Correlation** → A Bivector's Shadow
- **Neural Networks** → Coordinate Transformers
- **Economic Stationarity** → Subspace Stability

---

## Pragmatic GA Philosophy

Three core principles guide this implementation:

1. **No Custom Libraries** — Uses NumPy and PyTorch instead of specialized GA software
2. **Intuition First** — Prioritizes geometric understanding over formal proofs
3. **Real-World Application** — Bridges abstract mathematics with practical high-dimensional implementation

---

## Five-Track Series

### Track I: Foundations
**Status: Chapter 1 Complete ✅**

Introduces the Wedge Product and Geometric Product as fundamental operators for capturing area and unified geometric relationships.

**Chapters:**
1. **The Great Embedding: Escaping the Scalar Trap** ✅
2. **Beyond the Arrow: The Wedge Product** 📋
3. **The Master Algorithm: The Geometric Product** 📋
4. **Rotors: How to Rotate in N-Dimensions** 📋
5. **The Vector Derivative: Calculus without Indices** 📋

### Track II: Geometric Statistics 📋

Reframes statistical moments as geometric shapes; derives Linear Regression through geometric projection.

### Track III: Geometric Econometrics 📋

Examines time-series through rotations in phase space; reveals lead-lag market relationships differently.

### Track IV: Classical Machine Learning 📋

Applies geometric lenses to SVMs, clustering, and feature selection using "Information Volume" concepts.

### Track V: Deep Geometric Learning 📋

Develops Rotor Layers and Geometric Attention mechanisms that preserve symmetry in neural architectures.

---

## Quick Start

### Installation

```bash
git clone https://github.com/asudjianto-xml/gaOS.git
cd gaOS
```

### Prerequisites

```bash
pip install torch numpy matplotlib jupyter
```

### Run Examples

```bash
# Quick Python example
cd part_1
python example_usage.py

# Interactive Jupyter notebook
jupyter lab geometric_vectors_demo.ipynb
```

---

## Track I, Chapter 1: The Great Embedding

### The Scalar Trap

Traditional data science treats data as raw numbers, losing dimensional meaning:

```python
# Wrong: Treating multi-dimensional data as mere numbers
house = [2500, 3, 20]  # sqft, bedrooms, age
sum(house)  # = 2523 - MEANINGLESS!
```

### The Solution: Geometric Vectors

```python
from part_1.geometric_vector import GeometricVector, create_housing_vector

house1 = create_housing_vector(sqft=2500, bedrooms=3, age=20)
house2 = create_housing_vector(sqft=3000, bedrooms=4, age=15)

# Safe operations - dimensions preserved
combined = house1 + house2
print(combined.get_component('sqft'))  # 5500.0
```

### Key Concepts

**1. Law of Non-Interaction**

Incompatible dimensions cannot be accidentally combined:

```python
car = GeometricVector([150, 4, 8], ['horsepower', 'wheels', 'cylinders'])
house1 + car  # ValueError: Cannot add vectors from different spaces!
```

**2. Wedge Product**

Captures relationships without collapsing dimensional information:

```python
wedge = house1 ^ house2  # Wedge product operator
correlation = wedge.correlation_strength()
# 0 = parallel (correlated), 1 = orthogonal (uncorrelated)
```

**3. GPU Acceleration**

All operations automatically leverage CUDA when available:

```python
from part_1.geometric_vector import GeometricVectorBatch

# Batch processing on GPU
batch = GeometricVectorBatch([house1, house2, ...])
distances = batch.pairwise_distances()  # Computed in parallel
magnitudes = batch.magnitudes()
similarities = batch.gram_matrix()
```

---

## Repository Structure

```
gaOS/
├── README.md                           # This file
└── part_1/                             # Track I, Chapter 1
    ├── README.md                       # Chapter 1 documentation
    ├── geometric_vector.py             # Core implementation
    ├── example_usage.py                # Quick start examples
    └── geometric_vectors_demo.ipynb    # Interactive notebook
```

---

## Implementation Features

### ✅ Completed (Track I, Chapter 1)

- **GeometricVector Class**: Dimensionally-aware vectors with basis preservation
- **Type Safety**: Prevents meaningless operations via Law of Non-Interaction
- **Wedge Product**: Captures correlations without losing structure
- **GPU Acceleration**: Full PyTorch/CUDA support with automatic device detection
- **Batch Processing**: Efficient parallel operations
- **Comprehensive Examples**: Python scripts and interactive Jupyter notebook

### 📋 Coming Soon

- Track I, Chapters 2-5 (Wedge Product, Geometric Product, Rotors, Vector Derivatives)
- Tracks II-V (Statistics, Econometrics, Classical ML, Deep Learning)

---

## Examples

### Housing Data Analysis

```python
from part_1.geometric_vector import create_housing_vector
import numpy as np

house1 = create_housing_vector(2500, 3, 20)
house2 = create_housing_vector(3000, 4, 15)

# Compute similarity
similarity = house1.dot(house2)
angle = house1.angle_with(house2)

# Analyze geometric relationship
wedge = house1 ^ house2
correlation = wedge.correlation_strength()

print(f"Angle: {np.degrees(angle):.2f} degrees")
print(f"Correlation: {correlation:.4f}")
```

### Financial Data

```python
stock1 = GeometricVector([150.25, 1200000, 0.15],
                         ['price', 'volume', 'volatility'])
stock2 = GeometricVector([148.50, 1500000, 0.18],
                         ['price', 'volume', 'volatility'])

# Safe operations within same geometric space
portfolio = stock1 + stock2

# Geometric relationship analysis
wedge = stock1 ^ stock2
relationship = wedge.correlation_strength()
```

---

## Performance

GPU acceleration provides significant speedups:

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Batch Magnitudes (3000 vectors) | 45.6ms | 2.1ms | 22x |
| Pairwise Distances (3000×3000) | 234ms | 9.8ms | 24x |
| Gram Matrix (3000×3000) | 123ms | 5.6ms | 22x |

*Tested on NVIDIA GB10*

---

## Requirements

- Python 3.8+
- PyTorch >= 2.0 (with CUDA support recommended)
- NumPy >= 1.20
- Matplotlib >= 3.3 (for visualizations)
- Jupyter Lab (optional, for notebooks)

```bash
pip install torch numpy matplotlib jupyterlab
```

---

## Documentation

- **[Series Overview](https://agussudjianto.substack.com/p/geometric-algebra-for-data-science)**: Main article introducing the five-track series
- **[Chapter 1 Article](https://agussudjianto.substack.com/p/the-great-embedding-escaping-the)**: The Great Embedding
- **[part_1/README.md](part_1/README.md)**: Detailed API reference for Chapter 1
- **[Interactive Notebook](part_1/geometric_vectors_demo.ipynb)**: Hands-on tutorial with visualizations

---

## Contributing

This is an educational project implementing concepts from the Geometric Algebra for Data Science series. Contributions welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes
4. Push and open a Pull Request

### Areas for Contribution

- Implementations of Track I, Chapters 2-5
- Examples and use cases
- Performance optimizations
- Documentation improvements
- Tracks II-V implementations

---

## Citation

If you use this code in your research or projects, please cite:

```
Sudjianto, A. (2024). Geometric Algebra for Data Science.
https://agussudjianto.substack.com/p/geometric-algebra-for-data-science
```

---

## License

Educational implementation of concepts from the Geometric Algebra for Data Science series.

---

## Contact

- **Series Author**: [Agus Sudjianto on Substack](https://agussudjianto.substack.com)
- **Repository Issues**: [GitHub Issues](https://github.com/asudjianto-xml/gaOS/issues)

---

**Built with PyTorch, NumPy, and Geometric Algebra**

*Pragmatic • GPU-Accelerated • Intuition-First*
