# Geometric Algebra for Data Science

Implementation of concepts from the [Geometric Algebra for Data Science](https://agussudjianto.substack.com/p/geometric-algebra-for-data-science) series by Agus Sudjianto.

---

## About

This repository provides practical Python/PyTorch implementations with GPU acceleration for the five-track series:

- **Track I: Foundations** - The Wedge Product and Geometric Product
- **Track II: Geometric Statistics** - Statistical moments as geometric shapes
- **Track III: Geometric Econometrics** - Time-series through rotations in phase space
- **Track IV: Classical Machine Learning** - SVMs, clustering, and Information Volume
- **Track V: Deep Geometric Learning** - Rotor Layers and Geometric Attention mechanisms

**Read the series:** https://agussudjianto.substack.com/p/geometric-algebra-for-data-science

---

## Current Status

**Track I, Chapter 1: The Great Embedding - Escaping the Scalar Trap** ✅
[Read the article](https://agussudjianto.substack.com/p/the-great-embedding-escaping-the)

**Track I, Chapter 2: Beyond the Arrow - The Wedge Product** ✅
Article scheduled for January 14, 2026

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/asudjianto-xml/gaOS.git
cd gaOS

# Install dependencies
pip install torch numpy matplotlib jupyter

# Run examples
cd part_1
python chapter_1_examples.py
python chapter_2_examples.py

# Or use Jupyter notebooks
jupyter lab chapter_1_demo.ipynb
jupyter lab chapter_2_wedge_product_demo.ipynb
```

---

## Repository Structure

```
gaOS/
├── README.md
└── part_1/                                  # Track I: Foundations
    ├── README.md                            # Documentation
    ├── geometric_vector.py                  # Chapter 1: Core implementation
    ├── chapter_1_examples.py                # Chapter 1: Examples
    ├── chapter_1_demo.ipynb                 # Chapter 1: Interactive notebook
    ├── chapter_2_wedge_product.py           # Chapter 2: Wedge product
    ├── chapter_2_examples.py                # Chapter 2: Examples
    └── chapter_2_wedge_product_demo.ipynb   # Chapter 2: Interactive notebook
```

---

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy >= 1.20
- Matplotlib >= 3.3
- Jupyter Lab (optional)

---

## Citation

```
Sudjianto, A. (2024). Geometric Algebra for Data Science.
https://agussudjianto.substack.com/p/geometric-algebra-for-data-science
```

---

## Contact

- **Series Author**: [Agus Sudjianto on Substack](https://agussudjianto.substack.com)
- **Repository Issues**: [GitHub Issues](https://github.com/asudjianto-xml/gaOS/issues)
