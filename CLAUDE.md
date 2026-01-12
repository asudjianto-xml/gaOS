# PyTorch GPU-Accelerated Functions

This document provides PyTorch functions optimized for GPU usage with automatic device detection.

## Environment Setup

Python and PyTorch are pre-installed in: `~/jupyterlab/ga_verify/venv/bin`

## Device Detection

```python
import torch

def get_device():
    """
    Detects and returns the best available device (CUDA GPU or CPU).

    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("GPU not available. Using CPU.")
    return device
```

## Basic Tensor Operations

```python
def create_tensor(shape, device=None):
    """
    Creates a random tensor on the specified device.

    Args:
        shape: Tuple defining tensor shape
        device: Target device (auto-detected if None)

    Returns:
        torch.Tensor: Random tensor on specified device
    """
    if device is None:
        device = get_device()
    return torch.randn(shape, device=device)


def matrix_multiply(a, b, device=None):
    """
    Performs matrix multiplication on GPU when available.

    Args:
        a: First matrix (torch.Tensor or list/array)
        b: Second matrix (torch.Tensor or list/array)
        device: Target device (auto-detected if None)

    Returns:
        torch.Tensor: Result of matrix multiplication
    """
    if device is None:
        device = get_device()

    # Convert to tensors if needed and move to device
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32, device=device)
    else:
        a = a.to(device)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32, device=device)
    else:
        b = b.to(device)

    return torch.matmul(a, b)
```

## Neural Network Layer

```python
class SimpleLayer(torch.nn.Module):
    """
    A simple fully connected layer with automatic device handling.
    """
    def __init__(self, input_size, output_size, device=None):
        super(SimpleLayer, self).__init__()
        self.device = device if device is not None else get_device()
        self.linear = torch.nn.Linear(input_size, output_size).to(self.device)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output after linear transformation and activation
        """
        x = x.to(self.device)
        return self.activation(self.linear(x))


def create_mlp(layer_sizes, device=None):
    """
    Creates a multi-layer perceptron with automatic device placement.

    Args:
        layer_sizes: List of integers defining layer dimensions
        device: Target device (auto-detected if None)

    Returns:
        torch.nn.Sequential: The MLP model
    """
    if device is None:
        device = get_device()

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(torch.nn.ReLU())

    model = torch.nn.Sequential(*layers).to(device)
    return model
```

## Batch Processing

```python
def process_batch(data, model, device=None):
    """
    Processes a batch of data through a model on GPU.

    Args:
        data: Input data (list, array, or tensor)
        model: PyTorch model
        device: Target device (auto-detected if None)

    Returns:
        torch.Tensor: Model output
    """
    if device is None:
        device = get_device()

    # Convert to tensor and move to device
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    # Ensure model is on correct device
    model = model.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(data)

    return output
```

## Optimization and Training

```python
def train_step(model, optimizer, data, targets, criterion, device=None):
    """
    Performs a single training step on GPU.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        data: Input data
        targets: Target labels
        criterion: Loss function
        device: Target device (auto-detected if None)

    Returns:
        float: Loss value
    """
    if device is None:
        device = get_device()

    # Move data to device
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32, device=device)
    else:
        data = data.to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
    else:
        targets = targets.to(device)

    # Ensure model is on device
    model = model.to(device)

    # Forward pass
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_model(model, data_loader, device=None):
    """
    Evaluates model performance on GPU.

    Args:
        model: PyTorch model
        data_loader: DataLoader with evaluation data
        device: Target device (auto-detected if None)

    Returns:
        float: Average loss
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = torch.nn.functional.mse_loss(outputs, targets)

            total_loss += loss.item()
            count += 1

    return total_loss / count if count > 0 else 0.0
```

## Utility Functions

```python
def benchmark_device(size=5000, iterations=10):
    """
    Benchmarks GPU vs CPU performance for matrix operations.

    Args:
        size: Matrix dimension
        iterations: Number of iterations to run

    Returns:
        dict: Timing results for GPU and CPU
    """
    import time

    results = {}

    # GPU benchmark
    if torch.cuda.is_available():
        device = torch.device('cuda')
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / iterations
        results['gpu'] = gpu_time

    # CPU benchmark
    device = torch.device('cpu')
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    cpu_time = (time.time() - start) / iterations
    results['cpu'] = cpu_time

    if 'gpu' in results:
        results['speedup'] = cpu_time / results['gpu']

    return results


def clear_gpu_cache():
    """
    Clears GPU cache to free memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared")
```

## Example Usage

```python
# Initialize device
device = get_device()

# Create tensors
tensor_a = create_tensor((1000, 1000), device)
tensor_b = create_tensor((1000, 1000), device)

# Matrix multiplication
result = matrix_multiply(tensor_a, tensor_b, device)

# Create neural network
model = create_mlp([784, 256, 128, 10], device)

# Training example
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

dummy_data = torch.randn(32, 784, device=device)
dummy_targets = torch.randn(32, 10, device=device)

loss = train_step(model, optimizer, dummy_data, dummy_targets, criterion, device)
print(f"Training loss: {loss:.4f}")

# Benchmark
print("\nBenchmarking GPU vs CPU:")
benchmark_results = benchmark_device(size=3000, iterations=5)
for key, value in benchmark_results.items():
    if key == 'speedup':
        print(f"{key}: {value:.2f}x")
    else:
        print(f"{key}: {value:.4f} seconds")

# Clean up
clear_gpu_cache()
```

## Notes

- All functions automatically detect and use GPU when available
- Functions fall back to CPU if GPU is not detected
- No Unicode characters used for Windows compatibility
- Device detection happens at runtime for maximum flexibility
- Use `torch.cuda.synchronize()` for accurate GPU timing benchmarks

---

# Geometric Algebra - Escaping the Scalar Trap

Based on the article: https://agussudjianto.substack.com/p/the-great-embedding-escaping-the

## The Problem: The Scalar Trap

Traditional machine learning treats data as raw scalar arrays, leading to meaningless operations:

```python
# THE TRAP: Treating incompatible dimensions as raw numbers
house_data = [2500, 3, 20]  # sqft, bedrooms, age
sum(house_data)  # = 2523 - MEANINGLESS!
# We just added square feet + bedrooms + years!
```

**The Scalar Trap** occurs when we lose dimensional awareness by treating multi-dimensional
data as simple lists of numbers.

## The Solution: Geometric Vectors

Instead of arrays, represent data as sums of orthogonal basis vectors:

**x = v1*e1 + v2*e2 + v3*e3**

Where:
- `ei` are unit vectors for distinct features (size, bedrooms, age)
- `vi` are scaling coefficients

## The Law of Non-Interaction

**Incompatible dimensions cannot be accidentally combined.**

```python
(2500*e1) + (4*e2) != 2504

# Components remain structurally separate!
```

## Using the GeometricVector Class

See `geometric_vector.py` for the full implementation.

### Basic Usage

```python
from geometric_vector import GeometricVector, create_housing_vector

# Create dimensionally-aware vectors
house1 = create_housing_vector(sqft=2500, bedrooms=3, age=20)
house2 = create_housing_vector(sqft=3000, bedrooms=4, age=15)

print(house1)
# GeometricVector(2500.0000*sqft + 3.0000*bedrooms + 20.0000*age)

# Safe addition - preserves dimensions
combined = house1 + house2
print(combined.get_component('sqft'))  # 5500.0
print(combined.get_component('bedrooms'))  # 7.0

# Attempting to add incompatible spaces raises an error
car = GeometricVector([150, 4, 8], ['horsepower', 'wheels', 'cylinders'])
try:
    invalid = house1 + car  # ValueError: Cannot add vectors from different spaces!
except ValueError as e:
    print(f"Protected by Law of Non-Interaction: {e}")
```

### Vector Operations

```python
# Magnitude (Euclidean norm)
mag = house1.magnitude()

# Normalization
unit_vector = house1.normalize()

# Dot product (similarity measure)
similarity = house1.dot(house2)

# Angle between vectors
angle_radians = house1.angle_with(house2)

# Projection
proj = house1.project_onto(house2)

# Scalar multiplication
scaled = house1 * 2.5
```

### Wedge Product - Capturing Relationships

The wedge product (^) captures the relationship between vectors without
collapsing dimensional information. It represents the oriented area/volume
spanned by vectors.

```python
# Wedge product for correlation analysis
wedge = house1 ^ house2  # or house1.wedge(house2)

# Measure correlation strength
correlation = wedge.correlation_strength()
# 0 = parallel/antiparallel (highly correlated)
# 1 = orthogonal (uncorrelated)

# Get the magnitude (area of parallelogram)
area = wedge.magnitude()

# Access the antisymmetric matrix representation
matrix = wedge.as_matrix()
```

The wedge product is useful for:
- Understanding feature correlations without losing dimensional structure
- Detecting multicollinearity in high-dimensional spaces
- Geometric interpretation of data relationships

### GPU-Accelerated Batch Processing

```python
from geometric_vector import GeometricVectorBatch

# Create multiple vectors
houses = [
    create_housing_vector(2500, 3, 20),
    create_housing_vector(3000, 4, 15),
    create_housing_vector(1800, 2, 30),
    create_housing_vector(2200, 3, 10),
]

# Batch processing on GPU
batch = GeometricVectorBatch(houses)

# Efficient batch operations
magnitudes = batch.magnitudes()  # All magnitudes at once
mean_house = batch.mean()  # Average vector
distances = batch.pairwise_distances()  # Distance matrix
gram = batch.gram_matrix()  # Similarity matrix
```

### Custom Basis Spaces

```python
# Create any basis space you need
stock = GeometricVector(
    [150.25, 1200000, 0.15],
    ['price', 'volume', 'volatility']
)

weather = GeometricVector(
    [72, 65, 1013],
    ['temperature', 'humidity', 'pressure']
)

# Each maintains its own dimensional integrity
print(stock.basis)  # ('price', 'volume', 'volatility')
print(weather.basis)  # ('temperature', 'humidity', 'pressure')
```

## Key Benefits

1. **Type Safety**: Prevents meaningless operations between incompatible dimensions
2. **Semantic Clarity**: Basis names make data interpretation explicit
3. **GPU Acceleration**: All operations leverage CUDA when available
4. **Mathematical Rigor**: Operations follow geometric algebra principles
5. **Batch Efficiency**: Process multiple vectors in parallel on GPU

## Running the Demo

```bash
# Activate the environment
source ~/jupyterlab/ga_verify/venv/bin/activate

# Run demonstrations
python geometric_vector.py
```

This will show:
1. The Scalar Trap problem and solution
2. Geometric operations (magnitude, dot product, wedge product)
3. GPU-accelerated batch processing

## Connection to Machine Learning

Traditional ML pipelines often ignore dimensional structure. GeometricVector enables:

- **Feature Engineering**: Maintain semantic meaning through transformations
- **Embedding Spaces**: Ensure embeddings respect dimensional boundaries
- **Loss Functions**: Design losses that respect geometric structure
- **Attention Mechanisms**: Compute attention over dimensionally-aware representations

## Further Reading

- Original article: https://agussudjianto.substack.com/p/the-great-embedding-escaping-the
- Geometric Algebra for Computer Science (Dorst, Fontijne, Mann)
- Clifford Algebra and its applications in physics
