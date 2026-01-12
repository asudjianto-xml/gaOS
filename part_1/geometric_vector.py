"""
Geometric Vector Implementation - Escaping the Scalar Trap

Based on "The Great Embedding" article, this module implements dimensionally-aware
vector operations using PyTorch with GPU acceleration.

The core idea: Data should be treated as geometric objects with orthogonal basis
vectors rather than raw scalar arrays, enforcing the Law of Non-Interaction.
"""

import torch
import numpy as np
from typing import List, Union, Optional


def get_device():
    """
    Detects and returns the best available device (CUDA GPU or CPU).

    Returns:
        torch.device: The device to use for computations
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("GPU not available. Using CPU.")
    return device


class GeometricVector:
    """
    A dimensionally-aware vector that maintains basis information.

    Represents vectors as: x = v1*e1 + v2*e2 + v3*e3 + ...
    where ei are orthogonal basis vectors and vi are coefficients.

    This prevents the "Scalar Trap" by ensuring incompatible dimensions
    cannot be accidentally combined.
    """

    def __init__(self, values: Union[List, np.ndarray, torch.Tensor],
                 basis_names: List[str],
                 device: Optional[torch.device] = None):
        """
        Initialize a geometric vector.

        Args:
            values: Coefficient values for each basis dimension
            basis_names: Names of basis vectors (e.g., ['sqft', 'bedrooms', 'age'])
            device: Target device (auto-detected if None)
        """
        if device is None:
            device = get_device()

        self.device = device

        # Convert values to tensor on device
        if isinstance(values, torch.Tensor):
            self.values = values.to(device).float()
        else:
            self.values = torch.tensor(values, dtype=torch.float32, device=device)

        # Store basis names
        self.basis = tuple(basis_names)  # Immutable for safety

        # Validate dimensions match
        if len(self.values) != len(self.basis):
            raise ValueError(f"Values length ({len(self.values)}) must match "
                           f"basis length ({len(self.basis)})")

    def __repr__(self):
        """String representation showing basis decomposition."""
        components = []
        values_cpu = self.values.cpu().numpy()
        for val, basis in zip(values_cpu, self.basis):
            components.append(f"{val:.4f}*{basis}")
        return "GeometricVector(" + " + ".join(components) + ")"

    def __str__(self):
        return self.__repr__()

    @property
    def dim(self):
        """Return the dimensionality of the vector."""
        return len(self.basis)

    def magnitude(self):
        """
        Compute the Euclidean magnitude/norm of the vector.

        Returns:
            torch.Tensor: Scalar magnitude
        """
        return torch.norm(self.values)

    def normalize(self):
        """
        Return a normalized version of this vector.

        Returns:
            GeometricVector: Unit vector in the same direction
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return GeometricVector(self.values / mag, list(self.basis), self.device)

    def add(self, other: 'GeometricVector') -> 'GeometricVector':
        """
        Add two geometric vectors (enforces Law of Non-Interaction).

        Args:
            other: Another GeometricVector

        Returns:
            GeometricVector: Sum of vectors

        Raises:
            ValueError: If vectors have different basis spaces
        """
        if self.basis != other.basis:
            raise ValueError(
                f"Cannot add vectors from different spaces!\n"
                f"This basis: {self.basis}\n"
                f"Other basis: {other.basis}\n"
                f"Violates the Law of Non-Interaction."
            )

        # Move other to same device if needed
        other_values = other.values.to(self.device)
        return GeometricVector(
            self.values + other_values,
            list(self.basis),
            self.device
        )

    def __add__(self, other):
        """Operator overload for +"""
        return self.add(other)

    def subtract(self, other: 'GeometricVector') -> 'GeometricVector':
        """Subtract two geometric vectors."""
        if self.basis != other.basis:
            raise ValueError("Cannot subtract vectors from different spaces!")

        other_values = other.values.to(self.device)
        return GeometricVector(
            self.values - other_values,
            list(self.basis),
            self.device
        )

    def __sub__(self, other):
        """Operator overload for -"""
        return self.subtract(other)

    def scale(self, scalar: float) -> 'GeometricVector':
        """
        Multiply vector by a scalar.

        Args:
            scalar: Scaling factor

        Returns:
            GeometricVector: Scaled vector
        """
        return GeometricVector(
            self.values * scalar,
            list(self.basis),
            self.device
        )

    def __mul__(self, scalar):
        """Operator overload for * (scalar multiplication)"""
        if isinstance(scalar, (int, float)):
            return self.scale(scalar)
        else:
            raise TypeError("Can only multiply GeometricVector by scalar")

    def __rmul__(self, scalar):
        """Reverse multiplication for scalar * vector"""
        return self.__mul__(scalar)

    def dot(self, other: 'GeometricVector') -> torch.Tensor:
        """
        Compute dot product (inner product).

        Args:
            other: Another GeometricVector

        Returns:
            torch.Tensor: Scalar dot product

        Raises:
            ValueError: If vectors have different basis spaces
        """
        if self.basis != other.basis:
            raise ValueError("Cannot compute dot product of vectors from different spaces!")

        other_values = other.values.to(self.device)
        return torch.dot(self.values, other_values)

    def wedge(self, other: 'GeometricVector') -> 'WedgeProduct':
        """
        Compute wedge product (outer product) - captures relationship without
        collapsing dimensional information.

        The wedge product represents the oriented area/volume spanned by vectors.

        Args:
            other: Another GeometricVector

        Returns:
            WedgeProduct: Antisymmetric product representing the bivector
        """
        if self.basis != other.basis:
            raise ValueError("Cannot compute wedge product of vectors from different spaces!")

        return WedgeProduct(self, other)

    def __xor__(self, other):
        """Operator overload for ^ (wedge product)"""
        return self.wedge(other)

    def project_onto(self, other: 'GeometricVector') -> 'GeometricVector':
        """
        Project this vector onto another vector.

        Args:
            other: Vector to project onto

        Returns:
            GeometricVector: Projection of self onto other
        """
        if self.basis != other.basis:
            raise ValueError("Cannot project vectors from different spaces!")

        other_mag_sq = torch.dot(other.values, other.values)
        if other_mag_sq == 0:
            raise ValueError("Cannot project onto zero vector")

        projection_scalar = torch.dot(self.values, other.values) / other_mag_sq
        return other.scale(projection_scalar.item())

    def angle_with(self, other: 'GeometricVector') -> float:
        """
        Compute angle between two vectors in radians.

        Args:
            other: Another GeometricVector

        Returns:
            float: Angle in radians [0, pi]
        """
        if self.basis != other.basis:
            raise ValueError("Cannot compute angle between vectors from different spaces!")

        cos_angle = self.dot(other) / (self.magnitude() * other.magnitude())
        # Clamp to avoid numerical errors
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        return torch.acos(cos_angle).item()

    def to_numpy(self):
        """Convert to numpy array (losing basis information)."""
        return self.values.cpu().numpy()

    def get_component(self, basis_name: str) -> float:
        """
        Get the coefficient for a specific basis dimension.

        Args:
            basis_name: Name of the basis vector

        Returns:
            float: Coefficient value
        """
        try:
            idx = self.basis.index(basis_name)
            return self.values[idx].item()
        except ValueError:
            raise ValueError(f"Basis '{basis_name}' not found in {self.basis}")


class WedgeProduct:
    """
    Represents a wedge product (bivector) of two vectors.

    The wedge product captures the oriented area spanned by two vectors,
    useful for understanding correlations and multicollinearity without
    losing dimensional information.
    """

    def __init__(self, v1: GeometricVector, v2: GeometricVector):
        """
        Create wedge product of two vectors.

        Args:
            v1: First geometric vector
            v2: Second geometric vector
        """
        self.v1 = v1
        self.v2 = v2
        self.device = v1.device

        # Store the antisymmetric matrix representation
        # This is the outer product minus its transpose
        outer = torch.outer(v1.values, v2.values)
        self.matrix = outer - outer.t()

    def __repr__(self):
        return f"WedgeProduct({self.v1.basis})"

    def magnitude(self):
        """
        Compute the magnitude of the wedge product.
        This represents the area of the parallelogram spanned by the vectors.

        Returns:
            torch.Tensor: Magnitude of the bivector
        """
        # Frobenius norm of the antisymmetric matrix
        return torch.norm(self.matrix) / torch.sqrt(torch.tensor(2.0, device=self.device))

    def as_matrix(self):
        """
        Return the matrix representation of the wedge product.

        Returns:
            torch.Tensor: Antisymmetric matrix
        """
        return self.matrix

    def correlation_strength(self):
        """
        Measure correlation strength between the two vectors.

        Returns:
            float: Value between 0 and 1, where 0 means parallel/antiparallel
                   and 1 means orthogonal
        """
        # Normalize by the product of magnitudes
        mag_product = self.v1.magnitude() * self.v2.magnitude()
        if mag_product == 0:
            return 0.0
        return (self.magnitude() / mag_product).item()


class GeometricVectorBatch:
    """
    Efficient batch processing of multiple geometric vectors on GPU.

    Stores vectors as a 2D tensor (batch_size x dim) for parallel operations.
    """

    def __init__(self, vectors: List[GeometricVector]):
        """
        Create a batch from a list of geometric vectors.

        Args:
            vectors: List of GeometricVector instances with same basis
        """
        if not vectors:
            raise ValueError("Cannot create empty batch")

        # Check all have same basis
        first_basis = vectors[0].basis
        if not all(v.basis == first_basis for v in vectors):
            raise ValueError("All vectors in batch must have same basis")

        self.basis = first_basis
        self.device = vectors[0].device

        # Stack into batch tensor
        self.values = torch.stack([v.values for v in vectors])

    def __repr__(self):
        return f"GeometricVectorBatch(size={len(self)}, dim={self.dim}, basis={self.basis})"

    def __len__(self):
        return self.values.shape[0]

    @property
    def dim(self):
        return self.values.shape[1]

    def magnitudes(self):
        """
        Compute magnitudes of all vectors in batch.

        Returns:
            torch.Tensor: Vector of magnitudes
        """
        return torch.norm(self.values, dim=1)

    def normalize(self):
        """
        Normalize all vectors in batch.

        Returns:
            GeometricVectorBatch: Batch of normalized vectors
        """
        mags = self.magnitudes().unsqueeze(1)
        normalized_vectors = [
            GeometricVector(vals, list(self.basis), self.device)
            for vals in (self.values / mags)
        ]
        return GeometricVectorBatch(normalized_vectors)

    def mean(self):
        """
        Compute mean vector of batch.

        Returns:
            GeometricVector: Mean vector
        """
        mean_values = torch.mean(self.values, dim=0)
        return GeometricVector(mean_values, list(self.basis), self.device)

    def pairwise_distances(self):
        """
        Compute pairwise distances between all vectors in batch.

        Returns:
            torch.Tensor: Distance matrix (batch_size x batch_size)
        """
        # Efficient batch distance computation
        return torch.cdist(self.values, self.values)

    def gram_matrix(self):
        """
        Compute Gram matrix (matrix of all pairwise dot products).

        Returns:
            torch.Tensor: Gram matrix (batch_size x batch_size)
        """
        return torch.matmul(self.values, self.values.t())


def create_housing_vector(sqft: float, bedrooms: int, age: int,
                         device: Optional[torch.device] = None) -> GeometricVector:
    """
    Create a geometric vector for housing data (example from article).

    Args:
        sqft: Square footage
        bedrooms: Number of bedrooms
        age: Age in years
        device: Target device

    Returns:
        GeometricVector: Housing data as geometric vector
    """
    return GeometricVector(
        [sqft, bedrooms, age],
        ['sqft', 'bedrooms', 'age'],
        device
    )


def demonstrate_scalar_trap():
    """
    Demonstrate the scalar trap problem and how GeometricVector prevents it.
    """
    print("=" * 60)
    print("DEMONSTRATING THE SCALAR TRAP")
    print("=" * 60)

    # Raw scalar approach (THE TRAP)
    print("\n1. THE SCALAR TRAP (Wrong Approach):")
    print("-" * 60)
    raw_data = [2500, 3, 20]  # sqft, bedrooms, age
    print(f"Raw data: {raw_data}")
    print(f"Sum of raw values: {sum(raw_data)}")
    print("Problem: 2500 + 3 + 20 = 2523 is meaningless!")
    print("We just added square feet + bedrooms + years!")

    # Geometric approach (THE SOLUTION)
    print("\n2. GEOMETRIC VECTOR APPROACH (Correct):")
    print("-" * 60)
    house1 = create_housing_vector(2500, 3, 20)
    print(f"House 1: {house1}")

    house2 = create_housing_vector(3000, 4, 15)
    print(f"House 2: {house2}")

    # Addition preserves dimensions
    print("\n3. ADDING VECTORS (Preserves Dimensions):")
    print("-" * 60)
    combined = house1 + house2
    print(f"Combined: {combined}")
    print("Each dimension stays separate!")
    print(f"Total sqft: {combined.get_component('sqft')}")
    print(f"Total bedrooms: {combined.get_component('bedrooms')}")
    print(f"Total age: {combined.get_component('age')}")

    # Attempting to add incompatible spaces
    print("\n4. LAW OF NON-INTERACTION (Safety Check):")
    print("-" * 60)
    car = GeometricVector([150, 4, 8], ['horsepower', 'wheels', 'cylinders'])
    print(f"Car vector: {car}")
    try:
        invalid = house1 + car
        print("ERROR: Should not reach here!")
    except ValueError as e:
        print(f"Correctly prevented: {e}")

    print("\n" + "=" * 60)


def demonstrate_geometric_operations():
    """
    Demonstrate various geometric operations.
    """
    print("\n" + "=" * 60)
    print("GEOMETRIC OPERATIONS")
    print("=" * 60)

    house1 = create_housing_vector(2500, 3, 20)
    house2 = create_housing_vector(3000, 4, 15)

    # Magnitude
    print("\n1. MAGNITUDE:")
    print("-" * 60)
    print(f"House 1 magnitude: {house1.magnitude():.4f}")
    print(f"House 2 magnitude: {house2.magnitude():.4f}")

    # Dot product
    print("\n2. DOT PRODUCT (Similarity):")
    print("-" * 60)
    dot_prod = house1.dot(house2)
    print(f"House1 · House2 = {dot_prod:.4f}")

    # Angle
    print("\n3. ANGLE BETWEEN VECTORS:")
    print("-" * 60)
    angle_rad = house1.angle_with(house2)
    angle_deg = np.degrees(angle_rad)
    print(f"Angle: {angle_rad:.4f} radians ({angle_deg:.2f} degrees)")

    # Wedge product
    print("\n4. WEDGE PRODUCT (Captures Relationship):")
    print("-" * 60)
    wedge = house1 ^ house2
    print(f"Wedge product magnitude: {wedge.magnitude():.4f}")
    print(f"Correlation strength: {wedge.correlation_strength():.4f}")
    print("(0 = parallel, 1 = orthogonal)")

    # Projection
    print("\n5. PROJECTION:")
    print("-" * 60)
    proj = house1.project_onto(house2)
    print(f"House1 projected onto House2: {proj}")

    print("\n" + "=" * 60)


def demonstrate_batch_processing():
    """
    Demonstrate efficient batch processing on GPU.
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING ON GPU")
    print("=" * 60)

    # Create batch of housing vectors
    houses = [
        create_housing_vector(2500, 3, 20),
        create_housing_vector(3000, 4, 15),
        create_housing_vector(1800, 2, 30),
        create_housing_vector(2200, 3, 10),
        create_housing_vector(2800, 3, 25),
    ]

    batch = GeometricVectorBatch(houses)
    print(f"\n{batch}")

    print("\n1. BATCH MAGNITUDES:")
    print("-" * 60)
    mags = batch.magnitudes()
    for i, mag in enumerate(mags):
        print(f"House {i+1}: {mag:.4f}")

    print("\n2. MEAN VECTOR:")
    print("-" * 60)
    mean_house = batch.mean()
    print(f"Average house: {mean_house}")

    print("\n3. PAIRWISE DISTANCES:")
    print("-" * 60)
    distances = batch.pairwise_distances()
    print(f"Distance matrix shape: {distances.shape}")
    print(f"Distance between House 1 and 2: {distances[0, 1]:.4f}")

    print("\n4. GRAM MATRIX (Similarity):")
    print("-" * 60)
    gram = batch.gram_matrix()
    print(f"Gram matrix shape: {gram.shape}")
    print(f"Similarity (dot product) House 1 and 2: {gram[0, 1]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_scalar_trap()
    demonstrate_geometric_operations()
    demonstrate_batch_processing()

    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)
