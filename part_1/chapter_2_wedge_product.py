"""
Chapter 2: Beyond the Arrow - The Wedge Product

Based on "Geometric Algebra for Data Science, Part 1 | Chapter 2"

The Wedge Product (∧) replaces the Cross Product, working in any dimension.
It measures independence and creates bivectors (oriented areas).

Key insight: u ∧ v = u ⊗ v - v ⊗ u (antisymmetric outer product)
"""

import torch
import numpy as np
from typing import Optional


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


def wedge_product_tensor(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Computes the Bivector representation of u ∧ v
    using the antisymmetric outer product.

    The Wedge Product measures the oriented area spanned by two vectors.
    Unlike the cross product, this works in ANY dimension.

    Mathematical definition:
        u ∧ v = u ⊗ v - v ⊗ u

    where ⊗ is the outer product.

    Args:
        u, v: Tensors of shape (..., D) where D is the dimension

    Returns:
        Bivector tensor of shape (..., D, D) - an antisymmetric matrix

    Properties of the result:
        - Antisymmetric: B[i,j] = -B[j,i]
        - Magnitude: Area of parallelogram spanned by u and v
        - Zero diagonal: B[i,i] = 0

    Example:
        >>> u = torch.tensor([1.0, 0.0, 0.0])  # X-axis
        >>> v = torch.tensor([0.0, 1.0, 0.0])  # Y-axis
        >>> B = wedge_product_tensor(u, v)
        >>> # B represents the XY plane with area 1
    """
    # 1. Create the outer product (u ⊗ v)
    # Use einsum for cleaner batch handling
    outer_product = torch.einsum('...i,...j->...ij', u, v)

    # 2. The wedge product is the antisymmetric part
    # u ∧ v = u ⊗ v - v ⊗ u
    bivector = outer_product - outer_product.transpose(-1, -2)

    return bivector


def wedge_magnitude(bivector: torch.Tensor) -> torch.Tensor:
    """
    Compute the magnitude (area) of a bivector.

    For a bivector B = u ∧ v, the magnitude is the area of the
    parallelogram spanned by u and v.

    Args:
        bivector: Antisymmetric matrix of shape (..., D, D)

    Returns:
        Magnitude tensor of shape (...)

    Note:
        The magnitude is computed as ||B||_F / sqrt(2) where ||·||_F
        is the Frobenius norm. The factor of sqrt(2) accounts for
        the antisymmetry (each area component is counted twice).
    """
    # Frobenius norm of the antisymmetric matrix
    frobenius_norm = torch.norm(bivector, p='fro', dim=(-2, -1))

    # Divide by sqrt(2) to account for antisymmetry
    return frobenius_norm / torch.sqrt(torch.tensor(2.0, device=bivector.device))


def are_vectors_independent(u: torch.Tensor, v: torch.Tensor,
                           threshold: float = 1e-6) -> bool:
    """
    Check if two vectors are linearly independent using the wedge product.

    Two vectors are linearly dependent (collinear) if their wedge product
    is zero, i.e., u ∧ v = 0.

    Args:
        u, v: Vectors of shape (D,)
        threshold: Tolerance for zero comparison

    Returns:
        True if vectors are linearly independent, False otherwise

    Example:
        >>> u = torch.tensor([1.0, 0.0])
        >>> v = torch.tensor([2.0, 0.0])  # Parallel to u
        >>> are_vectors_independent(u, v)
        False
    """
    bivector = wedge_product_tensor(u, v)
    magnitude = wedge_magnitude(bivector)
    return magnitude.item() > threshold


def independence_strength(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Measure how independent (orthogonal) two vectors are.

    Returns a value between 0 and 1:
        - 0: Vectors are parallel (linearly dependent)
        - 1: Vectors are orthogonal (maximally independent)

    Args:
        u, v: Vectors of shape (D,)

    Returns:
        Independence strength (0 to 1)

    Note:
        This normalizes the wedge product magnitude by the product
        of vector magnitudes, similar to how cosine similarity works
        for the dot product.
    """
    bivector = wedge_product_tensor(u, v)
    wedge_mag = wedge_magnitude(bivector)

    # Normalize by product of vector magnitudes
    u_mag = torch.norm(u)
    v_mag = torch.norm(v)

    if u_mag == 0 or v_mag == 0:
        return 0.0

    # This gives a value between 0 (parallel) and 1 (orthogonal)
    return (wedge_mag / (u_mag * v_mag)).item()


def extract_basis_planes(bivector: torch.Tensor) -> dict:
    """
    Extract the coefficients for all basis plane projections from a bivector.

    For a D-dimensional space, there are D(D-1)/2 basis planes.
    Each element B[i,j] represents the area projected onto the e_i ∧ e_j plane.

    Args:
        bivector: Antisymmetric matrix of shape (D, D)

    Returns:
        Dictionary mapping plane names (e.g., "e1∧e2") to coefficients

    Example:
        >>> u = torch.tensor([1.0, 0.0, 0.0])
        >>> v = torch.tensor([0.0, 1.0, 0.0])
        >>> B = wedge_product_tensor(u, v)
        >>> planes = extract_basis_planes(B)
        >>> planes["e0∧e1"]  # Should be 1.0 (XY plane)
    """
    dim = bivector.shape[-1]
    planes = {}

    # Extract upper triangular part (since B[i,j] = -B[j,i])
    for i in range(dim):
        for j in range(i + 1, dim):
            coefficient = bivector[i, j].item()
            plane_name = f"e{i}∧e{j}"
            planes[plane_name] = coefficient

    return planes


class BatchWedgeProduct:
    """
    Efficient batch processing of wedge products on GPU.

    Computes wedge products for batches of vector pairs in parallel.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize batch wedge product processor.

        Args:
            device: Target device (auto-detected if None)
        """
        self.device = device if device is not None else get_device()

    def compute_batch(self, u_batch: torch.Tensor,
                     v_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute wedge products for batches of vector pairs.

        Args:
            u_batch: Tensor of shape (batch_size, D)
            v_batch: Tensor of shape (batch_size, D)

        Returns:
            Bivector batch of shape (batch_size, D, D)
        """
        u_batch = u_batch.to(self.device)
        v_batch = v_batch.to(self.device)

        return wedge_product_tensor(u_batch, v_batch)

    def batch_magnitudes(self, u_batch: torch.Tensor,
                        v_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitudes (areas) for batch of vector pairs.

        Args:
            u_batch: Tensor of shape (batch_size, D)
            v_batch: Tensor of shape (batch_size, D)

        Returns:
            Magnitudes tensor of shape (batch_size,)
        """
        bivectors = self.compute_batch(u_batch, v_batch)
        return wedge_magnitude(bivectors)

    def batch_independence(self, u_batch: torch.Tensor,
                          v_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute independence strengths for batch of vector pairs.

        Args:
            u_batch: Tensor of shape (batch_size, D)
            v_batch: Tensor of shape (batch_size, D)

        Returns:
            Independence strengths tensor of shape (batch_size,)
        """
        # Ensure tensors are on the correct device
        u_batch = u_batch.to(self.device)
        v_batch = v_batch.to(self.device)

        bivectors = self.compute_batch(u_batch, v_batch)
        wedge_mags = wedge_magnitude(bivectors)

        u_mags = torch.norm(u_batch, dim=1)
        v_mags = torch.norm(v_batch, dim=1)

        # Normalize
        independence = wedge_mags / (u_mags * v_mags + 1e-8)

        return independence


def compare_with_cross_product():
    """
    Demonstrate the limitation of the cross product vs wedge product.

    Shows that:
    1. Cross product only works in 3D
    2. Wedge product works in any dimension
    3. Both give equivalent information in 3D
    """
    print("=" * 70)
    print("CROSS PRODUCT vs WEDGE PRODUCT")
    print("=" * 70)

    # Test in 3D
    print("\n1. In 3D (Cross Product Works):")
    print("-" * 70)
    u_3d = torch.tensor([1.0, 0.0, 0.0])
    v_3d = torch.tensor([0.0, 1.0, 0.0])

    # Cross product (numpy for simplicity)
    cross = torch.cross(u_3d, v_3d)
    print(f"u × v (cross product): {cross}")
    print(f"Magnitude: {torch.norm(cross):.4f}")

    # Wedge product
    bivector = wedge_product_tensor(u_3d, v_3d)
    wedge_mag = wedge_magnitude(bivector)
    print(f"\nu ∧ v (wedge product):\n{bivector}")
    print(f"Magnitude: {wedge_mag:.4f}")

    # Test in 4D
    print("\n2. In 4D (Cross Product FAILS):")
    print("-" * 70)
    u_4d = torch.randn(4)
    v_4d = torch.randn(4)

    print("Attempting cross product...")
    try:
        cross_4d = torch.cross(u_4d, v_4d)
        print(f"Cross product result: {cross_4d}")
    except RuntimeError as e:
        print(f"ERROR: {e}")

    print("\nUsing wedge product:")
    bivector_4d = wedge_product_tensor(u_4d, v_4d)
    print(f"Bivector shape: {bivector_4d.shape}")
    print(f"Magnitude (area): {wedge_magnitude(bivector_4d):.4f}")

    planes = extract_basis_planes(bivector_4d)
    print(f"\nNumber of basis planes in 4D: {len(planes)}")
    print("Plane projections:")
    for plane, coef in planes.items():
        print(f"  {plane}: {coef:.4f}")


if __name__ == "__main__":
    print("Chapter 2: Beyond the Arrow - The Wedge Product")
    print("=" * 70)

    # Run comparison
    compare_with_cross_product()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)
