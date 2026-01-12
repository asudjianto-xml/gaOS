"""
Chapter 2 Examples: Beyond the Arrow - The Wedge Product

Practical demonstrations of the wedge product in data science contexts.
"""

import torch
import numpy as np
from chapter_2_wedge_product import (
    wedge_product_tensor,
    wedge_magnitude,
    are_vectors_independent,
    independence_strength,
    extract_basis_planes,
    BatchWedgeProduct,
    get_device
)


def example_1_basic_3d():
    """
    Example from the article: Wedge product in 3D.

    Shows the XY plane representation as a bivector.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic 3D Wedge Product")
    print("=" * 70)

    u = torch.tensor([1.0, 0.0, 0.0])  # X-axis
    v = torch.tensor([0.0, 1.0, 0.0])  # Y-axis

    print(f"Vector u: {u}")
    print(f"Vector v: {v}")

    # Compute wedge product
    B = wedge_product_tensor(u, v)

    print(f"\nBivector (u ∧ v) as a Matrix:\n{B}")

    # Interpret the matrix
    print("\nInterpretation:")
    print(f"  B[0, 1] = {B[0, 1]:.1f}  -> Coefficient for XY-plane")
    print(f"  B[1, 0] = {B[1, 0]:.1f}  -> Antisymmetry (YX = -XY)")
    print(f"  B[0, 2] = {B[0, 2]:.1f}  -> No area in XZ-plane")
    print(f"  B[1, 2] = {B[1, 2]:.1f}  -> No area in YZ-plane")

    # Magnitude (area)
    area = wedge_magnitude(B)
    print(f"\nArea of parallelogram: {area:.4f}")


def example_2_4d_advantage():
    """
    Example from the article: Wedge product in 4D.

    Shows why this is superior to cross product for high dimensions.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: The 4D Advantage")
    print("=" * 70)

    # Create 4D feature vectors (height, weight, age, income)
    print("\nFeature space: [height, weight, age, income]")

    u_4d = torch.randn(4)
    v_4d = torch.randn(4)

    print(f"Vector u: {u_4d}")
    print(f"Vector v: {v_4d}")

    # Try wedge product
    B_4d = wedge_product_tensor(u_4d, v_4d)

    print(f"\n4D Bivector Shape: {B_4d.shape}")
    print(f"Area (magnitude): {wedge_magnitude(B_4d):.4f}")

    # Extract basis plane projections
    planes = extract_basis_planes(B_4d)
    print(f"\nNumber of basis planes in 4D: {len(planes)}")
    print("(In ND space, there are N(N-1)/2 basis planes)")

    print("\nProjected areas on basis planes:")
    for i, (plane, coef) in enumerate(planes.items()):
        if i < 3:  # Show first 3
            print(f"  {plane}: {coef:.4f}")
    print(f"  ... and {len(planes)-3} more planes")


def example_3_linear_independence():
    """
    Demonstrate using wedge product to detect linear independence.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Measuring Linear Independence")
    print("=" * 70)

    # Case 1: Independent vectors
    print("\n1. Orthogonal Vectors (Maximally Independent):")
    print("-" * 70)
    u1 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0])

    B1 = wedge_product_tensor(u1, v1)
    indep1 = independence_strength(u1, v1)

    print(f"u: {u1}")
    print(f"v: {v1}")
    print(f"Independence strength: {indep1:.4f}")
    print(f"Interpretation: {indep1:.1f} = Perfectly orthogonal")

    # Case 2: Parallel vectors
    print("\n2. Parallel Vectors (Linearly Dependent):")
    print("-" * 70)
    u2 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([2.0, 4.0, 6.0])  # 2 * u2

    B2 = wedge_product_tensor(u2, v2)
    indep2 = independence_strength(u2, v2)

    print(f"u: {u2}")
    print(f"v: {v2} (= 2 * u)")
    print(f"Independence strength: {indep2:.6f}")
    print(f"Interpretation: ~{indep2:.1f} = Collinear (dependent)")

    # Case 3: Partially dependent
    print("\n3. Partially Dependent Vectors:")
    print("-" * 70)
    u3 = torch.tensor([1.0, 0.0, 0.0])
    v3 = torch.tensor([0.8, 0.6, 0.0])  # 53 degree angle from u3

    B3 = wedge_product_tensor(u3, v3)
    indep3 = independence_strength(u3, v3)

    angle = torch.acos(torch.dot(u3, v3) / (torch.norm(u3) * torch.norm(v3)))
    print(f"u: {u3}")
    print(f"v: {v3}")
    print(f"Angle between vectors: {np.degrees(angle.item()):.1f} degrees")
    print(f"Independence strength: {indep3:.4f}")
    print(f"Interpretation: {indep3:.2f} = Partially independent")


def example_4_data_science_features():
    """
    Apply wedge product to detect multicollinearity in features.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Detecting Multicollinearity in Features")
    print("=" * 70)

    print("\nScenario: Housing data with potentially correlated features")
    print("-" * 70)

    # Simulate feature vectors (normalized)
    # Feature 1: [sqft, bedrooms, bathrooms, age]
    sqft_feature = torch.tensor([2500, 3, 2, 20], dtype=torch.float32)
    sqft_feature = sqft_feature / torch.norm(sqft_feature)

    # Feature 2: Highly correlated (bedrooms often correlate with sqft)
    bedroom_feature = torch.tensor([2400, 3.2, 2.1, 18], dtype=torch.float32)
    bedroom_feature = bedroom_feature / torch.norm(bedroom_feature)

    # Feature 3: Independent (age doesn't correlate)
    age_feature = torch.tensor([1800, 2, 1, 50], dtype=torch.float32)
    age_feature = age_feature / torch.norm(age_feature)

    # Check independence
    indep_12 = independence_strength(sqft_feature, bedroom_feature)
    indep_13 = independence_strength(sqft_feature, age_feature)

    print(f"\nIndependence: sqft ∧ bedrooms = {indep_12:.4f}")
    print(f"Independence: sqft ∧ age = {indep_13:.4f}")

    print(f"\nInterpretation:")
    if indep_12 < 0.3:
        print(f"  ⚠️  sqft and bedrooms are highly correlated (multicollinearity!)")
    if indep_13 > 0.5:
        print(f"  ✓  sqft and age are relatively independent")


def example_5_batch_processing():
    """
    Demonstrate GPU-accelerated batch wedge products.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: GPU-Accelerated Batch Processing")
    print("=" * 70)

    # Create batch processor
    batch_processor = BatchWedgeProduct()

    # Generate random feature pairs
    batch_size = 1000
    dim = 10

    print(f"\nProcessing {batch_size} pairs of {dim}-dimensional vectors")
    print("-" * 70)

    u_batch = torch.randn(batch_size, dim)
    v_batch = torch.randn(batch_size, dim)

    # Compute batch wedge products
    bivectors = batch_processor.compute_batch(u_batch, v_batch)
    print(f"Bivector batch shape: {bivectors.shape}")

    # Compute batch magnitudes
    areas = batch_processor.batch_magnitudes(u_batch, v_batch)
    print(f"Areas shape: {areas.shape}")
    print(f"Mean area: {areas.mean():.4f}")
    print(f"Std area: {areas.std():.4f}")

    # Compute batch independence
    independence = batch_processor.batch_independence(u_batch, v_batch)
    print(f"\nIndependence statistics:")
    print(f"  Mean: {independence.mean():.4f}")
    print(f"  Std: {independence.std():.4f}")
    print(f"  Min: {independence.min():.4f}")
    print(f"  Max: {independence.max():.4f}")

    # Find most and least independent pairs
    most_indep_idx = torch.argmax(independence)
    least_indep_idx = torch.argmin(independence)

    print(f"\nMost independent pair (index {most_indep_idx}):")
    print(f"  Independence: {independence[most_indep_idx]:.4f}")

    print(f"Least independent pair (index {least_indep_idx}):")
    print(f"  Independence: {independence[least_indep_idx]:.4f}")


def example_6_high_dimensional():
    """
    Show wedge product working in very high dimensions (like NLP embeddings).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: High-Dimensional Embeddings (100D)")
    print("=" * 70)

    print("\nSimulating word embeddings in 100-dimensional space")
    print("-" * 70)

    # Simulate word embeddings
    word1_embedding = torch.randn(100)
    word2_embedding = torch.randn(100)
    word3_embedding = word1_embedding + 0.1 * torch.randn(100)  # Similar to word1

    # Normalize
    word1_embedding = word1_embedding / torch.norm(word1_embedding)
    word2_embedding = word2_embedding / torch.norm(word2_embedding)
    word3_embedding = word3_embedding / torch.norm(word3_embedding)

    # Measure independence
    indep_12 = independence_strength(word1_embedding, word2_embedding)
    indep_13 = independence_strength(word1_embedding, word3_embedding)

    print(f"\nIndependence (word1 ∧ word2): {indep_12:.4f}")
    print(f"Independence (word1 ∧ word3): {indep_13:.4f}")

    print(f"\nInterpretation:")
    print(f"  word1 and word2 are {'independent' if indep_12 > 0.5 else 'related'}")
    print(f"  word1 and word3 are {'independent' if indep_13 > 0.5 else 'related'}")

    # Compute areas
    B_12 = wedge_product_tensor(word1_embedding, word2_embedding)
    B_13 = wedge_product_tensor(word1_embedding, word3_embedding)

    area_12 = wedge_magnitude(B_12)
    area_13 = wedge_magnitude(B_13)

    print(f"\nArea spanned (word1 ∧ word2): {area_12:.4f}")
    print(f"Area spanned (word1 ∧ word3): {area_13:.4f}")


def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 70)
    print("Chapter 2: Beyond the Arrow - The Wedge Product")
    print("Practical Examples for Data Science")
    print("=" * 70)

    # Run all examples
    example_1_basic_3d()
    example_2_4d_advantage()
    example_3_linear_independence()
    example_4_data_science_features()
    example_5_batch_processing()
    example_6_high_dimensional()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("1. Wedge Product works in ANY dimension (unlike cross product)")
    print("2. It measures linear independence (area = 0 means dependent)")
    print("3. Bivectors retain full geometric information (not just a scalar)")
    print("4. GPU acceleration makes this practical for large datasets")
    print("5. Applications: multicollinearity detection, feature selection,")
    print("   embedding analysis, and more")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
