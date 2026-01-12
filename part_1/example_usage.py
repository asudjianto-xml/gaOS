"""
Quick Start Example - Geometric Vector Implementation

This script demonstrates the key concepts from "The Great Embedding" article
using the GeometricVector class with GPU acceleration.
"""

from geometric_vector import (
    GeometricVector,
    create_housing_vector,
    GeometricVectorBatch,
    get_device
)


def main():
    print("=" * 70)
    print("GEOMETRIC VECTORS - ESCAPING THE SCALAR TRAP")
    print("=" * 70)

    # Initialize device
    device = get_device()
    print()

    # ========================================================================
    # PART 1: THE SCALAR TRAP
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: THE SCALAR TRAP - Why Raw Arrays Are Dangerous")
    print("=" * 70)

    print("\nScenario: Housing data")
    print("-" * 70)

    # The wrong way (scalar trap)
    raw_house1 = [2500, 3, 20]  # sqft, bedrooms, age
    raw_house2 = [3000, 4, 15]

    print(f"House 1 (raw): {raw_house1}")
    print(f"House 2 (raw): {raw_house2}")
    print(f"\nNaive sum: {[a+b for a, b in zip(raw_house1, raw_house2)]}")
    print("Problem: We're treating sqft + bedrooms + age as if they're the same unit!")

    # The right way (geometric vectors)
    print("\n" + "-" * 70)
    print("Using GeometricVector (the correct approach):")
    print("-" * 70)

    house1 = create_housing_vector(2500, 3, 20)
    house2 = create_housing_vector(3000, 4, 15)

    print(f"\nHouse 1: {house1}")
    print(f"House 2: {house2}")

    combined = house1 + house2
    print(f"\nCombined: {combined}")
    print("\nEach dimension stays separate and meaningful!")
    print(f"  Total sqft: {combined.get_component('sqft')}")
    print(f"  Total bedrooms: {combined.get_component('bedrooms')}")
    print(f"  Total age: {combined.get_component('age')}")

    # ========================================================================
    # PART 2: LAW OF NON-INTERACTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: LAW OF NON-INTERACTION - Dimensional Safety")
    print("=" * 70)

    print("\nTrying to add incompatible vector spaces:")
    print("-" * 70)

    car = GeometricVector([150, 4, 8], ['horsepower', 'wheels', 'cylinders'])
    print(f"\nCar vector: {car}")
    print(f"House vector: {house1}")

    try:
        invalid = house1 + car
        print("\nERROR: Should never reach here!")
    except ValueError as e:
        print(f"\nCORRECTLY PREVENTED:")
        print(f"  {e}")
        print("\nThe Law of Non-Interaction protects us from meaningless operations!")

    # ========================================================================
    # PART 3: GEOMETRIC OPERATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 3: GEOMETRIC OPERATIONS")
    print("=" * 70)

    # Magnitude
    print("\n1. Magnitude (Euclidean norm):")
    print("-" * 70)
    print(f"   |house1| = {house1.magnitude():.4f}")
    print(f"   |house2| = {house2.magnitude():.4f}")

    # Dot product
    print("\n2. Dot Product (similarity):")
    print("-" * 70)
    dot_prod = house1.dot(house2)
    print(f"   house1 . house2 = {dot_prod:.4f}")

    # Angle
    print("\n3. Angle Between Vectors:")
    print("-" * 70)
    angle_rad = house1.angle_with(house2)
    import numpy as np
    angle_deg = np.degrees(angle_rad)
    print(f"   Angle = {angle_rad:.4f} radians ({angle_deg:.2f} degrees)")

    # Wedge product (the key innovation)
    print("\n4. Wedge Product (captures relationship without collapsing):")
    print("-" * 70)
    wedge = house1 ^ house2  # Using ^ operator
    print(f"   Magnitude: {wedge.magnitude():.4f}")
    print(f"   Correlation strength: {wedge.correlation_strength():.4f}")
    print("   (0 = parallel/correlated, 1 = orthogonal/uncorrelated)")

    # Projection
    print("\n5. Vector Projection:")
    print("-" * 70)
    proj = house1.project_onto(house2)
    print(f"   house1 projected onto house2:")
    print(f"   {proj}")

    # Scalar multiplication
    print("\n6. Scalar Multiplication:")
    print("-" * 70)
    scaled = house1 * 2.5
    print(f"   house1 * 2.5 = {scaled}")

    # ========================================================================
    # PART 4: GPU BATCH PROCESSING
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 4: GPU-ACCELERATED BATCH PROCESSING")
    print("=" * 70)

    print("\nCreating batch of 5 houses:")
    print("-" * 70)

    houses = [
        create_housing_vector(2500, 3, 20),
        create_housing_vector(3000, 4, 15),
        create_housing_vector(1800, 2, 30),
        create_housing_vector(2200, 3, 10),
        create_housing_vector(2800, 3, 25),
    ]

    batch = GeometricVectorBatch(houses)
    print(f"{batch}")

    # Batch operations
    print("\n1. Batch Magnitudes (computed in parallel on GPU):")
    print("-" * 70)
    mags = batch.magnitudes()
    for i, mag in enumerate(mags):
        print(f"   House {i+1}: {mag:.4f}")

    print("\n2. Mean Vector:")
    print("-" * 70)
    mean_house = batch.mean()
    print(f"   Average: {mean_house}")

    print("\n3. Pairwise Distances:")
    print("-" * 70)
    distances = batch.pairwise_distances()
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Distance House 1 <-> House 2: {distances[0, 1]:.4f}")
    print(f"   Distance House 1 <-> House 3: {distances[0, 2]:.4f}")

    print("\n4. Gram Matrix (similarity matrix via dot products):")
    print("-" * 70)
    gram = batch.gram_matrix()
    print(f"   Gram matrix shape: {gram.shape}")
    print(f"   Similarity House 1 <-> House 2: {gram[0, 1]:.4f}")

    # ========================================================================
    # PART 5: PRACTICAL APPLICATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 5: PRACTICAL APPLICATIONS")
    print("=" * 70)

    print("\nGeometric Vectors enable:")
    print("-" * 70)
    print("  1. Type-safe feature engineering")
    print("  2. Dimensionally-aware embeddings")
    print("  3. Geometric loss functions")
    print("  4. Correlation analysis via wedge products")
    print("  5. GPU-accelerated batch operations")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  * The Scalar Trap: Treating multi-dimensional data as raw numbers")
    print("  * Solution: Geometric vectors with basis awareness")
    print("  * Law of Non-Interaction: Incompatible dimensions can't mix")
    print("  * Wedge Product: Captures relationships without losing structure")
    print("  * GPU Acceleration: All operations leverage CUDA when available")

    print("\nFor more details, see:")
    print("  - CLAUDE.md (documentation)")
    print("  - geometric_vector.py (full implementation)")
    print("  - https://agussudjianto.substack.com/p/the-great-embedding-escaping-the")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
