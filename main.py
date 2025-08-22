#!/usr/bin/env python3
"""
Main pipeline for density sensitivity ML analysis.

This script implements the complete workflow:
1. Loop into each subset folder (currently ACONF only)
2. Make Coulomb matrices for all molecules in that folder
3. Note the size of the largest molecule
4. Line by line through ref file, for each reaction:
   a) Combine matrices according to stoichiometry
   b) Diagonalize the combined matrix
   c) Pad with zeros to largest molecule size
   d) Append charge and mult of product
5. Store 1D arrays in both human-readable and model-compatible formats
"""

import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple

# Import functions from our files
from generate_cm import create_cm
from preprocess import parse_ref_file, combine_cm
from diagonalize_matrices import diagonalize_matrix, analyze_eigenvalue_distributions, create_ml_dataframe
from pad_and_metadata import pad_eigenvalue_arrays, parse_info_file, get_product_properties, create_final_features


def create_all_coulomb_matrices(subset_path: str) -> Dict[str, np.ndarray]:
    """
    Create Coulomb matrices for all molecules in a subset folder.
    
    Args:
        subset_path: Path to subset folder (e.g., "new_structures/ACONF")
        
    Returns:
        Dictionary mapping molecule names to their Coulomb matrices
    """
    
    coulomb_matrices = {}
    
    # Get all molecule directories
    molecule_dirs = [d for d in os.listdir(subset_path) 
                    if os.path.isdir(os.path.join(subset_path, d)) 
                    and not d.startswith('.')]
    
    print(f"Creating Coulomb matrices for {len(molecule_dirs)} molecules...")
    
    for molecule_name in sorted(molecule_dirs):
        molecule_path = os.path.join(subset_path, molecule_name)
        
        # Find XYZ file in molecule directory
        xyz_files = [f for f in os.listdir(molecule_path) if f.lower().endswith('.xyz')]
        
        if not xyz_files:
            print(f"  Warning: No XYZ file found for {molecule_name}")
            continue
            
        xyz_file = os.path.join(molecule_path, xyz_files[0])
        
        # Create Coulomb matrix
        try:
            coulomb_matrix = create_cm(xyz_file)
            coulomb_matrices[molecule_name] = coulomb_matrix
            print(f"  {molecule_name}: {coulomb_matrix.shape}")
        except Exception as e:
            print(f"  Error creating Coulomb matrix for {molecule_name}: {e}")
    
    print(f"✅ Created {len(coulomb_matrices)} Coulomb matrices")
    return coulomb_matrices


def setup_subset_data(subset_path: str) -> Tuple[Dict[str, np.ndarray], Dict, List]:
    """
    Load all necessary data for processing a subset.
    
    Args:
        subset_path: Path to subset folder
        
    Returns:
        Tuple of (coulomb_matrices, info_data, reactions)
    """
    
    # Step 1: Create all Coulomb matrices for this subset
    coulomb_matrices = create_all_coulomb_matrices(subset_path)
    
    # Step 2: Parse info file for charge/spin data
    info_path = os.path.join(subset_path, "info")
    info_data = parse_info_file(info_path)
    print(f"✅ Loaded info for {len(info_data)} systems")
    
    # Step 3: Parse ref file for reactions
    ref_path = os.path.join(subset_path, "ref")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
        
    reactions = parse_ref_file(ref_path)
    print(f"✅ Loaded {len(reactions)} reactions from ref file")
    
    return coulomb_matrices, info_data, reactions


def process_single_reaction(reaction_id: int, 
                           systems: List[str], 
                           coeffs: List[int], 
                           ref_val: float,
                           coulomb_matrices: Dict[str, np.ndarray], 
                           info_data: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Process a single reaction: combine matrices, diagonalize, extract metadata.
    
    Args:
        reaction_id: 1-indexed reaction identifier
        systems: List of molecule names in reaction
        coeffs: List of stoichiometric coefficients
        ref_val: Reference energy value
        coulomb_matrices: Dict of molecule name -> Coulomb matrix
        info_data: Dict of molecule charge/spin data
        
    Returns:
        Tuple of (eigenvalues, metadata_dict)
    """
    
    print(f"  Processing reaction {reaction_id}: {systems} {coeffs} -> {ref_val}")
    
    # a) Get Coulomb matrices for this reaction
    reaction_matrices = [coulomb_matrices[system] for system in systems]
    print(f"    Matrix shapes: {[m.shape for m in reaction_matrices]}")
    
    # b) Combine matrices according to stoichiometry
    combined_matrix = combine_cm(reaction_matrices, coeffs)
    print(f"    Combined matrix shape: {combined_matrix.shape}")
    
    # c) Diagonalize the combined matrix
    eigenvalues = diagonalize_matrix(combined_matrix)
    print(f"    Eigenvalues shape: {eigenvalues.shape}")
    
    # d) Get product molecule's charge and spin
    product_charge, product_spin = get_product_properties(systems, coeffs, info_data)
    
    # Create metadata
    metadata = {
        'reaction_id': reaction_id,
        'systems': systems,
        'coefficients': coeffs,
        'product_charge': product_charge,
        'product_spin': product_spin,
        'ref_value': ref_val,
        'eigenvalue_size': len(eigenvalues)
    }
    
    return eigenvalues, metadata


def process_subset_reactions(subset_name: str, subset_path: str) -> Tuple[List[np.ndarray], List[float], List[Dict]]:
    """
    Process all reactions in a subset.
    
    Args:
        subset_name: Name of the subset (e.g., "ACONF")
        subset_path: Path to subset folder
        
    Returns:
        Tuple of (eigenvalue_arrays, ref_values, reaction_metadata)
    """
    
    print(f"\n=== Processing {subset_name} Reactions ===")
    
    # Setup: Load all data for this subset
    coulomb_matrices, info_data, reactions = setup_subset_data(subset_path)
    
    # Process each reaction
    eigenvalue_arrays = []
    ref_values = []
    reaction_metadata = []
    
    for i, (systems, coeffs, ref_val) in enumerate(reactions):
        try:
            eigenvalues, metadata = process_single_reaction(
                reaction_id=i + 1,  # 1-indexed
                systems=systems,
                coeffs=coeffs,
                ref_val=ref_val,
                coulomb_matrices=coulomb_matrices,
                info_data=info_data
            )
            
            # Store results
            eigenvalue_arrays.append(eigenvalues)
            ref_values.append(ref_val)
            reaction_metadata.append(metadata)
            
        except Exception as e:
            print(f"  ❌ Error processing reaction {i+1}: {e}")
            continue
    
    print(f"\\n✅ Successfully processed {len(eigenvalue_arrays)} reactions")
    
    # Create DataFrame for data exploration
    print("\\n📊 Creating DataFrame for data exploration...")
    ml_df = create_ml_dataframe(eigenvalue_arrays, ref_values, reaction_metadata)
    
    # Clear Coulomb matrices to free memory
    del coulomb_matrices
    
    return eigenvalue_arrays, ref_values, reaction_metadata



def save_results(feature_matrix: np.ndarray, 
                targets: np.ndarray, 
                reaction_metadata: List[Dict],
                subset_name: str,
                max_eigenvalue_size: int):
    """
    Save results in both human-readable and model-compatible formats.
    
    Args:
        feature_matrix: Final feature matrix
        targets: Target values
        reaction_metadata: Reaction metadata
        subset_name: Name of subset
        max_eigenvalue_size: Size used for padding
    """
    
    print(f"\\n=== Saving Results for {subset_name} ===")
    
    # Create feature names
    feature_names = [f'eigenval_{i}' for i in range(max_eigenvalue_size)] + ['product_charge', 'product_spin']
    
    # Model-compatible format (.npy files)
    np.save(f"{subset_name}_features.npy", feature_matrix)
    np.save(f"{subset_name}_targets.npy", targets)
    
    # Human-readable format (.csv)
    metadata_df = pd.DataFrame(reaction_metadata)
    metadata_df.to_csv(f"{subset_name}_metadata.csv", index=False)
    
    # Complete package (.pkl)
    complete_data = {
        'subset_name': subset_name,
        'features': feature_matrix,
        'targets': targets,
        'metadata': metadata_df,
        'feature_names': feature_names,
        'num_eigenvalues': max_eigenvalue_size,
        'num_metadata': 2
    }
    
    with open(f"{subset_name}_complete.pkl", "wb") as f:
        pickle.dump(complete_data, f)
    
    print(f"✅ Saved model-compatible: {subset_name}_features.npy, {subset_name}_targets.npy")
    print(f"✅ Saved human-readable: {subset_name}_metadata.csv") 
    print(f"✅ Saved complete package: {subset_name}_complete.pkl")
    
    # Display summary
    print(f"\\n=== {subset_name} Summary ===")
    print(f"Samples: {len(targets)}")
    print(f"Features: {feature_matrix.shape[1]} ({max_eigenvalue_size} eigenvalues + 2 metadata)")
    print(f"Target range: {np.min(targets):.3f} to {np.max(targets):.3f}")


def main():
    """
    Main pipeline function.
    
    Currently processes ACONF subset only. 
    TODO: Expand outer loop to handle all 55 subsets.
    """
    
    print("🚀 Density Sensitivity ML Pipeline")
    print("=" * 50)
    
    # Configuration
    base_path = "new_structures"
    subsets_to_process = ["ACONF"]  # TODO: Expand to all 55 subsets
    
    for subset_name in subsets_to_process:
        print(f"\\n🔄 Processing subset: {subset_name}")
        
        subset_path = os.path.join(base_path, subset_name)
        
        if not os.path.exists(subset_path):
            print(f"❌ Subset path not found: {subset_path}")
            continue
        
        try:
            # Process all reactions in this subset
            eigenvalue_arrays, ref_values, reaction_metadata = process_subset_reactions(
                subset_name, subset_path
            )
            
            # Create final feature matrix
            feature_matrix, targets = create_final_features(
                eigenvalue_arrays, ref_values, reaction_metadata, subset_name
            )
            
            # Save results
            max_eigenvalue_size = feature_matrix.shape[1] - 2  # Subtract metadata features
            save_results(feature_matrix, targets, reaction_metadata, subset_name, max_eigenvalue_size)
            
            print(f"✅ Completed processing {subset_name}")
            
        except Exception as e:
            print(f"❌ Error processing {subset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\\n🎉 Pipeline completed!")
    print("Ready for Random Forest training! 🌲🤖")


if __name__ == "__main__":
    main()
