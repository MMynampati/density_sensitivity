#!/usr/bin/env python3
"""
Padding and metadata integration for eigenvalue arrays.

This module takes diagonalized eigenvalue arrays and:
1. Pads them to consistent length (max size = 20)
2. Adds charge and spin information from molecular systems
3. Prepares final feature arrays for machine learning
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Tuple, Optional

def pad_eigenvalue_arrays(eigenvalue_arrays: List[np.ndarray], target_size: int = None) -> np.ndarray:
    """
    Pad eigenvalue arrays with zeros to consistent dimensions.
    
    Args:
        eigenvalue_arrays: List of 1D eigenvalue arrays of different sizes
        target_size: Target size for padding. If None, uses max size in arrays
        
    Returns:
        2D numpy array where each row is a padded eigenvalue array
    """
    
    if target_size is None:
        target_size = max(len(arr) for arr in eigenvalue_arrays)
    
    print(f"Padding {len(eigenvalue_arrays)} arrays to size {target_size}")
    
    padded_arrays = []
    
    for i, arr in enumerate(eigenvalue_arrays):
        if len(arr) > target_size:
            print(f"  Warning: Array {i} has size {len(arr)} > target {target_size}, truncating")
            padded = arr[:target_size]
        else:
            # Pad with zeros at the end
            padded = np.zeros(target_size)
            padded[:len(arr)] = arr
        
        padded_arrays.append(padded)
        
        if i < 3:  # Show first few for verification
            print(f"  Array {i}: {len(arr)} -> {len(padded)} (first 3: {padded[:3]})")
    
    return np.array(padded_arrays)

def parse_info_file(info_path: str) -> Dict[str, Dict]:
    """
    Parse the info file to get charge and spin multiplicity for each system.
    
    Args:
        info_path: Path to the info file
        
    Returns:
        Dictionary mapping system names to their properties
    """
    
    system_info = {}
    
    try:
        with open(info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    system_name = parts[0]
                    charge = int(parts[1])
                    mult = int(parts[2])
                    
                    system_info[system_name] = {
                        'charge': charge,
                        'spin_multiplicity': mult
                    }
    
    except Exception as e:
        print(f"Warning: Could not parse info file {info_path}: {e}")
    
    return system_info

def get_product_properties(systems: List[str], coeffs: List[int], info_data: Dict) -> Tuple[int, int]:
    """
    Get charge and spin of the product molecule (negative coefficient system).
    
    Args:
        systems: List of system names in reaction
        coeffs: List of stoichiometric coefficients  
        info_data: Dictionary from parse_info_file with charge/spin data
        
    Returns:
        Tuple of (charge, spin_multiplicity) for the product molecule

    # NOTE: this function need to handle cases when there is multiple products (- coefficients)
    # right now it only retrun charge & mult of first one 
    """
    
    # Find the system with negative coefficient (the product)
    product_system = None
    for system, coeff in zip(systems, coeffs):
        if coeff < 0:
            product_system = system
            break
    
    if product_system is None:
        raise ValueError(f"No negative coefficient found in reaction: {systems} {coeffs}")
    
    if product_system not in info_data:
        raise ValueError(f"Product system {product_system} not found in info data")
    
    charge = info_data[product_system]['charge']
    spin_mult = info_data[product_system]['spin_multiplicity']
    
    return charge, spin_mult


def get_swarm_binary_labels(subset_name: str, reaction_metadata: List[Dict], threshold: float = 2.0) -> Tuple[List[bool], List[float]]:
    """
    Get binary density sensitivity labels from SWARM file.
    
    Args:
        subset_name: Name of subset (e.g., "ACONF")
        reaction_metadata: List of reaction metadata dictionaries
        threshold: S value threshold for density sensitivity
        
    Returns:
        Tuple of (binary_labels, s_values)
    """
    
    print(f"\n🔍 Looking up density sensitivity for {subset_name}...")
    
    # Load SWARM data
    swarm_path = "../BurkeLab/all_v2_SWARM.csv"
    if not os.path.exists(swarm_path):
        raise FileNotFoundError(f"SWARM file not found: {swarm_path}")
    
    swarm_df = pd.read_csv(swarm_path)
    pbe_df = swarm_df[swarm_df['calctype'] == 'PBE'].copy()
    
    binary_labels = []
    s_values = []
    
    for metadata in reaction_metadata:
        rxn_id = metadata['reaction_id']
        
        # Find matching row in SWARM data
        mask = (pbe_df['setname'] == subset_name) & (pbe_df['rxnidx'] == rxn_id)
        matches = pbe_df[mask]
        
        if len(matches) == 0:
            raise ValueError(f"No SWARM entry found for {subset_name} reaction {rxn_id}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple SWARM entries found for {subset_name} reaction {rxn_id}")
        
        # Get S value and apply threshold
        s_value = matches.iloc[0]['S']
        s_values.append(s_value)
        binary_labels.append(s_value >= threshold)
        
        print(f"   Reaction {rxn_id}: S = {s_value:.4f} → {'SENSITIVE' if s_value >= threshold else 'NOT SENSITIVE'}")
    
    # Summary
    num_sensitive = sum(binary_labels)
    total_reactions = len(binary_labels)
    
    print(f"\n📊 {subset_name} Density Sensitivity Summary:")
    print(f"   Total reactions: {total_reactions}")
    print(f"   Sensitive (S ≥ {threshold}): {num_sensitive}")
    print(f"   Not sensitive (S < {threshold}): {total_reactions - num_sensitive}")
    print(f"   S value range: {min(s_values):.4f} to {max(s_values):.4f}")
    
    if num_sensitive == 0 or num_sensitive == total_reactions:
        print(f"   ⚠️  Warning: All reactions have the same label (class imbalance)")
    
    return binary_labels, s_values


def create_final_features(eigenvalue_arrays: List[np.ndarray], 
                         ref_values: List[float], 
                         reaction_metadata: List[Dict],
                         subset_name: str,
                         use_swarm_labels: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create final feature matrix with padding and metadata.
    
    Args:
        eigenvalue_arrays: List of eigenvalue arrays
        ref_values: List of reference values (used for backward compatibility)
        reaction_metadata: List of reaction metadata
        subset_name: Name of subset for analysis
        use_swarm_labels: If True, use SWARM-based binary labels; if False, use ref_values
        
    Returns:
        Tuple of (feature_matrix, targets)
    """
    
    print(f"\n=== Creating Final Features for {subset_name} ===")
    
    # Import here to avoid circular imports
    from diagonalize_matrices import analyze_eigenvalue_distributions
    
    # Analyze eigenvalue array sizes
    size_stats = analyze_eigenvalue_distributions(eigenvalue_arrays, subset_name)
    max_size = size_stats['recommended_padding']
    
    # Pad eigenvalue arrays to consistent size
    print(f"\nPadding arrays to size {max_size}...")
    padded_eigenvalues = pad_eigenvalue_arrays(eigenvalue_arrays, target_size=max_size)
    print(f"✅ Padded eigenvalues shape: {padded_eigenvalues.shape}")
    
    # Extract metadata features (charge and spin of product)
    metadata_features = []
    for meta in reaction_metadata:
        meta_row = [
            meta['product_charge'],
            meta['product_spin']
        ]
        metadata_features.append(meta_row)
    
    metadata_features = np.array(metadata_features)
    print(f"✅ Metadata features shape: {metadata_features.shape}")
    
    # Combine eigenvalues + metadata
    feature_matrix = np.hstack([padded_eigenvalues, metadata_features])
    
    # Create targets based on choice
    if use_swarm_labels:
        print(f"\n🔍 Using SWARM-based binary classification labels...")
        # Get SWARM-based binary labels
        binary_labels, s_values = get_swarm_binary_labels(subset_name, reaction_metadata)
        
        # Update metadata with SWARM information
        updated_metadata = []
        for metadata, label, s_val in zip(reaction_metadata, binary_labels, s_values):
            new_metadata = metadata.copy()
            new_metadata['subset'] = subset_name
            new_metadata['density_sensitive'] = label
            new_metadata['swarm_s_value'] = s_val
            new_metadata['original_ref_value'] = metadata['ref_value']
            updated_metadata.append(new_metadata)
        
        # Save SWARM-labeled data
        print(f"\n💾 Saving SWARM-labeled data for {subset_name}...")
        
        # Save binary targets
        binary_targets_file = f"{subset_name}_binary_targets.npy"
        np.save(binary_targets_file, np.array(binary_labels, dtype=int))
        print(f"✅ Saved binary targets: {binary_targets_file}")
        
        # Save updated metadata
        metadata_df = pd.DataFrame(updated_metadata)
        metadata_file = f"{subset_name}_swarm_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        print(f"✅ Saved SWARM metadata: {metadata_file}")
        
        targets = np.array(binary_labels, dtype=int)  # Convert boolean to 0/1
        print(f"✅ Binary targets shape: {targets.shape}")
        print(f"✅ Class distribution: {np.sum(targets)} sensitive, {len(targets) - np.sum(targets)} not sensitive")
    else:
        print(f"\n📊 Using reference values for regression...")
        targets = np.array(ref_values)
        print(f"✅ Regression targets shape: {targets.shape}")
    
    print(f"✅ Final feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, targets


