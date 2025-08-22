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


def create_final_features(eigenvalue_arrays: List[np.ndarray], 
                         ref_values: List[float], 
                         reaction_metadata: List[Dict],
                         subset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create final feature matrix with padding and metadata.
    
    Args:
        eigenvalue_arrays: List of eigenvalue arrays
        ref_values: List of reference values
        reaction_metadata: List of reaction metadata
        subset_name: Name of subset for analysis
        
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
    targets = np.array(ref_values)
    
    print(f"✅ Final feature matrix shape: {feature_matrix.shape}")
    print(f"✅ Targets shape: {targets.shape}")
    
    return feature_matrix, targets


