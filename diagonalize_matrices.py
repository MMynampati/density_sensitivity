#!/usr/bin/env python3
"""
Matrix diagonalization module for density sensitivity analysis.

This module takes combined Coulomb matrices and converts them to 1D eigenvalue arrays
for machine learning input. It can either load existing combined matrices or create
them from scratch using the preprocess.py functions.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Tuple, Optional

# Import functions from preprocess.py
from preprocess import parse_ref_file, combine_cm


def diagonalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Diagonalize an nxn matrix and return eigenvalues as a 1D array of length n.
    note - not technically a diagonalization (diagonalization is a process of finding the eigenvalues 
    AND eigenvectors of a matrix), but we only care about the eigenvalues for our use case
    
    Args:
        matrix: nxn numpy array to diagonalize
        
    Returns:
        1D array of eigenvalues in ascending order     *NOTE: updated so ordering is deterministic

    Notes: 
        -   eigh returns eigenvalues in ascending order (as opposed to the paper, which sorts in desc) 
        -   paper takes absolute value of eigenvalues, but we don't
    """
    #   check that matrix is symmetric
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Matrix is not symmetric. Eigenvalues may not be real.")
    
    #   eigh returns eigenvalues in ascending order
    eigenvalues = np.linalg.eigh(matrix)[0]         
    
    # Handle complex eigenvalues - take real part for now
    # (Coulomb matrices should be real symmetric, so eigenvalues should be real)  
        ## NOTE: look into this - check to make sure this is true for reaction level matrices

    #if we find non-real parts to a eigenvalue something has probably gone wrong- so then throw an error
    if np.iscomplexobj(eigenvalues):
        eigenvalues = np.real(eigenvalues)

    
    return eigenvalues



def create_ml_dataframe(eigenvalue_arrays: List[np.ndarray], 
                        ref_values: List[float], 
                        reaction_metadata: List[Dict] = None) -> pd.DataFrame:
    """
    Create a pandas DataFrame for data exploration and validation.
    
    Args:
        eigenvalue_arrays: List of eigenvalue arrays
        ref_values: List of reference values
        reaction_metadata: Optional list of reaction metadata dicts
        
    Returns:
        DataFrame with eigenvalue arrays and reference values
    """
    
    df = pd.DataFrame({
        "eigenvalues": eigenvalue_arrays,
        "ref_values": ref_values
    })
    
    # Add useful metadata columns if provided
    if reaction_metadata:
        df['reaction_id'] = [meta.get('reaction_id', i+1) for i, meta in enumerate(reaction_metadata)]
        df['systems'] = [str(meta.get('systems', '')) for meta in reaction_metadata]
        df['product_charge'] = [meta.get('product_charge', 0) for meta in reaction_metadata]
        df['product_spin'] = [meta.get('product_spin', 1) for meta in reaction_metadata]
        df['eigenvalue_size'] = [meta.get('eigenvalue_size', len(eigenvalue_arrays[i])) 
                                for i, meta in enumerate(reaction_metadata)]
    else:
        # Basic metadata if reaction_metadata not provided
        df['eigenvalue_size'] = [len(arr) for arr in eigenvalue_arrays]
    
    print(f"Created DataFrame with {len(df)} samples")
    print("Sample eigenvalue array shapes:")
    for i, arr in enumerate(df['eigenvalues'][:3]):
        print(f"  Sample {i}: {arr.shape}")
    
    return df


def analyze_eigenvalue_distributions(eigenvalue_arrays: List[np.ndarray], setname: str = "Dataset") -> Dict:
    """
    Analyze the distribution of eigenvalue array sizes to help with padding decisions.
    
    Args:
        eigenvalue_arrays: List of eigenvalue arrays
        setname: Dataset name for display
        
    Returns:
        Dictionary with size statistics
    """
    
    sizes = [len(arr) for arr in eigenvalue_arrays]
    
    stats = {
        'num_arrays': len(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'mean_size': np.mean(sizes),
        'size_distribution': dict(zip(*np.unique(sizes, return_counts=True))),
        'recommended_padding': max(sizes)
    }
    
    print(f"\n=== Eigenvalue Array Size Analysis ({setname}) ===")
    print(f"Number of arrays: {stats['num_arrays']}")
    print(f"Minimum size: {stats['min_size']}")
    print(f"Maximum size: {stats['max_size']}")
    print(f"Average size: {stats['mean_size']:.1f}")
    print(f"Size distribution: {stats['size_distribution']}")
    print(f"Recommended padding size: {stats['recommended_padding']}")
    
    return stats
