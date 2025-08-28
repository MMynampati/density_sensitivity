#!/usr/bin/env python3
"""
Main pipeline for density sensitivity ML analysis.

PREVIOUS:
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

NEW:
This script implements the complete workflow:
1. Load coulomb matrices from the dictionary    ----------------> done
2. Go through the subsets, and for each reaction in each subset:  ----------------> done
    a) combine matrices according to stoichiometry    ----------------> done
    b) diagonalize the combined matrix     ----------------> done
    c) update max size if exceeded   ----------------> done
3. Store all eigenvalue vectors in a dictionary    ----------------> done

4. Go through the subsets again, and for each reaction in each subset:
    a) pad [eigenvalue vector] with zeros to the max size
    b) append charge and mult of product
    c) add a label for the reaction 
    d) add to the final dataframe
5. Get training 
"""
import os
import numpy as np
import pandas as pd
import pickle
from statistics import mean, median, stdev
from typing import List, Dict, Tuple

# # Import functions from our files
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
    Processes multiple subsets for density sensitivity analysis.
    """
    print("Density Sensitivity ML Pipeline")

    # Configuration
    base_path = "new_structures"

    # load the pickle containing coulomb matrices for all setnames 
    with open("final_dict_allsets.pkl", "rb") as f:
        final_dict = pickle.load(f)

    # 1st loop = loop over setnames (key of final dict), and the coulomb matrices of each setname (inner dict) 
    # - ignoring PA26 since it has issues, ignoring WATER27 since mihira said 
    # go through ref file of each setname, parse and return every row of ref file 
    # for each row of ref file, get respective coulomb matrices and combine based on stochiometry coeffs
    # if could not combine due to shape missmatch, use a zero matrix as a placeholder
    # diagonalize the combined matrix 
    # keep track of largest 'vector of eigenval'
    # store  all 'vectors of eigenvals' for each setname in a dict
    # print largest size of 'vector of eigenvals' and which subset/systems it is from 
    # print summary stats of length of 'vectors of eigenvals' 

    reaction_dicts = {} 
    sizes = []           #global list to store size of all vectors 
    max_size = -1
    max_setname = None
    max_systems = None

    for setname, innerDict in final_dict.items(): 
        if setname not in ["PA26", "WATER27"]:  
            print("parsing ", setname) 
            innerDict = {k.lower(): v for k, v in innerDict.items()}  #convert to keys to lowercase to prevent name missmatch 
            setname_path = os.path.join(base_path, setname)           # path for setname folder
            ref_path = os.path.join(setname_path, "ref")              # path for ref file of the setname
            rows = parse_ref_file(ref_path)                           # get all rows in the ref file of the setname 

            eigenval_vect_list = []     #list holds all vectors of eigenvals for current setname
            for systems, coeffs, ref_val in rows:
                coulomb_matrices = [innerDict[s.lower()] for s in systems]   #convert to lowercase to prevent name missmatch 
                try:
                    C = combine_cm(coulomb_matrices, coeffs)    # combine 
                except ValueError as e:                         #if shape missmatch
                    S = max(M.shape[0] for M in coulomb_matrices)
                    print(f"--[{setname}] shape mismatch for systems={systems} coeffs={coeffs} - {e}. Using {S}x{S} zero placeholder.")
                    C = np.zeros((S, S), dtype=coulomb_matrices[0].dtype)

                #now diagonalizing the combined matrix (C)
                eigenval_vect = diagonalize_matrix(C)
                eigenval_vect_list.append(eigenval_vect) # add all 'vector of eigenvals'of this setname to a list

                len_eig = len(eigenval_vect)
                sizes.append(len_eig)    #adding its size to out global list of sizes 
                if len_eig > max_size:        #update if larger
                    max_size = len_eig
                    max_setname = setname
                    max_systems = systems
                
            # put all 'vectors of eigenvals' for this setname in a dict 
            reaction_dicts[setname] = eigenval_vect_list

    print("\nDone storing vectors of eigen vals in a Dict....")

    print(f"\n\nLargest eigenvector size: {max_size}")
    print(f"  setname: {max_setname}")
    print(f"  systems: {max_systems}")

    print("\n\nstatistics about length of vectors....")
    print(f"n = {len(sizes)} toal vectors (samples)")
    print(f"min length= {min(sizes)}")
    print(f"max length= {max(sizes)}")
    print(f"mean length= {mean(sizes):.3f}")
    print(f"median length= {median(sizes)}")
    print(f"std = {stdev(sizes):.3f}")


    available_subsets = reaction_dicts.keys()          
    print(f"\n\n{len(available_subsets)} available subsets")       #now we have 53 setnames, since 2 were excluded. 
                                                                  # need to iterate over these in 2nd loop 

    # # 2nd loop 
    # for subset_name in available_subsets:
    #     print(f"Processing subset: {subset_name}......")
    #     subset_path = os.path.join(base_path, subset_name)

    #     try:
    #         print("hiiii")
    #         # # Process all reactions in this subset
    #         # eigenvalue_arrays, ref_values, reaction_metadata = process_subset_reactions(subset_name, subset_path)
            
    #         # # Create final feature matrix
    #         # feature_matrix, targets = create_final_features(eigenvalue_arrays, ref_values, reaction_metadata, subset_name)
            
    #         # # Save results
    #         # max_eigenvalue_size = feature_matrix.shape[1] - 2  # Subtract metadata features
    #         # save_results(feature_matrix, targets, reaction_metadata, subset_name, max_eigenvalue_size)
            
    #         # successful_subsets.append(subset_name)
    #         # print(f"✅ Completed processing {subset_name}")
            
    #     except Exception as e:
    #         print("hiiii")
    #         continue
    
    print(f"\nPipeline completed!")


if __name__ == "__main__":
    main()
