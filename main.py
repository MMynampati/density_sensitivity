#!/usr/bin/env python3
"""
Main pipeline for density sensitivity ML analysis.

NEW:
This script implements the complete workflow:
1. Load coulomb matrices from the dictionary 
2. 1st loop - Go through the subsets, and for each reaction in each subset:  
    a) combine matrices according to stoichiometry    
    b) diagonalize the combined matrix   
    c) update max size if exceeded  
    d)append meta data (charge/spin/size of eigenvalue vectoe )
3. Store all in a dictionary 

4. 2nd loop - Go through the subsets again, and for each reaction in each subset:
    a) pad [eigenvalue vector] with zeros to the max size
    b) append charge and mult of product and eigenvalue vector size
    c) add a boolean label for the reaction from SWARM csv
    d) add to the final dataframe

5. Get complete feature matrix and target labels
6. optional - get balanced features and target labels
"""
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from statistics import mean, median, stdev
from typing import List, Dict, Tuple

# # Import functions from our files
from generate_cm import create_cm
from preprocess import parse_ref_file, combine_cm
from diagonalize_matrices import diagonalize_matrix, analyze_eigenvalue_distributions, create_ml_dataframe
from pad_and_metadata import pad_eigenvalue_arrays, parse_info_file, get_product_properties, create_final_features

def main():
    """
    Main pipeline function.
    Processes multiple subsets for density sensitivity analysis.
    """
    print("Density Sensitivity ML Pipeline")

    # Configuration
    base_path = "new_structures"
    # base_path = "/Users/nedamohseni/Downloads/new_structures"

    # load the pickle containing coulomb matrices for all setnames 
    with open("final_dict_allsets.pkl", "rb") as f:
        final_dict = pickle.load(f)

    # 1st loop = loop over setnames (key of final dict), and the coulomb matrices of each setname (inner dict) 
    # - ignoring PA26 since it has product & reactanct shape missmatch issue, ignoring WATER27 since mihira said 
    # go through ref file of each setname, parse and return every row of ref file  ex)  [(['B_T', 'B_G'], [-1, 1], 0.598), ....]
    # for each row of ref file, get respective coulomb matrices and combine based on stochiometry coeffs
    # if could not combine due to shape missmatch, use a zero matrix as a placeholder
    # diagonalize the combined matrix 
    # keep track of largest 'vector of eigenval'
    # store  all 'vectors of eigenvals' for each setname in a dict
    # print largest size of 'vector of eigenvals' and which subset/systems it is from 
    # print summary stats of length of 'vectors of eigenvals' 
    # also storing metadata for each sample. such as ref values / list of system & coeffs / size of eigenval vector 

    reaction_dicts = {}      # dict to store combined matrices and meta data for each reaction (setname)
    all_sizes = []           #global list to store size of all vectors 
    max_size = -1
    max_setname = None
    max_systems = None

    # 1st loop
    for setname, innerDict in final_dict.items(): 
        if setname not in ["PA26", "WATER27"]:     # ignore                  
            print("parsing ", setname, "....") 
            innerDict = {k.lower(): v for k, v in innerDict.items()}  # convert to keys to lowercase to prevent name missmatch 
            setname_path = os.path.join(base_path, setname)           # path for setname folder
            ref_path = os.path.join(setname_path, "ref")              # path for ref file of the setname
            rows = parse_ref_file(ref_path)                           # get all rows in the ref file of the setname 
            info_path = os.path.join(setname_path, "info")            # path for info file of the setname
            info_dict = parse_info_file(info_path)                    # returns a dict containing charge/spin of all systems of this setname 

            eigenval_vect_list = []     # list holds all vectors of eigenvals for current setname
            refs = []                   # list of all ref values for curr setname
            systems_list = []           # list of all system pairs/tuples for curr setname
            coeffs_list = []            # list of all coeff lists for curr setname
            sizes_list = []             # list of length of each eigenvalue vector for current setname
            charge_list = []            # list of 
            spin_list = []              #

            for systems, coeffs, ref_val in rows:                            #looping over each row of ref file for this setname
                coulomb_matrices = [innerDict[s.lower()] for s in systems]   #get coulomb matrix for all systems in this row - convert system name to lowercase to prevent name missmatch 
                try:
                    C = combine_cm(coulomb_matrices, coeffs)                 # combine 
                except ValueError as e:                                      # if shape missmatch (only happens for 4 samples (SIE4x4))
                    S = max(M.shape[0] for M in coulomb_matrices)            #get largest size S
                    print(f"--[{setname}] shape mismatch for systems={systems} coeffs={coeffs} - {e}. Using {S}x{S} zero placeholder.")
                    C = np.zeros((S, S), dtype=coulomb_matrices[0].dtype)    #create a zero vector of size S as placeholder

                #now diagonalizing the combined matrix (C)
                eigenval_vect = diagonalize_matrix(C)
                eigenval_vect_list.append(eigenval_vect) # add all 'vector of eigenvals'of this setname to a list

                len_eig = len(eigenval_vect)      # length
                all_sizes.append(len_eig)         # adding its size to out global list of sizes 
                if len_eig > max_size:            # update max length if larger, keep track of setname/system for max length 
                    max_size = len_eig
                    max_setname = setname
                    max_systems = systems

                #charge/mult 
                charge, spin_mult = get_product_properties(systems, coeffs, info_dict)  #get charge and spin for this row 
                # print(coeffs)
                # print(systems, coeffs)
                # print(charge, spin_mult)

                # meta data 
                refs.append(ref_val)
                systems_list.append(systems)            
                coeffs_list.append(coeffs)              
                sizes_list.append(len(eigenval_vect))   
                charge_list.append(charge)
                spin_list.append(spin_mult)

            # put all 'vectors of eigenvals' and metadata for this setname in a dict 
            reaction_dicts[setname] = {"eigenval_vectors": eigenval_vect_list, "refs": refs,
                                        "systems": systems_list, "coeffs": coeffs_list,           
                                        "sizes": sizes_list, 
                                        "charge" : charge_list, "spin" : spin_list }

    print("\nDone storing vectors of eigen vals in a Dict....")
    # this is how data is stored
    # {"aconf" : {"eigenval_vectors":  [vector 1, 2, ... , vector 15]
    #                 "refs": [ref1, ref2, ....., ref15] 
    #                 "systems": [systems1, 2, ..., systems15 ]
    #                 "coeffs": [coeffs1, 2, ..., coeffs15 ]
    #                 "sizes": [size1, 2, ..., size15 ]
    #                 "charges": [charge1, 2, ..., charge15 ]
    #                 "spin": [spin1, 2, ..., spin15 ]
    #}}

    print(f"\n\nLargest eigenvector size: {max_size}")
    print(f"  setname: {max_setname}")
    print(f"  systems: {max_systems}")

    print("\n\nstatistics about length of vectors....")
    print(f"n = {len(all_sizes)} toal vectors (samples)")
    print(f"min length= {min(all_sizes)}")
    print(f"max length= {max(all_sizes)}")
    print(f"mean length= {mean(all_sizes):.3f}")
    print(f"median length= {median(all_sizes)}")
    print(f"std = {stdev(all_sizes):.3f}\n")

    # to make the pandas df
    rows = []
    for setname, d in reaction_dicts.items():
        if not isinstance(d, dict) or "eigenval_vectors" not in d:
            continue

        n = len(d["eigenval_vectors"])
        for i in range(n):
            rows.append({   "setname": setname,
                            "idx_within_set": i,
                            "vector": d["eigenval_vectors"][i],  
                            "size": d["sizes"][i],
                            "ref": d["refs"][i],
                            "systems": d["systems"][i],
                            "coeffs": d["coeffs"][i], 
                            "charges": d["charge"][i], 
                            "spin": d["spin"][i] })

    df = pd.DataFrame(rows)
    print(df.head())

    #to store as a csv
    df.to_csv("Descriptor1/reaction_vectors.csv", index=False)


    available_subsets = reaction_dicts.keys()          
    print(f"\n\n{len(available_subsets)} available subsets")       #now we have 53 setnames, since 2 were excluded. 
                                                                  # need to iterate over these in 2nd loop 

    feature_matrix = []
    targets = []
    
    # Load SWARM data for S values
    swarm_df = pd.read_csv("../density_sensitivity/all_v2_SWARM.csv")
    # swarm_df = pd.read_csv("/Users/nedamohseni/Desktop/density_sensitivity/all_v2_SWARM.csv")

    pbe_df = swarm_df[swarm_df['calctype'] == 'PBE'].copy()                 #filter to only PBE data
    
    # 2nd loop 
    for index, row in df.iterrows():
        setname = row['setname']
        idx_within_set = row['idx_within_set']
        eigenvalues = row['vector']
        charge = row['charges']
        spin = row['spin']
        size = row['size']

        # add some stats about eigenvals   (mean/std)
        mean_eig = np.mean(eigenvalues)
        std_eig = np.std(eigenvalues)
        
        # 1. Pad eigenvalue vector to max_size (144)
        padded_vector = np.zeros(max_size)
        padded_vector[:len(eigenvalues)] = eigenvalues
        
        # 2. Append charge and spin and size to create feature vector
        feature_vector = np.append(padded_vector, [charge, spin,size])
        # feature_vector = np.append(padded_vector, [charge, spin,size, mean_eig, std_eig])        # uncomment to add mean / std of eigenvals
        feature_matrix.append(feature_vector)
        
        # 3. Get S value from SWARM file
        mask = (pbe_df['setname'] == setname) & (pbe_df['rxnidx'] == idx_within_set + 1)  # +1 because rxnidx is 1-indexed in SWARM file
        matches = pbe_df[mask]
        
        if len(matches) == 0:
            print(f"Warning: No SWARM entry found for {setname} reaction {idx_within_set + 1}")
            # Use a default value or skip this sample
            s_value = 0.0  # Default to not sensitive
        else:
            s_value = matches.iloc[0]['S']
        
        # 4. Create binary target (S >= 2.0 is sensitive)
        binary_target = 1 if s_value >= 2.0 else 0
        targets.append(binary_target)
        
        if index < 5:  # Print first few for verification
            print(f"Sample {index}: {setname} rxn {idx_within_set + 1}, S={s_value:.4f} -> {'SENSITIVE' if binary_target else 'NOT SENSITIVE'}")
    
    # Convert to numpy arrays
    feature_matrix = np.array(feature_matrix)
    targets = np.array(targets)

    print(feature_matrix[:5])
    print(targets[:5])

    print(f"\nFinal feature matrix shape: {feature_matrix.shape}")
    print(f"Final targets shape: {targets.shape}")
    print(f"Class distribution: {np.sum(targets)} sensitive, {len(targets) - np.sum(targets)} not sensitive")
    
    # Save the complete dataset
    np.save("Descriptor1/Descriptor1_complete_features.npy", feature_matrix)
    np.save("Descriptor1/Descriptor1_complete_targets.npy", targets)
    print("✅ Saved complete dataset: Descriptor1_complete_features.npy, Descriptor1_complete_targets.npy")


    # OPTIONAL *** BALANCE THE DATASET
   # Create a 1:1 balanced dataset
    num_sensitive = np.sum(targets)
    num_not_sensitive = len(targets) - num_sensitive
    min_class_size = min(num_sensitive, num_not_sensitive)

    sensitive_indices = np.where(targets == 1)[0]
    not_sensitive_indices = np.where(targets == 0)[0]

    #randomly sample sensitive indices to match the number of not sensitive indices
    sensitive_indices = np.random.choice(sensitive_indices, size=min_class_size, replace=False)
    not_sensitive_indices = np.random.choice(not_sensitive_indices, size=min_class_size, replace=False)

    #combine indices
    balanced_indices = np.concatenate([sensitive_indices, not_sensitive_indices])
    balanced_feature_matrix = feature_matrix[balanced_indices]
    balanced_targets = targets[balanced_indices]

    print(f"Balanced feature matrix shape: {balanced_feature_matrix.shape}")
    print(f"Balanced targets shape: {balanced_targets.shape}")
    print(f"Class distribution: {np.sum(balanced_targets)} sensitive, {len(balanced_targets) - np.sum(balanced_targets)} not sensitive")

    #save balanced dataset
    np.save("Descriptor1/Descriptor1_balanced_features.npy", balanced_feature_matrix)
    np.save("Descriptor1/Descriptor1_balanced_targets.npy", balanced_targets)
    print("✅ Saved balanced dataset: Descriptor1_balanced_features.npy, Descriptor1_balanced_targets.npy")
    
    
    
    print(f"\nPipeline completed!")


if __name__ == "__main__":
    main()
