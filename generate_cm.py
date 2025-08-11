import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.data import chemical_symbols
from dscribe.descriptors import CoulombMatrix


def create_cm(xyz_file):
    '"creates coulomb matrix for a .xyz file"'
    atoms = read(xyz_file, format="xyz")
    n = len(atoms)

    # generate coulomb Matrix (output is flattened)
    cm = CoulombMatrix(n_atoms_max=n, permutation="sorted_l2")  
    cm_flat = cm.create(atoms)  

    #convert to 2d  
    cm_matrix = cm_flat.reshape((n, n))                     
    return cm_matrix


def create_dict(setname_path, save_heatmaps=False):
    """ takes a single folder (setname folder) and Builds a dictionary 
    {'setname_pat'h: {'subfolder': Coulomb matrix}} from .xyz files in subfolders
    optionally store heatmap of each coulomb matrix 
    TO-Do : create dict containing all 55 setnames, store folder name as the keys, instead of paths? """

    result = {setname_path: {}}

    for subfolder in sorted(os.listdir(setname_path)):

        subdir = os.path.join(setname_path, subfolder)

        if not os.path.isdir(subdir):
            continue

        xyz_file = [f for f in os.listdir(subdir) if f.lower().endswith(".xyz")][0]
        xyz_path = os.path.join(subdir, xyz_file)

        cm_matrix = create_cm(xyz_path) 
        result[setname_path][subfolder] = cm_matrix #store in dict

        if save_heatmaps: # creates a folder and store all heatmaps for ACONF 
            script_dir = os.path.dirname(os.path.abspath(__file__))
            heatmap_dir = os.path.join(script_dir, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            heatmap_path = os.path.join(heatmap_dir, f"{subfolder}.png")
            sns.heatmap(cm_matrix, cmap="viridis")
            plt.title(f"Coulomb Matrix: {subfolder}")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

    return result


setname_path = "....../new_structures/ACONF"
final_dict = create_dict(setname_path, save_heatmaps=False)

print(final_dict)



