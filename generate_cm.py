import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import pickle

from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.data import chemical_symbols
from dscribe.descriptors import CoulombMatrix

current_dir = os.path.dirname(os.path.abspath(__file__))

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
    {'setname': {'subfolder': Coulomb matrix}} from .xyz files in subfolders
    optionally store heatmap of each coulomb matrix 
    TO-Do : create dict containing all 55 setnames"""

    setname = os.path.basename(os.path.normpath(setname_path))  #get setname 
    result = {setname: {}}

    for subfolder in sorted(os.listdir(setname_path)):
        print("making cm for ", subfolder)
        subdir = os.path.join(setname_path, subfolder)

        if not os.path.isdir(subdir):
            continue

        xyz_file = [f for f in os.listdir(subdir) if f.lower().endswith(".xyz")][0]
        xyz_path = os.path.join(subdir, xyz_file)

        cm_matrix = create_cm(xyz_path) 
        result[setname][subfolder] = cm_matrix #store in dict

        if save_heatmaps: # creates a folder and store all heatmaps for ACONF 
            heatmap_dir = os.path.join(current_dir, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            heatmap_path = os.path.join(heatmap_dir, f"{subfolder}.png")
            sns.heatmap(cm_matrix, cmap="viridis")
            plt.title(f"Coulomb Matrix: {subfolder}")
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()

    return result



BASE_PATH = "....../new_structures"
setname_path = os.path.join(BASE_PATH, "ACONF")

final_dict = create_dict(setname_path, save_heatmaps=False)

# save dictionary as pickle
with open(os.path.join(current_dir, "final_dict.pkl"), "wb") as f:
    pickle.dump(final_dict, f)

# print(final_dict)

# TODO: 
# make a function similar to create_dict to handle all 55 setnames or 
# make a loop to run create_dict for all setnames and append them all to a single dict

# list of 55 setnames ? (double check )
dblist = [
    "ACONF","ADIM6","AHB21","AL2X6","ALK8","ALKBDE10","Amino20x4","BH76","BH76RC","BHDIV10","BHPERI","BHROT27",
    "BSR36","BUT14DIOL","C60ISO","CARBHB12","CDIE20","CHB6","DARC","DC13","DIPCS10","FH51","G21EA","G21IP","G2RC","HAL59",
    "HEAVY28","HEAVYSB11","ICONF","IDISP","IL16","INV24","ISO34","ISOL24","MB16-43","MCONF","NBPRC","PA26",
    "PArel","PCONF21","PNICO23","PX13","RC21","RG18","RSE43","S22","S66","SCONF","SIE4x4","TAUT15","UPU23","W4-11","WATER27","WCPT18","YBDE18"
]


