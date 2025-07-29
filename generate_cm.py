from ase import Atoms
from dscribe.descriptors import CoulombMatrix
from ase.data import chemical_symbols
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# path to coord file
coord_path = '..../ACONF/B_G/coord'


def parse_coord_file(filepath):
    """Parses a coord file and returns an ASE Atoms object"""
    atoms = []
    positions = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('$') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            try:
                x, y, z = map(float, parts[:3])
                symbol = parts[3].lower().capitalize()
                if symbol not in chemical_symbols:
                    continue
                positions.append([x, y, z])
                atoms.append(symbol)
            except ValueError:
                continue
    return Atoms(symbols=atoms, positions=positions)


molecule = parse_coord_file(coord_path)

# generate coulomb Matrix (output is flattened)
cm = CoulombMatrix(n_atoms_max=len(molecule), permutation="sorted_l2")
coulomb_matrix = cm.create(molecule)

 # convert to 2d 
n = len(molecule)  # num atoms
cm_matrix = coulomb_matrix.reshape((n, n))
print(cm_matrix)

# display heatmap
sns.heatmap(cm_matrix, cmap="viridis")
plt.title("Coulomb Matrix Heatmap")
plt.savefig("cm_heatmap.png")