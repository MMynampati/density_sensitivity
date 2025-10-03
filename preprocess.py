import pickle
import os 
import numpy as np 
import pandas as pd


def parse_ref_line(line: str):
    """
    gets a single line in the ref file,
    ex)   1   B_T B_G -1  1   0.598
    Returns a tuple containing (list of systems, list of Stoichiometry coefficients, Ref. value)  
    ex) (['B_T', 'B_G'], [-1, 1], 0.598)
    """

    toks = line.strip().split()
   
    if not toks or toks[0].startswith("#"):  # ignore header
        return None

    row = toks[1:]                 # drop row index (first token)
    ref_value = float(row[-1])     # last token (ref value) 
    data = row[:-1]                # middle tokens : systems(len k) + coeffs(len k) ,  total len : 2k 

    if len(data) % 2 != 0 or len(data) == 0:
        raise ValueError(f"expected even # of tokens in : {line}")

    k = len(data) // 2
    systems = data[:k]                     # split first half
    coeffs  = [int(x) for x in data[k:]]   # split 2nd half

    return systems, coeffs, ref_value


def parse_ref_file(ref_path: str):
    """takes path of a ref file, parese each line and return all lines,
     each line represented as a tuple containing
    (list of systems, list of Stoichiometry coefficients, Ref. value) ."""
    rows = []
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            out = parse_ref_line(line)
            if out is not None:
                rows.append(out)
    return rows


def _sort_by_row_l2(M: np.ndarray) -> np.ndarray:
    """
    Symmetrically sort matrix M by descending row L2 norm.
    Applies the same permutation to rows and columns.
    """
    # row L2 norms
    norms = np.linalg.norm(M, axis=1)
    # sort by descending order, stable to preserve relative order on ties
    order = np.argsort(-norms, kind="mergesort")
    return M[order][:, order]



def _block_diag_repeat(M: np.ndarray, k: int, sign: int) -> np.ndarray:
    """ 
    used to repeat coulomb matrix of a  molecule k times
    k copies of sign*M on a block diagonal (no magnitude scaling).
    """
    n = M.shape[0]
    out = np.zeros((k*n, k*n), dtype=M.dtype)
    for i in range(k):
        out[i*n:(i+1)*n, i*n:(i+1)*n] = sign * M
    return out

def _block_diag_many(blocks):
    """Takes matrices with same stoichiometry coeff sign (+) or (-) and combines them together.
    Works in case of when we have 3 systems/coeffs in a line, e.g. 2 + coeff, 1 - coeff 
    or 2 - coeff, 1 + coeff.
    """
    # if a single matrix for + or -, return itself
    if len(blocks) == 1: 
        return blocks[0]

    # adds shape of matrices with same sign 
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n), dtype=blocks[0].dtype)

    off = 0
    for b in blocks:
        m = b.shape[0]
        out[off:off+m, off:off+m] = b
        off += m

    return out

def combine_cm(matrices, coeffs) -> np.ndarray:
    """combine reactant(-) and product(+) matrices.
    if shapes are same and stochiometric coeffs are all +1/-1, calculates a weightd sum 
    (scales reactant matrices by -1), adds it to product matrix(es)

    else if diff shape OR stochiometric coeffs are not +1/-1:
    separates matrices with + coeff vs - coeff and stores in list 
    if multiple + matirces : combines all + matrices (adds them as block diagonals)
    if multiple - matrices : combines all - matrices (adds them as block diagonals)

    output: 
          final combined matrix of a single row in ref file (product + (-)(reactant))
    """
    
    same_shape = len({m.shape for m in matrices}) == 1
    unit_coeffs = all(abs(c) == 1 for c in coeffs)
    # if all systems have same shape and |stochiometric coeff| = 1
    if same_shape and unit_coeffs:
        acc = np.zeros_like(matrices[0])
        for M, c in zip(matrices, coeffs):
            acc += c * M
        return acc
    
    # if not 
    # append matrix to eight + matrix list or - matrix list based on respective coeff sign
    pos_blocks, neg_blocks = [], []
    for M, c in zip(matrices, coeffs):
        if c == 0:
            continue
        k = abs(c)
        E = _block_diag_repeat(M, k, +1)  # repeat only; no sign here (repeat a molecule k times)
        # print(E.shape)
        # print(E)
        (pos_blocks if c > 0 else neg_blocks).append(E)   #assign to either pos or neg

    if not pos_blocks or not neg_blocks:
        raise ValueError("Ref line is unbalanced: need at least one + and one - term.")

    # pack each side
    # if multiple matirces for + or - , add them as block diagonal 
    # at the end there will be 1 matrix of + matrices, one matrix of - matrices, 
    # each will have a shape that is sum of shape of matrices its built from 
    P = _block_diag_many(pos_blocks) 
    N = _block_diag_many(neg_blocks)

    # print("\nshape of p : ", P.shape)
    # print(P)
    # print("\nshape of N : ", N.shape)
    # print(N)
    # print("\n\n\n")

    # symmetric sort by row L2 norm on p & N before subtraction
    P = _sort_by_row_l2(P)
    N = _sort_by_row_l2(N)

    if P.shape != N.shape:
        raise ValueError(f"Shape mismatch after packing sides: P{P.shape} vs N{N.shape}. ")

    #finally return difference
    return P - N

BASE_PATH = "......./new_structures"
# BASE_PATH = "/Users/nedamohseni/Downloads/new_structures"

if __name__ == "__main__":

   # dict of coulomb matrices (all setnames)
    with open("final_dict_allsets.pkl", "rb") as f:
        final_dict = pickle.load(f)

    print(len(final_dict.keys()))

    np.set_printoptions(precision=3, suppress=True, linewidth=200)   # for better visuals
    
    # for checking matrix operations
    print(final_dict['BSR36']['c2h6'].shape)
    print(final_dict['BSR36']['c2h6'])
    print("\n\n")
    print(final_dict['BSR36']['h1'].shape)
    print(final_dict['BSR36']['h1'])
    print("\n\n")
    print(final_dict['BSR36']['ch4'].shape)
    print(final_dict['BSR36']['ch4'])

    # loop over setnames (key of final dict) (setnames)
    reaction_dicts = {} 
    for setname, innerDict in final_dict.items(): 
        if setname not in ["PA26"]:  #there are issues in PA26, i'm skipping it 
            print(setname) 
            innerDict = {k.lower(): v for k, v in innerDict.items()}
            setname_path = os.path.join(BASE_PATH, setname)  # path for setname folder
            ref_path = os.path.join(setname_path, "ref")     # path for ref file of the setname
            rows = parse_ref_file(ref_path)                  # get all rows in the ref file of the setname 

            combined_matrices = []
            refs = []

            for systems, coeffs, ref_val in rows:
                # print(systems,coeffs)
                coulomb_matrices = [innerDict[s.lower()] for s in systems]
                try:
                    C = combine_cm(coulomb_matrices, coeffs)    # combine 
                except ValueError as e:
                    S = max(M.shape[0] for M in coulomb_matrices)
                    print(f"[{setname}] shape mismatch for systems={systems} coeffs={coeffs} - {e}. Using {S}x{S} zero placeholder.")
                    C = np.zeros((S, S), dtype=coulomb_matrices[0].dtype)

                combined_matrices.append(C)                           # add all matrices of this setname to a list
                refs.append(ref_val)                                  # add all ref values of this setname , not sure if they're useful ????
                # break


            # put all combined matrices in a dict 
            reaction_dicts[setname] = {"matrices":  combined_matrices, "refs": refs} 


    print("\n\n\n", reaction_dicts['BSR36']["matrices"][0].shape)   #print first row of aconf 
    print(reaction_dicts['BSR36']["matrices"][0])   #print first row of aconf 

    print("\n\n\n", reaction_dicts['TAUT15']["matrices"][0].shape)   #print first row of aconf 
    print(reaction_dicts['TAUT15']["matrices"][0])   #print first row of aconf 

    from diagonalize_matrices import diagonalize_matrix
    print("\n\nafter diagonalizing: ")
    print(diagonalize_matrix(reaction_dicts['TAUT15']["matrices"][0]))


    # this is how data  (combined matrices) is stored
    # {"aconf" : {"matrices":  [matrix 1, 2, ... , matrix 15]
    #                 "refs": [ref1, ref2, ....., ref15] }}s


    # # prints all the combined matrices : 
    # for k,v in reaction_dicts.items(): 
    #     print(v["matrices"])


    # # storing them into pandas:
    # df = pd.DataFrame({
    #     "matrix": reaction_dicts["ACONF"]["matrices"],  # list of np.ndarrays
    #     "ref":    reaction_dicts["ACONF"]["refs"],      # list of floats
    # })

    # print(df.head())  



