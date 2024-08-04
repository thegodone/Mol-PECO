#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-09-12 16:14:56
# @Author  : mengji (zmj_xy@sjtu.edu.cn)
# @Link    : http://example.org
# @Version : $Id$

import os, torch
import numpy as np
import torch.nn.functional as F
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcCoulombMat
from scipy import sparse as sp
from typing import Any, Iterator, List, Optional, Tuple, Union, cast, IO



# from deepchem data_utils
def pad_array(x: np.ndarray,
              shape: Union[Tuple, int],
              fill: float = 0.0,
              both: bool = False) -> np.ndarray:

    x = np.asarray(x)
    if not isinstance(shape, tuple):
        shape = tuple(shape for _ in range(x.ndim))
    pad = []
    for i in range(x.ndim):
        diff = shape[i] - x.shape[i]
        assert diff >= 0
        if both:
            a, b = divmod(diff, 2)
            b += a
            pad.append((a, b))
        else:
            pad.append((0, diff))
    pad = tuple(pad)  # type: ignore
    x = np.pad(x, pad, mode='constant', constant_values=fill)
    return x


def cal_coulomb_matrix(mol, max_atoms = 28):
    try:
        #featurizer = dc.feat.CoulombMatrix(max_atoms = max_atoms)
        #cou_matrix = featurizer.coulomb_matrix(mol)
        AllChem.EmbedMolecule(mol ,AllChem.ETKDGv3())
        cou_matrix = CalcCoulombMat(mol)
        # convert into an array of array (matrix)
        cou_matrix = np.array(cou_matrix)
        # padding step
        cou_matrix = pad_array(cou_matrix, max_atoms)

        if np.isinf(cou_matrix).sum()!=0:
            print(Chem.MolToSmiles(mol), "invalid coulomb matrix")
            return None, False
        return cou_matrix, True
    except Exception as e:
        return None, False

def laplace_decomp(A, max_freqs = 10):
    # Laplacian
    n = A.shape[0]
    N = sp.diags(A.sum(axis = 0).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L)
    # print(EigVals.shape, EigVecs.shape)
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        EigVecs = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
        
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)

    EigVals = EigVals.repeat(n, 1)
    return EigVals, EigVecs

if __name__ == '__main__':
    smiles = "CCCCOC(=O)CC(CC(=O)OCCCC)(C(=O)OCCCC)OC(=O)C"
    mol = Chem.MolFromSmiles(smiles)
    print(mol.GetNumAtoms())
    mol = Chem.AddHs(mol)
    print(mol.GetNumAtoms())
    cou_matrix = cal_coulomb_matrix(mol)
    print(cou_matrix)
    print(cou_matrix[0].shape)

