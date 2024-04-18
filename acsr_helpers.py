"""
This instantiates the ACSR metadata in memory as required by triton kernels.
"""
import triton
import triton.language as tl
import torch
import numpy as np
from dataclasses import dataclass
from typing import Type, TypeAlias
from functools import reduce

## Here we have all the type-aliases used within this script. 
##   Only use if current script fails remove otherwise.

## Helper mask generators for unit testing and actual performance analysis.
## s -> sequence length, p -> sparsity_parameter.
def create_blocked_mask(s : int, p  : int):
    mask = [[0 for _ in range(s)] for _ in range(s)]

    for i in range(s):
        for j in range(s):
            if i < p:  ## We are dealing with the first block here.
                if j < p:
                    mask[i][j] = 1 
            else:
                ## Now we have to compute if (i,j) belong to a block.
                block_num = i // p 
                start_col = (block_num-1)*p
                end_col = start_col + 2*p
                if start_col <= j and  j < end_col:
                    mask[i][j] = 1

    return mask

## This gives the mapping from dense -> sparse.
@dataclass
class AffineIndices:
    linear_transformation : float 
    translation : int
    nnz : int

## This gives the mapping from sparse -> dense.
@dataclass
class AffineIndicesInt:
    linear_transformation : int
    translation : int
    nnz : int

class ACSR:
    def __init__(self, 
                affine_indices :  list[Type[AffineIndices]] , 
                affine_indices_int : list[Type[AffineIndicesInt]]):
        self.affine_indices = affine_indices
        self.affine_indices_int = affine_indices_int

    def get_dTos_linear_transformations(self) -> list[float]:
        return list(map(lambda x: x.linear_transformation, self.affine_indices))

    def get_sTod_linear_transformations(self) -> list[int]:
        return list(map(lambda x: x.linear_transformation, self.affine_indices_int))

    def get_dTos_translations(self) -> list[float]:
        return list(map(lambda x : x.translation, self.affine_indices))

    def get_sTod_translations(self) -> list[int]:
        return list(map(lambda x : x.translation, self.affine_indices_int))

    def get_nnzs(self) -> list[float]:
        return list(map(lambda x: x.nnz, self.affine_indices))

## We create all the necessary metadata required for the ACSR.
def instantiate_metadata(mask, BLOCK_HEIGHT : int):
    affine_indices_dTos : list[Type[AffineIndices]] = [] ## Dense to sparse mapping.
    affine_indices_sToD : list[Type[AffineIndicesInt]] = [] ## Sparse to dense mapping.
    nnzs : list[int] = [] ## Number of non-zero values.
    ## TODO, need to add optimisations: span-specialisation and transformation-alignment.
    for row in len(mask):
        ## We grab the first two non-zero elements.
        a = -1
        b = -1
        for idx, col in enumerate(mask[row]):
            if col == 1 and a ==-1:
                a = idx
            elif col == 1 and a != -1 and b == -1:
                b = idx
                break

        assert a != -1 and b != -1, "Incorrect unpacking of affine-indices."

        ## Here we solve a system of linear equations.
        affine_indices = np.linalg.solve(np.array([a, 1], [b, 1]), np.array([0,1]))
        curr_nnz = reduce(lambda x,y : x+y, mask[row],0)
        affine_indices_dTos.append(AffineIndices(affine_indices[0],affine_indices[1], curr_nnz))
        affine_indices_sToD.append(AffineIndicesInt(round(1/affine_indices[0]), -affine_indices[1], curr_nnz))
        nnzs.append(curr_nnz)

        return ACSR(affine_indices_dTos, affine_indices_sToD)

## Now we have to send everything to the GPU!
def create_acsr(mask : list[list[int]], BLOCK_HEIGHT : int):
    acsr : ACSR = instantiate_metadata(mask, BLOCK_HEIGHT)
    ## We create 5 torch arrays to give to the GPU.
    dTos_linear_transformations = torch.FloatTensor(acsr.get_dTos_linear_transformations())
    dTos_translations = torch.Tensor(acsr.get_dTos_translations())
    sTod_linear_transformations = torch.Tensor(acsr.get_sTod_linear_transformations())
    sTod_translations = torch.Tensor(acsr.get_sTod_translations())
    nnzs = torch.Tensor(acsr.get_nnzs())
    return (dTos_linear_transformations,dTos_translations,
                sTod_linear_transformations,sTod_translations,nnzs)  


if __name__ == "__main__":
    ## Over here, some simple unit tests.
    n = 10
    m = 10
    k = 10
    p = 2
    BLOCK_HEIGHT = 2
    mask = create_blocked_mask(n, p)
    a,b,c,d,e = create_acsr(mask, BLOCK_HEIGHT)
    print(f'dTos a: {a}')
    print(f'dTos b: {b}')
    print(f'sTod a: {c}')
    print(f'sTod b: {d}')
    print(f'nnzs: {e}')





