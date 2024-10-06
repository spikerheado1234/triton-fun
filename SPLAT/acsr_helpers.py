"""
This instantiates the ACSR metadata in memory as required by triton kernels.
"""
import triton
import triton.language as tl
import torch
import numpy as np
from dataclasses import dataclass
from typing import Type 
from functools import reduce
from math import ceil

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

def create_windowed_mask(s : int, p : int) -> list[list[int]]:
    mask = [[0 for _ in range(s)] for _ in range(s)]

    for i in range(s):
        for j in range(s):
            if i-p <= j and j <= i+p:
                mask[i][j] = 1

    return mask

def create_causal_windowed_mask(s : int, p : int) -> list[list[int]]:
    mask = [[0 for _ in range(s)] for _ in range(s)]

    for i in range(s):
        for j in range(s):
            if i-p <= j and j <= i+p:
                if i >= j:
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
                affine_indices_int : list[Type[AffineIndicesInt]],
                span_specialised_loop_idxs : list[tuple[int]]):
        self.affine_indices = affine_indices
        self.affine_indices_int = affine_indices_int
        self.span_specialised_loop_idxs = span_specialised_loop_idxs

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

    def get_span_specialised_data(self) -> list[tuple[int]]:
        return self.span_specialised_loop_idxs

## We create all the necessary metadata required for the ACSR.
def instantiate_metadata(mask, BLOCK_HEIGHT : int):
    affine_indices_dTos : list[Type[AffineIndices]] = [] ## Dense to sparse mapping.
    affine_indices_sToD : list[Type[AffineIndicesInt]] = [] ## Sparse to dense mapping.
    nnzs : list[int] = [] ## Number of non-zero values.
    span_specialised_loop_data : list[tuple[int]] = [(len(mask[0]), -1) for _ in range(ceil(len(mask) / BLOCK_HEIGHT))]

    ## TODO, need to add optimisation: transformation-alignment.
    for row in range(len(mask)):
        ## We grab the first two non-zero elements.
        a = -1
        b = -1
        curr_nnz = reduce(lambda x,y : x+y, mask[row],0)
        if curr_nnz > 1:
            for idx, col in enumerate(mask[row]):
                if col == 1 and a ==-1:
                    a = idx
                elif col == 1 and a != -1 and b == -1:
                    b = idx
                    break

            assert a != -1 and b != -1, "Incorrect unpacking of affine-indices."

            ## Here we solve a system of linear equations.
            affine_indices = np.linalg.solve(np.array([[a, 1], [b, 1]]), np.array([0,1]))
            affine_indices_dTos.append(AffineIndices(affine_indices[0],affine_indices[1], curr_nnz))
            affine_indices_sToD.append(AffineIndicesInt(round(1/affine_indices[0]), abs(affine_indices[1]), curr_nnz))
            nnzs.append(curr_nnz)
        else:
            affine_indices = (1, 0)
            affine_indices_dTos.append(AffineIndices(affine_indices[0],affine_indices[1], curr_nnz))
            affine_indices_sToD.append(AffineIndicesInt(round(1/affine_indices[0]), abs(affine_indices[1]), curr_nnz))
            nnzs.append(curr_nnz)

        ## Populate data for optimisations.

        ## Populate span_specialisation loop data.
        sTod_trf = round(1/affine_indices[0])
        sTod_translation = abs(affine_indices[1])
        curr_start = span_specialised_loop_data[row // BLOCK_HEIGHT][0]
        curr_end = span_specialised_loop_data[row // BLOCK_HEIGHT][1]
        span_specialised_loop_data[row // BLOCK_HEIGHT] = (min(curr_start, sTod_translation),
                                                           max(curr_end, curr_nnz*sTod_trf + sTod_translation))

    return ACSR(affine_indices_dTos, affine_indices_sToD, span_specialised_loop_data)

## Now we have to send everything to the GPU!
def create_acsr(mask : list[list[int]], BLOCK_HEIGHT : int, GPU_ID : int):
    acsr : ACSR = instantiate_metadata(mask, BLOCK_HEIGHT)
    ## We create 5 torch arrays to give to the GPU.
    dTos_linear_transformations = torch.FloatTensor(acsr.get_dTos_linear_transformations()).to(GPU_ID)
    dTos_translations = torch.Tensor(acsr.get_dTos_translations()).to(GPU_ID)
    sTod_linear_transformations = torch.Tensor(acsr.get_sTod_linear_transformations()).to(GPU_ID)
    sTod_translations = torch.Tensor(acsr.get_sTod_translations()).to(GPU_ID)
    nnzs = torch.Tensor(acsr.get_nnzs()).to(GPU_ID)
    trailing_dim_acsr = max([reduce(lambda a,b: a+b, row, 0) for row in mask])
    ## Metadata for optimisations.
    span_spec_loop_start = torch.Tensor(list(map(lambda x: x[0], acsr.get_span_specialised_data()))).to(GPU_ID)
    span_spec_loop_end = torch.Tensor(list(map(lambda x: x[1], acsr.get_span_specialised_data()))).to(GPU_ID)
    return (dTos_linear_transformations,dTos_translations,
                sTod_linear_transformations,sTod_translations,nnzs,trailing_dim_acsr,
                ## Optimisation metadata.
                span_spec_loop_start, span_spec_loop_end)  


if __name__ == "__main__":
    ## Over here, some simple unit tests.
    n = 10
    m = 10
    k = 10
    p = 2
    BLOCK_HEIGHT = 2
    GPU_ID = 0
    mask = create_blocked_mask(n, p)
    a,b,c,d,e,f = create_acsr(mask, BLOCK_HEIGHT, GPU_ID)
    print(f'dTos a: {a}')
    print(f'dTos b: {b}')
    print(f'sTod a: {c}')
    print(f'sTod b: {d}')
    print(f'nnzs: {e}')
