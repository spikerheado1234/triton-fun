"""
This is a fun implementation of the RSDDMM kernel described in my paper, SPLAT.
"""

import triton
import triton.language as tl
import torch 

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
    

## This is a matrix multiplication of: m*k by k*n -> m*n matrix. NOTE, this is a general mat-mul kernel. 
@triton.jit
def rsddmm_kernel(x_ptr, y_ptr, 
                    out_ptr, acsr_metadata, 
                    m, n, k, BLOCK_SIZE : tl.constexpr):
    pass

## Currently mask is just a 2-d array that we extract all the ACSR metadata from.
def rsddmm_launcher(x : torch.Tensor, y : torch.Tensor, mask):
    pass

def truth(x : torch.Tensor, y: torch.Tensor):
    return torch.einsum('ab,bc -> ac')

## Define checker later, figure out good practice. TODO.
def is_correct(out_torch : torch.Tensor, out_rsddmm : torch.Tensor, mask):
    pass

## Multiply a: m*k and k*n matrix.
def test(m: int, k : int, n : int):
    ## Some simple test-cases for me to try out.
    assert m==n, "We only need to consider the case when m=n."
    mask = create_blocked_mask(m) ## Alt: n.
    left = torch.randn((m,k))
    right = torch.randn((k,n))

    torch_output = truth(left, right)
    rsddmm_output = rsddmm_launcher(left, right)
    assert is_correct(torch_output, rsddmm_output, mask), "Input is not within the threshold of correctness!"