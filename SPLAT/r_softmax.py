import triton
import triton.language as tl
import torch
import time
from typing import Any
from acsr_helpers import create_acsr, create_blocked_mask, create_windowed_mask
from functools import reduce
from math import log2, ceil

import pdb

@triton.jit
def r_softmax_kernel(
    x_ptr, out_ptr, 
    dTos_linear_trf, dTos_translations, 
    sTod_linear_trf, sTod_translations, nnzs,
    m, n, true_trailing_dim : tl.constexpr, power_two_trailing_dim: tl.constexpr, BLOCK_SIZE_X : tl.constexpr
):

    bx = tl.program_id(axis=0)
    ## TODO(ahangupta): use num_blocks in order to have another parameter to tune.
    num_blocks = tl.num_programs(axis=0)
    bx_start = bx*BLOCK_SIZE_X

    ## Extract the metadata out first.
    block_translations = tl.load(
        ## Ptrs
        sTod_translations + tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start < m,
        ## Default value.
        other=0.0
    ).reshape(BLOCK_SIZE_X, 1)

    block_linear_trfs = tl.load(
        ## Ptrs
        sTod_linear_trf + tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start < m,
        ## Default value.
        other=1
    ).reshape(BLOCK_SIZE_X, 1)

    block_nnzs = tl.load(
        ## Ptrs
        nnzs + tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start < m,
        ## Default value.
        other=0.0
    ).reshape(BLOCK_SIZE_X, 1)

    ## We load the data.
    edge_idx = block_nnzs 

    ## RQ: what are the implications of unrolling the loop out into pointers vs. retaining the original loop?
    ##   How will the generated code differ?
    ptrs = bx_start*true_trailing_dim + tl.arange(0, power_two_trailing_dim)[None, :] + tl.arange(0, BLOCK_SIZE_X)[:, None]*true_trailing_dim
    
    mask_ptrs = tl.arange(0, power_two_trailing_dim)[None, :] + tl.zeros((BLOCK_SIZE_X, 1), dtype=tl.int64) < edge_idx
    mask_ptrs = mask_ptrs & (bx_start + tl.arange(0, BLOCK_SIZE_X)[:, None] < m)

    rows = tl.load(x_ptr + ptrs, mask=mask_ptrs, other=-1e9)

    ## For numerical stability.
    max_val = tl.max(rows, axis=-1).reshape(BLOCK_SIZE_X, 1)
    rows -= max_val

    numerator = tl.exp(rows)
    denominator = tl.sum(numerator, axis=-1).reshape(BLOCK_SIZE_X, 1)

    softmax_out = numerator / denominator

    tl.store(out_ptr + ptrs, softmax_out, mask=mask_ptrs)

def rsoftmax_preamble(mask : list[list[int]], output_shape: tuple[int], BLOCK_SIZE_X : int, GPU_ID : int):

    ## We have to pass in the next power of two as the trailing_dim.
    trailing_dim_pow_two = 2**ceil(log2(output_shape[-1]))

    full_shape = output_shape

    ## First we create the output tensor.
    output : torch.Tensor = torch.empty(full_shape, dtype=torch.float32).to(GPU_ID)

    ## Finally, we can launch the kernel
    grid_dim = (triton.cdiv(len(mask), BLOCK_SIZE_X),)

    return (
        grid_dim, output, full_shape, trailing_dim_pow_two
        )

def rsoftmax_launcher(
        x : torch.Tensor, output : torch.Tensor, 
        dTos_linear_transformations : torch.Tensor, dTos_translations : torch.Tensor,
        sTod_linear_transformations : torch.Tensor, sTod_translations : torch.Tensor,
        acsr_trailing_dim_true : int, acsr_trailing_dim_power_two: int, 
        nnzs : torch.Tensor,
        grid_dim : tuple[int], BLOCK_SIZE_X : int
         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


    #torch.cuda.synchronize()
    r_softmax_kernel[grid_dim](x,output,
                        dTos_linear_transformations,dTos_translations, 
                        sTod_linear_transformations,sTod_translations,nnzs,
                        x.shape[0],x.shape[1],acsr_trailing_dim_true, 
                        acsr_trailing_dim_power_two, BLOCK_SIZE_X=BLOCK_SIZE_X, num_warps=2
                        )
    #torch.cuda.synchronize()

    ## We return the sTod arrays for correctness checking only.
    return (output, sTod_linear_transformations, sTod_translations, nnzs)

def is_correct(
        inp_tensor : torch.Tensor, out_rsoftmax : torch.Tensor, 
        sTod_linear_transofrmations : torch.Tensor, sTod_translations : torch.Tensor, 
        nnzs: torch.Tensor, mask : list[list[int]]
        ) -> bool:

    out_rsoftmax_list = out_rsoftmax.tolist()
    sTod_linear_transformations_list = sTod_linear_transofrmations.tolist()
    sTod_translations_list = sTod_translations.tolist()
    nnzs_list = nnzs.tolist()

    num_deviations : int = 0
    mse_error : float = 0

    for row in range(len(mask)):
        curr_sum = 0
        for nnz_col_id in range(len(out_rsoftmax_list[0])):
            ## We convert to the dense index.
            if nnz_col_id < nnzs_list[row]:
                ## Now, we manually compute ground truth.
                curr_sum += out_rsoftmax_list[row][nnz_col_id]
        if abs(curr_sum - 1) >= 1e-3:
            num_deviations += 1
            mse_error += abs(curr_sum - 1)

    if num_deviations > 0:
        print(f'test case failed average mse: {mse_error}')
        return False
    else:
        print(f'test case passed!')
        return True

## Compute the softmax over mask. -> mask is size [m, n]. This is its "dense" size.
def test(
        m: int, n : int, mask : list[list[int]], 
        GPU_ID : int, BLOCK_SIZE_X : int,
        ):

    assert m==n, "We only need to consider the case when m=n."

    ## Create acsr.
    dTos_linear_transformations, dTos_translations, \
    sTod_linear_transformations, sTod_translations, nnzs, trailing_dim_acsr, \
    _, _ = create_acsr(
        mask, BLOCK_SIZE_X, GPU_ID
        )

    ## Call the softmax preamble.
    grid_dim, output, full_shape, trailing_dim_pow_two = rsoftmax_preamble(mask, (m, trailing_dim_acsr), BLOCK_SIZE_X, GPU_ID)

    inp : torch.Tensor = torch.randint(0, 100, full_shape,
                                       dtype=torch.float32).to(GPU_ID)

    ## Finally, launch the triton kernel.
    rspmm_output, sTod_linear_transformations, sTod_translations, nnzs = rsoftmax_launcher(
        inp, output, dTos_linear_transformations, dTos_translations, 
        sTod_linear_transformations, sTod_translations,
        trailing_dim_acsr, trailing_dim_pow_two, nnzs, grid_dim, 
        BLOCK_SIZE_X
        )

    ## Correctness check at the end.
    is_correct(
        inp, rspmm_output, 
        sTod_linear_transformations, 
        sTod_translations, nnzs, mask
        )

if __name__ == "__main__":
    ## Just a sample unit test over here.

    ## Small unit-tests
    def test_one():

        n: int = 10
        m: int = 10
        p: int = 2 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)


    def test_two():

        n: int = 10
        m: int = 10
        p: int = 5 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_three():

        n: int = 10
        m: int = 10
        p: int = 7 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)


    def test_four():

        n: int = 16
        m: int = 16
        p: int = 5 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_five():

        n: int = 16
        m: int = 16
        p: int = 16 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_six():

        n: int = 32
        m: int = 32
        p: int = 10 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_seven():

        n: int = 32
        m: int = 32
        p: int = 20 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_eight():

        n: int = 32
        m: int = 32
        p: int = 32 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_nine():

        n: int = 128
        m: int = 128
        p: int = 57 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    ## These are pretty small tests.
    test_one()
    test_two()
    test_three()
    test_four()
    test_five()
    test_six()
    test_seven()
    test_eight()
    test_nine()

    ## Larger tests.
    def test_ten():

        n: int = 1024
        m: int = 1024
        p: int = 256 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_eleven():

        n: int = 1024
        m: int = 1024
        p: int = 328 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    def test_twelve():

        n: int = 1024
        m: int = 1024
        p: int = 512 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, n, mask, GPU_ID, BLOCK_SIZE_X)

    test_ten()
    test_eleven()
    test_twelve()







