"""
This is a fun implementation of the RSDDMM kernel described in my paper, SPLAT.
"""

import triton
import triton.language as tl
from acsr_helpers import create_blocked_mask, create_acsr
from functools import reduce
import torch 
import pdb
import time
from typing import Any

## This is a matrix multiplication of: m*k by k*n -> m*n matrix. NOTE, this is a general mat-mul kernel. 
@triton.jit
def rsddmm_kernel(x_ptr, y_ptr, 
                    out_ptr, dTos_linear_trf, dTos_translations, 
                    sTod_linear_trf, sTod_translations, nnzs,
                    m, n, k, trailing_dim, tb_mapping_x, tb_mapping_y, 
                    BLOCK_SIZE_Y : tl.constexpr, BLOCK_SIZE_X : tl.constexpr):
    
    bx = tl.program_id(axis=0)
    by = tl.program_id(axis=1)
    batch_head_offset_x_input = by * m * k
    batch_head_offset_y_input = by * n * k
    batch_head_offset_output = by * m * trailing_dim

    ## We first unpack the tb_maps to uncover the top left x and y coordinate.
    bx_start = tl.load(tb_mapping_x+bx, mask=True)
    by_start = tl.load(tb_mapping_y+bx, mask=True)
    bx_start = bx_start.to(tl.int32)
    by_start = by_start.to(tl.int32)

    inner_tile_dim : tl.constexpr = 128

    x_ptrs = batch_head_offset_x_input + by_start*k + tl.arange(0, BLOCK_SIZE_Y)[:,None]*k + tl.arange(0, inner_tile_dim)[None,:]
    y_ptrs = batch_head_offset_y_input + bx_start + tl.arange(0, inner_tile_dim)[:,None]*n + tl.arange(0, BLOCK_SIZE_X)[None,:]

    accumulator = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=out_ptr.dtype.element_ty)

    for i in range(tl.cdiv(k, inner_tile_dim)):
        
        ## Let's do this naively at first.
        mask_x_ptrs = i*inner_tile_dim + tl.arange(0, inner_tile_dim)[None,:] < k ## The first constraint
        mask_x_ptrs = mask_x_ptrs & (tl.arange(0, BLOCK_SIZE_Y)[:,None] + by_start < m)
        mask_y_ptrs = i*inner_tile_dim + tl.arange(0, inner_tile_dim)[:, None] < k
        mask_y_ptrs = mask_y_ptrs & (tl.arange(0, BLOCK_SIZE_X)[None, :] + bx_start < n)
        x_tile = tl.load(x_ptr + x_ptrs, mask=mask_x_ptrs, other=0.0)
        y_tile = tl.load(y_ptr + y_ptrs, mask=mask_y_ptrs, other=0.0)

        accumulator += tl.dot(x_tile, y_tile)

        ## Increment x and y pointers here now.
        x_ptrs += inner_tile_dim
        y_ptrs += inner_tile_dim*n

    accumulator = accumulator.to(out_ptr.dtype.element_ty)

    ## This uses the sTOd affine-indices for scaling the indices of where to store.
    linear_transforms = tl.load(sTod_linear_trf+by_start+tl.arange(0,BLOCK_SIZE_Y), 
                                mask=by_start+tl.arange(0,BLOCK_SIZE_Y)<m, other=1.0)
    translations = tl.load(sTod_translations+by_start+tl.arange(0, BLOCK_SIZE_Y),
                           mask=by_start+tl.arange(0,BLOCK_SIZE_Y)<m,other=0.0)
    nnz = tl.load(nnzs+by_start+tl.arange(0,BLOCK_SIZE_Y), 
                  mask=by_start+tl.arange(0,BLOCK_SIZE_Y)<m, other=0.0)
    
    ## Now, we have to use these to recover the true-indices.

    ## We do this in 5 steps. 
    ## First: we compute the col_indices pertinent to this TB.
    ## Second: we scale the col_indices using the linear_transforms and translations array.
    ## Third: We convert the col_indices into ptrs.
    ## Fourth: We generate the mask.
    ## Fifth: We store into the ACSR array.

    ## Step 1

    ## Interestingly, this line throws a ValueError thinking its not wrapped
    ##   within a trion jitted function. We use tl.zeros instead.
    #col_idx = tl.full((BLOCK_SIZE_Y,), 0, tl.int32)
    col_idx = tl.zeros((BLOCK_SIZE_Y,), dtype=tl.int32)
    col_idx = col_idx[:,None] + tl.arange(0, BLOCK_SIZE_X)[None,:] + bx_start 

    ## Step 2
    col_idx /= linear_transforms[:,None] 
    ## Intresting bug. Setting interpreter=True, 
    ##   tl.int64 throws an error whilst torch.int64 does not. Turning off interpreter mode, the reverse is true.
    #col_idx -= translations[:,None].to(torch.int64)
    col_idx -= translations[:,None].to(tl.int64)

    ## Step 3 
    output_ptrs = col_idx + tl.arange(0, BLOCK_SIZE_Y)[:,None]*trailing_dim + by_start*trailing_dim
    ## Type casting required for tl.store compatibililty.
    ## Intresting bug. Setting interpreter=True, 
    ##   tl.int64 throws an error whilst torch.int64 does not. Turning off interpreter mode, the reverse is true.
    #output_ptrs = output_ptrs.to(torch.int64)
    output_ptrs = output_ptrs.to(tl.int64) + batch_head_offset_output

    ## Step 4. 
    ## First, we check for OOB conditions due to translations.
    output_mask = col_idx >= 0
    ## Next, we check if a column index maps to a valid contraction (modulo check).

    ## Unfortunately, broadcast semantics don't apply to the "==" operator.
    ##    So we have to do design a new bolean operator: ~op1 && ~op2
    '''For some reason, this is no longer working. We replace it with the equivalent col_idx % linear_transforms[:, None] == 0 check.
    op_one = (col_idx % linear_transforms[:, None]).to(tl.int64) > 0
    op_two = (col_idx % linear_transforms[:,None]).to(tl.int64) < 0
    output_mask = output_mask & ((not op_one) & (not op_two))
    '''
    output_mask = output_mask & (col_idx % linear_transforms[:, None].to(tl.int64) == 0)
    ## Lastly, we check for OOB due to exceeding nnz count.
    output_mask = output_mask & (col_idx < nnz[:,None])

    tl.store(out_ptr + output_ptrs, accumulator, mask=output_mask)

def naive_block_mappings(mask : list[list[int]], BLOCK_HEIGHT : int, BLOCK_WIDTH : int, GPU_ID : int) -> tuple[torch.Tensor, torch.Tensor]:
    x_coords = []
    y_coords = []

    ## We populate the anchor points.
    ##   We place x coord in x_coords
    ##   We place y coord in y_coords
    for block_number in range(max((len(mask)//BLOCK_HEIGHT), 1)):
        ## Now we have to find the min and max col_idxs for the block_number. ##
        min_col_idx = len(mask[0])
        max_col_idx = 0

        for row in range(block_number*BLOCK_HEIGHT, (block_number+1)*BLOCK_HEIGHT):
            for col in range(len(mask[0])):
                if row < len(mask): ## Check required due to irregular boundary conditions.
                    if mask[row][col]:
                        min_col_idx = min(min_col_idx, col)
                        max_col_idx = max(max_col_idx, col)

        ## Now, after we have found min_col_idx & max_col_idx, 
        ##    we compute the thread-block anchor-points.
        curr_idx = min_col_idx
        while curr_idx < max_col_idx:
            x_coords.append(curr_idx)
            y_coords.append(block_number*BLOCK_HEIGHT)
            curr_idx += BLOCK_WIDTH

    assert len(x_coords) == len(y_coords) and len(x_coords) > 0, "Issues with generating arrangement!"

    return (torch.tensor(x_coords, dtype=torch.int32).to(GPU_ID), torch.tensor(y_coords, dtype=torch.int32).to(GPU_ID))

## for now, we just do a simple naive tiling, TODO, change to SPLAT's special tiling later.
def gen_block_mappings(mask : list[list[int]], BLOCK_HEIGHT : int, 
        BLOCK_WIDTH : int, GPU_ID : int, is_naive : bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    return naive_block_mappings(mask, BLOCK_HEIGHT, BLOCK_WIDTH, GPU_ID)


def rsddmm_preamble(mask : list[list[int]], output_shape: tuple[int], BLOCK_SIZE_X : int,
                    BLOCK_SIZE_Y : int, GPU_ID : int, out_dtype : torch.dtype):

    output : torch.Tensor = torch.empty((output_shape), dtype=out_dtype).to(GPU_ID)

    ## Next, we compute the tiling blocks.
    tb_map_x, tb_map_y = gen_block_mappings(mask, BLOCK_SIZE_Y, BLOCK_SIZE_X, GPU_ID)

    assert tb_map_x.shape == tb_map_y.shape, "Incorrect tiling arrangement!"

    ## Finally, we can launch the kernel
    grid_dim = (tb_map_x.shape[0],output_shape[0]*output_shape[1])

    return (
        output, grid_dim, tb_map_x, tb_map_y
    )

def rsddmm_launcher(x : torch.Tensor, y : torch.Tensor, output : torch.Tensor,
                    dTos_linear_transformations : torch.Tensor, dTos_translations : torch.Tensor,
                    sTod_linear_transformations : torch.Tensor, sTod_translations : torch.Tensor,
                    trailing_dim : int, nnzs : torch.Tensor, grid_dim : tuple[int],
                    tb_map_x : torch.Tensor, tb_map_y : torch.Tensor, 
                    BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    rsddmm_start = time.time()
    rsddmm_kernel[grid_dim](x,y,output, 
                            dTos_linear_transformations,dTos_translations, 
                            sTod_linear_transformations,sTod_translations,nnzs,
                            x.shape[2],y.shape[3],x.shape[3], trailing_dim, tb_map_x, tb_map_y,
                            BLOCK_SIZE_Y=BLOCK_SIZE_Y, BLOCK_SIZE_X=BLOCK_SIZE_X, num_warps=2)
    rsddmm_end = time.time()
    print(f'time taken splat: {(rsddmm_end - rsddmm_start):.15f}')
    print(f'rsddmm kernel output shape: {output.shape}')
    ## We return the sTod arrays for correctness checking only.
    return (output, sTod_linear_transformations, sTod_translations, nnzs)

def truth(x : torch.Tensor, y: torch.Tensor, GPU_ID : int) -> torch.Tensor:
    return torch.einsum('bnqd, bndk -> bnqk', x, y)
    #return torch.matmul(x,y).to(GPU_ID)

## Define checker later, figure out good practice. TODO.
def is_correct(out_torch : torch.Tensor, out_rsddmm : torch.Tensor, 
                sTod_linear_transofrmations : torch.Tensor, 
                sTod_translations : torch.Tensor, nnzs: torch.Tensor, 
                batch_size : int, num_heads : int,
                mask : list[list[int]]) -> bool:
    out_torch_list = out_torch.tolist() ## Question: What are the sizes of these tensors?!
    out_rsddmm_list = out_rsddmm.tolist()
    sTod_linear_transformations_list = sTod_linear_transofrmations.tolist()
    sTod_translations_list = sTod_translations.tolist()
    nnzs_list = nnzs.tolist()

    num_deviations : int = 0
    mse_error : float = 0

    for b in range(batch_size):
        for h in range(num_heads):
            for row in range(len(mask)):
                for nnz_col_id in range(len(out_rsddmm_list[0][0][0])):
                    ## We convert to the dense index.
                    dense_col_id : int = round(nnz_col_id * sTod_linear_transformations_list[row] + sTod_translations_list[row])
                    if nnz_col_id < nnzs_list[row] and abs(out_torch_list[b][h][row][dense_col_id] - out_rsddmm_list[b][h][row][nnz_col_id]) > 1e-3:
                        #print(f'failed at: {row} {dense_col_id}')
                        mse_error += abs(out_torch_list[b][h][row][dense_col_id] - out_rsddmm_list[b][h][row][nnz_col_id])
                        num_deviations += 1

    if num_deviations > 0:
        print(f'test case failed average mse: {mse_error}')
        return False
    else:
        print(f'test case passed!')
        return True

## Multiply a: m*k and k*n matrix.
def test(m: int, k : int, n : int, num_heads : int, batch_size : int, 
         mask : list[list[int]], GPU_ID : int, BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int, out_dtype : torch.dtype):
    ## Some simple test-cases for me to try out.
    assert m==n, "We only need to consider the case when m=n."
    #left : torch.Tensor = torch.randn((m,k),dtype=torch.float32).to(GPU_ID)
    #right : torch.Tensor = torch.randn((k,n),dtype=torch.float32).to(GPU_ID)
    left : torch.Tensor = torch.randint(0, 100, (batch_size,num_heads,m,k),dtype=out_dtype).to(GPU_ID)
    right : torch.Tensor = torch.randint(0, 100, (batch_size,num_heads,k,n),dtype=out_dtype).to(GPU_ID)

    dTos_linear_transformations, dTos_translations, \
    sTod_linear_transformations, sTod_translations, nnzs, \
    acsr_trailing_dimension, _, _ = create_acsr(
        mask, BLOCK_SIZE_X, GPU_ID
        )
    
    output_tensor, grid_dim, \
    tb_map_x, tb_map_y = rsddmm_preamble(mask, (batch_size, num_heads, m, acsr_trailing_dimension), 
                                         BLOCK_SIZE_X, BLOCK_SIZE_Y, GPU_ID, out_dtype)

    ## Call the rsddmm launcher.
    rsddmm_output, sTod_linear_transformations, \
        sTod_translations, nnzs = rsddmm_launcher(left, right, output_tensor, 
                                                  dTos_linear_transformations, dTos_translations,
                                                  sTod_linear_transformations, sTod_translations,
                                                  acsr_trailing_dimension, nnzs, grid_dim, 
                                                  tb_map_x, tb_map_y, 
                                                  BLOCK_SIZE_Y, BLOCK_SIZE_X)
    
    ## Verify correctness.
    torch_output = truth(left, right, GPU_ID)
    is_correct(torch_output, rsddmm_output, 
               sTod_linear_transformations, sTod_translations, 
               nnzs, batch_size, num_heads, mask)

if __name__ == "__main__":
    ## Just a sample unit test over here.

    ## Small unit-tests
    def test_one():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 2 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, torch.bfloat16)

    def test_two():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 5 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_three():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 7 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_four():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 16
        m: int = 16
        k: int = 16
        p: int = 5 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_five():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 16
        m: int = 16
        k: int = 16
        p: int = 16 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_six():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 10 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_seven():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 20 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_eight():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 32 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_nine():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 128
        m: int = 128
        k: int = 128
        p: int = 57 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

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
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 256 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_eleven():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 328 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    def test_twelve():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 512 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16
        out_dtype : torch.dtype = torch.bfloat16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X, out_dtype)

    test_ten()
    test_eleven()
    test_twelve()
