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

## This is a matrix multiplication of: m*k by k*n -> m*n matrix. NOTE, this is a general mat-mul kernel. 
## TODO, remove the debug_tensor.
@triton.jit
def rsddmm_kernel(x_ptr, y_ptr, 
                    out_ptr, debug_tensor_ptr, dTos_linear_trf, dTos_translations, 
                    sTod_linear_trf, sTod_translations, nnzs,
                    m, n, k, trailing_dim, tb_mapping_x, tb_mapping_y, 
                    BLOCK_SIZE_Y : tl.constexpr, BLOCK_SIZE_X : tl.constexpr):
    
    bx = tl.program_id(axis=0)

    ## We first unpack the tb_maps to uncover the top left x and y coordinate.
    bx_start = tl.load(tb_mapping_x+bx, mask=True)
    by_start = tl.load(tb_mapping_y+bx, mask=True)
    bx_start = bx_start.to(tl.int32)
    by_start = by_start.to(tl.int32)

    inner_tile_dim : tl.constexpr = 128

    x_ptrs = by_start*k + tl.arange(0, BLOCK_SIZE_Y)[:,None]*k + tl.arange(0, inner_tile_dim)[None,:]
    y_ptrs = bx_start + tl.arange(0, inner_tile_dim)[:,None]*n + tl.arange(0, BLOCK_SIZE_X)[None,:]

    accumulator = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)

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

    accumulator = accumulator.to(output_ptrs.dtype.element_ty)

    ## This uses the sTOd affine-indices for scaling the indices of where to store.
    linear_transforms = tl.load(sTod_linear_trf+by_start+tl.arange(0,BLOCK_SIZE_Y), 
                                mask=by_start+tl.arange(0,BLOCK_SIZE_Y)<m, other=0.0)
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
    output_ptrs = output_ptrs.to(tl.int64)

    ## Step 4. 
    ## First, we check for OOB conditions due to translations.
    output_mask = col_idx >= 0
    ## Next, we check if a column index maps to a valid contraction (modulo check).

    ## Unfortunately, broadcast semantics don't apply to the "==" operator.
    ##    So we have to do design a new bolean operator: ~op1 && ~op2
    ## Intresting bug. Setting interpreter=True, 
    ##   tl.int64 throws an error whilst torch.int64 does not. Turning off interpreter mode, the reverse is true.
    #op_one = (col_idx % linear_transforms[:, None]).to(torch.int64) > 0
    #op_two = (col_idx % linear_transforms[:,None]).to(torch.int64) < 0
    op_one = (col_idx % linear_transforms[:, None]).to(tl.int64) > 0
    op_two = (col_idx % linear_transforms[:,None]).to(tl.int64) < 0
    output_mask = output_mask & ((not op_one) & (not op_two))
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

def rsddmm_launcher(x : torch.Tensor, 
                    y : torch.Tensor,
                    mask : list[list[int]], GPU_ID : int, 
                    BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ## First we create the output tensor.

    ## compute the trailing dimension length of the ACSR.
    trailing_dim : int = max(list(map(lambda x: reduce(lambda a,b: a+b, x, 0), mask)))

    output : torch.Tensor = torch.empty((len(mask), trailing_dim), dtype=torch.float16).to(GPU_ID)

    debug_tensor : torch.Tensor = torch.empty((len(mask), trailing_dim)).to(GPU_ID)

    ## Next, we compute the tiling blocks.
    tb_map_x, tb_map_y = gen_block_mappings(mask, BLOCK_SIZE_Y, BLOCK_SIZE_X, GPU_ID)

    ## Finally, we instantiate the acsr metadata.
    dTos_linear_transformations, dTos_translations, sTod_linear_transformations, sTod_translations, nnzs = create_acsr(mask, BLOCK_SIZE_Y, GPU_ID)

    assert tb_map_x.shape == tb_map_y.shape, "Incorrect tiling arrangement!"

    ## Finally, we can launch the kernel
    grid_dim = (tb_map_x.shape[0],)

    torch.cuda.synchronize()
    rsddmm_start = time.time()
    rsddmm_kernel[grid_dim](x,y,output, debug_tensor, 
                            dTos_linear_transformations,dTos_translations, 
                            sTod_linear_transformations,sTod_translations,nnzs,
                            x.shape[0],y.shape[1],x.shape[1], trailing_dim, tb_map_x, tb_map_y,
                            BLOCK_SIZE_Y=BLOCK_SIZE_Y, BLOCK_SIZE_X=BLOCK_SIZE_X, num_warps=2)
    torch.cuda.synchronize()
    rsddmm_end = time.time()
    print(f'time taken splat: {(rsddmm_end - rsddmm_start):.15f}')

    ## We return the sTod arrays for correctness checking only.
    return (output, sTod_linear_transformations, sTod_translations, nnzs)

def truth(x : torch.Tensor, y: torch.Tensor, GPU_ID : int) -> torch.Tensor:
    return torch.matmul(x,y).to(GPU_ID)

## Define checker later, figure out good practice. TODO.
def is_correct(out_torch : torch.Tensor, out_rsddmm : torch.Tensor, 
                sTod_linear_transofrmations : torch.Tensor, 
                sTod_translations : torch.Tensor, nnzs: torch.Tensor, mask : list[list[int]]) -> bool:
    out_torch_list = out_torch.tolist() ## Question: What are the sizes of these tensors?!
    out_rsddmm_list = out_rsddmm.tolist()
    sTod_linear_transformations_list = sTod_linear_transofrmations.tolist()
    sTod_translations_list = sTod_translations.tolist()
    nnzs_list = nnzs.tolist()

    num_deviations : int = 0
    mse_error : float = 0

    for row in range(len(mask)):
        for nnz_col_id in range(len(out_rsddmm_list[0])):
            ## We convert to the dense index.
            dense_col_id : int = round(nnz_col_id * sTod_linear_transformations_list[row] + sTod_translations_list[row])
            if nnz_col_id < nnzs_list[row] and abs(out_torch_list[row][dense_col_id] - out_rsddmm_list[row][nnz_col_id]) > 1e-3:
                #print(f'failed at: {row} {dense_col_id}')
                mse_error += abs(out_torch_list[row][dense_col_id] - out_rsddmm_list[row][nnz_col_id])
                num_deviations += 1

    if num_deviations > 0:
        print(f'test case failed average mse: {mse_error}')
        return False
    else:
        print(f'test case passed!')
        return True

## Multiply a: m*k and k*n matrix.
def test(m: int, k : int, n : int, mask : list[list[int]], GPU_ID : int, BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int):
    ## Some simple test-cases for me to try out.
    assert m==n, "We only need to consider the case when m=n."
    left : torch.Tensor = torch.randn((m,k),dtype=torch.float16).to(GPU_ID)
    right : torch.Tensor = torch.randn((k,n)).to(GPU_ID)
    ## Compare against pytorch's einsum as ground truth.
    torch_output = truth(left, right, GPU_ID)

    ## Call the rsddmm launcher.
    rsddmm_output, sTod_linear_transformations, sTod_translations, nnzs = rsddmm_launcher(left, right, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)
    is_correct(torch_output, rsddmm_output, sTod_linear_transformations, sTod_translations, nnzs, mask)

if __name__ == "__main__":
    ## Just a sample unit test over here.

    ## Small unit-tests
    def test_one():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 2 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_two():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 5 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_three():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 7 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_four():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 16
        m: int = 16
        k: int = 16
        p: int = 5 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_five():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 16
        m: int = 16
        k: int = 16
        p: int = 16 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_six():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 10 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_seven():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 20 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_eight():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 32 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_nine():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 128
        m: int = 128
        k: int = 128
        p: int = 57 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

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

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_eleven():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 328 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_twelve():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 512 ## Sparsity parameter.
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    test_ten()
    test_eleven()
    test_twelve()
