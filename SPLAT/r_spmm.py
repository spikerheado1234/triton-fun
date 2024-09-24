import torch
import triton
import triton.language as tl
from functools import reduce
from typing import Any
import time
from acsr_helpers import create_acsr, create_blocked_mask, create_windowed_mask
import pdb

## This is a spmm kernel with the incoming ACSR (x_ptr) in row-major and row-compressed.
## This represents an: (mxk) x (kxn) matrix multiplication.
## Trailing dimension represents the SPARSE trailing dimension of the ACSR (left-matrix).
@triton.jit
def r_spmm_kernel_row_maj_row_comp(
    x_ptr, y_ptr, 
    out_ptr, dTos_linear_trf, dTos_translations, 
    sTod_linear_trf, sTod_translations, nnzs,
    m, n, k, trailing_dim, 
    ## ACSR metadata for optimisations.
    span_loop_start, span_loop_end,
    BLOCK_SIZE_Y : tl.constexpr, BLOCK_SIZE_X : tl.constexpr
    ):
    ## We will first do this naively.
    ## Extract the blockIdx.x/y indices.
    by = tl.program_id(axis=1)
    bx = tl.program_id(axis=0)
    bz = tl.program_id(axis=2)
    batch_head_offset_output = bz * m * n
    batch_head_offset_sparse_mat = bz * m * trailing_dim
    batch_head_offset_dense_mat = bz * n * k

    ## The anchor point, top-left corner of the TB.
    by_start = by*BLOCK_SIZE_Y
    bx_start = bx*BLOCK_SIZE_X

    ## Next we prepare the x and y pointers.
    inner_tile : tl.constexpr = 128
    ## We use dense_col_idxs and internally convert them to sparse coordinates within the inner loop of this kernel.
    dense_col_idxs = tl.arange(0, inner_tile)[None, :].to(tl.int64) + tl.zeros((BLOCK_SIZE_Y,), dtype=tl.int64)[:, None] 
    y_ptrs = batch_head_offset_dense_mat + bx_start + tl.arange(0, BLOCK_SIZE_X)[None, :] + tl.arange(0, inner_tile)[:, None]*n

    ## We load the ACSR metadata, as this will be used to check properties within the inner sparse loop.
    block_translations = tl.load(
        ## Ptrs
        sTod_translations + tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start < m,
        ## Default value.
        other=0.0
    ).reshape(BLOCK_SIZE_Y, 1)

    block_linear_trfs = tl.load(
        ## Ptrs
        sTod_linear_trf + tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start < m,
        ## Default value.
        other=1
    ).reshape(BLOCK_SIZE_Y, 1)

    block_nnzs = tl.load(
        ## Ptrs
        nnzs + tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start,
        ## Mask
        tl.arange(0, BLOCK_SIZE_Y)[None, :] + by_start < m,
        ## Default value.
        other=0.0
    ).reshape(BLOCK_SIZE_Y, 1)

    accumulator = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)

    ## Load metadata for optimsations.

    ## Opt-1: span-specialisation.
    loop_start : tl.constexpr = tl.load(span_loop_start + tl.program_id(axis=1), mask=True)
    loop_end : tl.constexpr = tl.load(span_loop_end + tl.program_id(axis=1), mask=True)
    ## Opt-2: transformation-alignment. (TODO(ahangupta): finish implementing at a later date.)

    ## Triton ast to ttir throws an exception if we don't re-assign loop_end to a temporary variable. -> This is probably a bug in the lowering process.
    loop_end_temp = loop_end

    for i in range(
        tl.floor(tl.div_rn(loop_start, inner_tile)).to(tl.int32), 
        tl.ceil(loop_end_temp / inner_tile).to(tl.int32)
        ):

        ## Constraint one, OOB along the leading dimensions.
        sparse_x_col_idxs = (tl.div_rn(dense_col_idxs - block_translations, block_linear_trfs)).to(tl.int64)
        mask_x_ptrs = tl.arange(0, BLOCK_SIZE_Y)[:,None] + by_start < m
        mask_x_ptrs = mask_x_ptrs & (sparse_x_col_idxs < block_nnzs)
        mask_x_ptrs = mask_x_ptrs & (sparse_x_col_idxs >= 0)
        ## Constraint two, mapped to valid indices in the ACSR.

        ## First check: linear_trf valid check.
        '''
        op_one = dense_col_idxs % block_linear_trfs > 0
        op_two = dense_col_idxs % block_linear_trfs < 0
        mask_x_ptrs = mask_x_ptrs & ((not op_one) & (not op_two))
        '''
        mask_x_ptrs = mask_x_ptrs & (dense_col_idxs % block_linear_trfs == 0)

        ## Finally, we can convert the x_ptrs to sparse points in the ACSR.
        sparse_x_ptrs = batch_head_offset_sparse_mat + sparse_x_col_idxs + by_start*trailing_dim + i*inner_tile + tl.arange(0, BLOCK_SIZE_Y)[:,None]*trailing_dim 

        ## The mask for the dense matrix. 

        ## First check, OOB from the horizontal (leading dimension).
        mask_y_ptrs = bx_start + tl.arange(0, BLOCK_SIZE_X)[None, :] < n
        ## Second check, OOB from the vertical (trailing dimension)
        mask_y_ptrs  = mask_y_ptrs & (i*inner_tile + tl.arange(0, inner_tile)[:, None] < k)
        ## We load the data.
        x_tile = tl.load(x_ptr + sparse_x_ptrs, mask=mask_x_ptrs, other=0.0)
        y_tile = tl.load(y_ptr + y_ptrs, mask=mask_y_ptrs, other=0.0) ## -> orientation of this isn't particularly good.
        ## Do the matmul and accumulate.
        accumulator += tl.dot(x_tile, y_tile)

        y_ptrs += inner_tile*n
        dense_col_idxs += i*inner_tile

    ## Write-back logic.
    
    ## Now, write-backs are straightfoward since the output is dense.
    write_ptrs = batch_head_offset_output + bx_start + tl.arange(0, BLOCK_SIZE_X)[None, :] + (by_start*n + tl.arange(0, BLOCK_SIZE_Y)[:, None]*n)
    ## Check whether leading dimension OOB.
    write_ptrs_mask = bx_start + tl.arange(0, BLOCK_SIZE_X)[None, :] < m
    write_ptrs_mask = write_ptrs_mask & (by_start + tl.arange(0, BLOCK_SIZE_Y)[:, None] < n)
    tl.store(out_ptr + write_ptrs, accumulator, mask=write_ptrs_mask)

def rspmm_preamble(mask : list[list[int]], output_shape : tuple[int],
                   BLOCK_SIZE_X : int, BLOCK_SIZE_Y : int, GPU_ID : int) -> tuple[torch.Tensor, tuple[int], int]:

    ## Now, left tensor is an ACSR. we need to generate its trailing dimension.
    trailing_dim_acsr = max(
        [reduce(lambda a,b: a+b, row, 0) for row in mask]
        )

    ## First we create the output tensor.
    output : torch.Tensor = torch.empty(output_shape, 
                                        dtype=torch.float32).to(GPU_ID)

    ## Finally, we can launch the kernel
    grid_dim = (triton.cdiv(output_shape[3], BLOCK_SIZE_X),triton.cdiv(output_shape[2], BLOCK_SIZE_Y),output_shape[0]*output_shape[1])

    return (
        output, grid_dim, trailing_dim_acsr
    )

## Over here, x is an ACSR and y is the Values tensor.
def rspmm_launcher(x : torch.Tensor, y : torch.Tensor, output : torch.Tensor,
                   dTos_linear_transformations : torch.Tensor, dTos_translations : torch.Tensor,
                   sTod_linear_transformations : torch.Tensor, sTod_translations : torch.Tensor,
                   span_loop_start : torch.Tensor, span_loop_end : torch.Tensor,
                   acsr_trailing_dim : int, nnzs : torch.Tensor,
                   grid_dim : tuple[int], 
                   BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    rsddmm_start = time.time()
    r_spmm_kernel_row_maj_row_comp[grid_dim](x,y,output, 
                                            dTos_linear_transformations,dTos_translations, 
                                            sTod_linear_transformations,sTod_translations,nnzs,
                                            x.shape[2],y.shape[3],y.shape[2], acsr_trailing_dim,
                                            ## ACSR metadata for optimisations.
                                            span_loop_start, span_loop_end,
                                            BLOCK_SIZE_Y=BLOCK_SIZE_Y, BLOCK_SIZE_X=BLOCK_SIZE_X, num_warps=2)
    #torch.cuda.synchronize()
    rsddmm_end = time.time()
    print(f'time taken splat: {(rsddmm_end - rsddmm_start):.15f}')
    print(f'rspmm kernel output shape: {output.shape}')
    ## We return the sTod arrays for correctness checking only.
    return (output, sTod_linear_transformations, sTod_translations, nnzs)

def is_correct(
        left_tensor : torch.Tensor, right_tensor : torch.Tensor,
        out_rspmm : torch.Tensor, sTod_linear_transofrmations : torch.Tensor, 
        sTod_translations : torch.Tensor, nnzs: torch.Tensor, num_heads : int, batch_size : int,
        mask : list[list[int]]
        ) -> bool:

    left_tensor_list = left_tensor.tolist()
    right_tensor_list = right_tensor.tolist()
    out_rspmm_list = out_rspmm.tolist()
    sTod_linear_transformations_list = sTod_linear_transofrmations.tolist()
    sTod_translations_list = sTod_translations.tolist()
    nnzs_list = nnzs.tolist()

    num_deviations : int = 0
    mse_error : float = 0

    def dot_product(left, right, row, col, linear_trf, translation, nnzs):
        ## How pythonic can really be here?
        accum = 0
        for i in range(nnzs):
            inner_idx = round(i*linear_trf + translation)  ## unsure if this is entirely correct, TODO(ahangupta): double-confirm this.
            accum += left[row][i] * right[inner_idx][col]

        return accum

    for b in range(batch_size):
        for h in range(num_heads):
            for dense_row in range(len(mask)):
                for dense_col in range(len(right_tensor_list[0][0][0])):
                    ## We convert to the dense index.
                    sparse_col = round(dense_col / sTod_linear_transformations_list[dense_row] - sTod_translations_list[dense_row])
                    ## Legality check for the sparse_col coordinate.
                    if sparse_col < nnzs_list[dense_row] and sparse_col >= 0 and dense_col % round(sTod_linear_transformations_list[dense_row]) == 0:
                        ## Now, we manually compute ground truth.
                        manual_answer = dot_product(left_tensor_list[b][h], right_tensor_list[b][h], 
                                                    dense_row, dense_col, 
                                                    sTod_linear_transformations_list[dense_row],
                                                    sTod_translations_list[dense_row], 
                                                    round(nnzs_list[dense_row])
                                                    )
                        if abs(manual_answer - out_rspmm_list[b][h][dense_row][dense_col]) > 1e-3:
                            mse_error += abs(manual_answer - out_rspmm_list[b][h][dense_row][dense_col])
                            num_deviations += 1

    if num_deviations > 0:
        print(f'test case failed average mse: {mse_error}')
        return False
    else:
        print(f'test case passed!')
        return True

## Multiply a: m*k and k*n matrix.
def test(m: int, k : int, n : int, num_heads : int, 
         batch_size : int, mask : list[list[int]], GPU_ID : Any, BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int):
    ## Some simple test-cases for me to try out.
    assert m==n, "We only need to consider the case when m=n."

    ## Create left and right random inputs.
    trailing_dim_acsr = max([reduce(lambda a,b: a+b, row, 0) for row in mask])

    left : torch.Tensor = torch.randint(0, 100, (
        batch_size, num_heads, m, trailing_dim_acsr
        ),dtype=torch.float32).to(GPU_ID)

    right : torch.Tensor = torch.randint(0, 100, (batch_size, num_heads, k,n),dtype=torch.float32).to(GPU_ID)

    ## Create acsr.
    dTos_linear_transformations, dTos_translations, \
    sTod_linear_transformations, sTod_translations, nnzs, _, \
    span_loop_start, span_loop_end = create_acsr(
        mask, BLOCK_SIZE_X, GPU_ID
        )

    ## Call rspmm preamble.
    output_tensor, grid_dim, trailing_dim_acsr = rspmm_preamble(mask, (batch_size, num_heads, m, n), BLOCK_SIZE_X, BLOCK_SIZE_Y, GPU_ID)

    ## Call the rsddmm launcher.
    rspmm_output, sTod_linear_transformations, sTod_translations, nnzs = rspmm_launcher(
        left, right, output_tensor,
        dTos_linear_transformations, dTos_translations,
        sTod_linear_transformations, sTod_translations,
        span_loop_start, span_loop_end,
        trailing_dim_acsr, nnzs, grid_dim, 
        BLOCK_SIZE_Y, BLOCK_SIZE_X
        )

    is_correct(
        left, right,
        rspmm_output, 
        sTod_linear_transformations, 
        sTod_translations, nnzs, 
        num_heads, batch_size, mask
        )

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

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)


    def test_two():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 10
        m: int = 10
        k: int = 10
        p: int = 5 ## Sparsity parameter.
        GPU_ID : Any = "cpu"
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

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

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

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

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_five():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 16
        m: int = 16
        k: int = 16
        p: int = 16 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_six():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 10 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)
        
        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_seven():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 20 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_eight():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 32
        m: int = 32
        k: int = 32
        p: int = 32 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    def test_nine():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 128
        m: int = 128
        k: int = 128
        p: int = 57 ## Sparsity parameter.
        GPU_ID : Any = 'cpu'
        num_heads : int = 2
        batch_size : int = 2
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, num_heads, batch_size, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

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

    ## Currently, these fail. TODO(ahangupta), figure out what's going on. Priority: low.
    ## Larger tests.
    def test_ten():
        ## Basice parameters to multiply: m*k by k*n -> m*n matrix.
        n: int = 1024
        m: int = 1024
        k: int = 1024
        p: int = 256 ## Sparsity parameter.
        GPU_ID : Any = 0
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
