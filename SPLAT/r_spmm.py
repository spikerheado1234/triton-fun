import torch
import triton
import triton.language as tl
from functools import reduce
import time
from acsr_helpers import create_acsr, create_blocked_mask, create_windowed_mask

## This is a naive spmm kernel with the incoming ACSR (x_ptr) in row-major and row-compressed.
## This represents an: (mxk) x (kxn) matrix multiplication.
## Trailing dimension represents the SPARSE trailing dimension of the ACSR (left-matrix).
@triton.jit
def rspmm_kernel_row_maj_row_comp(
    x_ptr, y_ptr, 
    out_ptr, dTos_linear_trf, dTos_translations, 
    sTod_linear_trf, sTod_translations, nnzs,
    m, n, k, trailing_dim, 
    BLOCK_SIZE_Y : tl.constexpr, BLOCK_SIZE_X : tl.constexpr
    ):
    ## We will first do this naively.
    
    ## Extract the blockIdx.x/y indices.
    by = tl.pid(axis=1)
    bx = tl.pid(axis=0)

    ## The anchor point, top-left corner of the TB.
    by_start = by*BLOCK_SIZE_Y
    bx_start = bx*BLOCK_SIZE_X

    ## Next we prepare the x and y pointers.
    inner_tile : tl.constexpr = 128

    ## We use dense_col_idxs and internally convert them to sparse coordinates within the inner loop of this kernel.
    dense_col_idxs = tl.arange(0, inner_tile)[None, :].to(tl.int64) + tl.zeros(0, BLOCK_SIZE_Y)[:, None].to(tl.int64)
    y_ptrs = bx_start*BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[None, :] + tl.arange(0, inner_tile)[:, None] 

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
        other=0.0
    ).reshape(BLOCK_SIZE_Y, 1)

    accumulator = tl.zeros((BLOCK_SIZE_Y, BLOCK_SIZE_X), dtype=tl.float32)

    for i in range(tl.cdiv(k, inner_tile)):

        ## Constraint one, OOB along the leading and trailing dimensions.
        mask_x_ptrs = i*inner_tile + tl.arange(0, inner_tile)[None,:] < k
        mask_x_ptrs = mask_x_ptrs & (tl.arange(0, BLOCK_SIZE_Y)[:,None] + by_start < m)
        ## Constraint two, mapped to valid indices in the ACSR.

        ## First check: linear_trf valid check.
        op_one = dense_col_idxs % block_linear_trfs > 0
        op_two = dense_col_idxs % block_linear_trfs < 0
        mask_x_ptrs = mask_x_ptrs & ((not op_one) & (not op_two))

        ## Second check: translation valid check.
        mask_x_ptrs = mask_x_ptrs & (dense_col_idxs - block_translations > 0)

        ## Finally, we can convert the x_ptrs to sparse points in the ACSR.  This is incorrect.
        sparse_x_ptrs = (tl.div_rn(dense_col_idxs - block_translations), block_linear_trfs).to(tl.int64) + by_start*trailing_dim + i*inner_tile + tl.arange(0, BLOCK_SIZE_Y)*trailing_dim 

        ## The mask for the dense matrix. 

        ## First check, OOB from the horizontal (leading dimension).
        mask_y_ptrs = bx_start * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[None, :] < n
        ## Second check, OOB from the vertical (trailing dimension)
        mask_y_ptrs  = mask_y_ptrs & (i*inner_tile + tl.arange(0, BLOCK_SIZE_Y)[:, None] < k)

        ## We load the data.
        x_tile = tl.load(x_ptr + sparse_x_ptrs, mask=mask_x_ptrs, other=0.0)
        y_tile = tl.load(y_ptr + y_ptrs, mask=mask_y_ptrs, other=0.0)

        ## Do the matmul and accumulate.
        accumulator += tl.dot(x_tile, y_tile)

        y_ptrs += inner_tile*n
        dense_col_idxs += i*inner_tile

    ## Need to implement write-back logic.

    ## Now, write-backs are straightfoward since the output is dense.
    write_ptrs = bx_start * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[None, :] + (by_start*BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[:, None]*n)
    ## Check whether leading dimension OOB.
    write_ptrs_mask = bx_start * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[None, :] < m
    write_ptrs_mask = write_ptrs_mask & (by_start * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[:, None] < n)

    tl.store(out_ptr + write_ptrs, accumulator, mask=write_ptrs_mask)

## Over here, x is an ACSR and y is the Values tensor.
def rspmm_launcher(x : torch.Tensor, 
                   y : torch.Tensor,
                   acsr_trailing_dim : int,
                   mask : list[list[int]], GPU_ID : int, 
                   BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    ## First we create the output tensor.
    output : torch.Tensor = torch.empty((x.shape[0], y.shape[-1]), dtype=torch.float32).to(GPU_ID)

    ## We instantiate the acsr metadata.
    dTos_linear_transformations, dTos_translations, sTod_linear_transformations, sTod_translations, nnzs = create_acsr(
        mask, BLOCK_SIZE_Y, GPU_ID
        )

    ## Finally, we can launch the kernel
    grid_dim = (triton.cdiv(y.shape[1], BLOCK_SIZE_X),triton.cdiv(x.shape[0], BLOCK_SIZE_Y))

    torch.cuda.synchronize()
    rsddmm_start = time.time()
    rspmm_kernel_row_maj_row_comp[grid_dim](x,y,output, 
                                            dTos_linear_transformations,dTos_translations, 
                                            sTod_linear_transformations,sTod_translations,nnzs,
                                            x.shape[0],y.shape[1],x.shape[1], acsr_trailing_dim,
                                            BLOCK_SIZE_Y=BLOCK_SIZE_Y, BLOCK_SIZE_X=BLOCK_SIZE_X, num_warps=2)
    torch.cuda.synchronize()
    rsddmm_end = time.time()
    print(f'time taken splat: {(rsddmm_end - rsddmm_start):.15f}')
    print(f'rspmm kernel output shape: {output.shape}')
    ## We return the sTod arrays for correctness checking only.
    return (output, sTod_linear_transformations, sTod_translations, nnzs)

def is_correct(
        left_tensor : torch.Tensor, right_tensor : torch.Tensor,
        out_rspmm : torch.Tensor, sTod_linear_transofrmations : torch.Tensor, 
        sTod_translations : torch.Tensor, nnzs: torch.Tensor, mask : list[list[int]]
        ) -> bool:

    out_rspmm_list = out_rspmm.tolist()
    sTod_linear_transformations_list = sTod_linear_transofrmations.tolist()
    sTod_translations_list = sTod_translations.tolist()
    nnzs_list = nnzs.tolist()

    num_deviations : int = 0
    mse_error : float = 0

    def dot_product(left, right, row, col):
        ## How pythonic can really be here?
        accum = 0
        for i in range(len(left[0])):
            for j in range(len(right)):
                accum += left[row][i] * right[j][col]

        return accum

    for row in range(len(mask)):
        for nnz_col_id in range(len(out_rspmm_list[0])):
            ## We convert to the dense index.
            dense_col_id : int = round(nnz_col_id * sTod_linear_transformations_list[row] + sTod_translations_list[row])
            if nnz_col_id < nnzs_list[row]:
                ## Now, we manually compute ground truth.
                manual_answer = dot_product(left_tensor, right_tensor, row, dense_col_id)
                if abs(manual_answer - out_rspmm_list[row][dense_col_id]) > 1e-3:
                    mse_error += abs(manual_answer - out_rspmm_list[row][dense_col_id])
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
    #left : torch.Tensor = torch.randn((m,k),dtype=torch.float32).to(GPU_ID)
    #right : torch.Tensor = torch.randn((k,n),dtype=torch.float32).to(GPU_ID)
    ## Now, left tensor is an ACSR. we need to generate its trailing dimension.
    trailing_dim_acsr = max([reduce(lambda a,b: a+b, row, 0) for row in mask])

    left : torch.Tensor = torch.randint(0, 100, (
        m, trailing_dim_acsr
        ),dtype=torch.float32).to(GPU_ID)

    right : torch.Tensor = torch.randint(0, 100, (k,n),dtype=torch.float32).to(GPU_ID)

    ## Call the rsddmm launcher.
    rspmm_output, sTod_linear_transformations, sTod_translations, nnzs = rspmm_launcher(
        left, right, mask, GPU_ID, 
        BLOCK_SIZE_Y, BLOCK_SIZE_X
        )

    is_correct(
        rspmm_output, 
        sTod_linear_transformations, 
        sTod_translations, nnzs, mask
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
        GPU_ID : int = 0
        BLOCK_SIZE_Y : int = 16
        BLOCK_SIZE_X : int = 16

        ## Instantiate a mask.
        mask = create_blocked_mask(n, p)

        test(m, k, n, mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

    import sys
    sys.exit()

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
