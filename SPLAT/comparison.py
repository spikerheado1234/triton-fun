## This is a script to compare the runtime performance of triton's 
## Block-sparse kernels against SPLAT's sddmm and spmm kernels.
##  Preliminary experiments.
import triton
import triton.ops
import triton.language as tl
import torch
import time
from acsr_helpers import create_blocked_mask, create_windowed_mask
from r_sddmm import rsddmm_launcher

## These are helper methods to create layouts for 
##   Triton block-sparse kernels which operate at 
##   a different granularity compared to SPLAT's masks.

def create_triton_blocksparse_blocked_layout(s : int, 
                                             p : int, block : int, 
                                             GPU_ID : int) -> torch.Tensor:
    num_blocks = triton.cdiv(s, block)

    layout = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]

    ## Set the necessary blocks to 1.
    for i in range(s):
        for j in range(s):
            if i < p:  ## We are dealing with the first block here.
                if j < p:
                    layout[i//block][j//block] = 1 
            else:
                ## Now we have to compute if (i,j) belong to a block.
                block_num = i // p 
                start_col = (block_num-1)*p
                end_col = start_col + 2*p
                if start_col <= j and  j < end_col:
                    layout[i//block][j//block] = 1
    return (torch.tensor(layout, dtype=torch.long)[None,:,:]).to(GPU_ID)

def create_triton_blocksparse_windowed_layout(s : int, p : int, 
                                              block : int, 
                                              GPU_ID : int) -> torch.Tensor:
    num_blocks = triton.cdiv(s, block)

    layout = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]

    for i in range(s):
        for j in range(s):
            if i-p <= j and j <= i+p:
                layout[i//block][j//block] = 1

    return (torch.tensor(layout, dtype=torch.long)[None,:,:]).to(GPU_ID)

def triton_block_sparse_sddmm(left : torch.Tensor, right : torch.Tensor, block : int, layout : list[list[int]]) -> torch.Tensor:

    assert left.device == right.device, "Issue with where the tensors are stored"
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True, device=left.device)

    for _ in range(5):
        sparse_dot_sdd_nt(left, right)

    torch.cuda.synchronize()
    triton_blocksparse_start = time.time()
    sparse_dot_sdd_nt(left, right)
    torch.cuda.synchronize()
    triton_blocksparse_end = time.time()

    print(f'time taken triton: {(triton_blocksparse_end-triton_blocksparse_start):.15f}')

    return sparse_dot_sdd_nt(left, right)

## This runs a benchmark at a sequence length, varying the sparsity parameter
##  in powers of two within this range.
def benchmark(pattern : str, sequence_length : int, 
              triton_block_size : int, BLOCK_SIZE_Y : int, 
              BLOCK_SIZE_X : int, GPU_ID : int):
    
    sparsity_parameter : int = 2
    while sparsity_parameter <= sequence_length:
        print(f'sparsity parameter: {sparsity_parameter}')
        left : torch.Tensor = torch.randint(0,100,(sequence_length, sequence_length), dtype=torch.float32).to(GPU_ID)
        right : torch.Tensor = torch.randint(0,100,(sequence_length, sequence_length),dtype=torch.float32).to(GPU_ID)
        ## We benchmark both the triton and SPLAT's kernels.
        if pattern == "Blocked":
            ## Create the two layouts: one for triton and one for SPLAT.
            triton_layout = create_triton_blocksparse_blocked_layout(sequence_length, 
                                                                    sparsity_parameter, 
                                                                    triton_block_size, GPU_ID)
            splat_mask = create_blocked_mask(sequence_length, sparsity_parameter)

            ## Call the two sddmm kernels.
            triton_block_sparse_sddmm(left[None, None, :, :], right[None, None, :, :], triton_block_size, triton_layout)
            rsddmm_launcher(left, right, splat_mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)
        elif pattern == "Windowed":
            ## Create the two layouts: one for triton and one for SPLAT.
            triton_layout = create_triton_blocksparse_windowed_layout(sequence_length, 
                                                                    sparsity_parameter, 
                                                                    triton_block_size, GPU_ID)
            splat_mask = create_windowed_mask(sequence_length, sparsity_parameter)

            ## Call the two sddmm kernels.
            triton_block_sparse_sddmm(left[None, None, :, :], right[None, None, :, :], triton_block_size, triton_layout)
            rsddmm_launcher(left, right, splat_mask, GPU_ID, BLOCK_SIZE_Y, BLOCK_SIZE_X)

        else:
            raise Exception("Not implemented!")

        sparsity_parameter *= 2

sequence_length : int = 1024
triton_block_size : int = 128
BLOCK_SIZE_X : int = 16
BLOCK_SIZE_Y : int = 16
GPU_ID : int = 0

benchmark("Windowed", sequence_length, triton_block_size, BLOCK_SIZE_Y, BLOCK_SIZE_X, GPU_ID)
