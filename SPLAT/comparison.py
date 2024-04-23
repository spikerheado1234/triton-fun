## This is a script to compare the runtime performance of triton's 
## Block-sparse kernels against SPLAT's sddmm and spmm kernels.
##  Preliminary experiments.
import triton
import triton.language as tl
import torch
import time

## These are helper methods to create layouts for 
##   Triton block-sparse kernels which operate at 
##   a different granularity compared to SPLAT's masks.

def create_triton_blocksparse_blocked_layout(s : int, p : int, block : int) -> list[list[int]]:
    num_blocks = triton.cdiv(float(s)/float(block))

    layout = [[0 for _ in range(num_blocks)] for _ in range(num_blocks)]

    ## Set the necessary blocks to 1.
    for i in range(s):
        for j in range(s):
            if i < p:  ## We are dealing with the first block here.
                if j < p:
                    layout[i/block][j/block] = 1 
            else:
                ## Now we have to compute if (i,j) belong to a block.
                block_num = i // p 
                start_col = (block_num-1)*p
                end_col = start_col + 2*p
                if start_col <= j and  j < end_col:
                    layout[i/block][j/block] = 1

    return layout

def triton_block_sparse_sddmm(left : torch.Tensor, right : torch.Tensor, block : int, layout : list[list[int]]) -> torch.Tensor:

    assert left.device == right.device, "Issue with where the tensors are stored"
    torch.cuda.device.synchronize()
    triton_blocksparse_start = time.time()
    sparse_dot_sdd_nt = triton.ops.blocksparse.matmul(layout, block, "sdd", trans_a=False, trans_b=True,
                                                      device=left.device)
    torch.cuda.device.synchronize()
    triton_blocksparse_end = time.time()

    print(f'time taken: {triton_blocksparse_end-triton_blocksparse_start}')

    return sparse_dot_sdd_nt(left, right)


## This runs a benchmark at a sequence length, varying the sparsity parameter
##  in powers of two within this range.
def benchmark(pattern : str, sequence_length : int, triton_block_size : int):
    
    sparsity_parameter : int = 2
    while sparsity_parameter < sequence_length:
        left : torch.Tensor = torch.randn((sequence_length, sequence_length))
        right : torch.Tensor = torch.randn((sequence_length, sequence_length))
        ## We benchmark both the triton and SPLAT's kernels.
        if pattern == "Blocked":
            triton_layout = create_triton_blocksparse_blocked_layout(sequence_length, 
                                                                    sparsity_parameter, 
                                                                    triton_block_size)
            triton_block_sparse_sddmm(left, right, triton_block_size, triton_layout)
        else:
            raise Exception("Not implemented!")

        sparsity_parameter *= 2

benchmark("Blocked", 1024, 128)
