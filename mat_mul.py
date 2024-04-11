## A fun matrix multiplication in triton!

import triton
import triton.language as tl
import torch

### Here we are doing an: m * k by k * n mat-mul.
@triton.jit
def mat_mul_kernel(x_ptr, y_ptr, output_ptr, m,n,k, BLOCK_SIZE: tl.constexpr):

    ## Let's see if the performance is up to par with this.
    blockIdxX = tl.program_id(axis=0)
    blockIdxY = tl.program_id(axis=1)

    ## How we will do is this is to load a tile and multiply.
    inner_tile_dim = 128

    ## This is a BLOCK_SIZE by inner_tile_dim block of pointers.
    x_ptrs = x_ptr + (blockIdxY*BLOCK_SIZE + (tl.arange(0, BLOCK_SIZE)*k))[:,None] + tl.arange(0, inner_tile_dim)[None,:]
    
    ## This is an inner_tile_dim by BLOCK_SIZE block of pointers.
    y_ptrs = y_ptr + (tl.arange(0, inner_tile_dim)[:,None] * n) + tl.arange(0, BLOCK_SIZE)[None,:] + blockIdxX*BLOCK_SIZE 

    ## Where we compute partial MACs.
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    ## This is for masking within the loop itself.
    masked_k = tl.arange(0, inner_tile_dim)

    for i in range(tl.cdiv(k, inner_tile_dim)):

        ## Masking now becomes a tad complex.
        masked_x_ptrs = masked_k[None,:] * i*inner_tile_dim < k
        masked_x_ptrs = masked_x_ptrs or (tl.arange(0, BLOCK_SIZE)[:,None] + blockIdxY*BLOCK_SIZE < m)
        masked_y_ptrs = masked_k[:,None] * i*inner_tile_dim < k
        masked_y_ptrs = masked_y_ptrs or (tl.arange(0,BLOCK_SIZE)[None,:] + blockIdxX*BLOCK_SIZE < n)

        x_tile = tl.load(x_ptrs, x_ptr, mask=masked_x_ptrs, other=0.0)
        y_tile = tl.load(y_ptrs, y_ptr, mask=masked_y_ptrs, other=0.0)

        accumulator += tl.dot(x_tile, y_tile)

        ## Compute the correct new indices.
        x_ptrs += inner_tile_dim*k
        y_ptrs += inner_tile_dim*n

    ## Compute where to store the answer here.
    ## This gets a 1024 by 1024 block of data 
    ## to store back to memory.
    c_ptrs = output_ptr + tl.arange(0,BLOCK_SIZE)[:,None]*n + tl.arange(0, BLOCK_SIZE)[None,:] + blockIdxX*BLOCK_SIZE + blockIdxY*BLOCK_SIZE*n

    ## And the x and y boundary conditions.
    c_mask = tl.arange(0, BLOCK_SIZE)[None,:] + blockIdxX*BLOCK_SIZE < n 
    c_mask = c_mask or (tl.arange(0, BLOCK_SIZE)[:,None]*n + blockIdxY*BLOCK_SIZE < m)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@nvtx.annotate("triton-matmul", color="purple")
def mat_mul_launcher(x : torch.Tensor, y : torch.Tensor, GPU_ID : int, BLOCK_SIZE):

    ## The dimensions should be the following.
    m,k_one = x.shape
    k_two,n = y.shape

    assert k_one == k_two, "Incorrect tensor sizes!"

    output = torch.empty((m,n), dtype=torch.float32).to(GPU_ID)

    k = k_one
    BLOCK_SIZE = 128

    grid = (triton.cdiv(m,BLOCK_SIZE), triton.cdiv(n, BLOCK_SIZE))

    mat_mul_kernel[grid](x,y,output,m,n,k, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)


@nvtx.annotate("torch-einsum", color="yello")
def torch_matmul(x : torch.Tensor, y : torch.Tensor):
    return torch.einsum('ab,bc -> ac', x, y)


## We spawn a BLOCK_SIZE x BLOCK_SIZE GPU tile.
BLOCK_SIZE = 128
GPU_ID = 0
n = 10
m = 10
k = 10
x = torch.rand((m,k), dtype=torch.float32).to(GPU_ID)
y = torch.rand((k, n), dtype=torch.float32).to(GPU_ID)

out_torch = torch_matmul(x,y)

out_triton = mat_mul_launcher(x,y,GPU_ID, BLOCK_SIZE)
torch.cuda.synchronize()

print(f'tensors equal: {torch.equal(out_torch, out_triton)}')


