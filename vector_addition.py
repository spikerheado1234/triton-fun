import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x,y,
                        output,elements,
                        BLOCK_SIZE = tl.constexpr):
    
    ## Get blockIdx.
    blockIdx = tl.program_id(axis=0)

    # Get all the pointers to index the inputs.
    offsets = blockIdx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < elements 

    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.load(y+offsets, mask=mask)

    output_valules = x_values + y_values

    tl.store(output + offsets, output_valules, mask=mask)


def vector_add_launcher(x : torch.Tensor, y : torch.Tensor, 
                        GPU_ID : int, BLOCK_SIZE = 1024):

    output = torch.empty_like(x).to(GPU_ID)

    assert x.shape == y.shape, "Shape incorrect"

    elements = x.numel()

    assert x.is_cuda and y.is_cuda and output.is_cuda, "Tensors must be on GPU." 

    grid = (elements, BLOCK_SIZE)

    vector_add_kernel[grid](x,y,output,BLOCK_SIZE=BLOCK_SIZE)
    
    return output 


BLOCK_SIZE=1024
size = 10
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
out = vector_add_launcher(x,y,0,BLOCK_SIZE)
