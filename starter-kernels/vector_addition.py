import torch
import triton
import triton.language as tl
import nvtx

@triton.jit
def vector_add_kernel(x_ptr,
                      y_ptr,
                      output_ptr,
                      n_elements,
                      BLOCK_SIZE : tl.constexpr,
                      ):

    ## Get blockIdx.
    blockIdx = tl.program_id(axis=0)

    # Get all the pointers to index the inputs.
    offsets = blockIdx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements 

    x_values = tl.load(x_ptr+offsets, mask=mask)
    y_values = tl.load(y_ptr+offsets, mask=mask)

    output_valules = x_values + y_values

    tl.store(output_ptr + offsets, output_valules, mask=mask)


@nvtx.annotate("triton_vector_add", color="purple")
def vector_add_launcher(x : torch.Tensor, y : torch.Tensor, 
                        GPU_ID : int, b_size = 1024):

    output = torch.empty_like(x).to(GPU_ID)

    assert x.shape == y.shape, "Shape incorrect"

    elements = x.numel()

    assert x.is_cuda and y.is_cuda and output.is_cuda, "Tensors must be on GPU." 

    ## For some reason, this doesn't work.
    #grid = (triton.cdiv(elements, BLOCK_SIZE),)
    grid = lambda meta : (triton.cdiv(elements, meta['BLOCK_SIZE']), )
    compiled_func = vector_add_kernel[grid](x,y,output,elements,BLOCK_SIZE=b_size)
    print(f'asm keys are: {compiled_func.asm.keys()}')
    with open("vector_ptx_dump", "w+") as f:
        f.write(compiled_func.asm["ptx"])
    with open("vector_triton_ir_dump", "w+") as f:
        f.write(compiled_func.asm["ttir"])
    with open("vector_llvm_ir_dump", "w+") as f:
        f.write(compiled_func.asm["llir"])
    with open("vector_triton_gpu_ir_dump", "w+") as f:
        f.write(compiled_func.asm["ttgir"])
    
    return output 


import time

@nvtx.annotate("torch_add", color="yellow")
def torch_add(x: torch.Tensor, y : torch.Tensor):
    return x+y

BLOCK_SIZE=1024
size = int(1e5)
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
torch.cuda.synchronize()
trit_start = time.time()
out = vector_add_launcher(x,y,0,BLOCK_SIZE)
torch.cuda.synchronize()
trit_end = time.time()
## Question is: what does this do? Launch another kernel or what?
torch.cuda.synchronize()
start = time.time()
torch_result = torch_add(x,y)
torch.cuda.synchronize()
end = time.time()
print(f'absolute error: {torch.max(torch.abs(torch_result - out))}')
print(f'time taken torch: {end-start:.3f} time taken triton: {trit_end-trit_start:.3f}')
