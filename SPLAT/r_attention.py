import torch
import torch.nn as nn
from acsr_helpers import create_acsr, create_blocked_mask, create_windowed_mask, create_causal_windowed_mask
from typing import AnyStr, Any
from r_softmax import rsoftmax_launcher, rsoftmax_preamble
from r_spmm import rspmm_launcher, rspmm_preamble
from r_sddmm import rsddmm_launcher, rsddmm_preamble

import pdb


class RegularAttention(nn.Module):

    def __init__(
            self, batch: int, seq_length: int, num_heads: 
            int, head_dim : int, mask : list[list[int]], 
            BLOCK_SIZE_Y : int, BLOCK_SIZE_X : int, GPU_ID : Any,
            out_dtype : torch.dtype
            ):
        super().__init__()

        ## Create the ACSR.
        dTos_linear_transformations, dTos_translations, \
        sTod_linear_transformations, sTod_translations, \
        nnzs, acsr_trailing_dim, \
        span_loop_start, span_loop_end = create_acsr(mask, BLOCK_SIZE_Y, GPU_ID)

        ## rSDDMM preamble.
        rsddmm_output, rsddmm_grid_dim, \
        rsddmm_tb_map_x, rsddmm_tb_map_y = rsddmm_preamble(mask, (batch, num_heads, seq_length, acsr_trailing_dim), 
                                                           BLOCK_SIZE_X, BLOCK_SIZE_Y, GPU_ID, out_dtype)

        ## rSoftmax preamble.
        rsoftmax_grid_dim, rsoftmax_output, rsoftmax_full_shape, \
            rsoftmax_trailing_dim_pow_two = rsoftmax_preamble(mask, (batch, num_heads, seq_length, acsr_trailing_dim), 
                                                              BLOCK_SIZE_X, GPU_ID, out_dtype)

        ## rSpMM preamble.
        rspmm_output, rspmm_grid_dim, rspmm_trailing_dim_acsr =  rspmm_preamble(mask, (batch, num_heads, seq_length, head_dim), 
                                                                                BLOCK_SIZE_X, BLOCK_SIZE_Y, GPU_ID, out_dtype)

        ## Set variables accordingly.

        ## First, we set all the acsr variables.
        self.dTos_linear_transformations = dTos_linear_transformations
        self.dTos_translations = dTos_translations
        self.sTod_linear_transformations = sTod_linear_transformations
        self.sTod_translations = sTod_translations
        self.acsr_trailing_dim = acsr_trailing_dim
        self.nnzs = nnzs
        
        ## ACSR metadata required for optimisations.
        self.span_loop_start = span_loop_start
        self.span_loop_end = span_loop_end
        
        ## Block dimensions
        self.BLOCK_SIZE_Y = BLOCK_SIZE_Y
        self.BLOCK_SIZE_X = BLOCK_SIZE_X

        ## rSDDMM specific vars/data.
        self.rsddmm_output = rsddmm_output
        self.rsddmm_grid_dim = rsddmm_grid_dim
        self.rsddmm_tb_map_x = rsddmm_tb_map_x
        self.rsddmm_tb_map_y = rsddmm_tb_map_y

        ## rSoftmax specific vars/data.
        self.rsoftmax_output = rsoftmax_output
        self.rsoftmax_grid_dim = rsoftmax_grid_dim
        self.rsoftmax_trailing_dim_pow_two = rsoftmax_trailing_dim_pow_two

        ## rSpMM specific vars/data.
        self.rspmm_output = rspmm_output
        self.rspmm_grid_dim = rspmm_grid_dim


    ## Input x should already be in query/key/value post RoPe application.
    def forward(self, x):
        q,k,v = x
        ## First, we launch the r-sddmm.
        rsddmm_launcher(q, k, self.rsddmm_output, 
                        self.dTos_linear_transformations, self.dTos_translations, 
                        self.sTod_linear_transformations, self.sTod_translations, 
                        self.acsr_trailing_dim, self.nnzs, 
                        self.rsddmm_grid_dim, self.rsddmm_tb_map_x, 
                        self.rsddmm_tb_map_y, self.BLOCK_SIZE_Y, self.BLOCK_SIZE_X)

        rsoftmax_launcher(self.rsddmm_output, self.rsoftmax_output, 
                          self.dTos_linear_transformations, self.dTos_translations,
                          self.sTod_linear_transformations, self.sTod_translations,
                          self.acsr_trailing_dim, self.rsoftmax_trailing_dim_pow_two, self.nnzs,
                          self.rsoftmax_grid_dim, self.BLOCK_SIZE_X)

        rspmm_launcher(self.rsoftmax_output, v, self.rspmm_output, 
                       self.dTos_linear_transformations, self.dTos_translations,
                       self.sTod_linear_transformations, self.sTod_translations,
                       self.span_loop_start, self.span_loop_end, self.acsr_trailing_dim,
                       self.nnzs, self.rspmm_grid_dim, self.BLOCK_SIZE_Y, self.BLOCK_SIZE_X)

        out = self.rspmm_output
        return out

def unit(q,k,v,w_q,w_k,w_v, attn : RegularAttention):
    q = torch.einsum('bsd, dnh -> bsnh', q, w_q)
    k = torch.einsum('bsd, dnh -> bsnh', k, w_k)
    v = torch.einsum('bsd, dnh -> bsnh', v, w_v)
    # q = torch.matmul(q, w_q)
    # k = torch.matmul(k, w_k)
    # v = torch.matmul(v, w_v)
    out = attn([q, k.transpose(2,3),v])
    return out


if __name__ == '__main__':
    ## Let's write one simple test to see if everything works end-to-end.

    batch : int = 32
    heads : int = 12
    seq_length : int = 1024
    head_dim : int = 64
    BLOCK_SIZE_X : int = 16
    BLOCK_SIZE_Y : int = 16
    GPU_ID : Any = 'cpu'
    p : int = 128  ## Sparsity parameter.
    out_dtype : torch.dtype = torch.float32
    mask : list[list[int]] = create_windowed_mask(seq_length, p)
    #mask : list[list[int]] = create_causal_windowed_mask(seq_length, p)

    attn = RegularAttention(batch, seq_length, heads, head_dim, mask, 
                            BLOCK_SIZE_Y, BLOCK_SIZE_X, GPU_ID, out_dtype)

    query = torch.rand((batch, seq_length, head_dim*heads))
    key = torch.rand((batch, seq_length, head_dim*heads))
    value = torch.rand((batch, seq_length, head_dim*heads))
    #query = torch.randint(0, 100, (batch, heads, seq_length, head_dim), dtype=out_dtype).to(GPU_ID)
    ## Key should have the outer two dims transposed.
    #key = torch.randint(0, 100, (batch, heads, head_dim, seq_length), dtype=out_dtype).to(GPU_ID)
    #value = torch.randint(0, 100, (batch, heads, seq_length, head_dim), dtype=out_dtype).to(GPU_ID)
    w_query = torch.rand((head_dim*heads, heads, head_dim), dtype=out_dtype).to(GPU_ID)
    w_key = torch.rand((head_dim*heads, heads, head_dim), dtype=out_dtype).to(GPU_ID)
    w_value = torch.rand((head_dim*heads, heads, head_dim), dtype=out_dtype).to(GPU_ID)
    unit(query,key,value,w_query,w_key,w_value, attn)
    unit(query,key,value,w_query,w_key,w_value, attn)

    import time

    a = time.time()
    unit(query,key,value,w_query,w_key,w_value, attn)
    b = time.time()
    print(f'time taken: {b-a}')
    #print(attn.forward([query, key, value]))
