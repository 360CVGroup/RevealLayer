import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Flux style RoPE application with robust broadcasting.
    """
    cos, sin = freqs_cis

    if cos.ndim == 2:
        # [L, D] -> [1, 1, L, D]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        # [B, L, D] -> [B, 1, L, D]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out

class FluxRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.ada_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size) 
        )

        nn.init.zeros_(self.ada_regressor[-1].weight)
        nn.init.zeros_(self.ada_regressor[-1].bias)

        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )

    def forward(self, x, t_emb, rotary_emb: Tuple[torch.Tensor, torch.Tensor]):
        batch_size, seq_len, _ = x.shape

        params = self.ada_regressor(t_emb).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)

        x_norm = self.norm1(x)
        x_mod = modulate(x_norm, shift_msa, scale_msa)
        
        qkv = self.to_qkv(x_mod)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rotary_emb(q, rotary_emb)
        k = apply_rotary_emb(k, rotary_emb)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        attn_out = self.to_out(attn_out)
        
        x = x + gate_msa * attn_out

        x_norm = self.norm2(x)
        x_mod = modulate(x_norm, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mod)
        
        x = x + gate_mlp * mlp_out

        return x

class HybridRefiner(nn.Module):
    def __init__(self, 
                 feature_dim=3072, 
                 pixel_dim=16, 
                 time_emb_dim=3072, 
                 attn_hidden=384, 
                 num_layers=4,
                 patch_size=2,
                 max_layers=32):
        super().__init__()
        
        self.attn_hidden = attn_hidden

        flat_pixel_dim = 64
        flat_mask_dim = 64

        self.input_proj = nn.Linear(feature_dim + flat_pixel_dim + flat_mask_dim, attn_hidden)
        
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, attn_hidden))
        self.layer_id_emb = nn.Embedding(max_layers, attn_hidden)

        self.blocks = nn.ModuleList([
            FluxRefinerBlock(attn_hidden, num_heads=8) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(attn_hidden, elementwise_affine=False, eps=1e-6)
        
        self.final_linear = nn.Linear(attn_hidden, feature_dim)

        nn.init.zeros_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward_feature_attn(
        self, 
        hidden_states: torch.Tensor,
        clean_tokens: torch.Tensor,
        split_sizes: List[int],
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:

        compute_dtype = hidden_states.dtype
        if not torch.is_floating_point(hidden_states):
            compute_dtype = torch.bfloat16

        hidden_states = hidden_states.to(compute_dtype)
        clean_tokens = clean_tokens.to(compute_dtype)
        temb = temb.to(compute_dtype)

        t_base = self.time_proj(temb).to(compute_dtype)

        x_splits = torch.split(hidden_states, split_sizes, dim=1)
        clean_splits = torch.split(clean_tokens, split_sizes, dim=1)

        cos, sin = image_rotary_emb
        cos_splits = torch.split(cos, split_sizes, dim=0)
        sin_splits = torch.split(sin, split_sizes, dim=0)

        refined_splits = []

        for i, x_i in enumerate(x_splits):
            rotary_i = (cos_splits[i], sin_splits[i])

            layer_idx = torch.tensor([i + 1], device=hidden_states.device)
            l_emb = self.layer_id_emb(layer_idx).to(compute_dtype)
            layer_cond = t_base + l_emb

            refiner_input = torch.cat([x_i, clean_splits[i].to(compute_dtype)], dim=-1)
            curr_x = self.input_proj(refiner_input)

            for block in self.blocks:
                curr_x = block(curr_x, t_emb=layer_cond, rotary_emb=rotary_i)

            refined_splits.append(curr_x)

        x_refined = torch.cat(refined_splits, dim=1)
        x_refined = self.final_norm(x_refined)
        output = self.final_linear(x_refined)
        return output