import einops
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torchvision
from torch.utils.checkpoint import checkpoint

from accelerate.utils import set_module_tensor_to_device
from diffusers.models.embeddings import apply_rotary_emb, FluxPosEmbed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin
import os

class MLPBlock(torchvision.ops.misc.MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, freqs_cis):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        B, L, C = input.shape
        x = self.ln_1(input)
        if freqs_cis is not None:
            query = x.view(B, L, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
            query = apply_rotary_emb(query, freqs_cis)
            query = query.transpose(1, 2).reshape(B, L, self.hidden_dim)
        x, _ = self.self_attention(query, query, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, freqs_cis):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input # + self.pos_embedding
        x = self.dropout(input)
        for l in self.layers:
            x = checkpoint(l, x, freqs_cis)
        x = self.ln(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, arch='vit-b/32'):
        super().__init__()
        self.arch = arch

        if self.arch == 'vit-b/32':
            ch = 768
            layers = 12
            heads = 12
        elif self.arch == 'vit-h/14':
            ch = 1280
            layers = 32
            heads = 16
        
        self.encoder = Encoder(
            seq_length=-1,
            num_layers=layers,
            num_heads=heads,
            hidden_dim=ch,
            mlp_dim=ch*4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.fc_in = nn.Linear(16, ch)
        self.fc_out = nn.Linear(ch, 256)
        user_home = '/home/jovyan'
        weights_base_dir = os.path.join(user_home, 'boomcheng-work-shcdt', 'zhaoshihao', 'pretrained_model', 'vit32')
        if self.arch == 'vit-b/32':
            from torchvision.models.vision_transformer import vit_b_32, ViT_B_32_Weights
            # vit = vit_b_32(weights=local_weight_path)
            # # vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
            weight_path = os.path.join(weights_base_dir, 'vit_b_32-d86f8d99.pth')
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"ViT-B/32 weight file not found: {weight_path}")
            print(f"[{os.getpid()}] Creating ViT-B/32 structure...")
            vit = vit_b_32(weights=None)
            print(f"[{os.getpid()}] Loading local weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            print(f"[{os.getpid()}] Applying weights to the model...")
            vit.load_state_dict(state_dict, strict=True)
            print(f"[{os.getpid()}] ViT-B/32 loaded successfully.")
        elif self.arch == 'vit-h/14':
            from torchvision.models.vision_transformer import vit_h_14, ViT_H_14_Weights
            vit = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)

        missing_keys, unexpected_keys = self.encoder.load_state_dict(vit.encoder.state_dict(), strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"ViT Encoder Missing keys: {missing_keys}")
            print(f"ViT Encoder Unexpected keys: {unexpected_keys}")
        del vit
    
    def forward(self, x, freqs_cis):
        out = self.fc_in(x)
        out = self.encoder(out, freqs_cis)
        out = checkpoint(self.fc_out, out)
        return out


def patchify(x, patch_size=8):
    if len(x.shape) == 4:
        bs, c, h, w = x.shape
        x = einops.rearrange(x, "b c (h p1) (w p2) -> b (c p1 p2) h w", p1=patch_size, p2=patch_size)
    elif len(x.shape) == 3:
        c, h, w = x.shape
        x = einops.rearrange(x, "c (h p1) (w p2) -> (c p1 p2) h w", p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, patch_size=8):
    if len(x.shape) == 4:
        bs, c, h, w = x.shape
        x = einops.rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=patch_size, p2=patch_size)
    elif len(x.shape) == 3:
        c, h, w = x.shape
        x = einops.rearrange(x, "(c p1 p2) h w -> c (h p1) (w p2)", p1=patch_size, p2=patch_size)
    return x


def crop_each_layer(hidden_states, use_layers, list_layer_box, H, W, pos_embedding):
    token_list = []
    cos_list, sin_list = [], []
    for layer_idx in range(hidden_states.shape[1]):
        if list_layer_box[layer_idx] is None:
            continue
        else:
            x1, y1, x2, y2 = list_layer_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2]
            c, h, w = layer_token.shape
            layer_token = layer_token.reshape(c, -1)
            token_list.append(layer_token)
            ids = prepare_latent_image_ids(-1, H * 2, W * 2, hidden_states.device, hidden_states.dtype)
            ids[:, 0] = use_layers[layer_idx]
            image_rotary_emb = pos_embedding(ids)
            pos_cos, pos_sin = image_rotary_emb[0].reshape(H, W, -1), image_rotary_emb[1].reshape(H, W, -1)
            cos_list.append(pos_cos[y1:y2, x1:x2].reshape(-1, 64))
            sin_list.append(pos_sin[y1:y2, x1:x2].reshape(-1, 64))
    token_list = torch.cat(token_list, dim=1).permute(1, 0)
    cos_list = torch.cat(cos_list, dim=0)
    sin_list = torch.cat(sin_list, dim=0)
    return token_list, (cos_list, sin_list)


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


class AutoencoderKLTransformerTraining(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(self):
        super().__init__()

        self.decoder_arch = 'vit'
        self.layer_embedding = 'rope'

        self.decoder = ViTEncoder()
        self.pos_embedding = FluxPosEmbed(theta=10000, axes_dim=(8, 28, 28))
        if 'rel' in self.layer_embedding or 'abs' in self.layer_embedding:
            self.layer_embedding = nn.Parameter(torch.empty(16, 2 + self.max_layers, 1, 1).normal_(std=0.02), requires_grad=True)

        def zero_module(module):
            """
            Zero out the parameters of a module and return it.
            """
            for p in module.parameters():
                p.detach().zero_()
            return module

    def encode(self, z_2d, box, use_layers):
        B, C, T, H, W = z_2d.shape

        z, freqs_cis = [], []
        for b in range(B):
            _z = z_2d[b]
            if 'vit' in self.decoder_arch:
                _use_layers = torch.tensor(use_layers[b], device=z_2d.device)
                if 'rel' in self.layer_embedding:
                    _use_layers[_use_layers > 2] = 2
                if 'rel' in self.layer_embedding or 'abs' in self.layer_embedding:
                    _z = _z + self.layer_embedding[:, _use_layers] # + self.pos_embedding
            if 'rope' not in self.layer_embedding:
                use_layers[b] = [0] * len(use_layers[b])
            _z, cis = crop_each_layer(_z, use_layers[b], box[b], H, W, self.pos_embedding) ### modified
            z.append(_z)
            freqs_cis.append(cis)

        return z, freqs_cis

    def decode(self, z, freqs_cis, box, H, W):
        B = len(z)
        pad = torch.zeros(4, H, W, device=z[0].device, dtype=z[0].dtype)
        pad[3, :, :] = -1
        x = []
        for b in range(B):
            _x = []
            _z = self.decoder(z[b].unsqueeze(0), freqs_cis[b]).squeeze(0)
            current_index = 0
            for layer_idx in range(len(box[b])):
                if box[b][layer_idx] == None:
                    _x.append(pad.clone())
                else:
                    x1, y1, x2, y2 = box[b][layer_idx]
                    x1_tok, y1_tok, x2_tok, y2_tok = x1 // 8, y1 // 8, x2 // 8, y2 // 8
                    token_length = (x2_tok - x1_tok) * (y2_tok - y1_tok)
                    tokens = _z[current_index:current_index + token_length]
                    pixels = einops.rearrange(tokens, "(h w) c -> c h w", h=y2_tok - y1_tok, w=x2_tok - x1_tok)
                    unpatched = unpatchify(pixels)
                    pixels = pad.clone()
                    pixels[:, y1:y2, x1:x2] = unpatched
                    _x.append(pixels)
                    current_index += token_length
            _x = torch.stack(_x, dim=1)
            x.append(_x)
        x = torch.stack(x, dim=0)
        return x
    
    def forward(self, z_2d, box, use_layers=None):
        z_2d = z_2d.transpose(0, 1).unsqueeze(0)
        use_layers = use_layers or [list(range(z_2d.shape[2]))]
        z, freqs_cis = self.encode(z_2d, box, use_layers)
        H, W = z_2d.shape[-2:]
        x_hat = self.decode(z, freqs_cis, box, H * 8, W * 8)
        assert x_hat.shape[0] == 1, x_hat.shape
        x_hat = einops.rearrange(x_hat[0], "c t h w -> t c h w")
        x_hat_rgb, x_hat_alpha = x_hat[:, :3], x_hat[:, 3:]
        return x_hat_rgb, x_hat_alpha
