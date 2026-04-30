import os
import math
import numpy as np
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm
from vae import PositionalEncoding, unscale_from_tanh
from data_utils import zero_pad_layers, layers_to_six_features_torch
from surrogate.models.pncformer import PnCFormer


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        denominator = (half - 1) if half > 1 else 1.0
        freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * (-math.log(10000.0) / denominator))
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0)
        return emb

class AdaLN(nn.Module):
    """Adaptive Layer Norm: Feature-wise Linear Modulation."""
    def __init__(self, cond_dim: int, width: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(cond_dim, width*2), nn.SiLU())
    def forward(self, cond): 
        h = self.fc(cond); gamma, beta = h.chunk(2, dim=-1)
        return gamma, beta


# ---------------------------------------------------------------------------
# Cross-Attention Module
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Standard cross-attention: query (UNet features) attends to context (band mask)."""
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 8, head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, context):
        # x_q: [B, L_q, C_q], context: [B, L_ctx, C_ctx]
        
        q = self.to_q(x_q)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Split into multi-head attention dimensions
        q = q.view(q.shape[0], q.shape[1], self.n_heads, -1).transpose(1, 2) # [B, n_heads, L_q, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.n_heads, -1).transpose(1, 2) # [B, n_heads, L_ctx, head_dim]
        v = v.view(v.shape[0], v.shape[1], self.n_heads, -1).transpose(1, 2) # [B, n_heads, L_ctx, head_dim]
        
        # Attention scores
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        # Weighted sum of values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(x_q.shape[0], x_q.shape[1], -1) # [B, L_q, C_q]
        return self.to_out(out)


# ---------------------------------------------------------------------------
# AttnResBlock1D (AdaLN + Cross-Attention hybrid)
# ---------------------------------------------------------------------------

class AttnResBlock1D(nn.Module):
    """1D Residual block with AdaLN (global cond) and Cross-Attention (band mask)."""
    def __init__(self, width: int, adaln_dim: int, context_dim: int, dropout: float, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        
        # AdaLN (global conditions)
        self.adaln = AdaLN(adaln_dim, width) 
        
        # Cross-Attention (sequential conditions)
        head_dim = width // n_heads
        self.attn = CrossAttention(query_dim=width, context_dim=context_dim, n_heads=n_heads, head_dim=head_dim)
        self.norm_attn = nn.LayerNorm(width)

        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adaln_vec, context_vec):
        # x: [B, C, L], adaln_vec: [B, D], context_vec: [B, K, D]

        gamma, beta = self.adaln(adaln_vec)
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Cross-Attention: [B,C,L] -> [B,L,C] -> attn -> [B,C,L]
        h_attn = self.norm_attn(h.permute(0, 2, 1))
        h_attn = h_attn + self.attn(h_attn, context_vec)
        h = h_attn.permute(0, 2, 1)

        # AdaLN modulation after attention
        h = h * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.drop(self.conv2(F.silu(self.norm2(h))))
        
        return x + h


# ---------------------------------------------------------------------------
# Band Mask Condition Encoders
# ---------------------------------------------------------------------------

class BandMaskCondEncoder(nn.Module):
    """Transformer encoder for band mask conditioning.
    Embeds discrete mask tokens and applies positional encoding + Transformer.
    """
    def __init__(self, n_classes: int, out_dim: int, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.out_dim = out_dim
        
        self.embedding = nn.Embedding(n_classes + 1, out_dim)  # 0:Pass, 1:Gap, 2:Defect, 3:Uncond
        
        self.pos_enc = PositionalEncoding(out_dim, dropout=0.0, max_len=1000)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=nhead, dim_feedforward=4*out_dim,
            activation="gelu", batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        
    def forward(self, band_mask: torch.Tensor):
        x = self.embedding(band_mask.long())
        x = self.pos_enc(x)
        # Manual layer loop avoids TransformerEncoder.forward() self.modules() traversal
        # that corrupts the C heap over many inference calls (PyTorch 2.x).
        for layer in self.transformer.layers:
            x = layer(x)
        if self.transformer.norm is not None:
            x = self.transformer.norm(x)

        return x


class BandMaskCondEncoder_DualConv(nn.Module):
    """Dual-channel encoder: original mask + defect-only binary mask,
    fused via Conv1d before Transformer encoding.
    """
    def __init__(self, n_classes: int, out_dim: int, n_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.out_dim = out_dim
        
        self.emb1 = nn.Embedding(n_classes + 1, out_dim)  # original mask
        self.emb2 = nn.Embedding(2, out_dim)                # defect-only binary
        
        # Local feature extractor: fuse two channels
        self.conv1d = nn.Sequential(
            nn.Conv1d(2 * out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(), nn.BatchNorm1d(out_dim)
        )
        
        self.pos_enc = PositionalEncoding(out_dim, dropout=0.0, max_len=1000)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=nhead, dim_feedforward=4*out_dim,
            activation="gelu", batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        
    def forward(self, band_mask: torch.Tensor):
        mask_ch1 = band_mask.long()
        mask_ch2 = (band_mask == 2).long()
        x1 = self.emb1(mask_ch1); x2 = self.emb2(mask_ch2)

        x_cat = torch.cat([x1, x2], dim=-1).permute(0, 2, 1)
        x_conv = self.conv1d(x_cat).permute(0, 2, 1)

        x = self.pos_enc(x_conv)
        # Manual layer loop (see BandMaskCondEncoder for rationale)
        for layer in self.transformer.layers:
            x = layer(x)
        if self.transformer.norm is not None:
            x = self.transformer.norm(x)
        return x

class UNet1D(nn.Module):
    def __init__(self, latent_dim: int, width: int, depth: int, dropout: float, cfg: Any):
        super().__init__()
        self.spatial = 16; self.in_ch = width; self.width = width
        d_cond_mat = 4
        self.time_emb = SinusoidalTimeEmbedding(width)
        self.material_proj = nn.Sequential(nn.Linear(d_cond_mat, width), nn.SiLU())
        self.ncells_embed = nn.Embedding(cfg.max_cells + 1, width)

        # AdaLN: project concatenated (t, mat, ncells) into a single cond vector
        adaln_dim_in = width * 3
        adaln_dim_out = width
        self.adaln_proj = nn.Linear(adaln_dim_in, adaln_dim_out)

        # Band mask encoder produces cross-attention context
        context_dim = width
        self.band_mask_enc = BandMaskCondEncoder(cfg.n_classes, context_dim)
        
        # U-Net body
        self.inp = nn.Linear(latent_dim, self.in_ch * self.spatial)
        
        self.downs = nn.ModuleList([
            AttnResBlock1D(width, adaln_dim_out, context_dim, dropout) for _ in range(depth)
        ])
        self.pools = nn.ModuleList([nn.AvgPool1d(2) for _ in range(depth)])
        self.mid = AttnResBlock1D(width, adaln_dim_out, context_dim, dropout)
        self.ups = nn.ModuleList([
            AttnResBlock1D(width, adaln_dim_out, context_dim, dropout) for _ in range(depth)
        ])
        self.upsample = nn.ModuleList([nn.Upsample(scale_factor=2, mode='nearest') for _ in range(depth)])
        self.outp = nn.Linear(self.in_ch * self.spatial, latent_dim)

    def forward(self, z_noisy, t, band_mask: torch.Tensor, material_conds, n_cells):
        B = z_noisy.size(0)
        # AdaLN condition vector
        t_emb = self.time_emb(t)
        mat_vec = self.material_proj(material_conds)
        ncells_vec = self.ncells_embed(n_cells)
        adaln_vec = F.silu(self.adaln_proj(torch.cat([t_emb, mat_vec, ncells_vec], dim=-1)))
        
        # Cross-attention context from band mask
        context_vec = self.band_mask_enc(band_mask)

        # U-Net forward
        feats = []; 
        for d in range(len(self.downs)):
            x = self.downs[d](x, adaln_vec, context_vec)
            feats.append(x)
            x = self.pools[d](x)
        
        x = self.mid(x, adaln_vec, context_vec)
        
        for d in reversed(range(len(self.ups))):
            x = self.upsample[d](x)
            if x.size(-1) != feats[d].size(-1): x = F.pad(x, (0, feats[d].size(-1) - x.size(-1)))
            x = x + feats[d]
            x = self.ups[d](x, adaln_vec, context_vec)

        x = x.view(B, -1)
        return self.outp(x)

class DiffusionTransformer(nn.Module):
    """
    DiT-style Transformer Backbone for 1D Latent Diffusion
    - Input: Noisy Latent z_t [B, latent_dim]
    - Conditions: t, mat, n_cells, band_mask
    - Modes: 'concat', 'adaln', 'adaln-zero', 'mhca'
    """
    def __init__(self, latent_dim: int, width: int, depth: int, heads: int, dropout: float, cfg: Any):
        super().__init__()
        
        self.width = width # 128
        self.latent_dim = latent_dim
        self.cond_mode = getattr(cfg, "cond_mode", "adaln-zero")
        print(f"[DiffusionTransformer] Conditioning Mode: {self.cond_mode.upper()}")
        
        # 1. Input Projection (z -> Tokens)
        self.seq_len = 16 
        self.input_proj = nn.Linear(latent_dim, self.seq_len * width)
        
        # 2. Condition Embeddings
        self.time_emb = SinusoidalTimeEmbedding(width)
        
        d_cond_mat = 4 
        self.material_proj = nn.Sequential(nn.Linear(d_cond_mat, width), nn.SiLU())
        
        self.ncells_embed = nn.Embedding(cfg.max_cells + 1, width)
        
        # 3. Band Mask Encoder (Transformer)
        enc_dim = getattr(cfg, "encoder_dim", width)
        enc_depth = getattr(cfg, "encoder_depth", 2)
        enc_heads = getattr(cfg, "encoder_heads", heads)
        
        if getattr(cfg, "use_dual_enc", False):
            print(f"[DiffusionTransformer] Using Dual-Channel BandMask Encoder (Dim: {enc_dim}, Depth: {enc_depth}, Heads: {enc_heads})")
            self.band_mask_enc = BandMaskCondEncoder_DualConv(cfg.n_classes, enc_dim, n_layers=enc_depth, nhead=enc_heads)
        else:
            print(f"[DiffusionTransformer] Using Standard BandMask Encoder (Dim: {enc_dim}, Depth: {enc_depth}, Heads: {enc_heads})")
            self.band_mask_enc = BandMaskCondEncoder(cfg.n_classes, enc_dim, n_layers=enc_depth, nhead=enc_heads)
        
        # 4. Positional Embedding
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, width) * 0.02)
        
        # 5. Determine Block Configuration based on Mode
        use_cross_attn = True
        use_adaln = True
        use_adaln_zero = False  # default: False (only True for "adaln-zero" mode)
        adaln_input_dim = width * 3 # Default Hybrid: t(1) + mat(1) + n(1) = 3 tokens width
        self.ctx_dim = enc_dim
        
        if self.cond_mode == "hybrid":
            # Hybrid: AdaLN(t, mat, n) & CrossAttn(mask)
            use_cross_attn = True
            use_adaln = True
            adaln_input_dim = width * 3
            
        elif self.cond_mode == "concat":
            # Concat: AdaLN(t), Concat(x, mat, n, mask)
            # CrossAttn not needed (SelfAttn handles interaction)
            use_cross_attn = False 
            use_adaln = True 
            adaln_input_dim = width # Only Time
            
            # Note: We must project the encoder dimension to model width if they differ,
            # because we are concatenating to the main sequence.
            if enc_dim != width:
                self.enc_proj = nn.Linear(enc_dim, width)
            
        elif self.cond_mode in ["adaln", "adaln-zero"]:
            # AdaLN: AdaLN(t, mat, n, pooled_mask)
            use_cross_attn = False
            use_adaln = True
            use_adaln_zero = (self.cond_mode == "adaln-zero")
            adaln_input_dim = width * 3 + enc_dim # t, mat, n, mask_vec
            
        elif self.cond_mode == "mhca":
            # MHCA: AdaLN(t), CrossAttn(mat, n, mask)
            use_cross_attn = True
            use_adaln = True
            adaln_input_dim = width # Only Time
            
            # Context dimension might need unification if mat/n width != enc_dim
            # But currently mat/n are projected to 'width'. 
            # If enc_dim != width, we might need projection for mat/n or mask.
            # Ideally context tokens should have same dimension.
            if enc_dim != width:
               self.enc_proj = nn.Linear(enc_dim, width)
               self.ctx_dim = width
            else:
               self.ctx_dim = width

        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")

        # 6. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerDiffBlock(width, heads, dropout, 
                                 context_dim=self.ctx_dim, 
                                 use_cross_attn=use_cross_attn,
                                 use_adaln=use_adaln,
                                 adaln_input_dim=adaln_input_dim,
                                 use_adaln_zero=use_adaln_zero) 
            for _ in range(depth)
        ])
        
        # 7. Final Layer
        self.norm_out = nn.LayerNorm(width)
        self.output_proj = nn.Sequential(
            nn.Linear(width * self.seq_len, latent_dim)
        )

    def forward(self, z_noisy, t, band_mask: torch.Tensor, material_conds, n_cells):
        B = z_noisy.shape[0]
        
        # 1. Input Tokens
        x = self.input_proj(z_noisy).view(B, self.seq_len, self.width) # [B, S, W]
        x = x + self.pos_emb
        
        # 2. Condition Features
        t_vec = self.time_emb(t)                # [B, W]
        m_vec = self.material_proj(material_conds) # [B, W]
        n_vec = self.ncells_embed(n_cells)      # [B, W]
        
        # Band Mask Sequence
        mask_seq = self.band_mask_enc(band_mask) # [B, K, enc_dim]
        
        # 3. Construct Wrapper Inputs based on Mode
        adaln_vec = None
        context_vec = None
        
        if self.cond_mode == "concat":
            # AdaLN: t only
            adaln_vec = t_vec # [B, W]
            
            # Tokens: mat, n, mask
            # Expand m, n to sequence [B, 1, W]
            m_tok = m_vec.unsqueeze(1)
            n_tok = n_vec.unsqueeze(1)
            
            # Project mask if needed
            if hasattr(self, 'enc_proj'):
                mask_seq = self.enc_proj(mask_seq)
                
            # Concatenate to Input X: [x, m, n, mask]
            # Transformer handles [B, S', W]
            x = torch.cat([x, m_tok, n_tok, mask_seq], dim=1) 
            
            context_vec = None # No Cross Attn
            
        elif self.cond_mode in ["adaln", "adaln-zero"]:
            # Pool Mask: Global Average Pooling
            mask_vec = mask_seq.mean(dim=1) # [B, enc_dim]
            
            # AdaLN: t, m, n, mask_vec
            adaln_vec = torch.cat([t_vec, m_vec, n_vec, mask_vec], dim=-1) # [B, 3W + enc_dim]
            
            context_vec = None # No Cross Attn
            
        elif self.cond_mode == "mhca":
            # AdaLN: t only
            adaln_vec = t_vec # [B, W]
            
            # Context: m, n, mask
            m_tok = m_vec.unsqueeze(1)
            n_tok = n_vec.unsqueeze(1)
            
            if hasattr(self, 'enc_proj'):
                mask_seq = self.enc_proj(mask_seq)
                
            context_vec = torch.cat([m_tok, n_tok, mask_seq], dim=1) # [B, 2+K, W]
            
        # 4. Process Blocks
        for block in self.blocks:
            x = block(x, adaln_vec, context_vec)
            
        # 5. Output Handling
        # If Concat mode, we must slice x back to original sequence length (first seq_len tokens)
        if self.cond_mode == "concat":
            x = x[:, :self.seq_len, :]
            
        x_out = self.norm_out(x)
        out = self.output_proj(x_out.flatten(1))
        
        return out

class TransformerDiffBlock(nn.Module):
    def __init__(self, width, heads, dropout, context_dim=None, use_cross_attn=True, use_adaln=True, adaln_input_dim=None, use_adaln_zero=False):
        super().__init__()
        if context_dim is None: context_dim = width
        self.use_cross_attn = use_cross_attn
        self.use_adaln = use_adaln
        self.use_adaln_zero = use_adaln_zero
        
        # 1. Self Attention
        self.norm1 = nn.LayerNorm(width)
        self.attn1 = nn.MultiheadAttention(width, heads, batch_first=True, dropout=dropout)
        
        # 2. Cross Attention
        if self.use_cross_attn:
            self.norm2 = nn.LayerNorm(width)
            self.attn2 = nn.MultiheadAttention(width, heads, batch_first=True, kdim=context_dim, vdim=context_dim, dropout=dropout)
        
        # 3. FFN
        self.norm3 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width*4, width),
            nn.Dropout(dropout)
        )
        
        # 4. AdaLN (Adaptive Layer Norm like)
        if self.use_adaln:
            if adaln_input_dim is None:
                adaln_input_dim = width * 3 
            
            self.num_norms = 3 if self.use_cross_attn else 2
            
            if self.use_adaln_zero:
                # 3 params per norm (gamma, beta, alpha)
                self.adaln = nn.Linear(adaln_input_dim, width * 3 * self.num_norms)
            else:
                # 2 params per norm (gamma, beta)
                self.adaln = nn.Linear(adaln_input_dim, width * 2 * self.num_norms)
                
            # Zero-init
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)
        
    def forward(self, x, adaln_vector=None, context=None):
        # x: [B, L, W]
        
        # 0. AdaLN Parameters
        gammas, betas, alphas = [], [], []
        
        if self.use_adaln and adaln_vector is not None:
            adaln_out = self.adaln(adaln_vector)
            if self.use_adaln_zero:
                chunk_size = adaln_out.size(-1) // (self.num_norms * 3)
                chunks = torch.split(adaln_out, chunk_size, dim=-1)
                for i in range(self.num_norms):
                    gammas.append(chunks[3*i])
                    betas.append(chunks[3*i+1])
                    alphas.append(chunks[3*i+2])
            else:
                chunk_size = adaln_out.size(-1) // (self.num_norms * 2)
                chunks = torch.split(adaln_out, chunk_size, dim=-1)
                for i in range(self.num_norms):
                    gammas.append(chunks[2*i])
                    betas.append(chunks[2*i+1])
        
        # Helper: Apply Modulate
        def modulate(feat, norm_idx):
            if self.use_adaln and adaln_vector is not None:
                return feat * (1 + gammas[norm_idx].unsqueeze(1)) + betas[norm_idx].unsqueeze(1)
            return feat

        def gate(attn_feat, norm_idx):
            if self.use_adaln_zero and adaln_vector is not None:
                return attn_feat * alphas[norm_idx].unsqueeze(1)
            return attn_feat
        
        # 1. Self Attn
        h = self.norm1(x)
        h = modulate(h, 0)
        attn_out, _ = self.attn1(h, h, h)
        attn_out = gate(attn_out, 0)
        x = x + attn_out
        
        # 2. Cross Attn
        if self.use_cross_attn:
            h = self.norm2(x)
            h = modulate(h, 1)
            attn_out, _ = self.attn2(h, context, context)
            attn_out = gate(attn_out, 1)
            x = x + attn_out
            norm_idx_ffn = 2
        else:
            norm_idx_ffn = 1
        
        # 3. FFN
        h = self.norm3(x)
        h = modulate(h, norm_idx_ffn)
        ffn_out = self.ffn(h)
        ffn_out = gate(ffn_out, norm_idx_ffn)
        x = x + ffn_out
        
        return x



# ---------------------------------------------------------------------------
# DDPM and Training
# ---------------------------------------------------------------------------

class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model with Classifier-Free Guidance."""
    def __init__(self, model: UNet1D, timesteps: int, beta_start: float, beta_end: float):
        super().__init__(); self.model = model; self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps); alphas = 1.0 - betas
        self.register_buffer("betas", betas); self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sc1 = self.sqrt_alphas_cumprod[t].unsqueeze(-1); sc2 = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sc1 * x0 + sc2 * noise, noise
    
    def forward(self, x0, t, band_mask: torch.Tensor, material_conds, n_cells):
        noise = torch.randn_like(x0)
        xt, true_noise = self.q_sample(x0, t, noise)
        
        # Pass all conditions to model
        predicted_noise = self.model(xt, t, band_mask, material_conds, n_cells)
        return predicted_noise, true_noise
        
    @torch.no_grad()
    def p_sample_cfg(self, x, t, band_mask: torch.Tensor, material_conds, n_cells, w: float):
        
        # 1. Conditional prediction
        eps_c = self.model(x, t, band_mask, material_conds, n_cells)
        
        # 2. Unconditional prediction
        # Uncond token = 3 (distinct from 0=Pass)
        null_band_mask = torch.full_like(band_mask, 3)
        # AdaLN conds (mat, n_cells) remain; only band_mask is replaced with null token
        eps_empty = self.model(x, t, null_band_mask, material_conds, n_cells)
        
        # 3. CFG application
        eps_cfg = eps_empty + w * (eps_c - eps_empty)
        
        # DDPM reverse diffusion
        alpha_bar = self.alphas_cumprod[t].unsqueeze(-1); alpha = self.alphas[t].unsqueeze(-1); beta = self.betas[t].unsqueeze(-1)
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_cfg)
        
        if (t == 0).all():
            return (x - torch.sqrt(1 - alpha_bar) * eps_cfg) / torch.sqrt(alpha_bar) 
        else:
            noise = torch.randn_like(x)
            x_prev = mean + torch.sqrt(beta) * noise 
            return x_prev

    @torch.no_grad()
    def sample(self, B: int, band_mask: torch.Tensor, material_conds: torch.Tensor, n_cells: torch.Tensor, z_dim: int, device: torch.device, w: float = 4.0):
        """Full DDPM sampling with CFG."""
        x = torch.randn(B, z_dim, device=device) 
        timesteps = torch.arange(self.T - 1, -1, -1, device=device, dtype=torch.long)
        
        for t_val in timesteps:
            tt = torch.full((B,), t_val, device=device, dtype=torch.long)
            x = self.p_sample_cfg(x, tt, band_mask, material_conds, n_cells, w)
            
        return x

    @torch.no_grad()
    def sample_ddim(self, B: int, band_mask: torch.Tensor, material_conds: torch.Tensor, n_cells: torch.Tensor,
                    z_dim: int, device: torch.device, w: float = 4.0, ddim_steps: int = 50, eta: float = 0.0):
        """
        Clean DDIM Sampling
        x_{t-1} = sqrt(alpha_bar_{t-1}) * pred_x0 + dir_xt + noise
        """
        # Integer linspace avoids float->int rounding producing duplicate indices
        timesteps = np.linspace(0, self.T - 1, ddim_steps, dtype=int).tolist()[::-1]

        z = torch.randn(B, z_dim, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else -1

            tt = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict Noise (CFG)
            eps_c = self.model(z, tt, band_mask, material_conds, n_cells)

            # Unconditional noise for CFG
            null_band_mask = torch.full_like(band_mask, 3)
            eps_empty = self.model(z, tt, null_band_mask, material_conds, n_cells)

            eps = eps_empty + w * (eps_c - eps_empty)

            # DDIM Constants
            alpha_bars = self.alphas_cumprod
            alpha_bar_t      = alpha_bars[t]
            alpha_bar_t_prev = alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Clamp to avoid NaN from negative sqrt argument (floating-point)
            sigma_ratio = torch.clamp(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev),
                min=0.0
            )
            sigma_t = eta * torch.sqrt(sigma_ratio)

            # Predict x0
            pred_x0 = (z - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # Compute Direction to x_t
            dir_coef = torch.clamp(1 - alpha_bar_t_prev - sigma_t ** 2, min=0.0)
            dir_xt   = torch.sqrt(dir_coef) * eps

            # Update z (x_{t-1})
            noise = torch.randn_like(z) if t_prev >= 0 else 0.0
            z = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise

            # Free intermediates to prevent VRAM accumulation
            del eps_c, eps_empty, eps, pred_x0, dir_xt, noise, tt, null_band_mask
            del alpha_bar_t, alpha_bar_t_prev, sigma_t, sigma_ratio, dir_coef

        return z


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}
        
    def state_dict(self):
        return self.shadow
        
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class LatentCondDataset(Dataset):
    def __init__(self, npz_path: str, scale_factor: float = 1.0):
        print(f"Loading VAE Latent Cache from {npz_path} (Scale Factor: {scale_factor})...")
        try:
            z = np.load(npz_path)
            self.Z = z["Z"].astype(np.float32) * scale_factor      # VAE mu * scale
            self.C_mat = z["C_mat"].astype(np.float32) # [E1, rho1, E2, rho2]
            self.N_cells = z["N_cells"].astype(np.int64) # Ncells
            self.M = z["M"].astype(np.int64)      # Band Mask
            
            if "F" in z.files:
                self.F = z["F"].astype(np.float32)
                # Ensure F is [N, K]
                if self.F.ndim == 1:
                    self.F = np.tile(self.F.reshape(1, -1), (self.Z.shape[0], 1))
            else:
                # Default F
                K = self.M.shape[1] if self.M.shape[0] > 0 else 500
                base_f = np.linspace(0, np.pi, K, dtype=np.float32)
                self.F = np.tile(base_f, (self.Z.shape[0], 1))
                
            n_total = self.Z.shape[0]
            n_defects = np.sum(np.any(self.M == 2, axis=1))
            n_normal = n_total - n_defects
            print(f"Loaded {n_total} samples. (Normal: {n_normal}, Defect: {n_defects})")
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            self.Z, self.C_mat, self.N_cells, self.M, self.F = [np.zeros((0, d)) for d in [1, 4, 1, 1, 1]]

    def __len__(self): 
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return {
            "z0": torch.from_numpy(self.Z[idx]),
            "material_conds": torch.from_numpy(self.C_mat[idx]), # [4]
            "n_cells": torch.tensor(self.N_cells[idx], dtype=torch.long), # [1]
            "band_mask": torch.from_numpy(self.M[idx]),           # [K]
            "freqs": torch.from_numpy(self.F[idx])                # [K]
        }


@torch.no_grad()
def validate_diffusion(ddpm, val_dl, cfg, device): 
    ddpm.eval()
    total_loss = 0.0; num_elements = 0
    
    with torch.no_grad():
        for batch in val_dl:
            z0 = batch["z0"].to(device)
            material_conds = batch["material_conds"].to(device)
            n_cells = batch["n_cells"].to(device)
            mask = batch["band_mask"].to(device)
            Bsz = z0.size(0)

            t = torch.randint(0, cfg.timesteps, (Bsz,), device=device).long()
            zt, noise = ddpm.q_sample(z0, t)
            pred = ddpm.model(zt, t, mask, material_conds, n_cells)
            
            loss = F.mse_loss(pred, noise, reduction='sum')
            total_loss += loss.item()
            num_elements += z0.numel() 

    avg_loss = total_loss / num_elements if num_elements > 0 else 0.0
    return avg_loss


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_latent_diffusion(cfg, device, vae_decoder=None, surrogate_classifier=None, train_paths=None, valid_paths=None, resume_path=None):
    print("\n--- Starting DDPM Training (Stage 2)")
    
    UNCOND_PROB = getattr(cfg, "uncond_prob", 0.1) 
    print(f"CFG Unconditional Dropout Probability: {UNCOND_PROB}")
    
    # GPU optimizations
    if torch.cuda.is_available():
        # 1. Enable TF32 (TensorFloat-32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 2. Enable CuDNN Benchmark
        torch.backends.cudnn.benchmark = True
        print("[System] GPU Optimization Enabled: TF32=True, Benchmark=True")
    
    # Data loaders
    scale_factor = getattr(cfg, "latent_scale_factor", 1.0)
    # paths can be passed from main.py
    
    tr_datasets = [LatentCondDataset(p, scale_factor=scale_factor) for p in train_paths]
    va_datasets = [LatentCondDataset(p, scale_factor=scale_factor) for p in valid_paths]
    
    tr_ds = ConcatDataset(tr_datasets)
    va_ds = ConcatDataset(va_datasets)
    
    print(f"[DDPM Data] Training Samples: {len(tr_ds)}")
    print(f"[DDPM Data] Validation Samples: {len(va_ds)}")
    
    # Weighted sampler for defect-enriched training
    
    n_defect_samples = 0; n_normal_samples = 0
    all_weights = []
    
    for ds in tr_datasets:
         try:
            M_local = ds.M
            defect_counts = np.sum(M_local == 2, axis=1)
            n_defect_samples += np.sum(defect_counts > 0)
            n_normal_samples += (len(ds) - np.sum(defect_counts > 0))
            
            multiplier = getattr(cfg, "defect_sample_weight", 3.0)
            w = 1.0 + (defect_counts.astype(np.float64) * multiplier)
            all_weights.append(w)
         except:
            all_weights = None
            break
            
    print(f"[DDPM Data] Training Breakdown: {n_defect_samples} Defect / {n_normal_samples} Normal")
    
    if all_weights is not None:
        sample_weights = np.concatenate(all_weights)
        s_weights = torch.from_numpy(sample_weights).float()
        sampler = torch.utils.data.WeightedRandomSampler(s_weights, num_samples=len(tr_ds), replacement=True)
        shuffle = False
        print("[Sampler] Global Weighted Random Sampler Created.")
    else:
        sampler = None
        shuffle = True
        print("[Sampler] Fallback to Random Shuffle (Weights calc failed).")

    train_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler, num_workers=getattr(cfg, "num_workers", 0), pin_memory=False, drop_last=True)
    val_dl   = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=getattr(cfg, "num_workers", 0), pin_memory=False)
    

    # Model selection
    backbone_type = getattr(cfg, "diffusion_backbone", "transformer")
    print(f"[DDPM] Using backbone: {backbone_type}")
    
    if backbone_type == "transformer":
        model = DiffusionTransformer(
            latent_dim=cfg.latent_dim,
            width=cfg.transformer_width,
            depth=cfg.transformer_depth,
            heads=cfg.transformer_heads,
            dropout=cfg.dropout,
            cfg=cfg
        ).to(device)
    else:
        model = UNet1D(cfg.latent_dim, cfg.unet_width, cfg.unet_depth, cfg.dropout, cfg).to(device) 
        
    ddpm = DDPM(model, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Initialize EMA
    ema = EMA(ddpm, decay=0.9999)
    print(f"[DDPM] EMA Enabled (Decay: 0.9999)")
    
    if resume_path and os.path.exists(resume_path):
        print(f"[DDPM] Resuming from checkpoint: {resume_path}")
        try:
            state = torch.load(resume_path, map_location=device, weights_only=True)
            if "ddpm" in state:
                ddpm.load_state_dict(state["ddpm"], strict=False)
                start_epoch = state.get("epoch", -1) + 1  # start from next epoch
                best_val_loss = state.get("best_val_loss", float('inf'))
                if "ema" in state:
                    ema.load_state_dict(state["ema"])
                    print("[DDPM] Successfully loaded EMA weights.")
                print(f"[DDPM] Successfully loaded weights. Resuming from epoch {start_epoch+1} with best_val_loss={best_val_loss:.6f}")
            else:
                ddpm.load_state_dict(state, strict=False)
                print("[DDPM] Successfully loaded legacy weights (no epoch info).")
        except Exception as e:
            print(f"[DDPM] Failed to load checkpoint: {e}")
            print("[DDPM] Starting from scratch.")

    trainable_params = list(ddpm.parameters())
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    # foreach=False avoids PyTorch 2.5.x GradScaler + LR Scheduler state_steps bug
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr_diffusion, weight_decay=weight_decay, foreach=False)
    
    # Scheduler
    num_training_steps = cfg.epochs_diffusion * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_training_steps, eta_min=1e-6)
    
    print(f"[DDPM] Scheduler: CosineAnnealingLR (Total: {num_training_steps} steps)")
    
    # AMP
    scaler = torch.amp.GradScaler('cuda')
    print("[DDPM] AMP Enabled.")
    
    # Training loop
    
    if start_epoch >= cfg.epochs_diffusion:
        print(f"[DDPM] Already reached target epochs ({start_epoch} >= {cfg.epochs_diffusion}). Skipping training.")
        out_name = f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt"
        return os.path.join(cfg.save_dir, out_name)
    
    for epoch in range(start_epoch, cfg.epochs_diffusion):
        ddpm.train()        
        pbar = tqdm(train_dl, desc=f"[DDPM] {epoch+1}/{cfg.epochs_diffusion}", total=len(train_dl), dynamic_ncols=True, mininterval=0.1)
        last_batch_loss = 0.0
        batch_idx = 0
        
        for batch in pbar:
            z0             = batch["z0"].to(device)
            material_conds = batch["material_conds"].to(device) 
            n_cells        = batch["n_cells"].to(device)
            mask           = batch["band_mask"].to(device) 

            # CFG training: dropout band mask to null token (3) with probability UNCOND_PROB
            Bsz = z0.size(0)
            dropout_mask = torch.rand(Bsz, device=device) < UNCOND_PROB
            if dropout_mask.any():
                mask_cond = mask.clone()
                mask_cond[dropout_mask] = 3
            else:
                mask_cond = mask

            opt.zero_grad(set_to_none=True)

            t = torch.randint(0, cfg.timesteps, (Bsz,), device=device).long()
            
            with torch.amp.autocast('cuda'):
                pred_noise, true_noise = ddpm(z0, t, band_mask=mask_cond, material_conds=material_conds, n_cells=n_cells)
                loss = F.mse_loss(pred_noise, true_noise)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            ema.update(ddpm)
            
            last_batch_loss = loss.item()
            pbar.set_postfix({"mae": f"{last_batch_loss:.4f}"})
            batch_idx += 1

            # Periodic memory cleanup
            if batch_idx % 100 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        # Validation with EMA weights
        ema.apply_shadow(ddpm)
        val_loss = validate_diffusion(ddpm, val_dl, cfg, device)
        # Restore active weights
        ema.restore(ddpm)
        
        print(f"Epoch {epoch+1} Val Loss (EMA): {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_name = f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt"
            
            # Temporarily apply EMA to save EMA weights as main state
            ema.apply_shadow(ddpm)
            state_dict = {
                "ddpm": ddpm.state_dict(),
                "ema": ema.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }
            torch.save(state_dict, os.path.join(cfg.save_dir, out_name))
            ema.restore(ddpm)
            
            print(f"Saved best model (loss={best_val_loss:.6f})")
    
    out_name = f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt"
    return os.path.join(cfg.save_dir, out_name)