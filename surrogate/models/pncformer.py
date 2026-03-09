from __future__ import annotations
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

        # To support very long seq, allow runtime slicing; max_len is a cap.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        L = x.size(1)
        if L > self.pe.size(1):
            # extend pe on-the-fly (rare)
            device = x.device
            d_model = x.size(2)
            max_len = L
            pe = torch.zeros(max_len, d_model, device=device)
            position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :L].to(x.device)
        return self.dropout(x + pe)


class PnCFormer(nn.Module):
    """
    Encoder: variable-length sequence of layer features (L, x_input_dim).
    Decoder: frequency queries (F, 1) as tokens.
    Output: (B, F) regression.
    """
    def __init__(self,
                 x_input_dim: int = 6,
                 f_input_dim: int = 1,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4,
                 dropout: float = 0.0,
                 output_seq_len: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.output_seq_len = output_seq_len

        self.x_embed = nn.Linear(x_input_dim, d_model)
        self.x_pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        self.f_embed = nn.Linear(f_input_dim, d_model)
        self.f_pos_enc = PositionalEncoding(d_model, dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, f: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None):
        # Encoder
        x_emb = self.x_embed(x)
        x_emb = self.x_pos_enc(x_emb)
        
        # 💡 Fix: Handle padding mask correctly (True=Pad) and convert to float to avoid NestedTensor issues
        if src_key_padding_mask is not None:
            enc_pad = torch.zeros_like(src_key_padding_mask, dtype=x.dtype)
            enc_pad = enc_pad.masked_fill(src_key_padding_mask, float('-inf'))
        else:
            enc_pad = None

        memory = self.encoder(x_emb, src_key_padding_mask=enc_pad)

        # Decoder
        f = f.unsqueeze(-1)                # (B, F, 1)
        f_emb = self.f_embed(f)
        f_emb = self.f_pos_enc(f_emb)
        
        # Note: memory_key_padding_mask for decoder cross-attention also takes float mask
        dec = self.decoder(tgt=f_emb, memory=memory, memory_key_padding_mask=enc_pad)
        out = self.fc_out(dec).squeeze(-1) # (B, F)
        return out