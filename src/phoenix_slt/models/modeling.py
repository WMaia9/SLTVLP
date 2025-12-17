"""Model architectures for PHOENIX-2014-T SLT (encoders and seq2seq)."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration

from phoenix_slt.config import D_MODEL, ENCODER_LR, FF_EXPANSION, KPTS_FEAT_DIM, MBART_DIM, N_HEADS, SIGLIP_DIM


class LayerNorm(nn.Module):
    """Wrapper around nn.LayerNorm to keep consistent eps and typing."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class PositionalEncoding(nn.Module):
    """Sine-cosine positional encodings added to sequence features."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


class FeedForwardModule(nn.Module):
    """Position-wise FFN used inside Squeezeformer blocks."""
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.ln = LayerNorm(d_model)
        self.lin1 = nn.Linear(d_model, inner_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(inner_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.lin1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.lin2(y)
        y = self.dropout(y)
        return y


class ConvModule(nn.Module):
    """Depthwise-separable conv block with GLU-style gating."""
    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = y.transpose(1, 2)
        y = self.pointwise_in(y)
        y, gate = y.chunk(2, dim=1)
        y = y * torch.sigmoid(gate)
        y = self.depthwise(y)
        y = self.bn(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return y


class MHABlock(nn.Module):
    """Pre-LN multi-head self-attention with dropout."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ln = LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.ln(x)
        attn_out, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask)
        y = self.dropout(attn_out)
        return y


class StochasticDepth(nn.Module):
    """Implements drop-path regularization for residual branches."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x, residual):
        if not self.training or self.p == 0.0:
            return residual + x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return residual + x * random_tensor / keep_prob


class SqueezeformerBlock(nn.Module):
    """Squeezeformer encoder block (FFN → MHA → Conv → FFN)."""
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        conv_kernel: int = 15,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mha = MHABlock(d_model, num_heads, dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.sd_ff1 = StochasticDepth(drop_path)
        self.sd_mha = StochasticDepth(drop_path)
        self.sd_conv = StochasticDepth(drop_path)
        self.sd_ff2 = StochasticDepth(drop_path)
        self.ln_out = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        delta = self.ff1(x)
        x = self.sd_ff1(0.5 * delta, x)
        delta = self.mha(x, key_padding_mask=key_padding_mask)
        x = self.sd_mha(delta, x)
        delta = self.conv(x)
        x = self.sd_conv(delta, x)
        delta = self.ff2(x)
        x = self.sd_ff2(0.5 * delta, x)
        x = self.ln_out(x)
        return x


def make_drop_path_list(num_layers: int, max_drop: float = 0.1):
    """Linearly scale drop-path probabilities across layers."""
    if num_layers <= 1:
        return [0.0]
    return [max_drop * (i / (num_layers - 1)) for i in range(num_layers)]


class TemporalTransformer(nn.Module):
    """Lightweight transformer encoder over temporal SigLIP features."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.ln(x)


class GatedFusion(nn.Module):
    """Fuse keypoint and visual embeddings via learnable gate."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x_k: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_k, x_v], dim=-1)
        h = torch.tanh(self.proj(h))
        g = torch.sigmoid(self.gate(h))
        out = g * x_k + (1 - g) * x_v
        return out


class SqueezeformerFusionEncoder(nn.Module):
    """Fuses keypoints and SigLIP embeddings with Squeezeformer backbone."""
    def __init__(
        self,
        kpts_in_dim: int = KPTS_FEAT_DIM,
        vlm_in_dim: int = SIGLIP_DIM,
        d_model: int = D_MODEL,
        num_layers: int = 4,
        num_heads: int = N_HEADS,
        ff_expansion: int = FF_EXPANSION,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.kpts_proj = nn.Linear(kpts_in_dim, d_model)
        self.vlm_proj = nn.Linear(vlm_in_dim, d_model)

        self.temporal_vlm = TemporalTransformer(
            d_model=d_model,
            num_heads=max(1, num_heads // 2),
            dropout=dropout,
        )

        self.fusion = GatedFusion(d_model)
        self.pos_enc = PositionalEncoding(d_model)

        drop_paths = make_drop_path_list(num_layers, max_drop=0.1)
        self.blocks = nn.ModuleList(
            [
                SqueezeformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_expansion=ff_expansion,
                    dropout=dropout,
                    conv_kernel=15,
                    drop_path=drop_paths[i],
                )
                for i in range(num_layers)
            ]
        )
        self.ln_out = LayerNorm(d_model)

    def forward(
        self,
        kpts: torch.Tensor,
        kpts_mask: torch.Tensor,
        siglip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_k = self.kpts_proj(kpts)
        x_v = self.vlm_proj(siglip)
        x_v = self.temporal_vlm(x_v)

        x_v = x_v.transpose(1, 2)
        x_v = F.interpolate(x_v, size=x_k.shape[1], mode="linear", align_corners=False)
        x_v = x_v.transpose(1, 2)

        x = self.fusion(x_k, x_v)
        x = self.pos_enc(x)
        key_padding_mask = kpts_mask == 0

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.ln_out(x)
        return x, key_padding_mask


class VLP_PretrainingModel(nn.Module):
    """Contrastive visual-language pretraining head over fusion encoder."""
    def __init__(self, encoder, text_model: str = "facebook/mbart-large-cc25"):
        super().__init__()
        self.visual_encoder = encoder
        mbart = MBartForConditionalGeneration.from_pretrained(text_model)
        self.text_encoder = mbart.model.encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.vis_head = nn.Linear(D_MODEL, 256)
        self.text_head = nn.Linear(MBART_DIM, 256)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    def encode_visual(self, batch) -> torch.Tensor:
        vis_feat, _ = self.visual_encoder(batch["kpts"], batch["kpts_mask"], batch["siglip"])
        mask = batch["kpts_mask"].unsqueeze(-1)
        vis_sum = (vis_feat * mask).sum(dim=1)
        vis_len = mask.sum(dim=1).clamp(min=1.0)
        vis_vec = vis_sum / vis_len
        vis_vec = self.vis_head(vis_vec)
        vis_vec = F.normalize(vis_vec, dim=-1)
        return vis_vec

    def encode_text(self, batch) -> torch.Tensor:
        with torch.no_grad():
            txt_out = self.text_encoder(
                input_ids=batch["labels"], attention_mask=batch["labels_mask"]
            )
            txt_feat = txt_out.last_hidden_state

        mask = batch["labels_mask"].unsqueeze(-1)
        txt_sum = (txt_feat * mask).sum(dim=1)
        txt_len = mask.sum(dim=1).clamp(min=1.0)
        txt_vec = txt_sum / txt_len
        txt_vec = self.text_head(txt_vec)
        txt_vec = F.normalize(txt_vec, dim=-1)
        return txt_vec

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vis_vec = self.encode_visual(batch)
        txt_vec = self.encode_text(batch)
        scale = self.logit_scale.exp()
        return vis_vec, txt_vec, scale


class SignTranslationModel(nn.Module):
    """End-to-end SLT model: fusion encoder + adapter + mBART decoder."""
    def __init__(self, encoder, text_model: str = "facebook/mbart-large-cc25"):
        super().__init__()
        self.encoder = encoder
        self.adapter = nn.Linear(D_MODEL, MBART_DIM)
        self.mbart = MBartForConditionalGeneration.from_pretrained(text_model)

        self.mbart.model.shared.requires_grad_(False)
        self.mbart.model.encoder.embed_positions.requires_grad_(False)
        self.mbart.model.decoder.embed_positions.requires_grad_(False)
        self.mbart.model.encoder.requires_grad_(False)

        for param in self.mbart.model.decoder.parameters():
            param.requires_grad = True

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, enc_mask = self.encoder(batch["kpts"], batch["kpts_mask"], batch["siglip"])
        enc_out = self.adapter(enc_out)
        outputs = self.mbart(
            inputs_embeds=enc_out,
            attention_mask=batch["kpts_mask"],
            labels=batch["labels"],
        )
        return outputs.loss, outputs.logits

    def generate(
        self, batch, tokenizer, max_new_tokens: int = 60
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            enc_out, _ = self.encoder(batch["kpts"], batch["kpts_mask"], batch["siglip"])
            enc_out = self.adapter(enc_out)
            generated_ids = self.mbart.generate(
                inputs_embeds=enc_out,
                attention_mask=batch["kpts_mask"],
                max_new_tokens=max_new_tokens,
                num_beams=5,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
                eos_token_id=tokenizer.eos_token_id,
            )
        return generated_ids
