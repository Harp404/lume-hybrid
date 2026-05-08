"""Lume-Hybrid model definition.

A 390M-parameter hybrid language model:
  - Even-indexed layers (0, 2, 4, ...): Mamba2 selective state-space mixer
  - Odd-indexed layers  (1, 3, 5, ...): Differential Attention (Ye et al., 2024)
  - SwiGLU FFN, RMSNorm pre-norm, RoPE on attention layers, tied embeddings.

Falls back to a pure-PyTorch Mamba implementation when CUDA Mamba kernels
are unavailable (functional but much slower; not recommended for training).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba2
    HAS_MAMBA_KERNELS = True
    MAMBA_VERSION = 2
except ImportError:
    try:
        from mamba_ssm import Mamba
        HAS_MAMBA_KERNELS = True
        MAMBA_VERSION = 1
    except ImportError:
        HAS_MAMBA_KERNELS = False
        MAMBA_VERSION = 0


def backend_summary() -> str:
    if HAS_MAMBA_KERNELS:
        return f"mamba_ssm v{MAMBA_VERSION} CUDA kernels available"
    return "no CUDA kernels; pure-PyTorch fallback active"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LumeConfig:
    vocab_size: int = 128256
    dim: int = 896
    n_layers: int = 24
    n_heads: int = 14
    head_dim: int = 64
    hidden_dim: int = 2432
    max_seq_len: int = 3072
    rope_theta: float = 1_000_000.0
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    learning_rate: float = 5e-4
    batch_size: int = 2
    grad_accum_steps: int = 128
    total_steps: int = 50_000
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    use_gradient_checkpointing: bool = True


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.to(x.dtype) * self.weight


def compute_rope_freqs(
    dim: int, max_seq: int, theta: float = 1_000_000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2)
    cos = cos[:s].view(1, 1, s, d // 2, 1)
    sin = sin[:s].view(1, 1, s, d // 2, 1)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack([
        x0 * cos.squeeze(-1) - x1 * sin.squeeze(-1),
        x0 * sin.squeeze(-1) + x1 * cos.squeeze(-1),
    ], -1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LumeConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Mixers
# ---------------------------------------------------------------------------
class DifferentialAttention(nn.Module):
    """Ye et al. 2024 — two softmax attentions, the second is subtracted."""

    def __init__(self, cfg: LumeConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dim = cfg.dim

        self.wq = nn.Linear(cfg.dim, 2 * self.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, 2 * self.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(2 * self.n_heads * cfg.head_dim)
        self.k_norm = RMSNorm(2 * self.n_heads * cfg.head_dim)

        self.lambda_q = nn.Parameter(torch.randn(self.n_heads, cfg.head_dim) * 0.01)
        self.lambda_k = nn.Parameter(torch.randn(self.n_heads, cfg.head_dim) * 0.01)
        self.lambda_bias = nn.Parameter(torch.tensor(-0.2))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_norm(self.wq(x))
        k = self.k_norm(self.wk(x))
        v = self.wv(x)

        q = q.view(b, s, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q1, q2 = q.chunk(2, dim=1)
        k1, k2 = k.chunk(2, dim=1)

        attn1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        attn2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)

        lam = torch.exp(torch.sum(self.lambda_q * self.lambda_k, dim=-1) + self.lambda_bias)
        lam = torch.clamp(lam, min=0.0, max=1.0).view(1, self.n_heads, 1, 1)

        diff_attn = attn1 - lam * attn2
        return self.wo(diff_attn.transpose(1, 2).contiguous().view(b, s, -1))


class PurePyTorchMamba(nn.Module):
    """Functional fallback when mamba_ssm is unavailable. Slower; not for training."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :s]
        x_in = F.silu(x_in).transpose(1, 2)
        return self.out_proj(x_in * F.silu(res))


class MambaWrapper(nn.Module):
    def __init__(self, cfg: LumeConfig):
        super().__init__()
        if HAS_MAMBA_KERNELS:
            M_Class = Mamba2 if MAMBA_VERSION == 2 else Mamba
            self.mamba = M_Class(
                d_model=cfg.dim, d_state=cfg.d_state,
                d_conv=cfg.d_conv, expand=cfg.expand,
            )
        else:
            self.mamba = PurePyTorchMamba(
                d_model=cfg.dim, d_state=cfg.d_state,
                d_conv=cfg.d_conv, expand=cfg.expand,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# ---------------------------------------------------------------------------
# Block + Model
# ---------------------------------------------------------------------------
class LumeBlock(nn.Module):
    def __init__(self, cfg: LumeConfig, idx: int):
        super().__init__()
        # Even-indexed -> Mamba; odd-indexed -> DifferentialAttention.
        # If Mamba kernels are unavailable, fall back to attention everywhere.
        self.use_attention = (idx % 2 == 1) or (not HAS_MAMBA_KERNELS)
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.mixer = DifferentialAttention(cfg) if self.use_attention else MambaWrapper(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        if self.use_attention:
            x = residual + self.mixer(x_norm, cos, sin)
        else:
            x = residual + self.mixer(x_norm)
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class LumeModel(nn.Module):
    def __init__(self, cfg: LumeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([LumeBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)
        cos, sin = compute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        h = self.embed(x)
        if self.training and self.cfg.use_gradient_checkpointing:
            for layer in self.layers:
                h = checkpoint(layer, h, self.cos, self.sin, use_reentrant=False)
        else:
            for layer in self.layers:
                h = layer(h, self.cos, self.sin)
        h = self.norm(h)
        logits = F.linear(h, self.embed.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        eos_token_id: int = 128001,
    ) -> torch.Tensor:
        """Naive autoregressive generation. For cached generation, see inference.py."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.max_seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
                if (next_token == eos_token_id).any():
                    break
        finally:
            if was_training:
                self.train()
        return idx

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
