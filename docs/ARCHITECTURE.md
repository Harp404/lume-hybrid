# Lume-Hybrid Architecture

## Overview

Lume-Hybrid is a 24-layer hybrid language model. Every other layer alternates between two mixer types:

- **Mamba2** (Dao & Gu, 2024) — selective state-space mixer with O(1) recurrent state
- **Differential Attention** (Ye et al., 2024) — softmax attention that subtracts a second softmax-attention term to suppress attention noise

Each block is a standard pre-norm residual block: `x = x + Mixer(RMSNorm(x))`, followed by `x = x + SwiGLU(RMSNorm(x))`. Output logits are produced by a tied projection back through the input embedding matrix.

## Configuration (`LumeConfig`)

| Field | Value | Notes |
|---|---|---|
| `vocab_size` | 128256 | Llama-3.2 tokenizer |
| `dim` | 896 | Residual stream width |
| `n_layers` | 24 | 12 Mamba2 + 12 DiffAttn |
| `n_heads` | 14 | Attention heads (per softmax path) |
| `head_dim` | 64 | |
| `hidden_dim` | 2432 | SwiGLU FFN inner width |
| `max_seq_len` | 3072 | RoPE cache size |
| `rope_theta` | 1,000,000 | RoPE base frequency |
| `d_state` | 128 | Mamba2 SSM state dim |
| `d_conv` | 4 | Mamba2 1-D conv width |
| `expand` | 2 | Mamba2 inner expansion factor |

Total parameter count: **390.7M**.

## Block alternation

```python
class LumeBlock(nn.Module):
    def __init__(self, cfg, idx):
        ...
        self.use_attention = (idx % 2 == 1) or (not HAS_MAMBA_KERNELS)
        ...
```

| Layer index | Mixer |
|---|---|
| 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 | Mamba2 |
| 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 | DifferentialAttention |

When CUDA Mamba kernels are unavailable, all layers fall back to attention so the model is still trainable on CPU/non-CUDA hardware (much slower).

## Differential Attention

The attention layer follows Ye et al. (2024). For input `x ∈ ℝ^{B×S×D}`:

1. Project `x` to `Q ∈ ℝ^{B×S×(2·H·d)}`, `K ∈ ℝ^{B×S×(2·H·d)}`, `V ∈ ℝ^{B×S×(H·d)}`. Note Q and K have **twice** as many heads as V — the doubled head dim is what gets split into the two attention paths.
2. RMSNorm over the full Q and K vectors.
3. Reshape to heads, apply RoPE to Q and K.
4. Split Q and K into two halves along the head axis: `(Q1, Q2)`, `(K1, K2)`.
5. Compute `attn1 = softmax(Q1 K1ᵀ / √d) V` and `attn2 = softmax(Q2 K2ᵀ / √d) V` (both causal, both share the same V).
6. Compute the difference scalar `λ = exp(λ_q · λ_k + λ_bias)`, clamped to `[0, 1]`. `λ_q, λ_k ∈ ℝ^{H×d}` are learned per-head vectors; `λ_bias` is a single learned scalar (initialized to -0.2).
7. Output: `out = wo((attn1 - λ · attn2).flatten)`.

The intuition: the second softmax learns to model the noise/distractor pattern, which is then subtracted out. Empirically this gives better attention sharpness with the same parameter count as standard MHA.

## Mamba2

Wrapped from `mamba_ssm`:

```python
Mamba2(d_model=cfg.dim, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
```

A pure-PyTorch fallback (`PurePyTorchMamba`) is provided for environments without the CUDA kernels — it is significantly slower and is for compatibility, not training.

## SwiGLU FFN

```python
y = w2(silu(w1(x)) * w3(x))
```

with `w1, w3 : dim → hidden_dim` and `w2 : hidden_dim → dim`, all bias-free.

## Normalization

RMSNorm everywhere. The implementation casts to fp32 for the variance computation, then casts back to the input dtype:

```python
norm = x_fp32 * rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
return norm.to(x.dtype) * weight
```

## Position encoding

RoPE on Q and K only, in attention layers only. `theta = 1_000_000` (long-context-friendly base). `cos`/`sin` tables are precomputed at model init for `max_seq_len = 3072` and stored as buffers.

Mamba2 layers do not use positional encoding — the recurrent state itself encodes order.

## Tied embeddings

The output projection reuses the input embedding matrix:

```python
logits = F.linear(h, self.embed.weight)
```

This saves ~115M parameters (vocab × dim) at the cost of a slight quality penalty that is well-documented in the literature.

## Initialization

All `nn.Linear` and `nn.Embedding` weights are initialized from `N(0, 0.02²)`. Mamba2 internal parameters (`A_log`, `D`, `dt_bias`, `dt_proj.bias`) use the defaults from `mamba_ssm` and are excluded from weight decay during training.

## Optimizer & schedule (training)

- AdamW, `(β1, β2) = (0.9, 0.95)`
- Three parameter groups:
  - Mamba internals → `weight_decay = 0`
  - Linear/embedding weights → `weight_decay = 0.1`
  - Biases and norm weights → `weight_decay = 0`
- Linear warmup over 2000 steps, then cosine decay to `0.1 × peak_lr`
- Peak LR `5e-4`, batch size `2 × 128` grad-accum × `3072` seq → ~786K tokens / step
- 50,000 steps total (~30B tokens at this configuration)
- Gradient clipping at `1.0`
- `bfloat16` autocast
- Gradient checkpointing on every layer
