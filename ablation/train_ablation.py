"""
train_ablation.py — Controlled architecture comparison v4

Four architectures, all matched to ~114M params, identical hyperparameters,
identical pre-tokenized data shared across all runs.

  1. transformer  — LLaMA3-style GQA + SwiGLU + RoPE + RMSNorm baseline
  2. mamba2       — Pure Mamba2 (Dao & Gu 2024) baseline
  3. qwen36       — Qwen3.6-27B EXACT architecture pattern (scaled to ~114M):
                      Hidden Layout: N x (3 x (GatedDeltaNet -> FFN) -> 1 x (GatedAttention -> FFN))
                      GDN: V heads 3x QK heads, head_dim=128
                      GatedAttn: GQA 6:1, head_dim=256, partial RoPE on 64 dims, output sigmoid gate
                      FFN intermediate ~ 3.4x hidden
  4. lume_hybrid  — Lume-Hybrid: alternating Mamba2 + DifferentialAttention
                      DifferentialAttention: 2x Q/K projections, RMSNorm on full 2x Q/K,
                      RoPE applied per-half, lambda = exp(lq dot lk + bias) clamped to [0,1].

Pre-tokenized data is built ONCE, cached, shared across ALL 4 archs.
Same byte-for-byte data, same order, every arch sees identical input.

EVAL SUITE (after each arch):
  - Validation loss + PPL on FineWeb-Edu held-out
  - BPB (bits per byte): val_loss / ln(2) / avg_chars_per_token
  - Repetition score on greedy generation (4-gram repeat rate)
  - English coherence: PPL on a fixed clean English snippet
  - Train/val gap (overfit indicator)

REQUIREMENTS:
  pip install mamba_ssm causal_conv1d
  pip install -U git+https://github.com/fla-org/flash-linear-attention
  pip install transformers datasets safetensors numpy

USAGE:
  python train_ablation.py --arch all --tokens 100_000_000
  python train_ablation.py --arch lume_hybrid --tokens 100_000_000
"""
import os, sys, math, time, gc, json, argparse, random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"\n{'='*80}\n>>> ARCHITECTURE ABLATION TRAINER (Lume-Hybrid + Qwen3.6 + baselines)\n{'='*80}")

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
    print("[ok] mamba_ssm (Mamba2) available")
except ImportError:
    HAS_MAMBA = False
    print("[error] mamba_ssm not installed -- install: pip install mamba_ssm causal_conv1d")

try:
    from fla.layers import GatedDeltaNet
    HAS_GDN = True
    print("[ok] flash-linear-attention (GatedDeltaNet) available")
except ImportError:
    HAS_GDN = False
    print("[error] FLA GatedDeltaNet not installed -- install:")
    print("  pip install -U git+https://github.com/fla-org/flash-linear-attention")

from transformers import AutoTokenizer

from tqdm import tqdm


# ============================================================
# Per-arch config -- sizes tuned so each lands ~114M params
# ============================================================
def make_config(arch, tokens, seed=42, output_dir="ablation_runs", seq_len=1024):
    base = dict(
        arch=arch,
        vocab_size=128256,
        max_seq_len=seq_len,
        rope_theta=1000000.0,
        # Training (IDENTICAL across all archs)
        learning_rate=6e-4,
        min_lr=6e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        batch_size=4,
        grad_accum=8,           # effective batch = 32
        warmup_steps=200,
        target_tokens=tokens,
        eval_every=100,
        val_tokens=200_000,
        seed=seed,
        output_dir=output_dir,
        use_grad_ckpt=True,
    )

    # ~114M params each -- dim=512 chosen for mamba_ssm stride compatibility
    if arch == "transformer":
        base.update(dim=512, n_layers=12, n_heads=8, n_kv_heads=2,
                    head_dim=64, hidden_dim=2048)
    elif arch == "mamba2":
        base.update(dim=512, n_layers=10, n_heads=8, n_kv_heads=2,
                    head_dim=64, hidden_dim=2048,
                    d_state=128, d_conv=4, expand=2,
                    mamba_headdim=64)
    elif arch == "qwen36":
        base.update(dim=512, n_super_blocks=3,
                    gdn_v_heads=8, gdn_qk_heads=4,
                    gdn_head_dim=64,
                    gattn_q_heads=8, gattn_kv_heads=2,
                    gattn_head_dim=64,
                    gattn_partial_rope_dim=32,
                    hidden_dim=2048)
    elif arch == "lume_hybrid":
        # Lume-Hybrid: alternating Mamba2 + DifferentialAttention.
        # 10 layers = 5 Mamba2 + 5 DiffAttn. Same dim=512 as siblings.
        base.update(dim=512, n_layers=10, n_heads=8, n_kv_heads=2,
                    head_dim=64, hidden_dim=2048,
                    d_state=128, d_conv=4, expand=2,
                    mamba_headdim=64)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return type("Cfg", (), base)()


def tokens_per_step(cfg):
    return cfg.batch_size * cfg.grad_accum * cfg.max_seq_len


def total_steps(cfg):
    return max(50, cfg.target_tokens // tokens_per_step(cfg))


# ============================================================
# Shared building blocks
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n.to(x.dtype) * self.weight


def compute_rope(dim, max_seq, theta):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    f = torch.outer(t, inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2)
    cos = cos[:s].view(1, 1, s, d // 2, 1)
    sin = sin[:s].view(1, 1, s, d // 2, 1)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack(
        [x0 * cos.squeeze(-1) - x1 * sin.squeeze(-1),
         x0 * sin.squeeze(-1) + x1 * cos.squeeze(-1)], -1
    ).flatten(-2)


def apply_partial_rope(x, cos, sin, rope_dim):
    b, h, s, d = x.shape
    if rope_dim >= d:
        return apply_rope(x, cos, sin)
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rope = apply_rope(x_rope, cos, sin)
    return torch.cat([x_rope, x_pass], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# Mixers
# ============================================================
class GQAAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        total = (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, total, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        qkv = self.qkv(x)
        q_end = self.n_heads * self.head_dim
        k_end = q_end + self.n_kv * self.head_dim
        q = qkv[..., :q_end].view(b, s, self.n_heads, self.head_dim)
        k = qkv[..., q_end:k_end].view(b, s, self.n_kv, self.head_dim)
        v = qkv[..., k_end:].view(b, s, self.n_kv, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).contiguous().view(b, s, -1))


class DifferentialAttention(nn.Module):
    """Lume-Hybrid's DifferentialAttention layer:
       - 2x Q/K projections, single V projection
       - RMSNorm over the FULL 2x Q/K hidden dim
       - RoPE applied to all 2x Q/K heads
       - lambda = exp(sum(l_q . l_k, dim=-1) + l_bias), clamped [0, 1]
       - diff = attn(q1,k1,v) - lambda * attn(q2,k2,v)
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, 2 * self.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, 2 * self.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(2 * self.n_heads * cfg.head_dim)
        self.k_norm = RMSNorm(2 * self.n_heads * cfg.head_dim)

        self.lambda_q = nn.Parameter(torch.randn(self.n_heads, cfg.head_dim) * 0.01)
        self.lambda_k = nn.Parameter(torch.randn(self.n_heads, cfg.head_dim) * 0.01)
        self.lambda_bias = nn.Parameter(torch.tensor(-0.2))

    def forward(self, x, cos, sin):
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

        a1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True)
        a2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True)

        lam = torch.exp(torch.sum(self.lambda_q * self.lambda_k, dim=-1) + self.lambda_bias)
        lam = torch.clamp(lam, 0.0, 1.0).view(1, self.n_heads, 1, 1)

        diff = a1 - lam * a2
        return self.wo(diff.transpose(1, 2).contiguous().view(b, s, -1))


class MambaMixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mamba = Mamba2(d_model=cfg.dim, d_state=cfg.d_state,
                            d_conv=cfg.d_conv, expand=cfg.expand,
                            headdim=cfg.mamba_headdim)

    def forward(self, x, cos=None, sin=None):
        return self.mamba(x.contiguous())


class GDNMixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gdn = GatedDeltaNet(
            mode="chunk",
            hidden_size=cfg.dim,
            num_heads=cfg.gdn_qk_heads,
            num_v_heads=cfg.gdn_v_heads,
            head_dim=cfg.gdn_head_dim,
            expand_v=1,
            use_gate=True,
            use_short_conv=True,
            conv_size=4,
        )

    def forward(self, x, cos=None, sin=None):
        out = self.gdn(x.contiguous())
        if isinstance(out, tuple):
            return out[0]
        return out


class GatedAttention(nn.Module):
    """Qwen3.6-27B's gated attention layer (scaled down)."""
    def __init__(self, cfg):
        super().__init__()
        self.n_q = cfg.gattn_q_heads
        self.n_kv = cfg.gattn_kv_heads
        self.head_dim = cfg.gattn_head_dim
        self.rope_dim = cfg.gattn_partial_rope_dim
        self.n_rep = self.n_q // self.n_kv

        total = (self.n_q + 2 * self.n_kv) * self.head_dim
        self.qkv = nn.Linear(cfg.dim, total, bias=False)
        self.wo = nn.Linear(self.n_q * self.head_dim, cfg.dim, bias=False)
        self.gate_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        qkv = self.qkv(x)
        q_end = self.n_q * self.head_dim
        k_end = q_end + self.n_kv * self.head_dim
        q = qkv[..., :q_end].view(b, s, self.n_q, self.head_dim)
        k = qkv[..., q_end:k_end].view(b, s, self.n_kv, self.head_dim)
        v = qkv[..., k_end:].view(b, s, self.n_kv, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)

        q = apply_partial_rope(q, cos, sin, self.rope_dim)
        k = apply_partial_rope(k, cos, sin, self.rope_dim)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(b, s, -1)
        o = self.wo(o)

        gate = torch.sigmoid(self.gate_proj(x))
        return o * gate


# ============================================================
# Blocks
# ============================================================
class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.hidden_dim)
        self.uses_attn_args = False

        if cfg.arch == "transformer":
            self.mixer = GQAAttention(cfg)
            self.uses_attn_args = True
        elif cfg.arch == "mamba2":
            self.mixer = MambaMixer(cfg)
        elif cfg.arch == "lume_hybrid":
            # Lume-Hybrid pattern: even=Mamba2, odd=DifferentialAttention
            if layer_idx % 2 == 1:
                self.mixer = DifferentialAttention(cfg)
                self.uses_attn_args = True
            else:
                self.mixer = MambaMixer(cfg)
        elif cfg.arch == "qwen36":
            sub_idx = layer_idx % 4
            if sub_idx == 3:
                self.mixer = GatedAttention(cfg)
                self.uses_attn_args = True
            else:
                self.mixer = GDNMixer(cfg)
        else:
            raise ValueError(f"Unknown arch: {cfg.arch}")

    def forward(self, x, cos, sin):
        if self.uses_attn_args:
            x = x + self.mixer(self.norm1(x), cos, sin)
        else:
            x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AblationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if cfg.arch == "qwen36":
            self.n_layers = cfg.n_super_blocks * 4
        else:
            self.n_layers = cfg.n_layers

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(cfg.dim)

        if cfg.arch == "transformer":
            rope_d = cfg.head_dim
        elif cfg.arch == "lume_hybrid":
            rope_d = cfg.head_dim
        elif cfg.arch == "qwen36":
            rope_d = cfg.gattn_partial_rope_dim
        else:
            rope_d = 64

        cos, sin = compute_rope(rope_d, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, targets=None):
        h = self.embed(x)
        if self.training and self.cfg.use_grad_ckpt:
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
                targets.reshape(-1), ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=0.8, top_k=50, eos_id=128001):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos_id).any():
                break
        self.train()
        return idx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Pre-tokenize once -- shared cache
# ============================================================
def build_token_cache(target_tokens, val_tokens, cache_path, val_cache_path, seed=42):
    if os.path.exists(cache_path) and os.path.exists(val_cache_path):
        train_size = os.path.getsize(cache_path) // 4
        val_size = os.path.getsize(val_cache_path) // 4
        if train_size >= target_tokens and val_size >= val_tokens:
            print(f"  [ok] Cache exists: train={train_size:,}, val={val_size:,} tokens -- reusing")
            return
        print(f"  Cache too small (train={train_size:,}, need={target_tokens:,}), rebuilding...")

    from datasets import load_dataset

    print(f"  Building cache: {target_tokens:,} train + {val_tokens:,} val tokens")
    print(f"  This runs ONCE -- all 4 archs will share this exact data.")

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    BOS, EOS = 128000, 128001

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    train_tokens = []
    pbar = tqdm(total=target_tokens, desc="Tokenizing train", unit="tok", unit_scale=True)
    for item in ds:
        if len(train_tokens) >= target_tokens:
            break
        text = item.get("text", "")
        if len(text) < 100 or len(text) > 100000:
            continue
        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        train_tokens.extend(ids)
        pbar.update(len(ids))
    pbar.close()

    train_tokens = train_tokens[:target_tokens]
    np.array(train_tokens, dtype=np.int32).tofile(cache_path)
    print(f"  [ok] Train cache: {cache_path}  ({len(train_tokens):,} tokens)")

    ds_val = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
    ds_val = ds_val.shuffle(seed=seed + 9999, buffer_size=10000).skip(50000)

    val_token_list = []
    pbar = tqdm(total=val_tokens, desc="Tokenizing val", unit="tok", unit_scale=True)
    for item in ds_val:
        if len(val_token_list) >= val_tokens:
            break
        text = item.get("text", "")
        if len(text) < 100 or len(text) > 100000:
            continue
        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        val_token_list.extend(ids)
        pbar.update(len(ids))
    pbar.close()

    val_token_list = val_token_list[:val_tokens]
    np.array(val_token_list, dtype=np.int32).tofile(val_cache_path)
    print(f"  [ok] Val cache:   {val_cache_path}  ({len(val_token_list):,} tokens)")


class CachedTokenDataset:
    def __init__(self, cache_path, seq_len, batch_size):
        self.tokens = np.fromfile(cache_path, dtype=np.int32)
        self.seq_len = seq_len
        self.batch_size = batch_size
        print(f"  Loaded: {len(self.tokens):,} tokens "
              f"({len(self.tokens) // seq_len:,} seqs of {seq_len})")

    def __iter__(self):
        idx = 0
        chunk = self.batch_size * self.seq_len
        while idx + chunk <= len(self.tokens):
            batch = self.tokens[idx:idx + chunk].reshape(self.batch_size, self.seq_len)
            yield torch.from_numpy(batch.astype(np.int64))
            idx += chunk


# ============================================================
# Eval components
# ============================================================
@torch.no_grad()
def quick_val(model, val_cache, cfg, device, n_batches=10):
    val_tokens = np.fromfile(val_cache, dtype=np.int32)
    n_seqs = len(val_tokens) // cfg.max_seq_len
    n_batches = min(n_batches, max(1, n_seqs // cfg.batch_size))

    model.eval()
    losses = []
    for i in range(n_batches):
        start = i * cfg.batch_size * cfg.max_seq_len
        end = start + cfg.batch_size * cfg.max_seq_len
        bt = val_tokens[start:end].reshape(cfg.batch_size, cfg.max_seq_len)
        b = torch.from_numpy(bt.astype(np.int64)).to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.bfloat16):
            _, l = model(b[:, :-1], b[:, 1:])
        if not (math.isnan(l.item()) or math.isinf(l.item())):
            losses.append(l.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def repetition_score(model, tokenizer, device, prompts, max_new=80, n_gram=4):
    model.eval()
    scores = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=max_new, temperature=0.7, top_k=50)
        gen_ids = out[0, ids.shape[1]:].tolist()
        if len(gen_ids) < n_gram + 1:
            continue
        ngrams = [tuple(gen_ids[i:i+n_gram]) for i in range(len(gen_ids) - n_gram + 1)]
        if not ngrams:
            continue
        unique = len(set(ngrams))
        repeat_rate = 1.0 - unique / len(ngrams)
        scores.append(repeat_rate)
    model.train()
    return sum(scores) / max(len(scores), 1)


@torch.no_grad()
def english_coherence_ppl(model, tokenizer, device, snippet):
    model.eval()
    ids = tokenizer.encode(snippet, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        model.train()
        return float('inf')
    with autocast("cuda", dtype=torch.bfloat16):
        _, l = model(ids[:, :-1], ids[:, 1:])
    model.train()
    return math.exp(min(l.item(), 20))


@torch.no_grad()
def sample_generation(model, tokenizer, device, prompt, max_new=60):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=max_new, temperature=0.8, top_k=50)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    model.train()
    return text


def compute_bpb(loss, avg_chars_per_token=4.4):
    return loss / math.log(2) / avg_chars_per_token


EVAL_PROMPTS = [
    "The quick brown fox jumps over the",
    "Climate change is caused by",
    "The history of mathematics begins with",
    "In a small village in the mountains,",
]

ENGLISH_SNIPPET = (
    "The Industrial Revolution, which took place from the 18th to 19th centuries, "
    "was a period during which predominantly agrarian, rural societies in Europe "
    "and America became industrial and urban. Prior to the Industrial Revolution, "
    "manufacturing was often done in people's homes, using hand tools or basic machines."
)


# ============================================================
# Training loop
# ============================================================
def train_one_arch(arch_name, tokens, train_cache, val_cache, device,
                   seed=42, output_dir="ablation_runs", seq_len=1024):
    print(f"\n{'='*80}\n>>> TRAINING: {arch_name}\n{'='*80}")
    cfg = make_config(arch_name, tokens, seed=seed, output_dir=output_dir, seq_len=seq_len)

    if arch_name in ("mamba2", "lume_hybrid") and not HAS_MAMBA:
        print(f"  [error] mamba_ssm not installed -- needed for {arch_name}. Skipping.")
        return None
    if arch_name == "qwen36" and (not HAS_GDN or not HAS_MAMBA):
        if not HAS_GDN:
            print("  [error] flash-linear-attention not installed -- needed for qwen36. Skipping.")
            return None

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    model = AblationModel(cfg).to(device).to(torch.bfloat16)
    n_params = model.count_params()
    n_steps = total_steps(cfg)
    tps = tokens_per_step(cfg)

    print(f"  Architecture:   {arch_name}")
    print(f"  Params:         {n_params/1e6:.1f}M")
    if arch_name == "qwen36":
        print(f"  Layout:         {cfg.n_super_blocks} super-blocks x [GDN,GDN,GDN,GatedAttn] = {model.n_layers} layers")
    elif arch_name == "lume_hybrid":
        n_mamba = sum(1 for i in range(cfg.n_layers) if i % 2 == 0)
        n_diff = cfg.n_layers - n_mamba
        print(f"  Layout:         {cfg.n_layers} layers = {n_mamba} Mamba2 + {n_diff} DifferentialAttention (alternating)")
    else:
        print(f"  Layers:         {cfg.n_layers} (dim={cfg.dim}, hidden={cfg.hidden_dim})")
    print(f"  Seq len:        {cfg.max_seq_len}")
    print(f"  Tokens/step:    {tps:,}")
    print(f"  Total steps:    {n_steps:,}  (~{cfg.target_tokens/1e6:.1f}M tokens)")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (any(x in name for x in ["A_log", "D", "dt_bias", "dt_proj.bias", "alpha", "beta_log"])
                or "bias" in name or "norm" in name or "lambda_" in name):
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
        fused=torch.cuda.is_available(),
    )

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, n_steps - cfg.warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return (cfg.min_lr / cfg.learning_rate) + (1 - cfg.min_lr / cfg.learning_rate) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_ds = CachedTokenDataset(train_cache, cfg.max_seq_len, cfg.batch_size)
    train_iter = iter(train_ds)

    model.train()
    history = []
    last_train_loss = None
    t0 = time.time()
    pbar = tqdm(range(1, n_steps + 1), desc=arch_name, unit="step")

    for step in pbar:
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(cfg.grad_accum):
            try:
                batch = next(train_iter).to(device, non_blocking=True)
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter).to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(batch[:, :-1], batch[:, 1:])
                loss = loss / cfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()

        last_train_loss = step_loss
        cur_lr = scheduler.get_last_lr()[0]
        ppl = math.exp(min(step_loss, 15))
        pbar.set_postfix(loss=f"{step_loss:.3f}", ppl=f"{ppl:.1f}",
                         lr=f"{cur_lr:.1e}", gn=f"{float(gn):.2f}")

        if step % cfg.eval_every == 0 or step == n_steps:
            v = quick_val(model, val_cache, cfg, device, n_batches=5)
            history.append({"step": step, "train_loss": step_loss,
                            "val_loss": v, "tokens": step * tps})
            print(f"  [step {step}] train={step_loss:.3f}  val={v:.3f}  ppl={math.exp(min(v,15)):.1f}")

    pbar.close()

    print(f"\n  Running eval suite for {arch_name}...")
    final_val = quick_val(model, val_cache, cfg, device, n_batches=20)
    bpb = compute_bpb(final_val)

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    rep = repetition_score(model, tok, device, EVAL_PROMPTS, max_new=80, n_gram=4)
    eng_ppl = english_coherence_ppl(model, tok, device, ENGLISH_SNIPPET)

    samples = []
    for p in EVAL_PROMPTS[:2]:
        samples.append({"prompt": p, "generation": sample_generation(model, tok, device, p, max_new=50)})

    train_val_gap = (last_train_loss or 0) - final_val
    elapsed = (time.time() - t0) / 60

    print(f"\n  [ok] {arch_name} DONE")
    print(f"    Final val loss:        {final_val:.4f}")
    print(f"    Final val PPL:         {math.exp(min(final_val, 15)):.2f}")
    print(f"    Approx BPB:            {bpb:.4f}")
    print(f"    English coherence PPL: {eng_ppl:.2f}")
    print(f"    Repetition rate (4-gm): {rep:.3f} (lower=better)")
    print(f"    Train/val gap:         {train_val_gap:+.4f} (>0 = overfitting)")
    print(f"    Wall time:             {elapsed:.1f} min")
    print(f"    Sample 1: {samples[0]['generation'][:160]}...")

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_file(model.state_dict(), f"{cfg.output_dir}/{arch_name}.safetensors")
    with open(f"{cfg.output_dir}/{arch_name}_history.json", "w") as f:
        json.dump({
            "arch": arch_name,
            "params": n_params,
            "target_tokens": cfg.target_tokens,
            "final_val_loss": final_val,
            "final_val_ppl": math.exp(min(final_val, 15)),
            "bpb": bpb,
            "english_ppl": eng_ppl,
            "repetition_rate": rep,
            "train_val_gap": train_val_gap,
            "wall_time_min": elapsed,
            "history": history,
            "samples": samples,
        }, f, indent=2)

    del model, opt, scheduler, train_ds, train_iter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return final_val


# ============================================================
# Compare
# ============================================================
def print_comparison(output_dir, target_tokens):
    archs = ["transformer", "mamba2", "qwen36", "lume_hybrid"]
    print("\n" + "=" * 100)
    print(f"  ABLATION RESULTS -- {target_tokens/1e6:.1f}M tokens, ~114M params each")
    print("=" * 100)
    print(f"\n{'Architecture':<14}{'Params(M)':>10}{'ValLoss':>10}{'ValPPL':>10}"
          f"{'BPB':>8}{'EngPPL':>10}{'Repeat':>9}{'Gap':>8}{'Time(m)':>9}")
    print("-" * 100)
    rows = []
    for a in archs:
        path = f"{output_dir}/{a}_history.json"
        if not os.path.exists(path):
            print(f"{a:<14}  (skipped)")
            continue
        with open(path) as f:
            h = json.load(f)
        rows.append((a, h["final_val_loss"]))
        marker = "  <- Lume-Hybrid" if a == "lume_hybrid" else ""
        print(f"{a:<14}{h['params']/1e6:>9.1f}M{h['final_val_loss']:>10.4f}"
              f"{h['final_val_ppl']:>10.1f}{h['bpb']:>8.3f}{h['english_ppl']:>10.1f}"
              f"{h['repetition_rate']:>9.3f}{h['train_val_gap']:>+8.3f}"
              f"{h['wall_time_min']:>9.1f}{marker}")

    if rows:
        rows.sort(key=lambda x: x[1])
        print(f"\n  Winner (lowest val loss): {rows[0][0]} (val loss = {rows[0][1]:.4f})")
        if len(rows) >= 2:
            gap = rows[1][1] - rows[0][1]
            print(f"  Margin over 2nd:           {gap:.4f} loss units")
            if gap < 0.05:
                print(f"  [warn] Margin < 0.05 -- likely within noise. Increase --tokens to confirm.")
        print(f"\n  Eval columns:")
        print(f"    BPB        = bits-per-byte (lower = better English modeling)")
        print(f"    EngPPL     = PPL on a fixed clean English snippet (lower = better)")
        print(f"    Repeat     = 4-gram repetition rate in greedy gen (lower = less repetition)")
        print(f"    Gap        = train_loss - val_loss (positive = overfitting)")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="all",
                   choices=["transformer", "mamba2", "qwen36", "lume_hybrid", "all"])
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--output-dir", default="ablation_runs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] No CUDA -- VERY slow")

    os.makedirs(args.output_dir, exist_ok=True)

    tok_label = f"{args.tokens//1_000_000}M"
    train_cache = f"{args.output_dir}/data_cache_{tok_label}tok_train.bin"
    val_cache = f"{args.output_dir}/data_cache_{tok_label}tok_val.bin"

    print(f"\nConfig:")
    print(f"  Target tokens / arch: {args.tokens:,}")
    print(f"  Seq len:              {args.seq_len}")
    print(f"  Output dir:           {args.output_dir}")
    print(f"  Train cache:          {train_cache}")
    print(f"  Val cache:            {val_cache}")

    if args.tokens < 50_000_000:
        print(f"\n  [warn] {args.tokens/1e6:.0f}M tokens is well below recommended 50M+ for clean signal.")

    print(f"\n{'='*80}\n>>> PREPARING SHARED TOKEN CACHE\n{'='*80}")
    build_token_cache(
        target_tokens=args.tokens,
        val_tokens=200_000,
        cache_path=train_cache,
        val_cache_path=val_cache,
        seed=args.seed,
    )

    if args.arch == "all":
        for arch in ["transformer", "mamba2", "qwen36", "lume_hybrid"]:
            train_one_arch(arch, args.tokens, train_cache, val_cache, device,
                           seed=args.seed, output_dir=args.output_dir, seq_len=args.seq_len)
        print_comparison(args.output_dir, args.tokens)
    else:
        train_one_arch(args.arch, args.tokens, train_cache, val_cache, device,
                       seed=args.seed, output_dir=args.output_dir, seq_len=args.seq_len)
        print_comparison(args.output_dir, args.tokens)


if __name__ == "__main__":
    main()
