# Architecture Ablation

Controlled comparison of Lume-Hybrid against three baselines at matched parameters and matched training data.

## Methodology

All four architectures were trained with:

- Identical pre-tokenized data cache (same byte order, same train/val split)
- Identical hyperparameters (lr=6e-4, batch=32 effective, AdamW, cosine schedule)
- Matched parameter count (~114M, within ±2% across all archs)
- 100M tokens of FineWeb-Edu (sample-10BT)
- RTX 4090, bfloat16, gradient checkpointing

The only variable was the architecture itself.

## Architectures

| Arch          | Params  | Description                                             |
|---------------|---------|---------------------------------------------------------|
| transformer   | 111.3M  | LLaMA3-style: GQA + SwiGLU + RoPE + RMSNorm, 12 layers  |
| mamba2        | 114.3M  | Pure Mamba2, 10 layers                                  |
| qwen36        | 115.7M  | Qwen3.6 pattern: 3 × [GDN, GDN, GDN, GatedAttn]         |
| lume_hybrid   | 113.6M  | Alternating Mamba2 + Differential Attention, 10 layers  |

## Results

| Arch            | Val Loss   | PPL      | BPB       | English PPL | Repetition |
|-----------------|------------|----------|-----------|-------------|------------|
| transformer     | 4.5404     | 93.7     | 1.489     | 37.2        | 0.045      |
| mamba2          | 4.6420     | 103.7    | 1.522     | 30.7        | 0.000      |
| qwen36          | 4.5037     | 90.3     | 1.477     | 29.6        | 0.029      |
| **lume_hybrid** | **4.3400** | **76.7** | **1.423** | 29.7        | 0.097      |

**Lume-Hybrid wins by 0.164 val loss units** over the runner-up (qwen36), well above the noise threshold (~0.05 loss units at this scale).

## Caveats

- 100M tokens is a small training budget. Rankings may differ at larger scale.
- Lume-Hybrid has higher repetition rate (0.097) than mamba2 (0.000) — the Differential Attention layers are still learning at this token count.
- Lume-Hybrid shows slightly more overfitting tendency (gap = 0.089 vs 0.001 for mamba2). Not concerning at this scale.
- Production-scale architectures (Qwen3.6-27B trained on trillions of tokens) are not directly comparable to these scaled-down versions trained on 100M tokens.

## Reproducing

```bash
python lume/ablation/train_ablation.py --arch all --tokens 100_000_000
```

Hardware: RTX 4090 or better, 24GB+ VRAM. Total run time ~2 hours.

Dependencies: torch 2.6, mamba_ssm 2.2.4, flash-linear-attention 0.3.2, transformers 4.44.2.

## Files

- [`train_ablation.py`](train_ablation.py) — the ablation training script (4 archs, shared token cache)
- [`results.json`](results.json) — final per-arch metrics from the run summarized above
- [`history/`](history/) — per-arch training-loss histories (`{arch}_history.json`), populated by `train_ablation.py` after each run
