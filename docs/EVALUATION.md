# Lume-Hybrid Evaluation Methodology

## Why bits-per-byte (BPB)?

Cross-tokenizer perplexity comparisons are misleading: a model with a larger vocabulary (and therefore fewer tokens per byte of text) will appear to have lower perplexity on the same corpus, even though it is no better at modeling the underlying language. To compare Lume-Hybrid (Llama-3.2 tokenizer, 128K vocab) against Pythia-410M (NeoX tokenizer, 50K vocab) fairly, we normalize to **bits per byte**:

```
BPB = (cross_entropy_in_nats / ln(2)) × (tokens_in_chunk / bytes_in_chunk)
```

Lower BPB means the model better predicts the underlying byte stream regardless of how the tokenizer chops it up. This is the standard normalization used in the BIG-bench and many recent scaling papers.

## Eval datasets

Six held-out validation streams, all drawn from publicly available corpora and **disjoint from any training data used for either Lume-Hybrid or Pythia**:

| Stream | What it tests |
|---|---|
| `slimpajama` | Mixed web text (general distribution) |
| `fineweb_edu` | High-quality educational web content |
| `code` | GitHub-style source code |
| `math` | Mathematical text (textbooks, papers) |
| `wikitext` | Encyclopedic prose |
| `pile` | Pile validation split (Pythia in-distribution) |

For each stream, we tokenize the validation chunks, run a single forward pass per chunk (no autoregressive sampling), compute mean cross-entropy, and convert to BPB using the per-chunk byte/token ratio.

## Baseline: Pythia-410M

We compare against [Pythia-410M-deduped](https://huggingface.co/EleutherAI/pythia-410m-deduped) at two checkpoints:

- **Step ~14400 (≈30B tokens):** matched-budget comparison. Same training-token count as Lume-Hybrid.
- **Step ~143000 (≈300B tokens):** the full Pythia run. Provides a reference for what 10× more training looks like.

Pythia is a standard, well-understood baseline and is the closest match in parameter count to Lume-Hybrid. Note that Pythia trained on the Pile (which includes GitHub) while Lume-Hybrid did not — this gives Pythia an in-distribution advantage on `code` and `pile`.

## Headline results

| Model              | Params | Tokens | slimpajama | fineweb_edu | code  | math  | wikitext | pile  |
|--------------------|--------|--------|------------|-------------|-------|-------|----------|-------|
| Lume-Hybrid (ours)  | 390M   | 30B    | 0.979      | **0.914**   | 0.728 | **1.005** | 1.257  | 0.959 |
| Pythia-410M        | 410M   | 30B    | **0.959**  | 0.954       | **0.571** | 1.012 | **1.193** | **0.805** |
| Pythia-410M        | 410M   | 300B   | 0.905      | 0.898       | 0.526 | 0.962 | 1.118    | 0.753 |

Lower is better. **Bold** = best of the matched-budget pair (top two rows).

## How to read this

1. **At matched 30B-token training budget**, Lume-Hybrid beats Pythia on `fineweb_edu` (0.914 vs 0.954) and ties on `math` (1.005 vs 1.012). On the other four streams, Pythia is ahead.

2. **The Pythia advantage on `code` and `pile` is largely a training-data confound.** Pythia trained on the Pile, which includes GitHub. Lume-Hybrid did not. We document this because the result is real but the interpretation is nuanced: Lume-Hybrid is not "worse at code reasoning"; it has seen less code-like data.

3. **At 300B tokens** (10× more training than either), Pythia leads across the board. This row is included not as a fair comparison but as a reference: it tells you what continued training of a 400M-class model can achieve, and is the natural benchmark Lume-Hybrid would be measured against if training were continued.

## What this evaluation does *not* show

- **Downstream task performance.** Perplexity is a proxy for language modeling ability. We did not run lm-eval-harness, MMLU, GSM8K, etc. The model is too small and undertrained for those benchmarks to be informative; it would post near-random scores.
- **Long-context performance.** All evaluation chunks fit within the 3072-token context window. We did not test in-context retrieval at long ranges, where the Mamba2/DiffAttn hybrid should theoretically shine.
- **Generation quality at scale.** See [results/samples.json](../results/samples.json) for qualitative side-by-side comparisons against Pythia.

## Inference benchmarks

Throughput and memory comparison between naive recompute (`O(L²)` re-run every step) and the cached generation in [`inference.py`](../inference.py) (KV cache for attention, history-accumulation for Mamba). All measurements: single A100 / consumer-class GPU, bfloat16, batch size 1.

| Prefill Length | Naive ms | Cached ms | Naive Memory | Cached Memory |
|----------------|----------|-----------|--------------|---------------|
| 256            | 3371     | 3417      | 2.43 GB      | 1.79 GB       |
| 512            | 3512     | 3457      | 2.51 GB      | 1.96 GB       |

The cache implementation reduces peak memory by 22–26%. Speed is roughly equivalent in this implementation: the attention path uses a true KV cache (linear in generated tokens), but the Mamba path currently re-runs over accumulated history rather than using the proper SSM step function. Implementing true `Mamba2.step()` caching is left as future work and would yield the expected ~10× speedup.

## Reproducing

```bash
# Get weights
# Lume-Hybrid weights are not published. Train locally first:
#   python train.py train --path lume_hybrid.safetensors
# then point evaluate.py at your local checkpoint

# Run perplexity suite
python evaluate.py
```

The eval script outputs the BPB table and writes raw numbers to JSON.
