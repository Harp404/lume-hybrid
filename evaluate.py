#!/usr/bin/env python
"""
run_eval_v6.py — Cached generation + creativity comparison.

Vs v5:
  1. KV cache + Mamba SSM state cache for Lume-Hybrid (inference.py).
     Should give ~5-30× speedup on long context generation.
     Falls back gracefully if mamba_ssm.step() not available.

  2. Creativity comparison — multiple temperatures + diverse prompts.
     - Low temp (0.3): factual, code
     - Medium temp (0.7): conversational
     - High temp (1.0): creative writing, brainstorming
     Lets you SEE how creative each model is at high temp.

  3. Longer generation samples (150 tokens) so you actually see what each model
     does over a longer span — repetition? coherence? wandering off?

  4. Multiple seeds per prompt so you can see variance.
"""
import os, sys, json, glob, time, gc, math, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate

from modeling import LumeConfig, LumeModel
from inference import generate_cached


# ============================================================
# Config
# ============================================================
LMEVAL_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,lambada_openai,wikitext"

BASELINES = [
    ("pythia_410m_step14000", "EleutherAI/pythia-410m-deduped", {"revision": "step14000"},  29.36, "EleutherAI/pythia-410m-deduped", 2048),
    ("pythia_410m_full",      "EleutherAI/pythia-410m-deduped", {},                        299.89, "EleutherAI/pythia-410m-deduped", 2048),
    ("opt_350m",              "facebook/opt-350m",              {},                        300.00, "facebook/opt-350m",               1900),
    ("smollm2_360m",          "HuggingFaceTB/SmolLM2-360M",     {},                       4000.00, "HuggingFaceTB/SmolLM2-360M",      8192),
]

PRETTY = {
    "lume-hybrid":                    ("Lume-Hybrid (ours)",   30.00,  "Mamba2+DiffAttn", "SlimPaj+FW-Edu+Stack"),
    "pythia_410m_step14000":  ("Pythia-410M",  29.36,  "Transformer",     "Pile"),
    "pythia_410m_full":       ("Pythia-410M",  299.89, "Transformer",     "Pile"),
    "opt_350m":               ("OPT-350M",     300.00, "Transformer",     "BookCorpus+CC"),
    "smollm2_360m":           ("SmolLM2-360M", 4000.00,"Transformer",     "FineWeb-Edu"),
}


# ============================================================
# Lume-Hybrid wrapper for lm-eval
# ============================================================
@register_model("lume-hybrid")
class LumeEval(LM):
    def __init__(self, checkpoint_path="lume_hybrid_weights/lume_hybrid.safetensors",
                 batch_size=4, tokenizer="meta-llama/Llama-3.2-1B", **kwargs):
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = LumeConfig()
        print(f"[lume] Loading {checkpoint_path}...")
        self.model = LumeModel(self.cfg).to(self._device)
        sd = load_file(checkpoint_path)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._batch_size = int(batch_size)

    @property
    def eot_token_id(self):  return 128001
    @property
    def max_length(self):    return self.cfg.max_seq_len
    @property
    def max_gen_toks(self):  return 256
    @property
    def batch_size(self):    return self._batch_size
    @property
    def device(self):        return self._device

    def tok_encode(self, s):       return self.tokenizer.encode(s, add_special_tokens=False)
    def tok_decode(self, toks):    return self.tokenizer.decode(toks)

    def _encode_pair(self, ctx, cont):
        c = self.tok_encode(ctx); n = self.tok_encode(cont)
        max_ctx = self.max_length - len(n) - 1
        c = c[-max_ctx:] if max_ctx > 0 else c[-1:]
        return c + n, len(n)

    @torch.no_grad()
    def loglikelihood(self, requests):
        results = []
        for i in range(0, len(requests), self._batch_size):
            batch = requests[i:i + self._batch_size]
            enc, clens = [], []
            for r in batch:
                ids, cl = self._encode_pair(r.args[0], r.args[1])
                enc.append(ids); clens.append(cl)
            mlen = max(len(e) for e in enc)
            padded = [e + [128002] * (mlen - len(e)) for e in enc]
            x = torch.tensor(padded, dtype=torch.long, device=self._device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = self.model(x)
            for j, (ids, cl) in enumerate(zip(enc, clens)):
                cs = len(ids) - cl
                cl_logits = logits[j, cs - 1:len(ids) - 1].float()
                cl_tokens = torch.tensor(ids[cs:], dtype=torch.long, device=self._device)
                lp = F.log_softmax(cl_logits, dim=-1)
                token_lp = lp[range(cl), cl_tokens]
                results.append((token_lp.sum().item(),
                                bool((cl_logits.argmax(-1) == cl_tokens).all().item())))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        out = []
        for r in requests:
            ids = self.tok_encode(r.args[0])
            total = 0.0
            stride = self.max_length - 1
            for s in range(0, max(1, len(ids) - 1), stride):
                ch = ids[s:s + self.max_length]
                if len(ch) < 2: continue
                inp = torch.tensor([ch], dtype=torch.long, device=self._device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = self.model(inp)
                lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
                tg = torch.tensor(ch[1:], dtype=torch.long, device=self._device)
                total += lp[range(len(tg)), tg].sum().item()
            out.append(total)
        return out

    def generate_until(self, requests):
        out = []
        for r in requests:
            ctx, gk = r.args
            ids = self.tok_encode(ctx)
            inp = torch.tensor([ids], dtype=torch.long, device=self._device)
            mx = gk.get("max_gen_toks", 128) if isinstance(gk, dict) else 128
            gen = generate_cached(self.model, inp, max_new_tokens=mx,
                                  temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                                  use_cache=True)
            out.append(self.tok_decode(gen[0][len(ids):].tolist()))
        return out


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================
# (1) Multi-domain perplexity  (same as v5)
# ============================================================
def load_eval_corpora():
    from datasets import load_dataset
    corpora = {}
    print("\n[Loading eval corpora]")

    try:
        ds = load_dataset("gmongaras/SlimPajama-627B_Reupload", split="train", streaming=True)
        ds = ds.skip(100000)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 200: break
            t = ex.get("text", "")
            if len(t.strip()) > 100: texts.append(t)
        corpora["slimpajama"] = texts
        print(f"  slimpajama: {len(texts)} docs")
    except Exception as e:
        print(f"  slimpajama: SKIP ({str(e)[:80]})")

    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                          split="train", streaming=True)
        ds = ds.skip(50000)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 200: break
            t = ex.get("text", "")
            if len(t.strip()) > 100: texts.append(t)
        corpora["fineweb_edu"] = texts
        print(f"  fineweb_edu: {len(texts)} docs")
    except Exception as e:
        print(f"  fineweb_edu: SKIP ({str(e)[:80]})")

    try:
        ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        ds = ds.skip(20000)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 200: break
            t = ex.get("text", "")
            if len(t.strip()) > 100: texts.append(t)
        corpora["math"] = texts
        print(f"  math: {len(texts)} docs")
    except Exception as e:
        print(f"  math: SKIP ({str(e)[:80]})")

    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:200]
        corpora["wikitext"] = texts
        print(f"  wikitext: {len(texts)} docs")
    except Exception as e:
        print(f"  wikitext: SKIP ({str(e)[:80]})")

    try:
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 200: break
            t = ex.get("text", "")
            if len(t.strip()) > 100: texts.append(t)
        corpora["pile"] = texts
        print(f"  pile: {len(texts)} docs")
    except Exception as e:
        print(f"  pile: SKIP ({str(e)[:80]})")

    return corpora


@torch.no_grad()
def eval_ppl_lume(model, tokenizer, texts, max_len=2048):
    device = next(model.parameters()).device
    total_loss, total_tok, total_bytes = 0.0, 0, 0
    for text in texts:
        b = len(text.encode("utf-8"))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2: continue
        for s in range(0, len(ids) - 1, max_len - 1):
            chunk = ids[s:s + max_len]
            if len(chunk) < 2: continue
            x = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
            tg = torch.tensor(chunk[1:], dtype=torch.long, device=device)
            total_loss += -lp[range(len(tg)), tg].sum().item()
            total_tok += len(tg)
        total_bytes += b
    avg_loss = total_loss / total_tok if total_tok else float("inf")
    return {"ppl": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
            "bpb": total_loss / (math.log(2) * total_bytes) if total_bytes else float("inf"),
            "loss": avg_loss, "tokens": total_tok, "bytes": total_bytes}


@torch.no_grad()
def eval_ppl_hf(model, tokenizer, texts, max_len=2048):
    device = next(model.parameters()).device
    total_loss, total_tok, total_bytes = 0.0, 0, 0
    for text in texts:
        b = len(text.encode("utf-8"))
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2: continue
        for s in range(0, len(ids) - 1, max_len - 1):
            chunk = ids[s:s + max_len]
            if len(chunk) < 2: continue
            x = torch.tensor([chunk], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out[0]
            lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
            tg = torch.tensor(chunk[1:], dtype=torch.long, device=device)
            total_loss += -lp[range(len(tg)), tg].sum().item()
            total_tok += len(tg)
        total_bytes += b
    avg_loss = total_loss / total_tok if total_tok else float("inf")
    return {"ppl": math.exp(avg_loss) if avg_loss < 20 else float("inf"),
            "bpb": total_loss / (math.log(2) * total_bytes) if total_bytes else float("inf"),
            "loss": avg_loss, "tokens": total_tok, "bytes": total_bytes}


def run_ppl_lume(checkpoint_path, corpora):
    print("\n" + "=" * 80 + "\n  Multi-domain Perplexity — Lume-Hybrid\n" + "=" * 80)
    free_gpu()
    cfg = LumeConfig()
    model = LumeModel(cfg).cuda().eval()
    sd = load_file(checkpoint_path)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    out = {}
    print(f"\n  {'Domain':<14}  {'Tokens':>10}  {'PPL':>10}  {'BPB':>8}")
    for name, texts in corpora.items():
        r = eval_ppl_lume(model, tok, texts, max_len=min(2048, cfg.max_seq_len))
        print(f"  {name:<14}  {r['tokens']:>10}  {r['ppl']:>10.2f}  {r['bpb']:>7.3f}")
        out[name] = r
    del model; free_gpu()
    return out


def run_ppl_hf(repo, kwargs, tok_repo, corpora, label):
    print(f"\n" + "=" * 80 + f"\n  Multi-domain Perplexity — {label}\n" + "=" * 80)
    free_gpu()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16, trust_remote_code=True, **kwargs,
        ).cuda().eval()
        tok = AutoTokenizer.from_pretrained(tok_repo)
    except Exception as e:
        print(f"  [skip] {e}")
        return {}
    out = {}
    max_pos = getattr(model.config, "max_position_embeddings", 2048)
    max_len = min(2048, max_pos)
    print(f"\n  {'Domain':<14}  {'Tokens':>10}  {'PPL':>10}  {'BPB':>8}")
    for n, texts in corpora.items():
        r = eval_ppl_hf(model, tok, texts, max_len=max_len)
        print(f"  {n:<14}  {r['tokens']:>10}  {r['ppl']:>10.2f}  {r['bpb']:>7.3f}")
        out[n] = r
    del model, tok; free_gpu()
    return out


# ============================================================
# (2) Inference benchmarks — Lume-Hybrid cached vs naive generation
# ============================================================
@torch.no_grad()
def bench_generation_lume_cached(model, prefill_len, gen_tokens=64,
                                 batch=1, warmup=1, runs=2):
    """Lume-Hybrid with KV + Mamba state cache."""
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(2, 32000, (batch, prefill_len), device=device)
    for _ in range(warmup):
        _ = generate_cached(model, x, max_new_tokens=gen_tokens,
                            temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                            use_cache=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = generate_cached(model, x, max_new_tokens=gen_tokens,
                            temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                            use_cache=True)
    torch.cuda.synchronize()
    total_dt = (time.time() - t0) / runs
    return {"prefill_len": prefill_len, "gen_tokens": gen_tokens,
            "total_ms": total_dt * 1000, "tok_per_sec": gen_tokens / total_dt,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "mode": "cached"}


@torch.no_grad()
def bench_generation_lume_naive(model, prefill_len, gen_tokens=64,
                                batch=1, warmup=1, runs=2):
    """Lume-Hybrid without cache (for comparison)."""
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(2, 32000, (batch, prefill_len), device=device)
    for _ in range(warmup):
        _ = generate_cached(model, x, max_new_tokens=gen_tokens,
                            temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                            use_cache=False)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = generate_cached(model, x, max_new_tokens=gen_tokens,
                            temperature=1.0, top_p=1.0, repetition_penalty=1.0,
                            use_cache=False)
    torch.cuda.synchronize()
    total_dt = (time.time() - t0) / runs
    return {"prefill_len": prefill_len, "gen_tokens": gen_tokens,
            "total_ms": total_dt * 1000, "tok_per_sec": gen_tokens / total_dt,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "mode": "naive"}


@torch.no_grad()
def bench_generation_hf(model, tokenizer, prefill_len, gen_tokens=64,
                        batch=1, warmup=1, runs=2):
    device = next(model.parameters()).device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vs = min(32000, model.config.vocab_size)
    x = torch.randint(2, vs, (batch, prefill_len), device=device)
    pad = tokenizer.eos_token_id or 0
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model.generate(x, max_new_tokens=gen_tokens, do_sample=False,
                               use_cache=True, pad_token_id=pad)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model.generate(x, max_new_tokens=gen_tokens, do_sample=False,
                               use_cache=True, pad_token_id=pad)
    torch.cuda.synchronize()
    total_dt = (time.time() - t0) / runs
    return {"prefill_len": prefill_len, "gen_tokens": gen_tokens,
            "total_ms": total_dt * 1000, "tok_per_sec": gen_tokens / total_dt,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "mode": "kv_cache"}


def run_inference_bench(checkpoint_path):
    print("\n" + "=" * 90)
    print("  INFERENCE BENCHMARKS — both Lume-Hybrid (cached + naive) and baselines")
    print("=" * 90)
    seq_lens = [256, 512, 1024, 1900]

    free_gpu()
    cfg = LumeConfig()
    lume = LumeModel(cfg).cuda().eval()
    sd = load_file(checkpoint_path)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    lume.load_state_dict(sd, strict=False)

    print(f"\n--- Lume-Hybrid GENERATION (naive vs cached) — same workload ---")
    print(f"\n{'PrefillLen':>11}  {'Naive ms':>10}  {'Cached ms':>11}  {'Speedup':>10}  "
          f"{'Naive GB':>10}  {'Cached GB':>11}")
    print("-" * 70)
    lume_naive = {}; lume_cached = {}
    for L in seq_lens:
        if L > cfg.max_seq_len: continue
        n = bench_generation_lume_naive(lume, L, gen_tokens=64)
        c = bench_generation_lume_cached(lume, L, gen_tokens=64)
        lume_naive[L] = n; lume_cached[L] = c
        speedup = n['total_ms'] / c['total_ms'] if c['total_ms'] > 0 else 0
        print(f"{L:>11}  {n['total_ms']:>10.1f}  {c['total_ms']:>11.1f}  "
              f"{speedup:>9.2f}x  {n['peak_gb']:>10.2f}  {c['peak_gb']:>11.2f}")

    del lume; free_gpu()

    hf_gen_all = {}
    for name, repo, extra, _, tok_repo, max_safe in BASELINES:
        free_gpu()
        label = PRETTY[name][0]
        print(f"\n[Loading {label}]")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, trust_remote_code=True, **extra,
            ).cuda().eval()
            tok = AutoTokenizer.from_pretrained(tok_repo)
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id or 0
        except Exception as e:
            print(f"  [skip] {e}"); continue
        max_pos = getattr(model.config, "max_position_embeddings", 4096)

        print(f"\n--- GENERATION: {label} (KV cache) ---")
        print(f"\n{'PrefillLen':>11}{'Total ms':>12}{'tok/s':>10}{'PeakMem GB':>14}")
        print("-" * 50)
        gen_res = {}
        for L in seq_lens:
            if L > max_pos - 64 or L > max_safe - 64: continue
            r = bench_generation_hf(model, tok, L, gen_tokens=64)
            gen_res[L] = r
            print(f"{L:>11}{r['total_ms']:>12.1f}{r['tok_per_sec']:>10.1f}{r['peak_gb']:>13.2f}")
        hf_gen_all[name] = gen_res
        del model, tok; free_gpu()

    return {"lume_naive": lume_naive, "lume_cached": lume_cached,
            "hf_generation": hf_gen_all}


# ============================================================
# (3) Creativity comparison — multiple temperatures
# ============================================================
CREATIVITY_PROMPTS = [
    # (tag, prompt, temperature, top_p, max_new)
    # Factual / analytic — should be deterministic-ish
    ("factual_capital",   "The capital of France is",                                                 0.3, 0.9, 30),
    ("factual_planet",    "The largest planet in our solar system is",                                0.3, 0.9, 30),
    # Code — low temp, structured
    ("code_fib",          "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",   0.3, 0.9, 100),
    ("code_sort",         "def quicksort(arr):\n    \"\"\"Sort the array using quicksort.\"\"\"\n",  0.3, 0.9, 120),
    # Conversational — medium temp
    ("conv_advice",       "What's a good way to learn programming? Here are some tips:\n1.",         0.7, 0.9, 150),
    ("conv_explain",      "Black holes are fascinating because",                                     0.7, 0.9, 150),
    # Creative writing — high temp, free
    ("creative_story",    "Once upon a time, in a small village by the sea, there lived a",          1.0, 0.95, 200),
    ("creative_dragon",   "The dragon opened its eyes and saw, for the first time in a thousand years,", 1.0, 0.95, 200),
    ("creative_robot",    "The robot asked, \"Do you think I'm alive?\" The scientist replied,",     1.0, 0.95, 200),
    # Brainstorm
    ("brainstorm_ideas",  "Here are five wild and creative ideas for a startup:\n1.",                1.0, 0.95, 250),
    # Reasoning
    ("reason_math",       "If a train leaves Paris at 8 AM going 100 km/h, and another leaves Lyon at 9 AM going 80 km/h in the opposite direction, ", 0.5, 0.9, 150),
]


def gen_with_lume(model, tokenizer, prompt, temperature, top_p, max_new, seed=42):
    """Cached generation via inference."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    out = generate_cached(
        model, x, max_new_tokens=max_new,
        temperature=temperature, top_p=top_p,
        repetition_penalty=1.15, use_cache=True,
    )
    gen_ids = out[0, x.shape[1]:].tolist()
    return tokenizer.decode(gen_ids)


def gen_with_hf(model, tokenizer, prompt, temperature, top_p, max_new, seed=42):
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    pad = tokenizer.eos_token_id or 0
    torch.manual_seed(seed)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p,
            repetition_penalty=1.15, pad_token_id=pad,
        )
    gen_ids = out[0, ids.shape[1]:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_creativity_eval(checkpoint_path):
    print("\n" + "=" * 100)
    print("  CREATIVITY COMPARISON")
    print("=" * 100)
    print("  Same prompts, same seeds, multiple temperatures.")
    print("  - Factual: temp=0.3 (should be deterministic)")
    print("  - Conversational: temp=0.7 (some variety)")
    print("  - Creative: temp=1.0 (high variety, see personality)")
    print("  - Repetition penalty: 1.15 across all\n")

    all_samples = {}

    # Lume-Hybrid
    free_gpu()
    print(f"\n{'─'*100}\n  ▶ Lume-Hybrid (ours) — 30B tokens, Mamba2+DiffAttn hybrid\n{'─'*100}")
    cfg = LumeConfig()
    lume = LumeModel(cfg).cuda().eval()
    sd = load_file(checkpoint_path)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    lume.load_state_dict(sd, strict=False)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    samples = {}
    for tag, prompt, temp, tp, mn in CREATIVITY_PROMPTS:
        try:
            text = gen_with_lume(lume, tok, prompt, temp, tp, mn)
        except Exception as e:
            text = f"[ERROR: {e}]"
        samples[tag] = {"prompt": prompt, "completion": text,
                        "temperature": temp, "top_p": tp}
        print(f"\n  [{tag}] (T={temp}) {prompt}")
        print(f"  → {text[:500]}")
    all_samples["lume-hybrid"] = samples
    del lume, tok; free_gpu()

    # HF baselines
    for name, repo, extra, _, tok_repo, _ in BASELINES:
        free_gpu()
        label = PRETTY[name][0]
        toks = PRETTY[name][1]
        print(f"\n{'─'*100}\n  ▶ {label} — {toks:.0f}B tokens, {PRETTY[name][2]}\n{'─'*100}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, trust_remote_code=True, **extra,
            ).cuda().eval()
            tok = AutoTokenizer.from_pretrained(tok_repo)
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id or 0
        except Exception as e:
            print(f"  [skip] {e}"); continue
        samples = {}
        for tag, prompt, temp, tp, mn in CREATIVITY_PROMPTS:
            try:
                text = gen_with_hf(model, tok, prompt, temp, tp, mn)
            except Exception as e:
                text = f"[ERROR: {e}]"
            samples[tag] = {"prompt": prompt, "completion": text,
                            "temperature": temp, "top_p": tp}
            print(f"\n  [{tag}] (T={temp}) {prompt}")
            print(f"  → {text[:500]}")
        all_samples[name] = samples
        del model, tok; free_gpu()

    return all_samples


# ============================================================
# (4) lm-eval standard
# ============================================================
def run_lm_eval_lume(checkpoint, tasks, batch_size, out_path):
    if os.path.exists(out_path):
        print(f"  [skip] Skipping Lume-Hybrid lm-eval"); return
    task_list = [t.strip() for t in tasks.split(",")]
    t0 = time.time()
    results = simple_evaluate(
        model="lume-hybrid", model_args=f"checkpoint_path={checkpoint}",
        tasks=task_list, num_fewshot=0, batch_size=batch_size,
    )
    if isinstance(results, dict): results.pop("samples", None)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [ok] done in {(time.time()-t0)/60:.1f} min")


def run_lm_eval_baseline(name, repo, extra, tasks, batch_size, out_dir):
    out_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(out_path):
        print(f"  [skip] Skipping {name}"); return
    ma = f"pretrained={repo},dtype=bfloat16"
    for k, v in extra.items():
        ma += f",{k}={v}"
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", ma, "--tasks", tasks, "--num_fewshot", "0",
           "--batch_size", str(batch_size), "--output_path", out_path]
    res = subprocess.run(cmd)
    print(f"  {'[ok]' if res.returncode == 0 else '[fail]'} {name}")


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="lume_hybrid_weights/lume_hybrid.safetensors")
    p.add_argument("--out-dir", default="results_v6")
    p.add_argument("--lume-batch", type=int, default=4)
    p.add_argument("--baseline-batch", type=int, default=8)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-creative", action="store_true")
    p.add_argument("--skip-lmeval", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    if not args.skip_ppl:
        print("\n" + "#" * 80 + "\n# (1) Perplexity\n" + "#" * 80)
        corpora = load_eval_corpora()
        if corpora:
            summary["lume_ppl"] = run_ppl_lume(args.checkpoint, corpora)
            for name, repo, extra, _, tok, _ in BASELINES:
                summary[f"{name}_ppl"] = run_ppl_hf(repo, extra, tok, corpora, PRETTY[name][0])
            with open(os.path.join(args.out_dir, "ppl.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)

    if not args.skip_bench:
        print("\n" + "#" * 80 + "\n# (2) Inference Benchmarks (cached vs naive)\n" + "#" * 80)
        summary["bench"] = run_inference_bench(args.checkpoint)
        with open(os.path.join(args.out_dir, "bench.json"), "w") as f:
            json.dump(summary["bench"], f, indent=2, default=str)

    if not args.skip_creative:
        print("\n" + "#" * 80 + "\n# (3) Creativity Comparison\n" + "#" * 80)
        summary["creative"] = run_creativity_eval(args.checkpoint)
        with open(os.path.join(args.out_dir, "creative.json"), "w") as f:
            json.dump(summary["creative"], f, indent=2, default=str)

    if not args.skip_lmeval:
        print("\n" + "#" * 80 + "\n# (4) lm-eval\n" + "#" * 80)
        run_lm_eval_lume(args.checkpoint, LMEVAL_TASKS, args.lume_batch,
                        os.path.join(args.out_dir, "lume_results.json"))
        free_gpu()
        for name, repo, extra, _, _, _ in BASELINES:
            free_gpu()
            run_lm_eval_baseline(name, repo, extra, LMEVAL_TASKS,
                                 args.baseline_batch, args.out_dir)

    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()