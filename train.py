"""Lume-Hybrid training, fine-tuning, and chat CLI.

Usage:
    python train.py train  --path lume_hybrid.safetensors
    python train.py tune   --path lume_hybrid.safetensors
    python train.py chat   --path lume_hybrid.safetensors

The model definition lives in modeling.py; the streaming dataset lives in
data.py. This file is the CLI entry point and the training loop.
"""
from __future__ import annotations

import argparse
import gc
import glob
import math
import os
import time

import torch
import wandb
from safetensors.torch import save_file, load_file
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from modeling import LumeConfig, LumeModel, backend_summary
from data import stream_datasets_interleaved


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Pretraining data mix
# ---------------------------------------------------------------------------
PHASE1_MIX = [
    ("gmongaras/SlimPajama-627B_Reupload", 0.5),
    ("HuggingFaceFW/fineweb-edu", 0.3),
    ("bigcode/the-stack-dedup", 0.2),
]
PHASE2_MIX = [
    ("HuggingFaceFW/fineweb-edu", 0.40),
    ("bigcode/the-stack-dedup", 0.30),
    ("open-web-math/open-web-math", 0.30),
]


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------
def build_optimizer(model: torch.nn.Module, cfg: LumeConfig) -> torch.optim.Optimizer:
    """Three parameter groups: Mamba internals (no decay), weights (decay), biases/norms (no decay)."""
    mamba_no_decay, weights_decay, biases_no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ("A_log", "D", "dt_bias", "dt_proj.bias")):
            mamba_no_decay.append(param)
        elif "bias" in name or "norm" in name:
            biases_no_decay.append(param)
        else:
            weights_decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": mamba_no_decay, "weight_decay": 0.0},
            {"params": weights_decay, "weight_decay": cfg.weight_decay},
            {"params": biases_no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )


def build_scheduler(opt: torch.optim.Optimizer, cfg: LumeConfig):
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=cfg.warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, cfg.total_steps - cfg.warmup_steps, eta_min=cfg.learning_rate * 0.1
            ),
        ],
        milestones=[cfg.warmup_steps],
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(step: int, model, optimizer, scheduler, path: str) -> None:
    """Save model weights as safetensors, plus full training state as a .pt sidecar."""
    print(f"  [save] step {step}")
    save_file(model.state_dict(), f"lume_step_{step}.safetensors")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
        },
        "training_state.pt",
    )
    save_file(model.state_dict(), path)
    print(f"  [save] checkpoint written: {path}")


def load_latest_checkpoint(model, optimizer, scheduler, cfg: LumeConfig):
    """Resume from the most recent lume_step_*.safetensors if one exists.

    Returns (start_step, samples_already_seen).
    """
    checkpoints = sorted(glob.glob("lume_step_*.safetensors"), key=os.path.getmtime)
    if not checkpoints:
        return 1, 0

    latest = checkpoints[-1]
    try:
        model.load_state_dict(load_file(latest))
        print(f"[resume] weights loaded from {latest}")
    except Exception as e:
        print(f"[resume] checkpoint corrupt or incompatible: {e}")
        return 1, 0

    if not os.path.exists("training_state.pt"):
        start_step = int(latest.split("_")[-1].split(".")[0]) + 1
        return start_step, 0

    state = torch.load("training_state.pt", weights_only=False)
    start_step = state["step"] + 1
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    samples_seen = (start_step - 1) * cfg.batch_size * cfg.grad_accum_steps
    print(f"[resume] training state restored at step {start_step}; skipping ~{samples_seen:,} samples")
    return start_step, samples_seen


# ---------------------------------------------------------------------------
# Evaluation hook
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_iter, device, num_batches: int = 5) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        try:
            batch = next(data_iter).to(device)
        except StopIteration:
            break
        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_training(args, cfg: LumeConfig, device: torch.device) -> None:
    print(f"[init] backend: {backend_summary()}")
    model = LumeModel(cfg).to(device)
    print(f"[init] params: {model.count_params() / 1e6:.2f}M | context: {cfg.max_seq_len}")

    wandb.init(
        project="lume-hybrid",
        name="run",
        config={
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.max_seq_len,
            "layers": cfg.n_layers,
            "dim": cfg.dim,
        },
    )
    try:
        model = torch.compile(model)
        print("[init] torch.compile enabled")
    except Exception:
        pass

    opt = build_optimizer(model, cfg)
    scheduler = build_scheduler(opt, cfg)
    start_step, samples_already_seen = load_latest_checkpoint(model, opt, scheduler, cfg)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    print("[data] warming up streams...")
    val_stream = iter(stream_datasets_interleaved(PHASE1_MIX, cfg.batch_size, cfg.max_seq_len))

    phase2_threshold = int(cfg.total_steps * 0.85)
    current_data = PHASE2_MIX if start_step > phase2_threshold else PHASE1_MIX
    train_stream = iter(
        stream_datasets_interleaved(
            current_data, cfg.batch_size, cfg.max_seq_len, samples_already_seen
        )
    )

    model.train()
    progress = tqdm(
        range(start_step, cfg.total_steps + 1),
        desc="Training",
        unit="step",
        initial=start_step,
        total=cfg.total_steps,
    )
    t0 = time.time()

    try:
        for step in progress:
            if step == phase2_threshold:
                progress.write("\n[phase 2] refinement starts")
                del train_stream
                gc.collect()
                torch.cuda.empty_cache()
                phase2_skip = (
                    (start_step - phase2_threshold) * cfg.batch_size * cfg.grad_accum_steps
                    if start_step > phase2_threshold
                    else 0
                )
                train_stream = iter(
                    stream_datasets_interleaved(
                        PHASE2_MIX, cfg.batch_size, cfg.max_seq_len, phase2_skip
                    )
                )

            step_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                try:
                    batch = next(train_stream).to(device)
                except (StopIteration, RuntimeError):
                    progress.write("[data] stream exhausted; restarting...")
                    train_stream = iter(
                        stream_datasets_interleaved(current_data, cfg.batch_size, cfg.max_seq_len)
                    )
                    batch = next(train_stream).to(device)

                with autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(batch[:, :-1], batch[:, 1:])
                    loss /= cfg.grad_accum_steps

                loss.backward()
                step_loss += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            scheduler.step()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            tokens_processed = cfg.batch_size * cfg.max_seq_len * cfg.grad_accum_steps
            tokens_per_sec = tokens_processed / dt
            train_ppl = math.exp(step_loss) if step_loss < 15 else 10_000.0
            current_lr = scheduler.get_last_lr()[0]

            wandb.log({
                "train/loss": step_loss,
                "train/perplexity": train_ppl,
                "train/grad_norm": grad_norm,
                "train/lr": current_lr,
                "perf/tokens_per_sec": tokens_per_sec,
                "perf/step_time_ms": dt * 1000,
                "global_step": step,
            })

            if step % 5 == 0:
                progress.set_postfix(
                    loss=f"{step_loss:.4f}", ppl=f"{train_ppl:.1f}", tps=f"{int(tokens_per_sec)}"
                )

            if step % 100 == 0:
                progress.write(
                    f"step {step} | loss {step_loss:.4f} | ppl {train_ppl:.1f} | "
                    f"grad {grad_norm:.2f} | {int(tokens_per_sec)} tok/s"
                )

            if math.isnan(step_loss) or math.isinf(step_loss):
                progress.write(f"[error] NaN/Inf at step {step}")
                save_checkpoint(step, model, opt, scheduler, "emergency_nan.safetensors")
                raise SystemExit(1)

            if step < 5_000:
                eval_interval = 250
            elif step < 15_000:
                eval_interval = 500
            else:
                eval_interval = 1_000

            if step % eval_interval == 0:
                v_loss = evaluate(model, val_stream, device)
                v_ppl = math.exp(v_loss) if v_loss < 20 else float("inf")
                progress.write(f"\n--- step {step} eval | loss {v_loss:.4f} | ppl {v_ppl:.2f} ---")

                prompt = "The future of AI is"
                sample_in = tok.encode(prompt, return_tensors="pt").to(device)
                out = model.generate(sample_in, max_new_tokens=50)
                gen_text = tok.decode(out[0], skip_special_tokens=True)
                progress.write(f"sample: {gen_text}\n")

                log_table = wandb.Table(columns=["step", "loss", "ppl", "prompt", "generated"])
                log_table.add_data(step, v_loss, v_ppl, prompt, gen_text.replace(prompt, ""))
                wandb.log({
                    "eval/loss": v_loss,
                    "eval/perplexity": v_ppl,
                    "eval/samples": log_table,
                    "global_step": step,
                })

                save_checkpoint(step, model, opt, scheduler, args.path)
                old_ckpt = f"lume_step_{step - 5_000}.safetensors"
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                model.train()

    except KeyboardInterrupt:
        print(f"\n[interrupt] saving emergency checkpoint at step {step}...")
        save_checkpoint(step, model, opt, scheduler, args.path)
        print("[interrupt] saved; you can resume from this checkpoint.")
        raise SystemExit(0)


def run_finetune(args, cfg: LumeConfig, device: torch.device) -> None:
    model = LumeModel(cfg).to(device)
    if os.path.exists(args.path):
        model.load_state_dict(load_file(args.path))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    stream = iter(stream_datasets_interleaved([("HuggingFaceH4/ultrachat_200k", 1.0)], 8, 2048))
    for i in range(500):
        try:
            batch = next(stream).to(device)
        except StopIteration:
            break
        with autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(batch[:, :-1], batch[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"tune {i}: {loss.item():.4f}")
    save_file(model.state_dict(), "lume_tuned.safetensors")


def run_chat(args, cfg: LumeConfig, device: torch.device) -> None:
    model = LumeModel(cfg).to(device)
    if not os.path.exists(args.path):
        raise SystemExit(f"checkpoint not found: {args.path}")

    print(f"[chat] loading {args.path}")
    sd = load_file(args.path)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print("[chat] weights loaded; type 'exit' to quit.")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    while True:
        prompt = input("you: ")
        if prompt.lower() == "exit":
            break
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=150)
        print("lume:", tok.decode(out[0][len(ids[0]):], skip_special_tokens=True))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lume-Hybrid training/inference CLI.")
    parser.add_argument("mode", choices=["train", "tune", "chat"])
    parser.add_argument(
        "--path", default="lume_hybrid.safetensors",
        help="Path to the canonical checkpoint file.",
    )
    args = parser.parse_args()

    cfg = LumeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    if args.mode == "train":
        run_training(args, cfg, device)
    elif args.mode == "tune":
        run_finetune(args, cfg, device)
    elif args.mode == "chat":
        run_chat(args, cfg, device)


if __name__ == "__main__":
    main()
