"""Streaming dataset loader for pretraining.

Interleaves multiple HuggingFace streaming datasets, tokenizes with the
Llama-3.2 tokenizer, packs into fixed-length batches, and supports skipping
ahead for resume-from-checkpoint.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import torch


# Token IDs from the Llama-3.2 tokenizer
BOS_TOKEN_ID = 128000
EOS_TOKEN_ID = 128001
PAD_TOKEN_ID = 128002

MIN_TEXT_CHARS = 100
MAX_TEXT_CHARS = 100_000
MIN_TOKENS = 32


def stream_datasets_interleaved(
    dataset_list: List[Tuple[str, float]],
    batch_size: int,
    seq_len: int,
    total_skipped_samples: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield padded batches from interleaved HuggingFace streaming datasets.

    Args:
        dataset_list: list of (dataset_name, sampling_probability) pairs.
        batch_size:   number of sequences per yielded batch.
        seq_len:      max sequence length (longer sequences are truncated).
        total_skipped_samples: rough number of samples to skip per dataset
            (proportional to the sampling probability) when resuming training.
    """
    from datasets import load_dataset, interleave_datasets
    from transformers import AutoTokenizer

    streams = []
    probabilities = []

    raw_probs = [p for _, p in dataset_list]
    total_p = sum(raw_probs)
    norm_probs = [p / total_p for p in raw_probs] if total_p > 0 else raw_probs

    for i, (name, prob) in enumerate(dataset_list):
        try:
            ds = load_dataset(name, split="train", streaming=True)
            column_names = getattr(ds, "column_names", [])
            target_col = next(
                (c for c in ["text", "content", "output"] if c in column_names), None
            )

            if target_col is None:
                continue

            ds = ds.select_columns([target_col])
            if target_col != "text":
                ds = ds.rename_column(target_col, "text")

            if total_skipped_samples > 0:
                dataset_skip = int(total_skipped_samples * norm_probs[i])
                ds = ds.skip(dataset_skip)
                print(f"  [data] connected: {name} (skipped ~{dataset_skip} samples)")
            else:
                print(f"  [data] connected: {name}")

            streams.append(ds)
            probabilities.append(prob)
        except Exception as e:
            print(f"  [data] failed: {name} | {e}")

    if not streams:
        raise RuntimeError("No data streams available.")

    combined = interleave_datasets(
        streams, probabilities=probabilities, stopping_strategy="first_exhausted"
    )
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    buffer: List[List[int]] = []
    iterator = iter(combined)

    while True:
        try:
            item = next(iterator)
        except StopIteration:
            break

        text = item.get("text", "")
        if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
            continue

        tokens = [BOS_TOKEN_ID] + tok.encode(text, add_special_tokens=False) + [EOS_TOKEN_ID]
        if len(tokens) < MIN_TOKENS:
            continue

        buffer.append(tokens[:seq_len])

        if len(buffer) == batch_size:
            max_len = max(len(t) for t in buffer)
            padded = [t + [PAD_TOKEN_ID] * (max_len - len(t)) for t in buffer]
            yield torch.tensor(padded, dtype=torch.long)
            buffer = []
