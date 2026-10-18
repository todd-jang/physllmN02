#!/usr/bin/env python3
"""N02 Main Evaluation — Activation Capture Pipeline.

Captures block-boundary hidden states and computes the three registered N02 observables:
1. Length-adjusted block update norm: ||h_(l+1) - h_l||_F / sqrt(T)
2. Input-update cosine: cos(h_l, Δh_l)
3. Relative norm change: (||h_(l+1)||_F - ||h_l||_F) / ||h_l||_F
4. Log norm ratio: ln(||h_(l+1)||_F) - ln(||h_l||_F)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import transformers

from norm_metrics import norm_change_metrics

CAPTURE_POINT = "residual_stream_block_boundary_pre_norm"

DEFAULT_MODELS = {
    "gpt2": "gpt2",
    "pythia-160m": "EleutherAI/pythia-160m",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "gemma-2-2b": "google/gemma-2-2b",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text_fingerprint(frame: pd.DataFrame) -> str:
    payload = "\n".join(
        f"{tid}\n{text}"
        for tid, text in zip(
            frame["text_id"].astype(str), frame["text"].astype(str)
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_transformer_blocks(
    model: torch.nn.Module,
) -> tuple[torch.nn.ModuleList, str]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "gpt2"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers, "gpt_neox_pythia"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "llama_qwen_gemma"
    raise RuntimeError(
        f"Unsupported architecture for model {type(model).__name__}"
    )


def first_hidden_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item
    raise TypeError(
        f"Could not extract [B,T,D] hidden state from {type(output)}"
    )


@torch.inference_mode()
def extract_metrics_single_text(
    tokenizer: Any,
    model: torch.nn.Module,
    model_key: str,
    manifest_row: dict[str, Any],
    device: torch.device,
    max_length: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = str(manifest_row["text"])
    encoded = tokenizer(
        text, return_tensors="pt", truncation=False, add_special_tokens=True
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    n_tokens = int(encoded["input_ids"].shape[1])
    if n_tokens > max_length:
        raise ValueError(
            f"Token count {n_tokens} exceeds max_length={max_length}"
        )

    blocks, architecture = resolve_transformer_blocks(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def pre_hook(module: torch.nn.Module, args: Any) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Failed to capture input to block 0")
        captured[0] = (
            args[0].detach().to("cpu", dtype=torch.float32).clone()[0]
        )

    handles.append(blocks[0].register_forward_pre_hook(pre_hook))

    for idx, block in enumerate(blocks):

        def post_hook(
            module: torch.nn.Module, args: Any, output: Any, i: int = idx
        ) -> None:
            captured[i + 1] = (
                first_hidden_tensor(output)
                .detach()
                .to("cpu", dtype=torch.float32)
                .clone()[0]
            )

        handles.append(block.register_forward_hook(post_hook))

    try:
        outputs = model(**encoded, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    n_blocks = len(blocks)
    if len(captured) != n_blocks + 1 or any(
        i not in captured for i in range(n_blocks + 1)
    ):
        raise RuntimeError(
            f"Expected {n_blocks + 1} states; captured {len(captured)}"
        )
    if not torch.isfinite(outputs.logits).all():
        raise FloatingPointError("Non-finite logits detected")

    states = [captured[i].to(dtype=torch.float64) for i in range(n_blocks + 1)]
    for i, h in enumerate(states):
        if not torch.isfinite(h).all():
            raise FloatingPointError(f"Non-finite hidden state at layer {i}")

    rows = []
    for l in range(n_blocks):
        h_in = states[l]
        h_out = states[l + 1]
        delta = h_out - h_in
        in_norm = torch.linalg.vector_norm(h_in).item()
        out_norm = torch.linalg.vector_norm(h_out).item()
        norm_changes = norm_change_metrics(in_norm, out_norm)
        delta_norm = torch.linalg.vector_norm(delta).item()
        dot = torch.sum(h_in * delta).item()
        cosine = (
            dot / (in_norm * delta_norm)
            if in_norm > 0 and delta_norm > 0
            else float("nan")
        )

        rows.append({
            "model_key": model_key,
            "text_id": str(manifest_row["text_id"]),
            "topic_id": str(manifest_row["topic_id"]),
            "topic_domain": str(manifest_row["topic_domain"]),
            "length_tier": str(manifest_row["length_tier"]),
            "style": str(manifest_row["style"]),
            "layer_idx": int(l),
            "n_layers": int(n_blocks),
            "rho_block": l / max(n_blocks - 1, 1),
            "n_tokens": n_tokens,
            "hidden_dim": int(h_in.shape[-1]),
            "capture_point": CAPTURE_POINT,
            "input_norm_fro": in_norm,
            "output_norm_fro": out_norm,
            "update_norm_fro": delta_norm,
            "update_norm_per_token": delta_norm / math.sqrt(max(n_tokens, 1)),
            "update_rms_per_element": delta_norm
            / math.sqrt(max(n_tokens * h_in.shape[-1], 1)),
            "input_update_cosine": cosine,
            **norm_changes,
        })

    meta = {
        "n_tokens": n_tokens,
        "architecture": architecture,
        "n_layers": n_blocks,
        "hidden_dim": int(states[0].shape[-1]),
        "capture_point": CAPTURE_POINT,
    }
    return rows, meta


def run_activation_capture(
    manifest_path: Path,
    out_dir: Path,
    model_keys: list[str] | None = None,
    device_name: str | None = None,
    dtype_name: str = "float32",
) -> None:
    manifest_path = manifest_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    df = pd.read_csv(manifest_path, keep_default_na=False)
    fingerprint = sha256_text_fingerprint(df)

    print(f"Loaded Manifest: {manifest_path.name}")
    print(f"  Rows: {len(df)} | SHA-256: {manifest_sha256}")
    print(f"  Text Fingerprint: {fingerprint}")

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, dtype_name)
    print(f"Execution Device: {device} | Model dtype: {dtype}")
    metrics_csv = out_dir / "N02_activation_metrics.csv"
    status_csv = out_dir / "N02_capture_status.csv"

    completed_jobs = set()
    if status_csv.exists():
        st_df = pd.read_csv(status_csv)
        completed_jobs = set(
            zip(
                st_df["model_key"].astype(str),
                st_df["text_id"].astype(str),
                strict=False,
            )
        )
        print(f"Found {len(completed_jobs)} existing completed jobs.")

    active_models = model_keys or list(DEFAULT_MODELS.keys())
    for m_key in active_models:
        model_id = DEFAULT_MODELS.get(m_key, m_key)
        print(f"\n==========================================")
        print(f"Loading Model: {m_key} ({model_id})")
        print(f"==========================================")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device)
        model.eval()

        batch_rows: list[dict[str, Any]] = []
        status_rows: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            tid = str(row["text_id"])
            if (m_key, tid) in completed_jobs:
                continue

            rows, meta = extract_metrics_single_text(
                tokenizer, model, m_key, row.to_dict(), device
            )
            batch_rows.extend(rows)
            status_rows.append({
                "model_key": m_key,
                "text_id": tid,
                "n_tokens": meta["n_tokens"],
                "status": "SUCCESS",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

            if len(status_rows) >= 10:
                # Flush checkpoint
                pd.DataFrame(batch_rows).to_csv(
                    metrics_csv,
                    mode="a" if metrics_csv.exists() else "w",
                    header=not metrics_csv.exists(),
                    index=False,
                )
                pd.DataFrame(status_rows).to_csv(
                    status_csv,
                    mode="a" if status_csv.exists() else "w",
                    header=not status_csv.exists(),
                    index=False,
                )
                batch_rows.clear()
                status_rows.clear()
                print(f"  Progress: {idx+1}/{len(df)} texts captured.")

        if status_rows:
            pd.DataFrame(batch_rows).to_csv(
                metrics_csv,
                mode="a" if metrics_csv.exists() else "w",
                header=not metrics_csv.exists(),
                index=False,
            )
            pd.DataFrame(status_rows).to_csv(
                status_csv,
                mode="a" if status_csv.exists() else "w",
                header=not status_csv.exists(),
                index=False,
            )

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nActivation capture run complete!")
    print(f"Metrics saved to: {metrics_csv}")


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = WORKSPACE_ROOT / "data/frozen/N02_Main_Evaluation_FROZEN_Manifest_v1.0.3.csv"
DEFAULT_OUT_DIR = WORKSPACE_ROOT / "results/activations"


def main():
    parser = argparse.ArgumentParser(
        description="Capture residual activations for N02 frozen manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "pythia-160m"],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model precision for the forward pass (default: float32)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    if not manifest_path.exists():
        alt = WORKSPACE_ROOT / manifest_path
        if alt.exists():
            manifest_path = alt

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = WORKSPACE_ROOT / out_dir

    run_activation_capture(
        manifest_path, out_dir, args.models, args.device, args.dtype
    )


if __name__ == "__main__":
    main()
