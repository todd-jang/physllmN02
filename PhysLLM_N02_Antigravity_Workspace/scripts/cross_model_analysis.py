#!/usr/bin/env python3
"""N02 Cross-Model Residual Dynamics Analysis.

Generates cross-architecture comparison tables and diagnostics for the
three captured models (gpt2, pythia-160m, qwen2.5-0.5b), following the
normalization rules in N02_Architecture_Scope_v1.0.md.

Outputs:
  - Layer-wise profile comparison (rho_block normalized)
  - Style isotropy invariance check (H4)
  - RMSNorm vs LayerNorm aggregate comparison (H3)
  - Per-model summary statistics

Usage:
  python scripts/cross_model_analysis.py
  python scripts/cross_model_analysis.py --metrics results/activations/N02_activation_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_METRICS = WORKSPACE_ROOT / "results/activations/N02_activation_metrics.csv"
DEFAULT_OUT_DIR = WORKSPACE_ROOT / "reports/analysis"

# Architecture metadata not present in the metrics CSV
MODEL_META = {
    "gpt2": {"norm_type": "LayerNorm", "residual": "serial", "attn": "MHA", "family": "gpt2"},
    "pythia-160m": {"norm_type": "LayerNorm", "residual": "parallel", "attn": "MHA", "family": "pythia"},
    "qwen2.5-0.5b": {"norm_type": "RMSNorm", "residual": "serial", "attn": "GQA", "family": "qwen2.5"},
    "qwen2.5-1.5b": {"norm_type": "RMSNorm", "residual": "serial", "attn": "GQA", "family": "qwen2.5"},
    "qwen2.5-3b": {"norm_type": "RMSNorm", "residual": "serial", "attn": "GQA", "family": "qwen2.5"},
    "gemma-2-2b": {"norm_type": "RMSNorm", "residual": "serial", "attn": "GQA", "family": "gemma2"},
}


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = [
        "model_key", "text_id", "topic_id", "layer_idx", "n_layers",
        "rho_block", "n_tokens", "hidden_dim",
        "input_norm_fro", "output_norm_fro", "update_norm_fro",
        "update_norm_per_token", "update_rms_per_element",
        "input_update_cosine", "output_input_norm_ratio",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def layer_profile_by_rho(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate layer-wise metrics using rho_block (relative depth)."""
    agg = df.groupby(["model_key", "layer_idx"]).agg(
        rho_block=("rho_block", "first"),
        n_layers=("n_layers", "first"),
        hidden_dim=("hidden_dim", "first"),
        mean_cosine=("input_update_cosine", "mean"),
        median_cosine=("input_update_cosine", "median"),
        std_cosine=("input_update_cosine", "std"),
        mean_norm_ratio=("output_input_norm_ratio", "mean"),
        std_norm_ratio=("output_input_norm_ratio", "std"),
        mean_update_rms=("update_rms_per_element", "mean"),
        mean_update_per_token=("update_norm_per_token", "mean"),
        n_texts=("text_id", "nunique"),
    ).reset_index()
    return agg.sort_values(["model_key", "layer_idx"])


def style_isotropy_check(df: pd.DataFrame) -> pd.DataFrame:
    """H4: Check style-level invariance of dimensionless observables."""
    if "style" not in df.columns:
        raise ValueError("'style' column required for isotropy check")

    agg = df.groupby(["model_key", "style"]).agg(
        mean_cosine=("input_update_cosine", "mean"),
        std_cosine=("input_update_cosine", "std"),
        mean_norm_ratio=("output_input_norm_ratio", "mean"),
        std_norm_ratio=("output_input_norm_ratio", "std"),
        mean_update_rms=("update_rms_per_element", "mean"),
        n_texts=("text_id", "nunique"),
    ).reset_index()
    return agg


def norm_type_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """H3: Compare LayerNorm vs RMSNorm models on dimensionless metrics."""
    df = df.copy()
    df["norm_type"] = df["model_key"].map(
        lambda m: MODEL_META.get(m, {}).get("norm_type", "unknown")
    )

    # Use only terminal layers (rho_block >= 0.9) for phase inversion comparison
    terminal = df[df["rho_block"] >= 0.9].copy()

    result_rows = []
    for norm_type in ["LayerNorm", "RMSNorm"]:
        subset = terminal[terminal["norm_type"] == norm_type]
        if subset.empty:
            continue
        for model_key in subset["model_key"].unique():
            ms = subset[subset["model_key"] == model_key]
            result_rows.append({
                "norm_type": norm_type,
                "model_key": model_key,
                "terminal_mean_cosine": ms["input_update_cosine"].mean(),
                "terminal_std_cosine": ms["input_update_cosine"].std(),
                "terminal_mean_norm_ratio": ms["output_input_norm_ratio"].mean(),
                "terminal_mean_update_rms": ms["update_rms_per_element"].mean(),
                "n_terminal_layers": ms["layer_idx"].nunique(),
                "n_texts": ms["text_id"].nunique(),
            })
    return pd.DataFrame(result_rows)


def phase_inversion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Identify layers with negative mean cosine (phase inversion)."""
    profile = layer_profile_by_rho(df)
    inv = profile[profile["mean_cosine"] < 0].copy()
    inv = inv.sort_values(["model_key", "layer_idx"])
    return inv[["model_key", "layer_idx", "rho_block", "n_layers",
                "mean_cosine", "median_cosine", "mean_norm_ratio", "n_texts"]]


def model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model global summary."""
    agg = df.groupby("model_key").agg(
        n_layers=("n_layers", "first"),
        hidden_dim=("hidden_dim", "first"),
        n_texts=("text_id", "nunique"),
        total_obs=("text_id", "count"),
        global_mean_cosine=("input_update_cosine", "mean"),
        global_mean_norm_ratio=("output_input_norm_ratio", "mean"),
        global_mean_update_rms=("update_rms_per_element", "mean"),
    ).reset_index()
    agg["norm_type"] = agg["model_key"].map(
        lambda m: MODEL_META.get(m, {}).get("norm_type", "unknown")
    )
    agg["attn_type"] = agg["model_key"].map(
        lambda m: MODEL_META.get(m, {}).get("attn", "unknown")
    )
    return agg


def run_analysis(metrics_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(metrics_path)

    models_found = sorted(df["model_key"].unique())
    print(f"Models in dataset: {models_found}")
    print(f"Total observations: {len(df)}")

    # 1. Layer profile
    profile = layer_profile_by_rho(df)
    profile_path = out_dir / "N02_cross_model_layer_profile.csv"
    profile.to_csv(profile_path, index=False, float_format="%.6f")
    print(f"\n[1] Layer profile saved: {profile_path.name}")

    # 2. Style isotropy (H4)
    isotropy = style_isotropy_check(df)
    isotropy_path = out_dir / "N02_cross_model_style_isotropy.csv"
    isotropy.to_csv(isotropy_path, index=False, float_format="%.6f")
    print(f"[2] Style isotropy saved: {isotropy_path.name}")

    # Style variation diagnostic
    print("\n--- H4 Style Isotropy Diagnostic ---")
    for model in models_found:
        ms = isotropy[isotropy["model_key"] == model]
        cosine_range = ms["mean_cosine"].max() - ms["mean_cosine"].min()
        ratio_range = ms["mean_norm_ratio"].max() - ms["mean_norm_ratio"].min()
        print(f"  {model}: cosine range={cosine_range:.4f}, ratio range={ratio_range:.4f}")
        if cosine_range < 0.05 and ratio_range < 0.05:
            print(f"    → PASS: style-invariant within tolerance")
        else:
            print(f"    → REVIEW: variation exceeds 0.05 threshold")

    # 3. Norm type comparison (H3)
    norm_comp = norm_type_comparison(df)
    norm_comp_path = out_dir / "N02_cross_model_norm_type_comparison.csv"
    norm_comp.to_csv(norm_comp_path, index=False, float_format="%.6f")
    print(f"\n[3] Norm type comparison saved: {norm_comp_path.name}")

    print("\n--- H3 RMSNorm vs LayerNorm Terminal Layers ---")
    for _, row in norm_comp.iterrows():
        print(f"  {row['norm_type']:>10} | {row['model_key']:<15} | "
              f"cos={row['terminal_mean_cosine']:+.4f} | "
              f"ratio={row['terminal_mean_norm_ratio']:.4f} | "
              f"rms={row['terminal_mean_update_rms']:.4f}")

    # 4. Phase inversion
    phase_inv = phase_inversion_summary(df)
    phase_path = out_dir / "N02_cross_model_phase_inversion.csv"
    phase_inv.to_csv(phase_path, index=False, float_format="%.6f")
    print(f"\n[4] Phase inversion layers saved: {phase_path.name}")
    if not phase_inv.empty:
        print("\n--- Phase Inversion Layers (negative cosine) ---")
        for _, row in phase_inv.iterrows():
            print(f"  {row['model_key']:<15} layer {int(row['layer_idx']):>2} "
                  f"(rho={row['rho_block']:.2f}) cos={row['mean_cosine']:+.4f} "
                  f"ratio={row['mean_norm_ratio']:.4f}")

    # 5. Model summary
    summary = model_summary(df)
    summary_path = out_dir / "N02_cross_model_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"\n[5] Model summary saved: {summary_path.name}")

    print("\n--- Global Model Summary ---")
    for _, row in summary.iterrows():
        print(f"  {row['model_key']:<15} | L={int(row['n_layers']):>2} D={int(row['hidden_dim']):>5} "
              f"| {row['norm_type']:>9} | cos={row['global_mean_cosine']:+.4f} "
              f"| ratio={row['global_mean_norm_ratio']:.4f} "
              f"| rms={row['global_mean_update_rms']:.6f}")

    # Save run metadata
    meta = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_file": str(metrics_path.name),
        "models_analyzed": models_found,
        "total_observations": int(len(df)),
        "scope_document": "N02_Architecture_Scope_v1.0.md",
        "normalization_rules": [
            "rho_block for cross-depth comparison",
            "dimensionless ratios (cosine, norm_ratio) for cross-model comparison",
            "update_rms_per_element for dim-normalized magnitude",
        ],
    }
    meta_path = out_dir / "N02_cross_model_analysis_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[meta] Analysis metadata saved: {meta_path.name}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="N02 Cross-Model Residual Dynamics Analysis"
    )
    parser.add_argument(
        "--metrics", type=Path, default=DEFAULT_METRICS,
        help="Path to N02_activation_metrics.csv"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for analysis results"
    )
    args = parser.parse_args()

    metrics_path = args.metrics
    if not metrics_path.exists():
        alt = WORKSPACE_ROOT / metrics_path
        if alt.exists():
            metrics_path = alt

    run_analysis(metrics_path, args.out_dir)


if __name__ == "__main__":
    main()
