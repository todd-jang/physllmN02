#!/usr/bin/env python3
"""Deterministic, non-destructive audit for a PhysLLM N02 candidate manifest.

This tool deliberately NEVER freezes a dataset. It separates machine-verifiable
checks from content review and emits CANDIDATE_NEEDS_REVIEW or AUDIT_FAIL.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "topic_id", "topic_domain", "operation_id", "length_tier", "style",
    "text_id", "text", "role", "pilot_excluded", "sampling_unit",
    "generation_seed",
]
EXPECTED_STYLES = {
    "general_prose", "question_dialogue", "academic", "code_plus_natural_language"
}
LENGTH_BANDS = {
    "short": (16, 64),
    "medium": (65, 128),
    "long": (129, 256),
}
TOKEN_COLUMNS = {
    "gpt2": ("tokens_gpt2", "in_band_gpt2", "gpt2"),
    "pythia-160m": ("tokens_pythia-160m", "in_band_pythia-160m", "EleutherAI/pythia-160m"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalise_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def normalise_code_template(text: str) -> str:
    """Reduce numeric/string literal variation to detect repeated code skeletons."""
    kept = []
    for line in str(text).splitlines():
        if line.strip().startswith("#"):
            continue
        line = re.sub(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])", "NUM", line)
        line = re.sub(r'("[^"]*"|\'[^\']*\')', "STR", line)
        line = line.strip()
        if line:
            kept.append(line)
    return "\n".join(kept)


def to_bool_series(series: pd.Series) -> pd.Series:
    def convert(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        return str(v).strip().lower() in {"true", "1", "yes", "y"}
    return series.map(convert)


def audit_manifest(
    manifest_path: Path,
    out_dir: Path,
    retokenize: bool = False,
    review_ledger: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    raw_hash = sha256_file(manifest_path)
    df = pd.read_csv(manifest_path, dtype={"topic_id": str, "text_id": str})
    findings: list[dict[str, str]] = []
    hard_checks: dict[str, bool] = {}

    def finding(severity: str, check: str, message: str, text_id: str = "") -> None:
        findings.append({"severity": severity, "check": check, "text_id": text_id, "message": message})

    def check(name: str, passed: bool, message: str) -> None:
        hard_checks[name] = bool(passed)
        finding("INFO" if passed else "ERROR", name, message)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    check("required_columns", not missing_cols, f"Missing columns: {missing_cols}" if missing_cols else "All required columns are present.")

    if missing_cols:
        # Write a valid, explicit partial report rather than crashing mid-way.
        report = {
            "schema_version": "1.0",
            "status": "AUDIT_FAIL",
            "manifest_path": str(manifest_path),
            "manifest_sha256": raw_hash,
            "rows": int(len(df)),
            "hard_checks": hard_checks,
            "human_review_complete": False,
            "frozen_for_confirmatory_analysis": False,
            "findings": findings,
            "note": "Required columns are missing; later checks were not run.",
        }
        (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(findings).to_csv(out_dir / "findings.csv", index=False)
        return report

    # Null / empty checks
    missing_values = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    empty_text = int(df["text"].fillna("").astype(str).str.strip().eq("").sum())
    check("no_missing_required_values", missing_values == 0 and empty_text == 0,
          f"Rows with missing required values={missing_values}; empty text rows={empty_text}.")

    check("expected_row_count", len(df) == 400, f"Observed {len(df)} rows; expected 400 for this candidate design.")
    topic_counts = df["topic_id"].value_counts(dropna=False)
    check("expected_topic_clusters", df["topic_id"].nunique(dropna=False) == 100,
          f"Observed {df['topic_id'].nunique(dropna=False)} unique topic_id values; expected 100.")
    cluster_sizes_ok = bool((topic_counts == 4).all()) and len(topic_counts) == 100
    check("four_rows_per_cluster", cluster_sizes_ok,
          f"Clusters with row count !=4: {int((topic_counts != 4).sum())}.")

    style_counts = df["style"].value_counts(dropna=False).to_dict()
    observed_styles = set(df["style"].dropna().astype(str))
    styles_ok = observed_styles == EXPECTED_STYLES and all(int(style_counts.get(s, 0)) == 100 for s in EXPECTED_STYLES)
    check("style_balance", styles_ok,
          f"Observed style counts={style_counts}; expected exactly 100 of each: {sorted(EXPECTED_STYLES)}.")

    dup_ids = int(df["text_id"].duplicated(keep=False).sum())
    check("unique_text_id", dup_ids == 0, f"Rows involved in duplicate text_id values={dup_ids}.")
    exact_dups = int(df["text"].astype(str).duplicated(keep=False).sum())
    normalized = df["text"].map(normalise_text)
    norm_dups = int(normalized.duplicated(keep=False).sum())
    check("no_exact_duplicate_text", exact_dups == 0, f"Rows involved in exact duplicate texts={exact_dups}.")
    check("no_normalized_duplicate_text", norm_dups == 0, f"Rows involved in whitespace/case-normalized duplicate texts={norm_dups}.")

    per_topic_metadata_ok = True
    metadata_problems = []
    for topic_id, group in df.groupby("topic_id", dropna=False):
        for col in ("topic_domain", "operation_id", "length_tier", "generation_seed", "sampling_unit"):
            if group[col].nunique(dropna=False) != 1:
                per_topic_metadata_ok = False
                metadata_problems.append(f"{topic_id}:{col} varies")
    check("cluster_metadata_consistency", per_topic_metadata_ok,
          "; ".join(metadata_problems[:20]) if metadata_problems else "Topic metadata is constant within each topic_id cluster.")

    roles = sorted(df["role"].dropna().astype(str).unique().tolist())
    role_candidate_ok = roles == ["candidate_main_eval_not_frozen"]
    check("candidate_role_not_frozen", role_candidate_ok,
          f"Observed role values={roles}; this workspace expects candidate_main_eval_not_frozen.")
    sampling_ok = set(df["sampling_unit"].dropna().astype(str)) == {"topic_id_cluster"}
    check("cluster_sampling_unit", sampling_ok,
          f"Observed sampling_unit values={sorted(set(df['sampling_unit'].dropna().astype(str)))}.")
    pilot_flags = to_bool_series(df["pilot_excluded"])
    check("pilot_prompts_excluded", bool(pilot_flags.all()),
          f"Rows with pilot_excluded != true: {int((~pilot_flags).sum())}.")

    # Validate stored token counts/bands independently of any tokenizer downloads.
    band_rows = []
    band_errors = []
    for model, (token_col, in_band_col, _) in TOKEN_COLUMNS.items():
        if token_col not in df.columns or in_band_col not in df.columns:
            finding("WARNING", "stored_token_counts", f"No stored counts for {model}; independent tokenizer check is needed.")
            continue
        stored_in_band = to_bool_series(df[in_band_col])
        model_mismatches = []
        for _, row in df.iterrows():
            tier = str(row["length_tier"])
            count = row[token_col]
            lo_hi = LENGTH_BANDS.get(tier)
            expected = False
            if lo_hi is not None and pd.notna(count):
                expected = lo_hi[0] <= int(count) <= lo_hi[1]
            ok = bool(expected and stored_in_band.loc[row.name])
            band_rows.append({"text_id": row["text_id"], "model": model, "length_tier": tier,
                              "stored_token_count": count, "stored_in_band": bool(stored_in_band.loc[row.name]),
                              "tier_band_ok_from_stored_count": bool(expected), "combined_ok": ok})
            if not ok:
                model_mismatches.append(str(row["text_id"]))
        if model_mismatches:
            band_errors.extend(model_mismatches)
        check(f"stored_length_bands_{model}", not model_mismatches,
              f"Rows failing stored count + declared tier check for {model}: {len(model_mismatches)}.")

    # Code syntax. Parsing is necessary but not sufficient for semantic validity.
    code_mask = df["style"].eq("code_plus_natural_language")
    syntax_failures = []
    template_to_ids: dict[str, list[str]] = defaultdict(list)
    row_template_info: dict[str, dict[str, Any]] = {}
    for _, row in df[code_mask].iterrows():
        text = str(row["text"])
        try:
            ast.parse(text)
            syntax_ok = True
        except SyntaxError as exc:
            syntax_ok = False
            syntax_failures.append((str(row["text_id"]), f"{exc.__class__.__name__}: {exc.msg}, line={exc.lineno}"))
            finding("ERROR", "python_syntax", syntax_failures[-1][1], str(row["text_id"]))
        template = normalise_code_template(text)
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]
        template_to_ids[template_hash].append(str(row["text_id"]))
        row_template_info[str(row["text_id"])] = {"syntax_ok": syntax_ok, "template_hash": template_hash}
    check("code_text_python_syntax", len(syntax_failures) == 0,
          f"Python AST parse failures among code_plus_natural_language rows={len(syntax_failures)} / {int(code_mask.sum())}.")

    repeated_groups = {k: v for k, v in template_to_ids.items() if len(v) >= 3}
    max_template_repetition = max((len(v) for v in template_to_ids.values()), default=0)
    if repeated_groups:
        finding("WARNING", "code_template_repetition",
                f"{len(repeated_groups)} normalized code templates occur >=3 times; maximum repetition={max_template_repetition}. This flags review priority, not an automatic failure.")
    finding("REVIEW_REQUIRED", "semantic_review",
            "All 400 rows require explicit content review; syntax and metadata checks cannot establish topic/operation/interpretation correctness.")

    # Optional independent retokenization
    retokenize_result: dict[str, Any] = {"requested": bool(retokenize), "status": "NOT_RUN"}
    if retokenize:
        try:
            from transformers import AutoTokenizer
            independent_rows = []
            mismatch_counts = {}
            for model_key, (stored_count_col, stored_band_col, model_name) in TOKEN_COLUMNS.items():
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                actual = []
                for _, row in df.iterrows():
                    try:
                        count = len(tokenizer.encode(str(row["text"]), add_special_tokens=True))
                    except Exception as exc:  # noqa: BLE001 - capture row failures in audit log
                        count = None
                        finding("ERROR", "retokenize_exception", f"{model_key}: {exc}", str(row["text_id"]))
                    actual.append(count)
                    tier = str(row["length_tier"])
                    lo_hi = LENGTH_BANDS.get(tier)
                    actual_in_band = bool(count is not None and lo_hi and lo_hi[0] <= count <= lo_hi[1])
                    stored = row[stored_count_col] if stored_count_col in df.columns else None
                    independent_rows.append({
                        "text_id": row["text_id"], "model": model_key,
                        "stored_token_count": stored,
                        "independent_token_count": count,
                        "count_matches_stored": bool(count is not None and pd.notna(stored) and int(stored) == int(count)),
                        "length_tier": tier, "independent_in_band": actual_in_band,
                    })
                if stored_count_col in df.columns:
                    mismatches = sum(1 for i, row in df.iterrows() if actual[i] is not None and pd.notna(row[stored_count_col]) and int(actual[i]) != int(row[stored_count_col]))
                    mismatch_counts[model_key] = int(mismatches)
                    check(f"independent_retokenization_{model_key}", mismatches == 0,
                          f"Independent count mismatches versus stored counts for {model_key}={mismatches}.")
                else:
                    mismatch_counts[model_key] = None
                actual_bad = sum(1 for i, row in df.iterrows() if actual[i] is None or not (LENGTH_BANDS.get(str(row["length_tier"]), (1, 0))[0] <= actual[i] <= LENGTH_BANDS.get(str(row["length_tier"]), (1, 0))[1]))
                check(f"independent_length_bands_{model_key}", actual_bad == 0,
                      f"Independent tokenizer counts outside declared length band for {model_key}={actual_bad}.")
            pd.DataFrame(independent_rows).to_csv(out_dir / "independent_tokenizer_counts.csv", index=False)
            retokenize_result = {"requested": True, "status": "COMPLETED", "stored_count_mismatches": mismatch_counts,
                                 "count_convention": "len(tokenizer.encode(text, add_special_tokens=True)"}
        except Exception as exc:  # tokenizer download or import failure does not imply data pass
            retokenize_result = {"requested": True, "status": "FAILED_TO_RUN", "error": f"{type(exc).__name__}: {exc}"}
            finding("WARNING", "retokenization_not_completed", retokenize_result["error"])

    # Build or update a review queue. Existing rows are matched by text_id; review status is never auto-approved.
    existing_review = None
    if review_ledger and review_ledger.is_file():
        try:
            existing_review = pd.read_csv(review_ledger, dtype={"text_id": str})
        except Exception as exc:  # noqa: BLE001
            finding("WARNING", "review_ledger_read", f"Could not read review ledger: {exc}")
    if existing_review is not None and "text_id" in existing_review.columns:
        review_state = existing_review.set_index("text_id").to_dict(orient="index")
    else:
        review_state = {}

    review_rows = []
    row_check_rows = []
    for _, row in df.iterrows():
        tid = str(row["text_id"])
        previous = review_state.get(tid, {})
        status = str(previous.get("review_status", "NOT_REVIEWED"))
        reviewer = str(previous.get("reviewer", "")) if pd.notna(previous.get("reviewer", "")) else ""
        rationale = str(previous.get("rationale", "")) if pd.notna(previous.get("rationale", "")) else ""
        reviewed_at = str(previous.get("reviewed_at_utc", "")) if pd.notna(previous.get("reviewed_at_utc", "")) else ""
        template_info = row_template_info.get(tid, {"syntax_ok": None, "template_hash": ""})
        semantic_review_complete = bool(
            status.upper() in {"APPROVED", "NEEDS_REVISION", "REJECTED"}
            and reviewer.strip() not in {"", "nan"}
            and rationale.strip() not in {"", "nan"}
            and reviewed_at.strip() not in {"", "nan"}
        )
        review_rows.append({
            "text_id": tid, "topic_id": row["topic_id"], "topic_domain": row["topic_domain"],
            "operation_id": row["operation_id"], "length_tier": row["length_tier"],
            "style": row["style"], "text": row["text"],
            "syntax_ok_if_code": template_info["syntax_ok"],
            "normalized_code_template_hash": template_info["template_hash"],
            "template_repeated_3plus": bool(template_info["template_hash"] in repeated_groups),
            "review_status": status, "reviewer": reviewer, "rationale": rationale,
            "reviewed_at_utc": reviewed_at,
            "semantic_review_complete_for_row": semantic_review_complete,
        })
        row_check_rows.append({
            "text_id": tid, "topic_id": row["topic_id"], "style": row["style"],
            "python_syntax_ok": template_info["syntax_ok"],
            "semantic_review_status": status,
            "semantic_review_complete_for_row": bool(status.upper() in {"APPROVED", "REJECTED"} and reviewer.strip() and rationale.strip()),
        })

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(out_dir / "review_queue.csv", index=False)
    pd.DataFrame(row_check_rows).to_csv(out_dir / "row_checks.csv", index=False)
    pd.DataFrame(findings).to_csv(out_dir / "findings.csv", index=False)
    if band_rows:
        pd.DataFrame(band_rows).to_csv(out_dir / "stored_length_band_checks.csv", index=False)

    reviewed_all = bool(len(review_df) == len(df) and review_df["semantic_review_complete_for_row"].all())
    approved_all = bool(len(review_df) == len(df) and review_df.apply(
        lambda r: str(r["review_status"]).upper() == "APPROVED"
        and bool(r["semantic_review_complete_for_row"]), axis=1
    ).all())
    errors = [f for f in findings if f["severity"] == "ERROR"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_filename": manifest_path.name,
        "manifest_sha256": raw_hash,
        "status": "AUDIT_FAIL" if errors or not all(hard_checks.values()) else "CANDIDATE_NEEDS_REVIEW",
        "frozen_for_confirmatory_analysis": False,
        "human_review_complete": reviewed_all,
        "all_rows_approved": approved_all,
        "rows": int(len(df)),
        "unique_topics": int(df["topic_id"].nunique(dropna=False)),
        "styles": style_counts,
        "length_tiers": df["length_tier"].value_counts(dropna=False).to_dict(),
        "code_rows": int(code_mask.sum()),
        "normalized_code_template_count": int(len(template_to_ids)),
        "normalized_code_template_groups_repeated_3plus": int(len(repeated_groups)),
        "max_template_repetition": int(max_template_repetition),
        "retokenization": retokenize_result,
        "hard_checks": hard_checks,
        "errors_count": int(len(errors)),
        "warnings_count": int(sum(f["severity"] == "WARNING" for f in findings)),
        "review_required_rows": int(len(df) - int(review_df["semantic_review_complete_for_row"].sum())),
        "findings": findings,
        "interpretation": "Machine audit only. Semantic correctness and release approval are separate gates. This script never freezes a dataset.",
    }
    (out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "audit_summary.md").write_text(make_markdown_summary(report), encoding="utf-8")
    return report


def make_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# N02 Manifest Audit Summary",
        "",
        f"- **Status:** `{report.get('status')}`",
        f"- **Frozen:** `{report.get('frozen_for_confirmatory_analysis')}` (this tool never freezes datasets)",
        f"- **Manifest:** `{report.get('manifest_filename')}`",
        f"- **SHA-256:** `{report.get('manifest_sha256')}`",
        f"- **Rows / topics:** {report.get('rows')} / {report.get('unique_topics')}",
        f"- **Code rows:** {report.get('code_rows')}",
        f"- **Normalized code templates:** {report.get('normalized_code_template_count')}; repeated 3+ times: {report.get('normalized_code_template_groups_repeated_3plus')}; max repetition: {report.get('max_template_repetition')}",
        f"- **Rows still needing review:** {report.get('review_required_rows')}",
        "",
        "## Required interpretation",
        "This is a machine audit, not a content approval. All candidate rows must receive explicit semantic review. Do not begin confirmatory analysis while the manifest is a candidate.",
        "",
        "## Checks",
    ]
    for name, passed in report.get("hard_checks", {}).items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", "See `findings.csv`, `row_checks.csv`, `review_queue.csv`, and `audit_report.json` for details.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--retokenize", action="store_true", help="Independently count tokens with HF tokenizers; requires transformers/network.")
    parser.add_argument("--review-ledger", type=Path, default=None, help="Optional ledger with text_id, review_status, reviewer, rationale columns.")
    args = parser.parse_args()
    report = audit_manifest(args.manifest, args.out_dir, args.retokenize, args.review_ledger)
    print(f"STATUS: {report['status']}")
    print(f"MANIFEST_SHA256: {report.get('manifest_sha256')}")
    print(f"ROWS: {report.get('rows')} | TOPICS: {report.get('unique_topics')} | REVIEW_REQUIRED: {report.get('review_required_rows')}")
    print(f"FROZEN: {report.get('frozen_for_confirmatory_analysis')}")
    print(f"REPORT: {(args.out_dir / 'audit_report.json').resolve()}")
    return 0 if report["status"] != "AUDIT_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
