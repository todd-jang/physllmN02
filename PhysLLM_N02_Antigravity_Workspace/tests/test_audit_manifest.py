import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_manifest import audit_manifest, normalise_code_template, normalise_text  # noqa: E402


def test_normalise_text_ignores_case_and_whitespace():
    assert normalise_text("  A\nB  ") == normalise_text("a b")


def test_code_template_normalisation_reduces_numeric_variants():
    first = "values = [2, 3]\nresult = max(values) - min(values)"
    second = "values = [20, 30]\nresult = max(values) - min(values)"
    assert normalise_code_template(first) == normalise_code_template(second)


def test_audit_never_freezes_candidate(tmp_path):
    styles = ["general_prose", "question_dialogue", "academic", "code_plus_natural_language"]
    rows = []
    for i in range(1, 101):
        topic = f"T{i:03d}"
        for style in styles:
            if style == "code_plus_natural_language":
                text = f"values = [{i}, {i + 1}]\nresult = max(values) - min(values)\n# Explain result for domain {i}."
            else:
                text = f"Unique sample {i} with style {style}; describe domain {i}."
            rows.append({
                "topic_id": topic,
                "topic_domain": f"domain_{i}",
                "operation_id": "range",
                "length_tier": "short",
                "style": style,
                "text_id": f"{topic}_{style}",
                "text": text,
                "role": "candidate_main_eval_not_frozen",
                "pilot_excluded": True,
                "sampling_unit": "topic_id_cluster",
                "generation_seed": 123,
            })
    manifest = tmp_path / "candidate.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    report = audit_manifest(manifest, tmp_path / "out")
    assert report["rows"] == 400
    assert report["status"] == "CANDIDATE_NEEDS_REVIEW"
    assert report["frozen_for_confirmatory_analysis"] is False
    assert report["human_review_complete"] is False
    assert report["review_required_rows"] == 400
    assert (tmp_path / "out" / "review_queue.csv").exists()
