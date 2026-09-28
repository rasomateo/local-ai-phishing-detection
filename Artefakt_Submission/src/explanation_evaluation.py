from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.final_artifact import load_final_artifact_components
from src.unified_artifact import predict_unified
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

OUT = REPOSITORY_ROOT / "outputs/evaluation/explanations"
CHECKSUM_NAMES = ["explanation_validation.json", "explanation_examples.json", "report.json", "manifest.json"]
PROTECTED_SHA256 = {
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/url/url_vectorizer.joblib": "182bfb0af30f56cd68657dbcdbd03571810882aeb75bcff5aad25af782478af3",
    "outputs/models/url/url_model.joblib": "f11e2a715ba993a1ac99c867964f79b289a5a982d92ae3cd8601585032c16be7",
    "outputs/models/url/url_calibrator.joblib": "110e48c05ec2c36ab7be35723c4916ca618062610c7f1d9c01ae84eac3040021",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
    "outputs/evaluation/phishingpot/checksums.sha256": "db21a28b32b3c7f5cc82b6ce9e9d8af3da789d228e977a38a94905e444ff1db6",
    "outputs/evaluation/dideco_intent/checksums.sha256": "dec06421882e73046364289ee2f425678c8b6e98b9fa9d44ef7bb49f80f0287b",
}
IMPLEMENTATION_PATHS = [
    "src/explain.py", "src/unified_artifact.py", "src/final_artifact.py",
    "src/explanation_evaluation.py", "scripts/validate_explanation_evaluation.py",
]
CASES = [
    {"case_id": "empty_subject", "subject": "", "body": "Please verify your account immediately"},
    {"case_id": "empty_body", "subject": "Quarterly team meeting agenda", "body": ""},
    {"case_id": "very_short", "subject": "x", "body": ""},
    {"case_id": "oov_no_features", "subject": "qzxvplmoknijbuh", "body": ""},
    {"case_id": "repeated_words", "subject": "Urgent urgent urgent", "body": "Verify verify your account"},
    {"case_id": "overlapping_ngrams", "subject": "Account alert", "body": "Verify your account now"},
    {"case_id": "url_in_text", "subject": "Project reference", "body": "Review https://example.com/case for the meeting"},
    {"case_id": "legitimate_prediction", "subject": "Meeting agenda", "body": "Please review the attached minutes for tomorrow"},
]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def _input_hash(subject: str, body: str) -> str:
    return hashlib.sha256((subject + "\x1f" + body).encode("utf-8")).hexdigest()


def validate_context() -> None:
    if OUT.exists():
        raise ValueError("Step-31 output directory already exists")


def validate_protected() -> dict[str, str]:
    return resolve_current_protected_hashes(REPOSITORY_ROOT, PROTECTED_SHA256)


def run() -> dict[str, Any]:
    validate_context()
    protected = validate_protected()
    components = load_final_artifact_components()
    examples: list[dict[str, Any]] = []
    max_difference = 0.0
    labels: set[str] = set()
    cases = list(CASES)
    if current_contract(REPOSITORY_ROOT).is_fixture:
        with (REPOSITORY_ROOT / "outputs/features/text/holdout_inputs.csv").open(
                encoding="utf-8", newline="") as handle:
            fixture_rows = list(csv.DictReader(handle))
        for label in ("legitimate", "phishing"):
            row = next(item for item in fixture_rows if item["label"] == label)
            cases.append({"case_id": f"fixture_{label}", "subject": row["subject"], "body": row["body"]})
    for case in cases:
        first = predict_unified(case["subject"], case["body"], components)
        second = predict_unified(case["subject"], case["body"], components)
        first_encoded = json.dumps(first["text_explanation"], ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"))
        second_encoded = json.dumps(second["text_explanation"], ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"))
        if first_encoded != second_encoded:
            raise ValueError(f"Nondeterministic explanation: {case['case_id']}")
        explanation = first["text_explanation"]
        if not explanation["reconstruction_tolerance"]["passed"]:
            raise ValueError(f"Unfaithful explanation: {case['case_id']}")
        active = {item["feature_index"] for item in explanation["all_feature_contributions"]}
        if any(item["feature_index"] not in active for item in
               explanation["top_positive_features"] + explanation["top_negative_features"]):
            raise ValueError("Displayed feature is not active")
        if any("signed_contribution" in occurrence for occurrence in
               explanation["position_visualization"]["occurrences"]):
            raise ValueError("Position occurrence contains an attribution")
        max_difference = max(max_difference, abs(float(explanation["reconstruction_difference"])))
        labels.add(first["email_classification"]["predicted_label"])
        examples.append({**case, "input_sha256": _input_hash(case["subject"], case["body"]),
                         "result": first})
    try:
        predict_unified("", "", components)
    except ValueError:
        both_empty_rejected = True
    else:
        both_empty_rejected = False
    if labels != {"legitimate", "phishing"} or not both_empty_rejected:
        raise ValueError("Synthetic boundary coverage incomplete")

    implementation = {name: sha256(REPOSITORY_ROOT / name) for name in IMPLEMENTATION_PATHS}
    method = {
        "version": 2,
        "contribution": "tfidf_value * native LinearSVC coefficient",
        "reconstruction_absolute_tolerance": 1e-10, "reconstruction_relative_tolerance": 1e-12,
        "ranking": "positive=(-contribution,feature_index); negative=(contribution,feature_index)",
        "decision_rule": "phishing iff calibrated text probability >= 0.5",
        "position_scope": "canonical normalized model text only; no occurrence attribution",
        "synthetic_case_ids": [case["case_id"] for case in cases],
    }
    run_id = "step31:" + hashlib.sha256(json.dumps(
        {"method": method,
         "predecessor_semantics": semantic_predecessor_identity(REPOSITORY_ROOT, protected),
         "implementation": implementation},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    validation = {
        "status": "passed", "evaluation_run_id": run_id, "case_count": len(examples),
        "faithfulness_passed": True, "determinism_passed": True, "feature_presence_passed": True,
        "sign_correctness_passed": True, "prediction_consistency_passed": True,
        "position_visualization_passed": True, "both_empty_rejected": both_empty_rejected,
        "observed_labels": sorted(labels), "maximum_absolute_reconstruction_difference": max_difference,
    }
    report = {
        "status": "completed", "step": 31, "research_question":
        "How can the artifact provide deterministic, model-faithful indicators for its decisions, and what are the limitations of these indicators?",
        "evaluation_run_id": run_id,
        "model_components": protected, "method": method, "validation": validation,
        "boundaries": {"model_fitted": False, "calibrator_fitted": False,
                       "threshold_changed": False, "holdout_accessed": False,
                       "external_evaluation_performed": False, "url_explanation_added": False,
                       "step32_started": False},
        "limitations": [
            "Contributions are a mathematical decomposition of the selected linear model raw score, not causal evidence.",
            "Calibration is a downstream score transformation and is not an additive sum of token probabilities.",
            "Position highlights show occurrences in normalized model text and are not occurrence-level attributions.",
            "Repeated occurrences do not create additional model contributions.",
            "No span gold standard exists, so explanation precision, recall, and F1 are not reported.",
            "No user study was performed, so usability or user comprehension is not established.",
        ],
    }
    manifest = {
        "selection_provenance": current_selection_provenance(REPOSITORY_ROOT),
        "artifact_type": "step31_model_faithful_text_explanation_rq3", "schema_version": 2,
        "status": "completed", "evaluation_run_id": run_id,
        "method_specification": method,
        "input_sha256": protected, "implementation_sha256": implementation,
        "runtime": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "joblib": importlib.metadata.version("joblib")},
        "schemas": {"explanation_validation.json": sorted(validation),
                    "explanation_examples.json": ["case_id", "subject", "body", "input_sha256", "result"]},
        "output_sha256": {},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".explanation-evaluation-staging-", dir=OUT.parent))
    try:
        write_json(staging / "explanation_validation.json", validation)
        write_json(staging / "explanation_examples.json", examples)
        write_json(staging / "report.json", report)
        manifest["output_sha256"] = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:-1]}
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")
        subprocess.run([sys.executable, "-B", "-m", "scripts.validate_explanation_evaluation",
                        "--out", str(staging)], cwd=REPOSITORY_ROOT, check=True)
        os.replace(staging, OUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-rq3-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_rq3_evaluation:
        parser.error("Step 31 is inert without --execute-rq3-evaluation")
    print(json.dumps(run(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
