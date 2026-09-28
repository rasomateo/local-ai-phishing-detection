from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import math
import os
import platform
import shutil
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import sklearn

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import normalize_text_field
from src.final_artifact import load_final_artifact_components, predict_final_artifact
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

SPLIT_RUN_ID = ""
EXPECTED_TOTAL = 50_255
EXPECTED_LABELS = {"legitimate": 49_701, "phishing": 554}
HOLDOUT = REPOSITORY_ROOT / "outputs/features/text/holdout_inputs.csv"
RAW_DATASET = REPOSITORY_ROOT / "outputs/merged_raw_dataset.csv"
URL_CV = REPOSITORY_ROOT / "outputs/datasets/url/canonical/cv_assignments.csv"
OUT = REPOSITORY_ROOT / "outputs/evaluation/internal_holdout"

EXPECTED_SHA256 = {
    "outputs/merged_raw_dataset.csv": "defa4d27b7b2947bd94e2c23dc56a62723f97818ddec712bc0125db69a6da5ff",
    "outputs/features/text/holdout_inputs.csv": "4f3d0e154eb2c7e00d38bffcc5bb3fbad25810c2d35a23cf86221e877fcf0187",
    "outputs/features/text/report.json": "d4c9a74f5619806ee070f8ad1b617d0e9ee29785dc3cdd1f01df7b4408e1d9db",
    "outputs/datasets/url/canonical/cv_assignments.csv": "594841ccb5ff56ca4d38f67ed68aebd21fac1353bb53b9106f561880505eedd7",
    "outputs/models/text/classifier/manifest.json": "7f7da1edb895bcc825b9f64b5f51c0bb4c125016498e5210116558346a4a2698",
    "outputs/models/text/classifier/report.json": "758d58a39fae02764688caa84fce171c40fcfacbc359f6ccc70b1980c4f46ad8",
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/manifest.json": "b20970982cbcb1279302037dbda16537915839f26567a7e3e74399c4b8e78752",
    "outputs/models/text/calibration/report.json": "981447079df35c2f8b25705514bd688ae18cef0c5ebb5866334889c53014be5b",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/url/manifest.json": "38f55151fe77385958355229453ebe07bc87a4f49f706401261d5df9308188d9",
    "outputs/models/url/report.json": "1291653da76d0b904e3c0e718e93c6cb964fffc4e9871f5db9b18317fde3ac4f",
    "outputs/models/url/url_vectorizer.joblib": "182bfb0af30f56cd68657dbcdbd03571810882aeb75bcff5aad25af782478af3",
    "outputs/models/url/url_model.joblib": "f11e2a715ba993a1ac99c867964f79b289a5a982d92ae3cd8601585032c16be7",
    "outputs/models/url/url_calibrator.joblib": "110e48c05ec2c36ab7be35723c4916ca618062610c7f1d9c01ae84eac3040021",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/models/final_artifact/architecture.json": "f730617eab8fd9295a740fd8f2bb280f66687368583d1c6fb3793d378bf68924",
    "outputs/models/final_artifact/manifest.json": "df012321b671c8d3be0edfddaa4ca7fa836ada6226c717ba15eacc58fb72b21f",
}
IMPLEMENTATION_PATHS = {
    "src/internal_holdout_evaluation.py",
    "scripts/validate_internal_holdout.py",
    "src/final_artifact.py",
}

HOLDOUT_COLUMNS = ["split_run_id", "record_id", "label", "subject", "body"]
URL_CV_COLUMNS = ["url_run_id", "canonical_url_id", "label", "leakage_group", "cv_fold"]
PREDICTION_COLUMNS = [
    "evaluation_run_id", "split_run_id", "record_id", "canonical_label",
    "text_raw_decision_score", "text_probability_phishing",
    "extracted_url_occurrence_count", "valid_url_occurrence_count", "unique_valid_url_count",
    "has_valid_url", "has_multiple_valid_url_occurrences", "has_multiple_distinct_valid_urls",
    "has_duplicate_normalized_url", "has_development_url_group_overlap",
    "development_overlap_valid_url_occurrence_count", "development_overlap_unique_url_count",
    "url_probability_max", "prediction",
]
METRIC_COLUMNS = [
    "evaluation_run_id", "configuration", "subset", "n", "legitimate_count", "phishing_count",
    "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1",
    "false_positive_rate", "false_negative_rate",
]
CONFUSION_COLUMNS = [
    "evaluation_run_id", "configuration", "tn", "fp", "fn", "tp",
    "negative_total", "positive_total", "total",
]
COVERAGE_COLUMNS = [
    "evaluation_run_id", "scope", "canonical_label", "emails_total",
    "emails_without_extracted_url", "emails_with_extracted_url", "emails_without_valid_url",
    "emails_with_valid_url", "emails_with_multiple_valid_url_occurrences",
    "emails_with_multiple_distinct_valid_urls", "emails_with_duplicate_normalized_urls",
    "emails_with_development_url_group_overlap", "extracted_url_occurrences", "valid_url_occurrences",
    "development_overlap_valid_url_occurrences", "distinct_scored_urls",
    "distinct_development_overlap_urls",
]
CHECKSUM_NAMES = [
    "holdout_predictions.csv", "configuration_metrics.csv", "confusion_matrices.csv",
    "url_coverage.csv", "report.json", "manifest.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_float(left: float, right: float) -> bool:
    return struct.pack("!d", float(left)) == struct.pack("!d", float(right))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def validate_execution_context() -> None:
    require(not OUT.exists() or not any(OUT.iterdir()), "Step-26 output directory is not empty")


def validate_protected_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    actual = resolve_current_protected_hashes(root, EXPECTED_SHA256)
    architecture_manifest = json.loads(
        (root / "outputs/models/final_artifact/manifest.json").read_text(encoding="utf-8")
    )
    require(__import__("re").fullmatch(r"step25c-[0-9a-f]{32}",
                                      str(architecture_manifest["architecture_run_id"])) is not None,
            "Unexpected Step-25C architecture run")
    return actual


def read_holdout(path: Path = HOLDOUT) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == HOLDOUT_COLUMNS, "Unexpected Step-17 Holdout schema")
        rows = list(reader)
    require(len(rows) == EXPECTED_TOTAL, "Holdout row count mismatch")
    require(all(row["record_id"] for row in rows), "Empty Holdout record_id")
    require(len({row["record_id"] for row in rows}) == EXPECTED_TOTAL, "Duplicate Holdout record_id")
    require({row["split_run_id"] for row in rows} == {SPLIT_RUN_ID}, "Holdout split_run_id mismatch")
    labels = Counter(row["label"] for row in rows)
    require(dict(labels) == EXPECTED_LABELS, "Holdout label distribution mismatch")
    return {row["record_id"]: row for row in rows}


def project_raw_holdout(raw_rows: Iterable[dict[str, str]],
                        holdout: dict[str, dict[str, str]]) -> dict[str, tuple[str, str]]:
    allowlist = set(holdout)
    seen: set[str] = set()
    original: dict[str, tuple[str, str]] = {}
    for raw in raw_rows:
        rid = raw["record_id"]
        if rid not in allowlist:
            continue
        require(rid not in seen, f"Duplicate raw Holdout record_id: {rid}")
        seen.add(rid)
        subject, body = raw["subject"], raw["body"]
        require(normalize_text_field(subject) == holdout[rid]["subject"],
                f"Raw-to-Step-17 subject mismatch: {rid}")
        require(normalize_text_field(body) == holdout[rid]["body"],
                f"Raw-to-Step-17 body mismatch: {rid}")
        original[rid] = (subject, body)
    require(seen == allowlist, f"Raw Holdout coverage mismatch; missing={len(allowlist - seen)}")
    return original


def stream_raw_holdout(holdout: dict[str, dict[str, str]], path: Path = RAW_DATASET) -> dict[str, tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected merged raw schema")
        return project_raw_holdout(reader, holdout)


def load_development_groups(path: Path = URL_CV) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == URL_CV_COLUMNS, "Unexpected Step-21B CV schema")
        rows = list(reader)
    expected = len(rows) if current_contract(REPOSITORY_ROOT).is_fixture else current_contract(REPOSITORY_ROOT).expected("url_development_count", 4001)
    require(len(rows) == expected, "Step-21B Development URL row count mismatch")
    require(len({row["canonical_url_id"] for row in rows}) == expected,
            "Step-21B canonical_url_id coverage mismatch")
    groups = {row["leakage_group"].strip().lower().rstrip(".") for row in rows}
    require("" not in groups, "Empty Step-21B leakage group")
    return groups


def matching_development_groups(normalized_url: str, groups: set[str]) -> tuple[str, ...]:
    host = (urlsplit(normalized_url).hostname or "").lower().rstrip(".")
    if not host:
        return ()
    try:
        ipaddress.ip_address(host)
        return (host,) if host in groups else ()
    except ValueError:
        return tuple(sorted(group for group in groups if host == group or host.endswith("." + group)))


def metric_values(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, int | float]:
    tp = sum(row["canonical_label"] == row[prediction_key] == "phishing" for row in rows)
    tn = sum(row["canonical_label"] == row[prediction_key] == "legitimate" for row in rows)
    fp = sum(row["canonical_label"] == "legitimate" and row[prediction_key] == "phishing" for row in rows)
    fn = sum(row["canonical_label"] == "phishing" and row[prediction_key] == "legitimate" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows), "legitimate_count": tn + fp, "phishing_count": fn + tp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def _url_maps(result: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    rows = result["url_model_evidence"]["occurrences"]
    valid_urls: list[str] = []
    scores: dict[str, float] = {}
    for row in rows:
        if row["normalized_url"]:
            require(row["scored"], "Valid URL was not scored")
            valid_urls.append(row["normalized_url"])
            scores[row["normalized_url"]] = float(row["url_model_score_phishing"])
    return valid_urls, scores


def evaluate_record(record_id: str, canonical: dict[str, str], original: tuple[str, str],
                    components: dict[str, Any],
                    groups: set[str]) -> dict[str, Any]:
    subject, body = original
    result = predict_final_artifact(subject, body, components)
    require(result["architecture"] == {
        "name": "step25c_text_led_email_classifier_with_auxiliary_url_evidence",
        "schema_version": 2, "fusion_applied": False,
    }, "Unexpected current architecture contract")
    email = result["email_classification"]
    require(email["decision_threshold"] == 0.5, "Decision threshold changed")
    valid_urls, score_by_url = _url_maps(result)
    unique_urls = list(dict.fromkeys(valid_urls))
    evidence = result["url_model_evidence"]
    require(evidence["valid_url_occurrence_count"] == len(valid_urls), "Valid URL count mismatch")
    require(evidence["scored_unique_url_count"] == len(unique_urls), "Unique URL count mismatch")
    maximum = evidence["maximum_url_model_score_phishing"]
    overlap_occurrences = [url for url in valid_urls if matching_development_groups(url, groups)]
    overlap_unique = set(overlap_occurrences)
    prediction = email["predicted_label"]
    require(prediction == ("phishing" if email["text_probability_phishing"] >= 0.5 else "legitimate"),
            "Decision rule mismatch")
    return {
        "split_run_id": canonical["split_run_id"], "record_id": record_id,
        "canonical_label": canonical["label"],
        "text_raw_decision_score": email["text_raw_decision_score"],
        "text_probability_phishing": email["text_probability_phishing"],
        "extracted_url_occurrence_count": evidence["extracted_url_count"],
        "valid_url_occurrence_count": len(valid_urls), "unique_valid_url_count": len(unique_urls),
        "has_valid_url": bool(unique_urls),
        "has_multiple_valid_url_occurrences": len(valid_urls) > 1,
        "has_multiple_distinct_valid_urls": len(unique_urls) > 1,
        "has_duplicate_normalized_url": len(valid_urls) > len(unique_urls),
        "has_development_url_group_overlap": bool(overlap_unique),
        "development_overlap_valid_url_occurrence_count": len(overlap_occurrences),
        "development_overlap_unique_url_count": len(overlap_unique),
        "url_probability_max": maximum, "prediction": prediction,
        "_valid_urls": set(unique_urls), "_overlap_urls": overlap_unique,
        "_score_by_url": score_by_url,
    }


def build_tables(rows: list[dict[str, Any]], run_id: str) -> tuple[list[dict[str, Any]], ...]:
    metrics: list[dict[str, Any]] = []
    by_subset = {
        "all": rows,
        "no_valid_url": [row for row in rows if not row["has_valid_url"]],
        "with_valid_url": [row for row in rows if row["has_valid_url"]],
    }
    for subset in ("all", "no_valid_url", "with_valid_url"):
        metrics.append({"evaluation_run_id": run_id, "configuration": "current_final_artifact",
                        "subset": subset, **metric_values(by_subset[subset], "prediction")})
    values = metrics[0]
    confusion = [{"evaluation_run_id": run_id, "configuration": "current_final_artifact",
                  **{key: values[key] for key in ("tn", "fp", "fn", "tp")},
                  "negative_total": values["tn"] + values["fp"],
                  "positive_total": values["fn"] + values["tp"], "total": values["n"]}]
    require((confusion[0]["negative_total"], confusion[0]["positive_total"], confusion[0]["total"]) ==
            (EXPECTED_LABELS["legitimate"], EXPECTED_LABELS["phishing"], EXPECTED_TOTAL),
            "Confusion totals mismatch")

    coverage: list[dict[str, Any]] = []
    for scope, label in (("overall", ""), ("legitimate", "legitimate"), ("phishing", "phishing")):
        selected = rows if not label else [row for row in rows if row["canonical_label"] == label]
        all_urls = set().union(*(row["_valid_urls"] for row in selected)) if selected else set()
        overlap_urls = set().union(*(row["_overlap_urls"] for row in selected)) if selected else set()
        coverage.append({
            "evaluation_run_id": run_id, "scope": scope, "canonical_label": label,
            "emails_total": len(selected),
            "emails_without_extracted_url": sum(row["extracted_url_occurrence_count"] == 0 for row in selected),
            "emails_with_extracted_url": sum(row["extracted_url_occurrence_count"] > 0 for row in selected),
            "emails_without_valid_url": sum(not row["has_valid_url"] for row in selected),
            "emails_with_valid_url": sum(row["has_valid_url"] for row in selected),
            "emails_with_multiple_valid_url_occurrences": sum(row["has_multiple_valid_url_occurrences"] for row in selected),
            "emails_with_multiple_distinct_valid_urls": sum(row["has_multiple_distinct_valid_urls"] for row in selected),
            "emails_with_duplicate_normalized_urls": sum(row["has_duplicate_normalized_url"] for row in selected),
            "emails_with_development_url_group_overlap": sum(row["has_development_url_group_overlap"] for row in selected),
            "extracted_url_occurrences": sum(row["extracted_url_occurrence_count"] for row in selected),
            "valid_url_occurrences": sum(row["valid_url_occurrence_count"] for row in selected),
            "development_overlap_valid_url_occurrences": sum(row["development_overlap_valid_url_occurrence_count"] for row in selected),
            "distinct_scored_urls": len(all_urls), "distinct_development_overlap_urls": len(overlap_urls),
        })
    return metrics, confusion, coverage


def make_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    spec = {
        "version": 2, "predecessor_semantics": semantic_predecessor_identity(root, protected),
        "split_run_id": SPLIT_RUN_ID, "configuration": "current_final_artifact",
        "decision_threshold": 0.5,
        "schemas": {"predictions": PREDICTION_COLUMNS, "metrics": METRIC_COLUMNS,
                    "confusion": CONFUSION_COLUMNS,
                    "coverage": COVERAGE_COLUMNS},
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "step26:" + hashlib.sha256(canonical).hexdigest()[:32]


def run(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    global SPLIT_RUN_ID, EXPECTED_TOTAL, EXPECTED_LABELS
    contract = current_contract(root)
    EXPECTED_TOTAL = contract.expected("text_holdout_count", 50_255)
    EXPECTED_LABELS = contract.expected("text_holdout_labels", {"legitimate": 49_701, "phishing": 554})
    SPLIT_RUN_ID = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    require(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository roots are not supported")
    validate_execution_context()
    protected = validate_protected_hashes(root)
    run_id = make_run_id(protected, root)

    holdout = read_holdout()
    raw_hash_before = sha256(RAW_DATASET)
    original = stream_raw_holdout(holdout)
    require(sha256(RAW_DATASET) == raw_hash_before == protected["outputs/merged_raw_dataset.csv"],
            "Merged raw dataset changed during projection")
    require(sha256(HOLDOUT) == protected["outputs/features/text/holdout_inputs.csv"],
            "Holdout changed during preflight")
    groups = load_development_groups()

    components = load_final_artifact_components()
    rows: list[dict[str, Any]] = []
    for record_id in sorted(holdout):
        rows.append(evaluate_record(record_id, holdout[record_id], original[record_id], components, groups))
    require(len(rows) == EXPECTED_TOTAL, "Prediction coverage mismatch")
    for row in rows:
        row["evaluation_run_id"] = run_id
    metrics, confusion, coverage = build_tables(rows, run_id)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="step26-", dir=OUT.parent))
    try:
        write_csv(staging / "holdout_predictions.csv", PREDICTION_COLUMNS, rows)
        write_csv(staging / "configuration_metrics.csv", METRIC_COLUMNS, metrics)
        write_csv(staging / "confusion_matrices.csv", CONFUSION_COLUMNS, confusion)
        write_csv(staging / "url_coverage.csv", COVERAGE_COLUMNS, coverage)
        first_hashes = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:4]}
        report = {
            "status": "completed", "evaluation_run_id": run_id,
            "population": {"total": EXPECTED_TOTAL, "by_label": EXPECTED_LABELS,
                           "split_run_id": SPLIT_RUN_ID},
            "configuration_metrics": metrics, "confusion_matrices": confusion,
            "url_coverage": coverage,
            "text_decision_invariant": {"checked_records": EXPECTED_TOTAL, "status": "passed"},
            "development_url_overlap": {
                "purpose": "descriptive transfer diagnostic only",
                "source": "Step-21B Development leakage groups; no URL Holdout data",
                "excludes_urls": False, "changes_predictions": False,
                "used_for_selection_or_thresholding": False,
            },
            "methodology": {
                "configuration": "Current Step-25C text-led artifact; URL evidence auxiliary only",
                "raw_projection": "Step-17 record_id allowlist; only allowed raw Subject/Body accessed; raw labels unused",
                "url_processing": "Original Subject/Body; extraction before canonical text normalization; all valid URLs scored",
                "positive_class": "phishing", "zero_denominator_metric_value": 0.0,
            },
            "limitations": [
                "Step-23 URL evidence was trained and calibrated on Phishload, not email-extracted URLs.",
                "Canonical email labels are not individual URL ground truth.",
                "Development URL overlap is descriptive and does not establish URL maliciousness.",
            ],
            "flags": {"model_fitted": False, "calibrator_fitted": False, "threshold_optimized": False,
                      "external_data_used": False, "url_holdout_used": False,
                      "rq2_or_rq3_started": False, "step27_started": False},
            "output_sha256": first_hashes,
        }
        write_json(staging / "report.json", report)
        manifest_hashes = {**first_hashes, "report.json": sha256(staging / "report.json")}
        implementation = {name: sha256(root / name) for name in IMPLEMENTATION_PATHS}
        manifest = {
            "selection_provenance": current_selection_provenance(root),
            "artifact_type": "step26_final_internal_email_holdout_evaluation", "artifact_version": 2,
            "evaluation_run_id": run_id,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__, "split_run_id": SPLIT_RUN_ID,
            "configuration_api": {"name": "current_final_artifact",
                                  "loader": "src.final_artifact.load_final_artifact_components",
                                  "predictor": "src.final_artifact.predict_final_artifact"},
            "thresholds": {"email_decision": 0.5},
            "step21b_overlap_source": "outputs/datasets/url/canonical/cv_assignments.csv",
            "protected_sha256": protected, "implementation_sha256": implementation, "schemas": {
                "holdout_predictions.csv": PREDICTION_COLUMNS,
                "configuration_metrics.csv": METRIC_COLUMNS,
                "confusion_matrices.csv": CONFUSION_COLUMNS,
                "url_coverage.csv": COVERAGE_COLUMNS,
            },
            "output_sha256": manifest_hashes,
            "self_hash_policy": "No JSON contains its own hash; checksums.sha256 excludes itself",
        }
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")

        from scripts.validate_internal_holdout import validate
        validate(root=root, out=staging, replay=True)
        require(not OUT.exists() or not any(OUT.iterdir()), "Step-26 output appeared during evaluation")
        if OUT.exists():
            OUT.rmdir()
        os.replace(staging, OUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-final-holdout", action="store_true",
                        help="Explicitly authorize the one-time canonical Holdout evaluation")
    args = parser.parse_args(argv)
    if not args.execute_final_holdout:
        parser.error("Holdout remains sealed; pass --execute-final-holdout only after explicit authorization")
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
