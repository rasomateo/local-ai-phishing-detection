from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes
from src.final_artifact import load_final_artifact_components, predict_final_artifact
from src.internal_holdout_evaluation import (
    CHECKSUM_NAMES, CONFUSION_COLUMNS, COVERAGE_COLUMNS, EXPECTED_LABELS, EXPECTED_SHA256,
    EXPECTED_TOTAL, IMPLEMENTATION_PATHS, METRIC_COLUMNS, PREDICTION_COLUMNS,
)
from src.execution_contract import current_contract
from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import normalize_text_field
from src.utils import REPOSITORY_ROOT

csv.field_size_limit(sys.maxsize)
OUT = REPOSITORY_ROOT / "outputs/evaluation/internal_holdout"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def same_float(left: float, right: float) -> bool:
    return struct.pack("!d", float(left)) == struct.pack("!d", float(right))


def read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == fields, f"Unexpected schema: {path.name}")
        return list(reader)


def as_bool(value: str) -> bool:
    check(value in {"true", "false"}, f"Invalid boolean: {value}")
    return value == "true"


def independent_metrics(items: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(item["label"] == item["prediction"] == "phishing" for item in items)
    tn = sum(item["label"] == item["prediction"] == "legitimate" for item in items)
    fp = sum(item["label"] == "legitimate" and item["prediction"] == "phishing" for item in items)
    fn = sum(item["label"] == "phishing" and item["prediction"] == "legitimate" for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(items), "legitimate_count": tn + fp, "phishing_count": fn + tp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(items) if items else 0.0,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def _overlap(normalized_url: str, groups: set[str]) -> bool:
    host = (urlsplit(normalized_url).hostname or "").lower().rstrip(".")
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return host in groups
    except ValueError:
        return any(host == group or host.endswith("." + group) for group in groups)


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT, replay: bool = True) -> dict[str, Any]:
    contract = current_contract(root)
    expected_total = contract.expected("text_holdout_count", EXPECTED_TOTAL)
    expected_labels = contract.expected("text_holdout_labels", EXPECTED_LABELS)
    split_run_id = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    protected = resolve_current_protected_hashes(root, EXPECTED_SHA256)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    check(manifest["artifact_version"] == 2, "Holdout schema version mismatch")
    check(manifest["selection_provenance"] == current_selection_provenance(root), "Selection provenance mismatch")
    check(manifest["protected_sha256"] == protected, "Protected inventory mismatch")
    final_manifest = json.loads((root / "outputs/models/final_artifact/manifest.json").read_text(encoding="utf-8"))
    check(re.fullmatch(r"step25c-[0-9a-f]{32}", final_manifest["architecture_run_id"]) is not None,
          "Non-canonical final-artifact run ID")
    check(manifest["split_run_id"] == report["population"]["split_run_id"] == split_run_id,
          "Split run mismatch")
    check(manifest["configuration_api"]["name"] == "current_final_artifact", "Configuration mismatch")
    check(manifest["thresholds"] == {"email_decision": 0.5}, "Threshold contract changed")
    check(manifest["schemas"] == {
        "holdout_predictions.csv": PREDICTION_COLUMNS,
        "configuration_metrics.csv": METRIC_COLUMNS,
        "confusion_matrices.csv": CONFUSION_COLUMNS,
        "url_coverage.csv": COVERAGE_COLUMNS,
    }, "Manifest schemas mismatch")
    serialized = json.dumps({"manifest": manifest, "report": report}, sort_keys=True).lower()
    for forbidden in ("configuration_" + "c", "paired_" + "comparison", "fusion " + "policy",
                      "historical " + "fusion"):
        check(forbidden not in serialized, f"Removed historical field present: {forbidden}")
    check(set(manifest["implementation_sha256"]) == IMPLEMENTATION_PATHS, "Implementation inventory mismatch")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")
    for name in CHECKSUM_NAMES[:-1]:
        check(manifest["output_sha256"][name] == digest(out / name), f"Output hash mismatch: {name}")
    check((out / "checksums.sha256").read_text(encoding="utf-8").splitlines() ==
          [f"{digest(out / name)}  {name}" for name in CHECKSUM_NAMES], "Checksum inventory mismatch")
    stored = read_rows(out / "holdout_predictions.csv", PREDICTION_COLUMNS)
    check(len(stored) == len({row["record_id"] for row in stored}) == expected_total,
          "Prediction coverage mismatch")
    check(Counter(row["canonical_label"] for row in stored) == Counter(expected_labels),
          "Label distribution mismatch")
    if not replay:
        return {"status": "validated", "evaluation_run_id": manifest["evaluation_run_id"], "records": len(stored)}

    with (root / "outputs/features/text/holdout_inputs.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == ["split_run_id", "record_id", "label", "subject", "body"],
              "Holdout schema mismatch")
        canonical = {row["record_id"]: row for row in reader}
    original: dict[str, tuple[str, str]] = {}
    with (root / "outputs/merged_raw_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == NORMALIZED_COLUMNS, "Raw schema mismatch")
        for row in reader:
            if row["record_id"] in canonical:
                check(normalize_text_field(row["subject"]) == canonical[row["record_id"]]["subject"] and
                      normalize_text_field(row["body"]) == canonical[row["record_id"]]["body"],
                      "Raw projection mismatch")
                original[row["record_id"]] = (row["subject"], row["body"])
    check(set(original) == set(canonical), "Raw holdout projection incomplete")
    with (root / "outputs/datasets/url/canonical/cv_assignments.csv").open(encoding="utf-8", newline="") as handle:
        groups = {row["leakage_group"].strip().lower().rstrip(".") for row in csv.DictReader(handle)}
    components = load_final_artifact_components()
    replayed: list[dict[str, Any]] = []
    stored_by_id = {row["record_id"]: row for row in stored}
    for record_id in sorted(canonical):
        result = predict_final_artifact(*original[record_id], components)
        email, evidence = result["email_classification"], result["url_model_evidence"]
        check(result["architecture"]["fusion_applied"] is False, "Fusion enabled")
        valid = [row["normalized_url"] for row in evidence["occurrences"] if row["normalized_url"]]
        unique = set(valid)
        overlap_occ = [url for url in valid if _overlap(url, groups)]
        overlap_unique = set(overlap_occ)
        expected = {
            "split_run_id": canonical[record_id]["split_run_id"], "canonical_label": canonical[record_id]["label"],
            "text_raw_decision_score": email["text_raw_decision_score"],
            "text_probability_phishing": email["text_probability_phishing"],
            "extracted_url_occurrence_count": evidence["extracted_url_count"],
            "valid_url_occurrence_count": len(valid), "unique_valid_url_count": len(unique),
            "has_valid_url": bool(unique), "has_multiple_valid_url_occurrences": len(valid) > 1,
            "has_multiple_distinct_valid_urls": len(unique) > 1,
            "has_duplicate_normalized_url": len(valid) > len(unique),
            "has_development_url_group_overlap": bool(overlap_unique),
            "development_overlap_valid_url_occurrence_count": len(overlap_occ),
            "development_overlap_unique_url_count": len(overlap_unique),
            "url_probability_max": evidence["maximum_url_model_score_phishing"],
            "prediction": email["predicted_label"],
        }
        actual = stored_by_id[record_id]
        for key, value in expected.items():
            if isinstance(value, bool):
                check(as_bool(actual[key]) == value, f"Boolean mismatch: {record_id} {key}")
            elif isinstance(value, int):
                check(int(actual[key]) == value, f"Integer mismatch: {record_id} {key}")
            elif isinstance(value, float):
                check(same_float(float(actual[key]), value), f"Float mismatch: {record_id} {key}")
            elif value is None:
                check(actual[key] == "", f"Null mismatch: {record_id} {key}")
            else:
                check(actual[key] == value, f"Value mismatch: {record_id} {key}")
        replayed.append({"label": canonical[record_id]["label"], "prediction": email["predicted_label"],
                         "valid": valid, "unique": unique, "overlap_occ": overlap_occ,
                         "overlap_unique": overlap_unique, "extracted": evidence["extracted_url_count"]})

    metric_rows = read_rows(out / "configuration_metrics.csv", METRIC_COLUMNS)
    check([(row["configuration"], row["subset"]) for row in metric_rows] ==
          [("current_final_artifact", subset) for subset in ("all", "no_valid_url", "with_valid_url")],
          "Metric ordering mismatch")
    subsets = {"all": replayed, "no_valid_url": [row for row in replayed if not row["unique"]],
               "with_valid_url": [row for row in replayed if row["unique"]]}
    for row in metric_rows:
        expected = independent_metrics(subsets[row["subset"]])
        for key, value in expected.items():
            check(int(row[key]) == value if isinstance(value, int) else same_float(float(row[key]), value),
                  f"Metric mismatch: {row['subset']} {key}")
    matrices = read_rows(out / "confusion_matrices.csv", CONFUSION_COLUMNS)
    check(len(matrices) == 1 and matrices[0]["configuration"] == "current_final_artifact",
          "Confusion configuration mismatch")
    check(not (out / ("paired_" + "comparison.csv")).exists(), "Historical paired comparison was produced")
    check(len(read_rows(out / "url_coverage.csv", COVERAGE_COLUMNS)) == 3, "URL coverage rows missing")
    check(report["text_decision_invariant"]["status"] == "passed", "Decision invariant failed")
    return {"status": "validated", "evaluation_run_id": manifest["evaluation_run_id"], "records": len(stored)}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, ensure_ascii=False, sort_keys=True))
