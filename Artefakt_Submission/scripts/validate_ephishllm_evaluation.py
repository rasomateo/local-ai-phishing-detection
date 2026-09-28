from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import normalize_text_field
from src.utils import REPOSITORY_ROOT
from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

OUT = REPOSITORY_ROOT / "outputs/evaluation/ephishllm"
RAW = REPOSITORY_ROOT / "data/raw/E-PhishLLM/ephishLLM.json"
CANONICAL = REPOSITORY_ROOT / "outputs/dataset_roles/ephishllm_english_external_evaluation.csv"
INTERNAL = REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv"
RUN = ""
SOURCE = "ephishllm_english"
THRESHOLD = 0.5
EXPECTED_LABELS = {"legitimate": 5_506, "phishing": 5_996}
EXPECTED_LANGUAGES = {"da": 4, "de": 2349, "en": 11502, "es": 35, "fr": 12,
                      "it": 2701, "ja": 2, "no": 3, "sv": 5, "zh-cn": 3}
EXPECTED_PROTECTED = {
    "data/raw/E-PhishLLM/ephishLLM.json": "56420b143e30343f6db723c3716e10abec9f72a67ba99ae2cc7c3a87b547f51f",
    "data/raw/E-PhishLLM/README.md": "da9b43209e87eaf6c5671ad1a577b5b67115a7e6a0c16fc9cf5a1024997e29c9",
    "outputs/dataset_acquisition_manifest.json": "38937b96b86d8e938f4ded324fa885ab41b41aeb0f8369f907fb34b19bbb147d",
    "outputs/dataset_roles/ephishllm_english_external_evaluation.csv": "d833c9c78f112d7c97ed4e221839dbf8bd594c77277258180f33424606c36b3b",
    "outputs/preprocessing_report.json": "cfbf68fdc5be8b8a1be28ba211278b277821fda92067cedd44d317b2e2c8b425",
    "outputs/preprocessed_dataset.csv": "cb8ab274790414f518f0b7bb4122ce8655194044bf23798442185c95226963b6",
    "outputs/splits/manifest.json": "471dfc7558b80ead96beab66e60054f28aacdfd4c9a9ba96aeac39497de9d23d",
    f"outputs/splits/runs/{RUN}/split_assignments.csv": "497acde2ed8ec6fd0a1d3d6f8c004398a0e2c3063d6771734b2d8eae41923b30",
    "outputs/leakage/reserved_sources/summary.json": "6b96cb2d71f8073c1743a23435701c57921b3d62744db4545d7a37631d155e67",
    "outputs/leakage/reserved_sources/exact_overlaps.csv": "9d08b43e504c871da049c516fe9430a8a96de2f66779aade95dd012053455379",
    "outputs/models/text/classifier/manifest.json": "7f7da1edb895bcc825b9f64b5f51c0bb4c125016498e5210116558346a4a2698",
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/manifest.json": "b20970982cbcb1279302037dbda16537915839f26567a7e3e74399c4b8e78752",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/final_artifact/architecture.json": "f730617eab8fd9295a740fd8f2bb280f66687368583d1c6fb3793d378bf68924",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
}
PREDICTION_COLUMNS = [
    "evaluation_run_id", "record_id", "source", "source_role", "language", "label",
    "text_raw_score", "calibrated_probability_phishing", "prediction", "correct",
]
METRIC_KEYS = [
    "n_total", "n_legitimate", "n_phishing", "tn", "fp", "fn", "tp", "accuracy",
    "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
]
CHECKSUM_NAMES = ["external_predictions.csv", "external_metrics.json", "report.json", "manifest.json"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == fields, f"Unexpected schema: {path.name}")
        return list(reader)


def stable_record_id(index: int) -> str:
    identity = "\x1f".join(("ephishllm", "english", "data/raw/E-PhishLLM/ephishLLM.json",
                             f"json_index={index}"))
    return "ephishllm:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def independent_raw_population(path: Path, root: Path = REPOSITORY_ROOT) -> tuple[list[dict[str, str]], Counter[str]]:
    contract = current_contract(root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_raw = contract.expected("ephish_raw_count", 16_616)
    check(isinstance(raw, list) and len(raw) == expected_raw, "Unexpected raw E-PhishLLM population")
    retained: list[dict[str, str]] = []
    languages: Counter[str] = Counter()
    for index, item in enumerate(raw):
        check(isinstance(item, dict) and set(item) == {"Subject", "Body", "Language", "type"},
              f"Unexpected raw schema at index {index}")
        check(isinstance(item["Subject"], str) and isinstance(item["Body"], str),
              f"Invalid text field at index {index}")
        check(isinstance(item["Language"], str), f"Invalid language at index {index}")
        check(type(item["type"]) is int and item["type"] in {0, 1}, f"Invalid label at index {index}")
        languages[item["Language"]] += 1
        if item["Language"] != "en":
            continue
        label = "legitimate" if item["type"] == 0 else "phishing"
        retained.append({
            "record_id": stable_record_id(index), "dataset": "ephishllm", "source_subset": "english",
            "source_file": "data/raw/E-PhishLLM/ephishLLM.json", "source_record": f"json_index={index}",
            "source_role": "external_evaluation", "split": "",
            "subject": normalize_text_field(item["Subject"]), "body": normalize_text_field(item["Body"]),
            "label": label, "raw_label": str(item["type"]),
            "label_provenance": "source_field:type;0=legitimate;1=phishing", "collection_period": "",
            "inclusion_reason": "reserved evaluation only; excluded from development and selection",
        })
    expected_languages = ({"en": contract.expected("ephish_retained_count", 11_502)}
                          if contract.is_fixture else EXPECTED_LANGUAGES)
    expected_labels = (contract.expected("ephish_labels", EXPECTED_LABELS)
                       if contract.is_fixture else EXPECTED_LABELS)
    check(dict(sorted(languages.items())) == expected_languages, "Language distribution mismatch")
    check(len(retained) == contract.expected("ephish_retained_count", 11_502) and
          Counter(r["label"] for r in retained) == Counter(expected_labels),
          "English population or label distribution mismatch")
    check(len({r["record_id"] for r in retained}) == len(retained), "Duplicate record IDs")
    check(len({(r["subject"], r["body"]) for r in retained}) == len(retained), "Exact text duplicates")
    check(all(r["subject"] or r["body"] for r in retained), "Fully empty English message")
    return retained, languages


def combine(subject: str, body: str) -> str:
    left, right = subject.strip(), body.strip()
    return f"{left}\n\n{right}" if left and right else left or right


def same_binary64(left: float, right: float) -> bool:
    return struct.pack("!d", float(left)) == struct.pack("!d", float(right))


def float_matches(stored: float, reproduced: float) -> tuple[bool, str]:
    if same_binary64(stored, reproduced):
        return True, "binary64_exact"
    return (math.isclose(stored, reproduced, rel_tol=1e-12, abs_tol=1e-12),
            "documented_tolerance_fallback")


def independent_metrics(pairs: list[tuple[str, str]]) -> dict[str, float | int]:
    tn = sum(a == p == "legitimate" for a, p in pairs)
    tp = sum(a == p == "phishing" for a, p in pairs)
    fp = sum(a == "legitimate" and p == "phishing" for a, p in pairs)
    fn = sum(a == "phishing" and p == "legitimate" for a, p in pairs)
    ratio = lambda n, d: n / d if d else 0.0
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "n_total": len(pairs), "n_legitimate": tn + fp, "n_phishing": fn + tp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": ratio(tn + tp, len(pairs)),
        "precision": precision, "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(fp, fp + tn), "false_negative_rate": ratio(fn, fn + tp),
    }


def metric_schema_matches(value: dict[str, Any]) -> bool:
    return len(value) == len(METRIC_KEYS) and set(value) == set(METRIC_KEYS)


def method_specification(protected: dict[str, str] = EXPECTED_PROTECTED) -> dict[str, Any]:
    return {
        "version": 1, "protected_sha256": protected,
        "source": SOURCE, "language_filter": "Language == en", "label_mapping": {"0": "legitimate", "1": "phishing"},
        "text_view": "Step-14 normalized subject and body joined by two newlines when both are nonempty",
        "decision_threshold": THRESHOLD, "decision_rule": "phishing iff p_text >= 0.5",
        "prediction_schema": PREDICTION_COLUMNS, "metric_schema": METRIC_KEYS,
        "metric_positive_class": "phishing", "zero_division": 0,
        "ordering": "record_id ascending", "float_serialization": ".17g",
    }


def expected_run_id(protected: dict[str, str] = EXPECTED_PROTECTED,
                    root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "step28:" + hashlib.sha256(encoded).hexdigest()[:32]


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    global RUN
    RUN = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    contract = current_contract(root)
    protected = resolve_current_protected_hashes(root, EXPECTED_PROTECTED)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["selection_provenance"] == current_selection_provenance(root),
          "Selection provenance chain mismatch")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "external_metrics.json").read_text(encoding="utf-8"))
    check(manifest["artifact_type"] == "step28_ephishllm_english_external_evaluation", "Artifact type mismatch")
    check(manifest["protected_sha256"] == protected, "Protected inventory mismatch")
    check(manifest["dataset_snapshot"] == {
        "path": "data/raw/E-PhishLLM/ephishLLM.json", "size_bytes": (root / "data/raw/E-PhishLLM/ephishLLM.json").stat().st_size,
        "sha256": protected["data/raw/E-PhishLLM/ephishLLM.json"], "upstream_version": None,
    }, "Dataset snapshot mismatch")
    check(manifest["schemas"] == {"external_predictions.csv": PREDICTION_COLUMNS,
                                  "external_metrics.json": METRIC_KEYS}, "Schema manifest mismatch")
    expected_run = expected_run_id(protected, root)
    check(manifest["method_specification"] == method_specification(protected),
          "Method specification mismatch")
    check({manifest["evaluation_run_id"], report["evaluation_run_id"]} == {expected_run}, "Run ID mismatch")
    expected_implementation = {
        "src/ephishllm_evaluation.py", "src/load_data.py", "src/preprocess.py",
        "src/prepare_reserved_evaluation.py", "src/final_artifact.py",
        "scripts/validate_ephishllm_evaluation.py",
    }
    check(set(manifest["implementation_sha256"]) == expected_implementation,
          "Unexpected implementation inventory")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")

    for name in CHECKSUM_NAMES[:-1]:
        check(manifest["output_sha256"][name] == digest(out / name), f"Manifest hash mismatch: {name}")
    checksum_lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    check(checksum_lines == [f"{digest(out / name)}  {name}" for name in CHECKSUM_NAMES],
          "Checksum inventory mismatch or recursive structure")

    raw_rows, _languages = independent_raw_population(root / "data/raw/E-PhishLLM/ephishLLM.json", root)
    canonical = rows(root / "outputs/dataset_roles/ephishllm_english_external_evaluation.csv", NORMALIZED_COLUMNS)
    check(raw_rows == canonical, "Independent raw/canonical population mismatch")
    external_ids = {row["record_id"] for row in canonical}
    external_pairs = {(row["subject"], row["body"]) for row in canonical}
    split_rows = rows(root / f"outputs/splits/runs/{RUN}/split_assignments.csv",
                      ["record_id", "label", "partition"])
    check(not external_ids.intersection(r["record_id"] for r in split_rows), "External ID in internal split")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (root / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == NORMALIZED_COLUMNS, "Internal corpus schema mismatch")
        for row in reader:
            check(row["source_role"] == "text_development" and row["dataset"] != "ephishllm",
                  "External source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    check(len(internal_ids) == contract.expected("text_total_count", 251_271), "Internal corpus coverage mismatch")
    check(not external_ids.intersection(internal_ids), "External/internal ID overlap")
    check(not external_pairs.intersection(internal_pairs), "External/internal exact text overlap")
    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    check(leakage["exact_overlaps_by_scope"]["internal|ephishllm"] == 0 and
          leakage["source_role_isolation_passed"] is True, "Step-16 leakage status mismatch")

    stored = rows(out / "external_predictions.csv", PREDICTION_COLUMNS)
    expected_total = contract.expected("ephish_retained_count", 11_502)
    expected_labels = contract.expected("ephish_labels", EXPECTED_LABELS) if contract.is_fixture else EXPECTED_LABELS
    check(len(stored) == len({r["record_id"] for r in stored}) == expected_total, "Prediction coverage mismatch")
    check([r["record_id"] for r in stored] == sorted(external_ids), "Prediction ID set/order mismatch")
    check({r["evaluation_run_id"] for r in stored} == {expected_run}, "Prediction run ID mismatch")
    check(Counter(r["label"] for r in stored) == Counter(expected_labels), "Prediction labels mismatch")

    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
          type(calibrator) is LogisticRegression, "Unexpected frozen component type")
    check(list(model.classes_) == ["legitimate", "phishing"] and list(calibrator.classes_) == [0, 1]
          and float(calibrator.coef_[0, 0]) > 0, "Frozen component orientation mismatch")

    by_id = {row["record_id"]: row for row in canonical}
    comparison_modes: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for actual in stored:
        source = by_id[actual["record_id"]]
        check((actual["source"], actual["source_role"], actual["language"], actual["label"]) ==
              (SOURCE, "external_evaluation", "en", source["label"]), "Prediction provenance mismatch")
        text = combine(normalize_text_field(source["subject"]), normalize_text_field(source["body"]))
        check(bool(text), "Fully empty message")
        raw = float(model.decision_function(vectorizer.transform([text]))[0])
        probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
        prediction = "phishing" if probability >= THRESHOLD else "legitimate"
        raw_ok, raw_mode = float_matches(float(actual["text_raw_score"]), raw)
        probability_ok, probability_mode = float_matches(float(actual["calibrated_probability_phishing"]), probability)
        check(raw_ok and probability_ok, f"Independent score mismatch: {actual['record_id']}")
        comparison_modes.update((raw_mode, probability_mode))
        check(actual["prediction"] == prediction, "Prediction mismatch")
        check(actual["correct"] in {"true", "false"} and
              (actual["correct"] == "true") == (prediction == source["label"]), "Correct flag mismatch")
        pairs.append((source["label"], prediction))

    expected_metrics = independent_metrics(pairs)
    check(metric_schema_matches(metrics) and metrics == expected_metrics and
          report["metrics"] == expected_metrics, "Metric reproduction mismatch")
    check(metrics["tn"] + metrics["fp"] == metrics["n_legitimate"] == expected_labels["legitimate"],
          "Negative confusion identity failed")
    check(metrics["tp"] + metrics["fn"] == metrics["n_phishing"] == expected_labels["phishing"],
          "Positive confusion identity failed")
    check(metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == metrics["n_total"] == expected_total,
          "Total confusion identity failed")
    check(all(value is False for value in report["confirmed_boundaries"].values()),
          "Methodological boundary flag is not false")
    check(report["inference"]["decision_threshold"] == THRESHOLD and
          report["inference"]["url_extraction_performed"] is False and
          report["inference"]["url_model_loaded"] is False and
          report["inference"]["url_model_inference_performed"] is False and
          report["inference"]["fusion_performed"] is False, "Inference boundary mismatch")
    comparison = ("documented_tolerance_fallback" if "documented_tolerance_fallback" in comparison_modes
                  else "binary64_exact")
    return {"status": "validated", "evaluation_run_id": expected_run, "records": len(stored),
            "float_comparison": comparison, **{key: expected_metrics[key] for key in ("tn", "fp", "fn", "tp")}}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, ensure_ascii=False, sort_keys=True))
