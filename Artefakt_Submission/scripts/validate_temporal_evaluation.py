from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.load_data import NORMALIZED_COLUMNS, iter_nazario_file_records
from src.preprocess import DUPLICATE_RULE, normalize_text_field
from src.utils import REPOSITORY_ROOT
from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

RUN = ""
SOURCE = "nazario_2025"
THRESHOLD = 0.5
CONFIDENCE = 0.95
Z = 1.959963984540054
OUT = REPOSITORY_ROOT / "outputs/evaluation/temporal"

EXPECTED_PROTECTED = {
    "data/raw/nazario/nazario2025.txt": "f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f",
    "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv": "5b10b1636606fcd9cb8851e977b1d6f293670b48f6fb8cc1903215e170cef8f6",
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
    "outputs/models/final_artifact/manifest.json": "df012321b671c8d3be0edfddaa4ca7fa836ada6226c717ba15eacc58fb72b21f",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
}
PREDICTION_COLUMNS = [
    "evaluation_run_id", "record_id", "source", "source_role", "collection_period", "label",
    "text_raw_score", "calibrated_probability_phishing", "prediction", "correct",
]
METRIC_KEYS = [
    "n", "tp", "fn", "recall", "miss_rate", "recall_ci_95_lower", "recall_ci_95_upper",
    "ci_method", "confidence_level", "z",
]
CHECKSUM_NAMES = ["temporal_predictions.csv", "temporal_metrics.json", "report.json", "manifest.json"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == columns, f"Unexpected schema: {path}")
        return list(reader)


def reproduce(root: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    canonical = read_csv(root / "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv", NORMALIZED_COLUMNS)
    raw = list(iter_nazario_file_records(root / "data/raw/nazario/nazario2025.txt", year=2025))
    retained: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in raw:
        row = {key: str(source.get(key, "")) for key in NORMALIZED_COLUMNS}
        row["subject"] = normalize_text_field(row["subject"])
        row["body"] = normalize_text_field(row["body"])
        pair = (row["subject"], row["body"])
        if pair not in seen:
            seen.add(pair)
            retained.append(row)
    counts = {"raw_records": len(raw), "exact_duplicate_exclusions": len(raw) - len(retained),
              "evaluated_records": len(retained), "parse_failures": 0}
    contract = current_contract(REPOSITORY_ROOT)
    check(counts == {"raw_records": contract.expected("nazario_2025_raw_count", 481),
                     "exact_duplicate_exclusions": contract.expected("nazario_2025_excluded_count", 26),
                     "evaluated_records": contract.expected("nazario_2025_retained_count", 455),
                     "parse_failures": 0}, "Population count mismatch")
    check(retained == canonical, "Independent population reproduction differs from canonical artifact")
    return retained, counts


def combine(subject: str, body: str) -> str:
    left, right = subject.strip(), body.strip()
    return f"{left}\n\n{right}" if left and right else left or right


def wilson(successes: int, total: int) -> tuple[float, float]:
    check(total > 0 and 0 <= successes <= total, "Invalid Wilson inputs")
    p = successes / total
    denominator = 1.0 + Z * Z / total
    center = (p + Z * Z / (2.0 * total)) / denominator
    margin = Z * math.sqrt(p * (1.0 - p) / total + Z * Z / (4.0 * total * total)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def same_float(left: float, right: float) -> bool:
    return struct.pack("!d", float(left)) == struct.pack("!d", float(right))


def float_matches(stored: float, reproduced: float) -> tuple[bool, str]:
    if same_float(stored, reproduced):
        return True, "binary64_exact"
    return math.isclose(stored, reproduced, rel_tol=1e-12, abs_tol=1e-15), "documented_tolerance_fallback"


def validate_isolation(root: Path, temporal: list[dict[str, str]]) -> None:
    temporal_ids = {row["record_id"] for row in temporal}
    temporal_pairs = {(row["subject"], row["body"]) for row in temporal}
    assignments = read_csv(root / f"outputs/splits/runs/{RUN}/split_assignments.csv",
                           ["record_id", "label", "partition"])
    check(not temporal_ids.intersection(row["record_id"] for row in assignments),
          "Temporal ID appears in internal assignments")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (root / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal schema")
        for row in reader:
            check(row["source_role"] == "text_development", "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    check(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271), "Internal ID coverage mismatch")
    check(not temporal_ids.intersection(internal_ids), "Temporal ID appears in internal corpus")
    check(not temporal_pairs.intersection(internal_pairs), "Temporal exact pair appears in internal corpus")


def independent_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    specification = {
        "version": 1, "predecessor_semantics": semantic_predecessor_identity(root, protected),
        "source": SOURCE, "prediction_schema": PREDICTION_COLUMNS, "metric_schema": METRIC_KEYS,
        "decision_threshold": THRESHOLD, "confidence_level": CONFIDENCE, "wilson_z": Z,
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "step27:" + hashlib.sha256(encoded).hexdigest()[:32]


def forbidden_metric_keys(value: Any) -> set[str]:
    forbidden = {"accuracy", "precision", "f1", "tn", "fp", "true_negatives", "false_positives"}
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(forbidden.intersection(value))
        for nested in value.values():
            found.update(forbidden_metric_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(forbidden_metric_keys(nested))
    return found


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    global RUN
    RUN = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    check(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository root unsupported")
    protected = resolve_current_protected_hashes(root, EXPECTED_PROTECTED)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["selection_provenance"] == current_selection_provenance(root),
          "Selection provenance chain mismatch")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "temporal_metrics.json").read_text(encoding="utf-8"))
    check(manifest["protected_sha256"] == protected, "Protected inventory mismatch")
    check(manifest["threshold"] == THRESHOLD, "Decision threshold changed")
    check(manifest["confidence_interval"] == {
        "method": "wilson_score_two_sided", "confidence_level": CONFIDENCE, "z": Z},
        "Confidence-interval contract mismatch")
    check(manifest["schemas"] == {"temporal_predictions.csv": PREDICTION_COLUMNS,
                                  "temporal_metrics.json": METRIC_KEYS}, "Output schema mismatch")
    check(list(metrics) == sorted(METRIC_KEYS), "Metric JSON key set or deterministic order mismatch")
    check(set(metrics) == set(METRIC_KEYS), "Positive-only metric schema mismatch")
    check(not forbidden_metric_keys(metrics) and not forbidden_metric_keys(report),
          "Forbidden binary metric found")

    expected_implementation = {
        "src/temporal_evaluation.py", "src/load_data.py", "src/preprocess.py",
        "src/final_artifact.py", "scripts/validate_temporal_evaluation.py",
    }
    check(set(manifest["implementation_sha256"]) == expected_implementation,
          "Unexpected implementation inventory")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")

    for name in CHECKSUM_NAMES[:-1]:
        check(manifest["output_sha256"][name] == digest(out / name), f"Manifest hash mismatch: {name}")
    lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    check(lines == [f"{digest(out / name)}  {name}" for name in CHECKSUM_NAMES],
          "Checksum inventory mismatch or recursive hash")

    expected_run = independent_run_id(protected, root)
    check(manifest["evaluation_run_id"] == report["evaluation_run_id"] == expected_run,
          "Evaluation run ID mismatch")
    temporal, counts = reproduce(root)
    validate_isolation(root, temporal)
    check(report["population"]["raw_records"] == counts["raw_records"] and
          report["population"]["exact_duplicate_exclusions"] == counts["exact_duplicate_exclusions"] and
          report["population"]["evaluated_records"] == counts["evaluated_records"] and
          report["population"]["duplicate_rule"] == DUPLICATE_RULE, "Report population mismatch")

    stored = read_csv(out / "temporal_predictions.csv", PREDICTION_COLUMNS)
    expected_ids = sorted(row["record_id"] for row in temporal)
    expected_retained = current_contract(root).expected("nazario_2025_retained_count", 455)
    check(len(stored) == len({row["record_id"] for row in stored}) == expected_retained, "Prediction coverage mismatch")
    check([row["record_id"] for row in stored] == expected_ids, "Prediction ID set/order mismatch")
    check({row["evaluation_run_id"] for row in stored} == {expected_run}, "Prediction run ID mismatch")

    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
          type(calibrator) is LogisticRegression, "Unexpected text component type")
    check(list(model.classes_) == ["legitimate", "phishing"] and list(calibrator.classes_) == [0, 1] and
          float(calibrator.coef_[0, 0]) > 0, "Text component orientation mismatch")

    temporal_by_id = {row["record_id"]: row for row in temporal}
    comparison_modes: set[str] = set()
    predictions: list[str] = []
    for actual in stored:
        source = temporal_by_id[actual["record_id"]]
        check((actual["source"], actual["source_role"], actual["collection_period"], actual["label"]) ==
              (SOURCE, "temporal_evaluation", "2025", "phishing"), "Prediction provenance mismatch")
        subject = normalize_text_field(source["subject"])
        body = normalize_text_field(source["body"])
        check(bool(subject or body), "Fully empty message")
        raw = float(model.decision_function(vectorizer.transform([combine(subject, body)]))[0])
        probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
        prediction = "phishing" if probability >= THRESHOLD else "legitimate"
        raw_ok, raw_mode = float_matches(float(actual["text_raw_score"]), raw)
        probability_ok, probability_mode = float_matches(
            float(actual["calibrated_probability_phishing"]), probability
        )
        check(raw_ok and probability_ok, f"Independent score mismatch: {actual['record_id']}")
        comparison_modes.update((raw_mode, probability_mode))
        check(actual["prediction"] == prediction, "Prediction mismatch")
        check(actual["correct"] in {"true", "false"} and
              (actual["correct"] == "true") == (prediction == "phishing"), "Correct flag mismatch")
        predictions.append(prediction)

    tp = sum(value == "phishing" for value in predictions)
    fn = sum(value == "legitimate" for value in predictions)
    lower, upper = wilson(tp, len(predictions))
    expected_metrics = {
        "n": len(predictions), "tp": tp, "fn": fn, "recall": tp / len(predictions),
        "miss_rate": fn / len(predictions), "recall_ci_95_lower": lower,
        "recall_ci_95_upper": upper, "ci_method": "wilson_score_two_sided",
        "confidence_level": CONFIDENCE, "z": Z,
    }
    check(metrics == expected_metrics and report["metrics"] == expected_metrics, "Metric reproduction mismatch")
    check(tp + fn == len(predictions) == expected_retained, "Positive-only count identity failed")
    check(report["inference"]["url_extraction_performed"] is False and
          report["inference"]["url_model_loaded"] is False and
          report["inference"]["url_model_inference_performed"] is False,
          "URL branch was not disabled")
    check(all(value is False for value in report["flags"].values()), "Methodological flag is not false")
    comparison = "documented_tolerance_fallback" if "documented_tolerance_fallback" in comparison_modes else "binary64_exact"
    return {"status": "validated", "evaluation_run_id": expected_run, "records": len(stored),
            "float_comparison": comparison, "tp": tp, "fn": fn}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, ensure_ascii=False, sort_keys=True))
