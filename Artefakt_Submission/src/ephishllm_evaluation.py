from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.load_data import NORMALIZED_COLUMNS
from src.prepare_reserved_evaluation import load_ephish
from src.preprocess import normalize_text_field
from src.final_artifact import _combine_subject_body
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

SPLIT_RUN_ID = ""
SOURCE_URL = "https://github.com/pajola/e-phishGen"
SOURCE = "ephishllm_english"
RAW_TOTAL = 16_616
NON_ENGLISH = 5_114
EXPECTED_TOTAL = 11_502
EXPECTED_LABELS = {"legitimate": 5_506, "phishing": 5_996}
DECISION_THRESHOLD = 0.5

RAW = REPOSITORY_ROOT / "data/raw/E-PhishLLM/ephishLLM.json"
CANONICAL = REPOSITORY_ROOT / "outputs/dataset_roles/ephishllm_english_external_evaluation.csv"
INTERNAL = REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv"
SPLITS = Path()
VECTORIZER = REPOSITORY_ROOT / "outputs/models/text/classifier/text_vectorizer.joblib"
MODEL = REPOSITORY_ROOT / "outputs/models/text/classifier/text_model.joblib"
CALIBRATOR = REPOSITORY_ROOT / "outputs/models/text/calibration/text_calibrator.joblib"
OUT = REPOSITORY_ROOT / "outputs/evaluation/ephishllm"

EXPECTED_SHA256 = {
    "data/raw/E-PhishLLM/ephishLLM.json": "56420b143e30343f6db723c3716e10abec9f72a67ba99ae2cc7c3a87b547f51f",
    "data/raw/E-PhishLLM/README.md": "da9b43209e87eaf6c5671ad1a577b5b67115a7e6a0c16fc9cf5a1024997e29c9",
    "outputs/dataset_acquisition_manifest.json": "38937b96b86d8e938f4ded324fa885ab41b41aeb0f8369f907fb34b19bbb147d",
    "outputs/dataset_roles/ephishllm_english_external_evaluation.csv": "d833c9c78f112d7c97ed4e221839dbf8bd594c77277258180f33424606c36b3b",
    "outputs/preprocessing_report.json": "cfbf68fdc5be8b8a1be28ba211278b277821fda92067cedd44d317b2e2c8b425",
    "outputs/preprocessed_dataset.csv": "cb8ab274790414f518f0b7bb4122ce8655194044bf23798442185c95226963b6",
    "outputs/splits/manifest.json": "471dfc7558b80ead96beab66e60054f28aacdfd4c9a9ba96aeac39497de9d23d",
    f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv": "497acde2ed8ec6fd0a1d3d6f8c004398a0e2c3063d6771734b2d8eae41923b30",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == columns, f"Unexpected schema: {path}")
        return list(reader)


def validate_canonical_rows(rows: list[dict[str, str]]) -> None:
    require(len(rows) == len({row["record_id"] for row in rows}) == EXPECTED_TOTAL,
            "Canonical E-PhishLLM ID coverage mismatch")
    require(Counter(row["label"] for row in rows) == Counter(EXPECTED_LABELS),
            "Canonical E-PhishLLM label distribution mismatch")
    require(len({(row["subject"], row["body"]) for row in rows}) == EXPECTED_TOTAL,
            "Canonical E-PhishLLM contains exact duplicate text pairs")
    for row in rows:
        require(row["dataset"] == "ephishllm" and row["source_subset"] == "english",
                "Unexpected E-PhishLLM source")
        require(row["source_file"] == "data/raw/E-PhishLLM/ephishLLM.json",
                "Unexpected E-PhishLLM source file")
        require(row["source_role"] == "external_evaluation" and not row["split"],
                "Unexpected E-PhishLLM role or split")
        require(row["label"] in EXPECTED_LABELS and row["raw_label"] in {"0", "1"},
                "Unexpected E-PhishLLM label")
        require((row["raw_label"] == "0") == (row["label"] == "legitimate"),
                "E-PhishLLM label mapping mismatch")
        require(row["label_provenance"] == "source_field:type;0=legitimate;1=phishing",
                "Unexpected E-PhishLLM label provenance")
        require(row["record_id"].startswith("ephishllm:") and len(row["record_id"]) == 74,
                "Unexpected E-PhishLLM record_id")
        require(normalize_text_field(row["subject"]) == row["subject"] and
                normalize_text_field(row["body"]) == row["body"], "Noncanonical E-PhishLLM text")
        require(bool(row["subject"] or row["body"]), "Fully empty E-PhishLLM message")


def reproduce_population(path: Path, canonical_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows, source_report = load_ephish(path)
    require(rows == canonical_rows, "Raw E-PhishLLM reproduction differs from canonical artifact")
    contract = current_contract(REPOSITORY_ROOT)
    expected_report = {
        "raw": RAW_TOTAL, "retained": EXPECTED_TOTAL,
        "languages": ({"en": EXPECTED_TOTAL} if contract.is_fixture else
                      {"da": 4, "de": 2349, "en": 11502, "es": 35, "fr": 12,
                       "it": 2701, "ja": 2, "no": 3, "sv": 5, "zh-cn": 3}),
        "labels": EXPECTED_LABELS, "empty_normalized_pair": 0,
        "source_sha256": contract.payload.get("raw_sha256", {}).get(
            "data/raw/E-PhishLLM/ephishLLM.json",
            EXPECTED_SHA256["data/raw/E-PhishLLM/ephishLLM.json"]),
    }
    require(source_report == expected_report, f"E-PhishLLM population mismatch: {source_report}")
    return rows, {
        "original_records": RAW_TOTAL, "non_english_exclusions": NON_ENGLISH,
        "english_records": EXPECTED_TOTAL, "n_legitimate": EXPECTED_LABELS["legitimate"],
        "n_phishing": EXPECTED_LABELS["phishing"], "empty_normalized_pair": 0,
        "intra_source_exact_duplicate_groups": 0, "intra_source_exact_excess_records": 0,
    }


def validate_isolation(rows: list[dict[str, str]], root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    external_ids = {row["record_id"] for row in rows}
    external_pairs = {(row["subject"], row["body"]) for row in rows}
    split_rows = read_csv(root / f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv",
                          ["record_id", "label", "partition"])
    require(not external_ids.intersection(row["record_id"] for row in split_rows),
            "E-PhishLLM record_id appears in an internal split")

    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (root / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal corpus schema")
        for row in reader:
            require(row["source_role"] == "text_development", "Reserved source in internal corpus")
            require(row["dataset"] != "ephishllm", "E-PhishLLM appears in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    require(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271), "Internal corpus ID coverage mismatch")
    require(not external_ids.intersection(internal_ids), "E-PhishLLM record_id overlaps internal corpus")
    require(not external_pairs.intersection(internal_pairs), "E-PhishLLM exact text overlaps internal corpus")

    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    require(leakage["exact_overlaps_by_scope"]["internal|ephishllm"] == 0,
            "Step-16 E-PhishLLM exact-overlap status changed")
    require(leakage["source_role_isolation_passed"] is True, "Step-16 role isolation failed")
    require(leakage["reserved_source_counts"]["ephishllm"]["retained"] == EXPECTED_TOTAL,
            "Step-16 E-PhishLLM population changed")
    return {"record_id_overlap_count": 0, "exact_text_overlap_count": 0,
            "step16_internal_ephishllm_overlap_count": 0, "status": "passed"}


def validate_protected_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    actual = resolve_current_protected_hashes(root, EXPECTED_SHA256)
    architecture = json.loads((root / "outputs/models/final_artifact/architecture.json").read_text(encoding="utf-8"))
    require(architecture["decision_threshold"] == DECISION_THRESHOLD and
            architecture["fusion_applied"] is False and
            architecture["email_decision_source"] ==
            "frozen_step19_text_model_plus_step20_inference_calibrator_only",
            "Step-25C text decision contract changed")
    step19 = json.loads((root / "outputs/models/text/classifier/manifest.json").read_text(encoding="utf-8"))
    step20 = json.loads((root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8"))
    contract = current_contract(root)
    require(step19["winner_config_id"] == step20["winner_config_id"], "Text configuration changed")
    if not contract.is_fixture:
        require(step19["winner_config_id"] == "subject_body|TFIDF_B|LinearSVC|C=2;class_weight=balanced", "Production text configuration changed")
    require(step19["flags"]["external_data_used"] is False and
            step20["flags"]["external_data_used"] is False, "External data used before evaluation")
    return actual


def load_text_components(root: Path = REPOSITORY_ROOT) -> tuple[Any, Any, Any]:
    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    require(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC,
            "Unexpected frozen text component type")
    require(type(calibrator) is LogisticRegression, "Unexpected frozen text calibrator type")
    require(list(model.classes_) == ["legitimate", "phishing"], "Text score orientation changed")
    require(list(calibrator.classes_) == [0, 1] and calibrator.coef_.shape == (1, 1) and
            float(calibrator.coef_[0, 0]) > 0, "Text calibrator orientation changed")
    return vectorizer, model, calibrator


def score_text(subject: str, body: str, components: tuple[Any, Any, Any]) -> tuple[float, float, str]:
    canonical_subject, canonical_body = normalize_text_field(subject), normalize_text_field(body)
    require(bool(canonical_subject or canonical_body), "Provide a non-empty subject or body")
    vectorizer, model, calibrator = components
    text = _combine_subject_body(canonical_subject, canonical_body)
    raw = float(model.decision_function(vectorizer.transform([text]))[0])
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    require(math.isfinite(raw) and math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "Invalid frozen text score")
    return raw, probability, "phishing" if probability >= DECISION_THRESHOLD else "legitimate"


def metric_values(items: Iterable[tuple[str, str]]) -> dict[str, float | int]:
    pairs = list(items)
    require(bool(pairs), "Cannot calculate metrics for an empty population")
    require(all(label in EXPECTED_LABELS and prediction in EXPECTED_LABELS
                for label, prediction in pairs), "Invalid label or prediction")
    tn = sum(label == prediction == "legitimate" for label, prediction in pairs)
    tp = sum(label == prediction == "phishing" for label, prediction in pairs)
    fp = sum(label == "legitimate" and prediction == "phishing" for label, prediction in pairs)
    fn = sum(label == "phishing" and prediction == "legitimate" for label, prediction in pairs)
    division = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision = division(tp, tp + fp)
    recall = division(tp, tp + fn)
    return {
        "n_total": len(pairs), "n_legitimate": tn + fp, "n_phishing": fn + tp,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": division(tn + tp, len(pairs)), "precision": precision, "recall": recall,
        "f1": division(2 * precision * recall, precision + recall),
        "false_positive_rate": division(fp, fp + tn),
        "false_negative_rate": division(fn, fn + tp),
    }


def fmt(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row[field]) for field in fields})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def validate_execution_context() -> None:
    require(not OUT.exists() or not any(OUT.iterdir()), "Step-28 output directory is not empty")


def method_specification(protected: dict[str, str]) -> dict[str, Any]:
    return {
        "version": 1, "protected_sha256": protected,
        "source": SOURCE, "language_filter": "Language == en", "label_mapping": {"0": "legitimate", "1": "phishing"},
        "text_view": "Step-14 normalized subject and body joined by two newlines when both are nonempty",
        "decision_threshold": DECISION_THRESHOLD, "decision_rule": "phishing iff p_text >= 0.5",
        "prediction_schema": PREDICTION_COLUMNS, "metric_schema": METRIC_KEYS,
        "metric_positive_class": "phishing", "zero_division": 0,
        "ordering": "record_id ascending", "float_serialization": ".17g",
    }


def make_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "step28:" + hashlib.sha256(encoded).hexdigest()[:32]


def run(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    global SPLIT_RUN_ID, SPLITS, RAW_TOTAL, NON_ENGLISH, EXPECTED_TOTAL, EXPECTED_LABELS
    contract = current_contract(root)
    RAW_TOTAL = contract.expected("ephish_raw_count", 16_616)
    NON_ENGLISH = contract.expected("ephish_non_english_count", 5_114)
    EXPECTED_TOTAL = contract.expected("ephish_retained_count", 11_502)
    EXPECTED_LABELS = contract.expected("ephish_labels", {"legitimate": 5_506, "phishing": 5_996})
    split_manifest = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))
    SPLIT_RUN_ID = split_manifest["run_id"]
    SPLITS = root / split_manifest["run_directory"] / "split_assignments.csv"
    require(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository roots are unsupported")
    validate_execution_context()
    protected = validate_protected_hashes(root)
    canonical = read_csv(CANONICAL, NORMALIZED_COLUMNS)
    validate_canonical_rows(canonical)
    reproduced, population = reproduce_population(RAW, canonical)
    leakage = validate_isolation(reproduced, root)
    components = load_text_components(root)
    run_id = make_run_id(protected, root)

    predictions: list[dict[str, Any]] = []
    for row in sorted(reproduced, key=lambda item: item["record_id"]):
        raw_score, probability, prediction = score_text(row["subject"], row["body"], components)
        predictions.append({
            "evaluation_run_id": run_id, "record_id": row["record_id"], "source": SOURCE,
            "source_role": row["source_role"], "language": "en", "label": row["label"],
            "text_raw_score": raw_score, "calibrated_probability_phishing": probability,
            "prediction": prediction, "correct": prediction == row["label"],
        })
    require(len(predictions) == EXPECTED_TOTAL, "Prediction coverage mismatch")
    metrics = metric_values((row["label"], row["prediction"]) for row in predictions)
    require(list(metrics) == METRIC_KEYS, "Unexpected metric schema")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="step28-", dir=OUT.parent))
    try:
        write_csv(staging / "external_predictions.csv", PREDICTION_COLUMNS, predictions)
        write_json(staging / "external_metrics.json", metrics)
        first_hashes = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:2]}
        report = {
            "status": "completed", "evaluation_run_id": run_id,
            "source_provenance": {
                "official_source": SOURCE_URL, "dataset": "ephishllm", "source": SOURCE,
                "source_role": "external_evaluation", "subset": "english", "language": "en",
                "raw_path": "data/raw/E-PhishLLM/ephishLLM.json",
                "canonical_path": "outputs/dataset_roles/ephishllm_english_external_evaluation.csv",
                "snapshot_version": "local file snapshot identified by path, byte size, schema, and SHA-256; no upstream commit or version tag recorded",
                "raw_size_bytes": RAW.stat().st_size,
                "record_id_schema": "ephishllm:<sha256 of repository-relative source identity>",
                "label_provenance": "source_field:type;0=legitimate;1=phishing",
            },
            "population": population,
            "leakage": {**leakage, "near_duplicate_scope":
                        "Not systematically assessed; no threshold was selected or tuned."},
            "inference": {
                "path": "frozen Step-19 vectorizer/model plus frozen Step-20 inference calibrator",
                "view": "subject_body", "decision_threshold": DECISION_THRESHOLD,
                "decision_rule": "phishing iff p_text >= 0.5", "url_extraction_performed": False,
                "url_model_loaded": False, "url_model_inference_performed": False,
                "fusion_performed": False,
            },
            "metrics": metrics,
            "limitations": [
                "E-PhishLLM is synthetic and does not establish general real-world performance.",
                "Cross-corpus differences may reflect source, language, prevalence, generation, or temporal factors; the result alone cannot identify a cause.",
                "The local snapshot has no recorded upstream commit or version tag.",
                "Non-exact near-duplicate or template-level overlap was not systematically assessed.",
                "This evaluation covers the frozen email-text decision path and provides no URL-branch performance evidence.",
            ],
            "research_interpretation": {
                "main_rq": "External synthetic-benchmark evidence for the email-text portion only; not URL or general deployment evidence.",
                "rq1": "Transfer evidence for the already selected subject_body configuration; model selection is not reopened.",
            },
            "confirmed_boundaries": {
                "training_performed": False, "model_selection_performed": False,
                "calibration_performed": False, "threshold_optimized": False,
                "fusion_modified": False, "url_branch_used": False,
                "internal_holdout_opened": False, "nazario_2025_re_evaluated": False,
                "external_results_used_for_tuning": False, "dideco_or_phishingpot_evaluated": False,
                "rq2_or_rq3_started": False, "thesis_numbers_updated": False, "step29_started": False,
            },
            "output_sha256": first_hashes,
        }
        write_json(staging / "report.json", report)
        output_hashes = {**first_hashes, "report.json": sha256(staging / "report.json")}
        implementation = {relative: sha256(root / relative) for relative in (
            "src/ephishllm_evaluation.py",
            "src/load_data.py",
            "src/preprocess.py",
            "src/prepare_reserved_evaluation.py",
            "src/final_artifact.py",
            "scripts/validate_ephishllm_evaluation.py",
        )}
        manifest = {
            "selection_provenance": current_selection_provenance(root),
            "artifact_type": "step28_ephishllm_english_external_evaluation", "artifact_version": 1,
            "evaluation_run_id": run_id, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "scikit_learn_version": sklearn.__version__,
            "source": SOURCE, "dataset_snapshot": {
                "path": "data/raw/E-PhishLLM/ephishLLM.json", "size_bytes": RAW.stat().st_size,
                "sha256": protected["data/raw/E-PhishLLM/ephishLLM.json"], "upstream_version": None,
            },
            "method_specification": method_specification(protected),
            "schemas": {"external_predictions.csv": PREDICTION_COLUMNS,
                        "external_metrics.json": METRIC_KEYS},
            "protected_sha256": protected, "implementation_sha256": implementation,
            "output_sha256": output_hashes,
            "self_hash_policy": "No JSON contains its own hash; checksums.sha256 excludes itself",
        }
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")

        from scripts.validate_ephishllm_evaluation import validate
        validate(root=root, out=staging)
        require(not OUT.exists() or not any(OUT.iterdir()), "Step-28 output appeared during evaluation")
        if OUT.exists():
            OUT.rmdir()
        os.replace(staging, OUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    validate_protected_hashes(root)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-external-evaluation", action="store_true",
                        help="Explicitly authorize the canonical English E-PhishLLM evaluation")
    args = parser.parse_args(argv)
    if not args.execute_external_evaluation:
        parser.error("External evaluation requires --execute-external-evaluation")
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
