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
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.load_data import NORMALIZED_COLUMNS, iter_nazario_file_records
from src.preprocess import DUPLICATE_RULE, normalize_text_field
from src.final_artifact import _combine_subject_body
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

SPLIT_RUN_ID = ""
SOURCE_URL = "https://monkey.org/~jose/phishing/"
SOURCE = "nazario_2025"
EXPECTED_RAW = 481
EXPECTED_EXCLUDED = 26
EXPECTED_RETAINED = 455
DECISION_THRESHOLD = 0.5
CONFIDENCE_LEVEL = 0.95
WILSON_Z = 1.959963984540054

RAW = REPOSITORY_ROOT / "data/raw/nazario/nazario2025.txt"
CANONICAL = REPOSITORY_ROOT / "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv"
INTERNAL = REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv"
SPLITS = Path()
VECTORIZER = REPOSITORY_ROOT / "outputs/models/text/classifier/text_vectorizer.joblib"
MODEL = REPOSITORY_ROOT / "outputs/models/text/classifier/text_model.joblib"
CALIBRATOR = REPOSITORY_ROOT / "outputs/models/text/calibration/text_calibrator.joblib"
OUT = REPOSITORY_ROOT / "outputs/evaluation/temporal"

EXPECTED_SHA256 = {
    "data/raw/nazario/nazario2025.txt": "f1fa7e0fe35c9a16f36d9aa20ade7e1d3908d1d0c1d917b8c53c8aa799ec1c8f",
    "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv": "5b10b1636606fcd9cb8851e977b1d6f293670b48f6fb8cc1903215e170cef8f6",
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


def reproduce_population(
    raw_rows: Iterable[dict[str, str]], canonical_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw = list(raw_rows)
    retained: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source in raw:
        normalized = {key: str(source.get(key, "")) for key in NORMALIZED_COLUMNS}
        normalized["subject"] = normalize_text_field(normalized["subject"])
        normalized["body"] = normalize_text_field(normalized["body"])
        pair = (normalized["subject"], normalized["body"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        retained.append(normalized)
    require(retained == canonical_rows, "Reproduced Nazario population differs from canonical artifact")
    counts = {"raw_records": len(raw), "exact_duplicate_exclusions": len(raw) - len(retained),
              "evaluated_records": len(retained), "parse_failures": 0}
    require(counts == {"raw_records": EXPECTED_RAW, "exact_duplicate_exclusions": EXPECTED_EXCLUDED,
                       "evaluated_records": EXPECTED_RETAINED, "parse_failures": 0},
            f"Nazario population count mismatch: {counts}")
    return retained, counts


def validate_canonical_rows(rows: list[dict[str, str]]) -> None:
    require(len(rows) == len({row["record_id"] for row in rows}) == EXPECTED_RETAINED,
            "Canonical Nazario ID coverage mismatch")
    for row in rows:
        require(row["dataset"] == "nazario" and row["source_subset"] == SOURCE,
                "Unexpected Nazario source")
        require(row["source_file"] == "data/raw/nazario/nazario2025.txt",
                "Unexpected Nazario source file")
        require(row["source_role"] == "temporal_evaluation" and row["collection_period"] == "2025",
                "Unexpected Nazario role or period")
        require(row["label"] == row["raw_label"] == "phishing", "Unexpected Nazario label")
        require(row["label_provenance"] == "source_corpus_membership:nazario_phishing",
                "Unexpected Nazario label provenance")
        require(row["record_id"].startswith("nazario:") and len(row["record_id"]) == 72,
                "Unexpected Nazario record_id")
        require(normalize_text_field(row["subject"]) == row["subject"] and
                normalize_text_field(row["body"]) == row["body"], "Noncanonical Nazario text")
        require(bool(row["subject"] or row["body"]), "Fully empty Nazario message")


def validate_isolation(rows: list[dict[str, str]], root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    temporal_ids = {row["record_id"] for row in rows}
    temporal_pairs = {(row["subject"], row["body"]) for row in rows}
    split_rows = read_csv(root / f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv",
                          ["record_id", "label", "partition"])
    split_ids = {row["record_id"] for row in split_rows}
    require(not temporal_ids.intersection(split_ids), "Nazario record_id appears in an internal split")

    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (root / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal corpus schema")
        for row in reader:
            require(row["source_role"] == "text_development", "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    require(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271), "Internal corpus ID coverage mismatch")
    require(not temporal_ids.intersection(internal_ids), "Nazario record_id appears in internal corpus")
    require(not temporal_pairs.intersection(internal_pairs), "Nazario exact text overlaps internal corpus")

    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    require(leakage["exact_overlaps_by_scope"]["internal|nazario_2025"] == 0,
            "Step-16 Nazario exact-overlap status changed")
    require(leakage["source_role_isolation_passed"] is True, "Step-16 role isolation failed")
    return {"record_id_overlap_count": 0, "exact_text_overlap_count": 0,
            "step16_internal_nazario_overlap_count": 0, "status": "passed"}


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
    canonical_subject = normalize_text_field(subject)
    canonical_body = normalize_text_field(body)
    require(bool(canonical_subject or canonical_body), "Provide a non-empty subject or body")
    vectorizer, model, calibrator = components
    text = _combine_subject_body(canonical_subject, canonical_body)
    raw = float(model.decision_function(vectorizer.transform([text]))[0])
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    require(math.isfinite(raw) and math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "Invalid frozen text score")
    return raw, probability, "phishing" if probability >= DECISION_THRESHOLD else "legitimate"


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    require(total > 0 and 0 <= successes <= total, "Invalid Wilson inputs")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def metric_values(predictions: Iterable[str]) -> dict[str, Any]:
    values = list(predictions)
    tp = sum(value == "phishing" for value in values)
    fn = sum(value == "legitimate" for value in values)
    require(tp + fn == len(values) and values, "Invalid positive-only predictions")
    lower, upper = wilson_interval(tp, len(values))
    return {
        "n": len(values), "tp": tp, "fn": fn, "recall": tp / len(values),
        "miss_rate": fn / len(values), "recall_ci_95_lower": lower,
        "recall_ci_95_upper": upper, "ci_method": "wilson_score_two_sided",
        "confidence_level": CONFIDENCE_LEVEL, "z": WILSON_Z,
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
    require(not OUT.exists() or not any(OUT.iterdir()), "Step-27 output directory is not empty")


def make_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    specification = {
        "version": 1,
        "predecessor_semantics": semantic_predecessor_identity(root, protected),
        "source": SOURCE, "prediction_schema": PREDICTION_COLUMNS, "metric_schema": METRIC_KEYS,
        "decision_threshold": DECISION_THRESHOLD, "confidence_level": CONFIDENCE_LEVEL, "wilson_z": WILSON_Z,
    }
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "step27:" + hashlib.sha256(encoded).hexdigest()[:32]


def run(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    global SPLIT_RUN_ID, SPLITS, EXPECTED_RAW, EXPECTED_EXCLUDED, EXPECTED_RETAINED
    contract = current_contract(root)
    EXPECTED_RAW = contract.expected("nazario_2025_raw_count", 481)
    EXPECTED_EXCLUDED = contract.expected("nazario_2025_excluded_count", 26)
    EXPECTED_RETAINED = contract.expected("nazario_2025_retained_count", 455)
    split_manifest = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))
    SPLIT_RUN_ID = split_manifest["run_id"]
    SPLITS = root / split_manifest["run_directory"] / "split_assignments.csv"
    require(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository roots are unsupported")
    validate_execution_context()
    protected = validate_protected_hashes(root)

    canonical_rows = read_csv(CANONICAL, NORMALIZED_COLUMNS)
    validate_canonical_rows(canonical_rows)
    reproduced, population = reproduce_population(
        iter_nazario_file_records(RAW, year=2025), canonical_rows
    )
    leakage = validate_isolation(reproduced, root)

    components = load_text_components(root)
    run_id = make_run_id(protected, root)
    predictions: list[dict[str, Any]] = []
    for row in sorted(reproduced, key=lambda item: item["record_id"]):
        raw_score, probability, prediction = score_text(row["subject"], row["body"], components)
        predictions.append({
            "evaluation_run_id": run_id, "record_id": row["record_id"], "source": SOURCE,
            "source_role": row["source_role"], "collection_period": row["collection_period"],
            "label": row["label"], "text_raw_score": raw_score,
            "calibrated_probability_phishing": probability, "prediction": prediction,
            "correct": prediction == row["label"],
        })
    require(len(predictions) == EXPECTED_RETAINED, "Prediction coverage mismatch")
    metrics = metric_values(row["prediction"] for row in predictions)
    require(list(metrics) == METRIC_KEYS, "Unexpected positive-only metric schema")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="step27-", dir=OUT.parent))
    try:
        write_csv(staging / "temporal_predictions.csv", PREDICTION_COLUMNS, predictions)
        write_json(staging / "temporal_metrics.json", metrics)
        first_hashes = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:2]}
        report = {
            "status": "completed", "evaluation_run_id": run_id,
            "source_provenance": {
                "official_source": SOURCE_URL, "raw_path": "data/raw/nazario/nazario2025.txt",
                "canonical_path": "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv",
                "dataset": "nazario", "source": SOURCE, "source_role": "temporal_evaluation",
                "collection_period": "2025", "label": "phishing",
                "label_provenance": "source_corpus_membership:nazario_phishing",
                "record_id_schema": "nazario:<sha256 of repository-relative source identity>",
            },
            "population": {**population, "duplicate_rule": DUPLICATE_RULE,
                           "empty_subject_records": sum(not row["subject"] for row in reproduced),
                           "empty_body_records": sum(not row["body"] for row in reproduced)},
            "leakage": {**leakage, "near_duplicate_scope":
                        "Not systematically assessed; no threshold was selected or tuned."},
            "inference": {
                "path": "frozen Step-19 vectorizer/model plus frozen Step-20 inference calibrator",
                "view": "subject_body", "decision_threshold": DECISION_THRESHOLD,
                "decision_rule": "phishing iff p_text >= 0.5", "url_extraction_performed": False,
                "url_model_loaded": False, "url_model_inference_performed": False,
            },
            "metrics": metrics,
            "required_statement": (
                f"Das Modell erkennt {100.0 * metrics['recall']:.6f} % der im reservierten "
                "Nazario-2025-Korpus enthaltenen Phishing-E-Mails."
            ),
            "confidence_interval_interpretation": (
                "The Wilson interval quantifies uncertainty in the observed recall estimate only; "
                "it does not quantify performance uncertainty for arbitrary future phishing campaigns."
            ),
            "limitations": [
                "The corpus is positive-only; accuracy, precision, F1, TN, FP, and false-positive behavior are undefined.",
                "Non-exact near-duplicate or template-level overlap was not systematically assessed.",
                "Results describe the reserved Nazario 2025 corpus and do not establish general future-campaign performance.",
            ],
            "flags": {
                "model_fitted": False, "calibrator_fitted": False, "threshold_optimized": False,
                "fusion_used": False, "url_branch_used": False, "holdout_opened": False,
                "holdout_evaluated": False, "training_data_modified": False,
                "external_data_used_for_tuning": False, "rq2_or_rq3_started": False,
                "step28_started": False,
            },
            "output_sha256": first_hashes,
        }
        write_json(staging / "report.json", report)
        manifest_hashes = {**first_hashes, "report.json": sha256(staging / "report.json")}
        implementation = {
            relative: sha256(root / relative) for relative in (
                "src/temporal_evaluation.py",
                "src/load_data.py",
                "src/preprocess.py",
                "src/final_artifact.py",
                "scripts/validate_temporal_evaluation.py",
            )
        }
        manifest = {
            "selection_provenance": current_selection_provenance(root),
            "artifact_type": "step27_nazario_2025_temporal_evaluation", "artifact_version": 1,
            "evaluation_run_id": run_id, "python_version": platform.python_version(),
            "numpy_version": np.__version__, "scikit_learn_version": sklearn.__version__,
            "source": SOURCE, "threshold": DECISION_THRESHOLD,
            "confidence_interval": {"method": "wilson_score_two_sided", "confidence_level": CONFIDENCE_LEVEL,
                                    "z": WILSON_Z},
            "schemas": {"temporal_predictions.csv": PREDICTION_COLUMNS,
                        "temporal_metrics.json": METRIC_KEYS},
            "protected_sha256": protected, "implementation_sha256": implementation,
            "output_sha256": manifest_hashes,
            "self_hash_policy": "No JSON contains its own hash; checksums.sha256 excludes itself",
        }
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")

        from scripts.validate_temporal_evaluation import validate
        validate(root=root, out=staging)
        require(not OUT.exists() or not any(OUT.iterdir()), "Step-27 output appeared during evaluation")
        if OUT.exists():
            OUT.rmdir()
        os.replace(staging, OUT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-temporal-evaluation", action="store_true",
                        help="Explicitly authorize the canonical Nazario 2025 temporal evaluation")
    args = parser.parse_args(argv)
    if not args.execute_temporal_evaluation:
        parser.error("Temporal evaluation requires --execute-temporal-evaluation")
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
