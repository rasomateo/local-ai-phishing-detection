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
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.load_data import NORMALIZED_COLUMNS
from src.prepare_reserved_evaluation import POT_COMMIT, POT_ROOT, load_phishingpot
from src.preprocess import DUPLICATE_RULE, normalize_text_field
from src.final_artifact import _combine_subject_body
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

SPLIT_RUN_ID = ""
SOURCE = "phishingpot"
SOURCE_URL = "https://github.com/rf-peixoto/phishing_pot"
EXPECTED_RAW = 8_614
EXPECTED_DUPLICATE_EXCLUSIONS = 648
EXPECTED_CANONICAL_RETAINED = 7_966
EXPECTED_EMPTY_RAW = 6
EXPECTED_EMPTY_EXCLUSIONS = 1
EXPECTED_EVALUATED = 7_965
EMPTY_REPRESENTATIVE_ID = "phishingpot:de3cc5649e8e5951078d42a204cf14e345b08e53ac4e11f5aa8c91388282bca9"
TRACKED_PATH_LIST_SHA256 = "d940fe055abda78f20fdd6117ef1fd5717b137521769dd93ae0b88a2eb4cc754"
PARSED_SNAPSHOT_SHA256 = "d0e132b2e6239bb83c77f34cef59cf65c396ae225d284dfeb700881b08a13684"
DECISION_THRESHOLD = 0.5
CONFIDENCE_LEVEL = 0.95
WILSON_Z = 1.959963984540054

CANONICAL = REPOSITORY_ROOT / "outputs/dataset_roles/phishingpot_external_evaluation.csv"
INTERNAL = REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv"
OUT = REPOSITORY_ROOT / "outputs/evaluation/phishingpot"

EXPECTED_SHA256 = {
    "data/raw/PhishingPot/phishing_pot/README.md": "fea48a02fe962a72a72f62c825da61f45bf07ad39708840ca8a9b20604f471ce",
    "outputs/dataset_acquisition_manifest.json": "38937b96b86d8e938f4ded324fa885ab41b41aeb0f8369f907fb34b19bbb147d",
    "outputs/dataset_roles/phishingpot_external_evaluation.csv": "39762e5d6af8c592ed93313695ec4113bcb5e8097c32eb2aa61df15fedb0d393",
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
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
}

PREDICTION_COLUMNS = [
    "evaluation_run_id", "record_id", "source", "source_role", "collection_period", "label",
    "text_raw_score", "calibrated_probability_phishing", "prediction", "correct",
]
METRIC_KEYS = [
    "n", "tp", "fn", "recall", "miss_rate", "recall_ci_95_lower", "recall_ci_95_upper",
    "ci_method", "confidence_level", "z",
]
CHECKSUM_NAMES = ["external_predictions.csv", "external_metrics.json", "report.json", "manifest.json"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == columns, f"Unexpected schema: {path}")
        return list(reader)


def validate_canonical_rows(rows: list[dict[str, str]]) -> None:
    require(len(rows) == len({row["record_id"] for row in rows}) == EXPECTED_RAW,
            "PhishingPot role artifact ID coverage mismatch")
    for row in rows:
        require(row["dataset"] == SOURCE and row["source_subset"] == "git_snapshot",
                "Unexpected PhishingPot source")
        require(row["source_file"] == f"data/raw/PhishingPot/phishing_pot/@{POT_COMMIT}",
                "Unexpected PhishingPot snapshot identity")
        require(row["source_role"] == "external_evaluation" and row["collection_period"] == "",
                "Unexpected PhishingPot role or collection period")
        require(row["label"] == row["raw_label"] == "phishing", "Unexpected PhishingPot label")
        require(row["label_provenance"] == "source_corpus_membership:phishing_pot",
                "Unexpected PhishingPot label provenance")
        require(row["record_id"].startswith("phishingpot:") and len(row["record_id"]) == 76,
                "Unexpected PhishingPot record_id")
        require(normalize_text_field(row["subject"]) == row["subject"] and
                normalize_text_field(row["body"]) == row["body"], "Noncanonical PhishingPot text")


def reproduce_population(raw_rows: Iterable[dict[str, str]], canonical_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw = list(raw_rows)
    require(raw == canonical_rows, "Raw PhishingPot reproduction differs from role artifact")
    empty_raw = [row for row in raw if not row["subject"] and not row["body"]]
    require(len(empty_raw) == EXPECTED_EMPTY_RAW, "Unexpected raw empty-message count")
    retained: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in raw:
        pair = (row["subject"], row["body"])
        if pair not in seen:
            seen.add(pair)
            retained.append(row)
    empty_retained = [row for row in retained if not row["subject"] and not row["body"]]
    require(len(raw) - len(retained) == EXPECTED_DUPLICATE_EXCLUSIONS and
            len(retained) == EXPECTED_CANONICAL_RETAINED, "Exact-dedup reconciliation mismatch")
    expected_empty_ids = [] if EXPECTED_EMPTY_EXCLUSIONS == 0 else [EMPTY_REPRESENTATIVE_ID]
    require([row["record_id"] for row in empty_retained] == expected_empty_ids,
            "Unexpected retained empty representative")
    evaluated = [row for row in retained if row["subject"] or row["body"]]
    require(len(retained) - len(evaluated) == EXPECTED_EMPTY_EXCLUSIONS and
            len(evaluated) == EXPECTED_EVALUATED, "Evaluation-only empty exclusion mismatch")
    return evaluated, {
        "raw_records": len(raw), "raw_empty_records": len(empty_raw),
        "raw_empty_record_ids": [row["record_id"] for row in empty_raw],
        "exact_duplicate_exclusions": len(raw) - len(retained),
        "canonical_retained": len(retained), "evaluation_only_empty_exclusions": EXPECTED_EMPTY_EXCLUSIONS,
        "evaluation_only_excluded_record_id": (EMPTY_REPRESENTATIVE_ID if EXPECTED_EMPTY_EXCLUSIONS else None),
        "evaluated_records": len(evaluated), "parse_failures": 0,
        "duplicate_rule": DUPLICATE_RULE,
        "evaluation_input_validity_rule": "at least one Step-14-normalized field subject or body is nonempty",
    }


def validate_source_snapshot(metadata: dict[str, Any]) -> dict[str, Any]:
    contract = current_contract(REPOSITORY_ROOT)
    expected_commit = contract.expected("phishingpot_commit", POT_COMMIT)
    expected = {
        "raw": EXPECTED_RAW, "retained": EXPECTED_RAW, "parse_failures": [],
        "empty_subject": contract.expected("phishingpot_empty_subject", 46),
        "empty_body": contract.expected("phishingpot_empty_body", 69),
        "empty_normalized_pair": EXPECTED_EMPTY_RAW,
        "commit": expected_commit,
        "tracked_path_list_sha256": contract.expected("phishingpot_path_digest", TRACKED_PATH_LIST_SHA256),
        "parsed_snapshot_content_sha256": contract.expected("phishingpot_content_digest", PARSED_SNAPSHOT_SHA256),
    }
    require(metadata == expected, f"PhishingPot snapshot metadata mismatch: {metadata}")
    return {**metadata, "official_source": SOURCE_URL, "collection_period": "undocumented",
            "file_inventory_verified": True}


def validate_isolation(rows: list[dict[str, str]], root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    external_ids = {row["record_id"] for row in rows}
    external_pairs = {(row["subject"], row["body"]) for row in rows}
    assignments = read_csv(root / f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv",
                           ["record_id", "label", "partition"])
    require(not external_ids.intersection(row["record_id"] for row in assignments),
            "PhishingPot record_id appears in an internal split")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with INTERNAL.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal schema")
        for row in reader:
            require(row["source_role"] == "text_development" and row["dataset"] != SOURCE,
                    "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    require(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271), "Internal corpus coverage mismatch")
    require(not external_ids.intersection(internal_ids), "PhishingPot ID overlaps internal corpus")
    require(not external_pairs.intersection(internal_pairs), "PhishingPot exact text overlaps internal corpus")
    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    require(leakage["exact_overlaps_by_scope"]["internal|phishingpot"] == 0 and
            leakage["source_role_isolation_passed"] is True, "Step-16 isolation changed")
    require(leakage["reserved_source_counts"]["phishingpot"]["retained"] == EXPECTED_RAW,
            "Step-16 PhishingPot population changed")
    return {"record_id_overlap_count": 0, "exact_text_overlap_count": 0,
            "step16_internal_phishingpot_overlap_count": 0, "status": "passed"}


def validate_protected_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    actual = resolve_current_protected_hashes(root, EXPECTED_SHA256)
    architecture = json.loads((root / "outputs/models/final_artifact/architecture.json").read_text(encoding="utf-8"))
    require(architecture["decision_threshold"] == DECISION_THRESHOLD and
            architecture["fusion_applied"] is False and architecture["email_decision_source"] ==
            "frozen_step19_text_model_plus_step20_inference_calibrator_only",
            "Step-25C text decision contract changed")
    step19 = json.loads((root / "outputs/models/text/classifier/manifest.json").read_text(encoding="utf-8"))
    step20 = json.loads((root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8"))
    require(step19["winner_config_id"] == step20["winner_config_id"], "Text configuration changed")
    if current_contract(root).is_production:
        require(step19["winner_config_id"] ==
                "subject_body|TFIDF_B|LinearSVC|C=2;class_weight=balanced", "Text configuration changed")
    require(step19["flags"]["external_data_used"] is False and step20["flags"]["external_data_used"] is False,
            "External data was used during model development")
    return actual


def load_text_components(root: Path = REPOSITORY_ROOT) -> tuple[Any, Any, Any]:
    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    require(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
            type(calibrator) is LogisticRegression, "Unexpected frozen component type")
    require(list(model.classes_) == ["legitimate", "phishing"] and
            list(calibrator.classes_) == [0, 1] and calibrator.coef_.shape == (1, 1) and
            float(calibrator.coef_[0, 0]) > 0, "Frozen component orientation changed")
    return vectorizer, model, calibrator


def score_text(subject: str, body: str, components: tuple[Any, Any, Any]) -> tuple[float, float, str]:
    subject, body = normalize_text_field(subject), normalize_text_field(body)
    require(bool(subject or body), "Fully empty message is not evaluable")
    vectorizer, model, calibrator = components
    raw = float(model.decision_function(vectorizer.transform([_combine_subject_body(subject, body)]))[0])
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    require(math.isfinite(raw) and math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "Invalid frozen text score")
    return raw, probability, "phishing" if probability >= DECISION_THRESHOLD else "legitimate"


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    require(total > 0 and 0 <= successes <= total, "Invalid Wilson inputs")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - margin),
            1.0 if successes == total else min(1.0, center + margin))


def metric_values(predictions: Iterable[str]) -> dict[str, Any]:
    values = list(predictions)
    tp, fn = values.count("phishing"), values.count("legitimate")
    require(values and tp + fn == len(values), "Invalid positive-only predictions")
    lower, upper = wilson_interval(tp, len(values))
    return {"n": len(values), "tp": tp, "fn": fn, "recall": tp / len(values),
            "miss_rate": fn / len(values), "recall_ci_95_lower": lower,
            "recall_ci_95_upper": upper, "ci_method": "wilson_score_two_sided",
            "confidence_level": CONFIDENCE_LEVEL, "z": WILSON_Z}


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
    require(not OUT.exists() or not any(OUT.iterdir()), "Step-29 output directory is not empty")


def method_specification(protected: dict[str, str]) -> dict[str, Any]:
    return {
        "version": 1, "protected_sha256": protected,
        "source": SOURCE, "source_commit": POT_COMMIT, "raw_records": EXPECTED_RAW,
        "exact_duplicate_rule": DUPLICATE_RULE,
        "exact_duplicate_exclusions": EXPECTED_DUPLICATE_EXCLUSIONS,
        "canonical_retained": EXPECTED_CANONICAL_RETAINED,
        "evaluation_input_validity_rule": "subject or body nonempty after Step-14 normalization",
        "evaluation_only_excluded_record_id": (EMPTY_REPRESENTATIVE_ID if EXPECTED_EMPTY_EXCLUSIONS else None),
        "evaluated_records": EXPECTED_EVALUATED, "prediction_schema": PREDICTION_COLUMNS,
        "metric_schema": METRIC_KEYS, "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "phishing iff p_text >= 0.5", "confidence_level": CONFIDENCE_LEVEL,
        "wilson_z": WILSON_Z, "ordering": "record_id ascending", "float_serialization": ".17g",
    }


def make_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "step29:" + hashlib.sha256(encoded).hexdigest()[:32]


def run(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    global SPLIT_RUN_ID, POT_COMMIT, TRACKED_PATH_LIST_SHA256, PARSED_SNAPSHOT_SHA256
    global EXPECTED_RAW, EXPECTED_DUPLICATE_EXCLUSIONS
    global EXPECTED_CANONICAL_RETAINED, EXPECTED_EMPTY_RAW, EXPECTED_EMPTY_EXCLUSIONS, EXPECTED_EVALUATED
    contract = current_contract(root)
    POT_COMMIT = contract.expected("phishingpot_commit", POT_COMMIT)
    TRACKED_PATH_LIST_SHA256 = contract.expected("phishingpot_path_digest", TRACKED_PATH_LIST_SHA256)
    PARSED_SNAPSHOT_SHA256 = contract.expected("phishingpot_content_digest", PARSED_SNAPSHOT_SHA256)
    EXPECTED_RAW = contract.expected("phishingpot_raw_count", EXPECTED_RAW)
    EXPECTED_DUPLICATE_EXCLUSIONS = contract.expected("phishingpot_duplicate_exclusions", EXPECTED_DUPLICATE_EXCLUSIONS)
    EXPECTED_CANONICAL_RETAINED = contract.expected("phishingpot_canonical_count", EXPECTED_CANONICAL_RETAINED)
    EXPECTED_EMPTY_RAW = contract.expected("phishingpot_empty_raw", EXPECTED_EMPTY_RAW)
    EXPECTED_EMPTY_EXCLUSIONS = contract.expected("phishingpot_empty_exclusions", EXPECTED_EMPTY_EXCLUSIONS)
    EXPECTED_EVALUATED = contract.expected("phishingpot_evaluated_count", EXPECTED_EVALUATED)
    SPLIT_RUN_ID = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    require(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository root unsupported")
    validate_execution_context()
    protected = validate_protected_hashes(root)
    raw_rows, source_metadata = load_phishingpot()
    snapshot = validate_source_snapshot(source_metadata)
    canonical = read_csv(CANONICAL, NORMALIZED_COLUMNS)
    validate_canonical_rows(canonical)
    evaluated, population = reproduce_population(raw_rows, canonical)
    leakage = validate_isolation(evaluated, root)
    components = load_text_components(root)
    run_id = make_run_id(protected, root)

    predictions: list[dict[str, Any]] = []
    for row in sorted(evaluated, key=lambda item: item["record_id"]):
        raw, probability, prediction = score_text(row["subject"], row["body"], components)
        predictions.append({"evaluation_run_id": run_id, "record_id": row["record_id"],
                            "source": SOURCE, "source_role": "external_evaluation", "collection_period": "",
                            "label": "phishing",
                            "text_raw_score": raw, "calibrated_probability_phishing": probability,
                            "prediction": prediction, "correct": prediction == "phishing"})
    metrics = metric_values(row["prediction"] for row in predictions)
    report = {
        "status": "completed", "evaluation_run_id": run_id,
        "source_provenance": {"official_source": SOURCE_URL,
                              "local_path": "data/raw/PhishingPot/phishing_pot",
                              "upstream_commit": POT_COMMIT, "collection_period": "undocumented",
                              "source_role": "external_evaluation", "source_subset": "git_snapshot",
                              "label": "phishing", "label_provenance": "source_corpus_membership:phishing_pot"},
        "snapshot": snapshot, "population": population, "leakage": {
            **leakage, "near_duplicate_scope": "not systematically assessed; Step-16C exact-only decision retained"},
        "inference": {"text_view": "Step-14 normalized subject_body", "decision_threshold": DECISION_THRESHOLD,
                      "decision_rule": "phishing iff p_text >= 0.5", "url_extraction_performed": False,
                      "url_model_loaded": False, "url_model_inference_performed": False,
                      "fusion_applied": False, "fit_performed": False},
        "metrics": metrics,
        "limitations": [
            "Positive-only corpus: accuracy, precision, F1, TN, FP, and FPR are not identifiable.",
            "Non-exact near-duplicate or template-level overlap was not systematically assessed.",
            "The Wilson interval describes this corpus, not arbitrary future phishing attacks.",
            "Cross-corpus differences versus Steps 26-28 are descriptive and not causal.",
        ],
        "boundaries": {"training_performed": False, "calibration_performed": False,
                       "threshold_optimized": False, "previous_evaluations_rerun": False,
                       "rq2_or_rq3_started": False, "thesis_updated": False},
    }

    implementation = {relative: sha256(root / relative) for relative in [
        "src/phishingpot_evaluation.py",
        "src/load_data.py", "src/preprocess.py", "src/prepare_reserved_evaluation.py",
        "src/final_artifact.py",
        "scripts/validate_phishingpot_evaluation.py"]}
    manifest = {
        "selection_provenance": current_selection_provenance(root),
        "artifact_type": "step29_phishingpot_external_evaluation", "schema_version": 1,
        "status": "completed", "evaluation_run_id": run_id,
        "method_specification": method_specification(protected),
        "protected_sha256": protected, "implementation_sha256": implementation,
        "dataset_snapshot": {"path": "data/raw/PhishingPot/phishing_pot", "upstream_commit": POT_COMMIT,
                             "tracked_eml_count": EXPECTED_RAW,
                             "tracked_path_list_sha256": TRACKED_PATH_LIST_SHA256,
                             "parsed_snapshot_content_sha256": PARSED_SNAPSHOT_SHA256,
                             "role_artifact_sha256": protected["outputs/dataset_roles/phishingpot_external_evaluation.csv"]},
        "schemas": {"external_predictions.csv": PREDICTION_COLUMNS, "external_metrics.json": METRIC_KEYS},
        "runtime": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "joblib": importlib.metadata.version("joblib")},
        "output_sha256": {},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".phishingpot-evaluation-staging-", dir=OUT.parent))
    try:
        write_csv(staging / "external_predictions.csv", PREDICTION_COLUMNS, predictions)
        write_json(staging / "external_metrics.json", metrics)
        write_json(staging / "report.json", report)
        manifest["output_sha256"] = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:3]}
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")
        subprocess.run([sys.executable, "-m", "scripts.validate_phishingpot_evaluation",
                        "--out", str(staging)], cwd=root, check=True)
        require(not OUT.exists(), "Step-29 output path appeared during staging")
        os.replace(staging, OUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "completed", "evaluation_run_id": run_id, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-external-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_external_evaluation:
        parser.error("Step 29 is inert without --execute-external-evaluation")
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
