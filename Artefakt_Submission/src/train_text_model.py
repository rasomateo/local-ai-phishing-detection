from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.features import build_text_views
from src.load_data import NORMALIZED_COLUMNS
from src.prepare_text_inputs import read_csv, sha256
from src.text_model_selection import current_input_identity, current_protected_paths, load_development
from src.utils import REPOSITORY_ROOT
from src.provenance import content_run_id
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
RUN_ID = ""
DEV_SHA = ""
HOLDOUT_SHA = ""
FLAGS = {"full_development_fit": True, "model_selection_performed": False, "holdout_used_for_fit": False, "holdout_used_for_metrics": False, "holdout_predictions_generated": False, "calibration_performed": False, "threshold_tuned": False, "url_data_used": False, "fusion_performed": False, "external_data_used": False, "rq2_data_used": False}
PROTECTED = {}


def refresh_active_provenance(root: Path) -> dict:
    global RUN_ID, DEV_SHA, HOLDOUT_SHA, PROTECTED
    identity = current_input_identity(root)
    RUN_ID = identity["split_run_id"]
    DEV_SHA = identity["development_sha256"]
    HOLDOUT_SHA = identity["holdout_integrity_sha256"]
    PROTECTED = current_protected_paths(root)
    return identity


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_winner(root: Path) -> tuple[dict, dict]:
    from scripts.validate_text_model_selection import validate as validate_selection

    selection = validate_selection(root)
    path = root / "outputs/benchmarks/text/winner.json"
    winner = json.loads(path.read_text(encoding="utf-8"))
    expected = {"split_run_id": RUN_ID, "model_family": "LinearSVC", "holdout_evaluated": False}
    for key, value in expected.items():
        require(winner.get(key) == value, f"Step-18 winner differs: {key}")
    require(selection["benchmark_run_id"] == winner["benchmark_run_id"] and selection["winner"] == winner["config_id"], "Validated selection/winner mismatch")
    return winner, selection


def validate_roles(root: Path, development_ids: set[str]) -> None:
    preprocessed = read_csv(root / "outputs/preprocessed_dataset.csv", NORMALIZED_COLUMNS)
    by_id = {row["record_id"]: row for row in preprocessed}
    require(len(by_id) == len(preprocessed), "Duplicate preprocessed ID")
    require(all(by_id.get(rid, {}).get("source_role") == "text_development" for rid in development_ids), "Non-text Development source role")
    for path in (root / "outputs/dataset_roles").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "record_id" in reader.fieldnames:
                require(not any(row["record_id"] in development_ids for row in reader), f"Reserved ID in Development: {path.name}")


def validate_development_frame(development, *, expected_total: int = 201016, expected_labels: dict | None = None) -> None:
    labels = expected_labels or {"legitimate": 198800, "phishing": 2216}
    require(len(development) == expected_total and Counter(development.label) == labels, "Development counts or labels differ")
    require(set(development.split_run_id) == {RUN_ID} and development.record_id.is_unique and development.record_id.ne("").all(), "Development split or ID mismatch")


def validate_partition_ids(development_ids: set[str], split_rows: list[dict]) -> None:
    canonical = {row["record_id"] for row in split_rows if row["partition"] == "development"}
    holdout = {row["record_id"] for row in split_rows if row["partition"] == "final_internal_holdout"}
    require(development_ids == canonical and not development_ids & holdout, "Development/Holdout identity mismatch")


def load_inputs(root: Path):
    refresh_active_provenance(root)
    winner, selection = validate_winner(root)
    development = load_development(root)
    contract = current_contract(root)
    validate_development_frame(
        development,
        expected_total=contract.expected("text_development_count", 201016),
        expected_labels=contract.expected("text_development_labels", {"legitimate": 198800, "phishing": 2216}),
    )
    splits = read_csv(root / f"outputs/splits/runs/{RUN_ID}/split_assignments.csv", ["record_id", "label", "partition"])
    validate_partition_ids(set(development.record_id), splits)
    validate_roles(root, set(development.record_id))
    return winner, development, selection


def audit_indices(record_ids, count: int = 256) -> list[int]:
    require(len(record_ids) >= count, "Audit subset too large")
    return sorted(range(len(record_ids)), key=lambda i: (hashlib.sha256(str(record_ids[i]).encode("utf-8")).hexdigest(), str(record_ids[i])))[:count]


def model_classes(model: LinearSVC) -> list[str]:
    classes = [str(value) for value in model.classes_]
    require(classes == ["legitimate", "phishing"], "Binary decision_function positive class is not phishing")
    return classes


def run(root: Path = REPOSITORY_ROOT) -> dict:
    start = time.perf_counter()
    winner, dev, selection = load_inputs(root)
    texts = build_text_views(dev[["subject", "body"]])[winner["view"]]
    parameters = {**winner["tfidf_parameters"], "ngram_range": tuple(winner["tfidf_parameters"]["ngram_range"]), "dtype": np.float32}
    vectorizer = TfidfVectorizer(**parameters)
    matrix = vectorizer.fit_transform(texts)
    model = LinearSVC(**winner["model_parameters"])
    require(model.get_params(deep=False) == winner["model_parameters"], "Effective LinearSVC parameters differ")
    model.fit(matrix, dev.label.to_numpy())
    classes = model_classes(model)
    require(matrix.shape == (current_contract(root).expected("text_development_count", 201016), len(vectorizer.vocabulary_)), "Vectorizer dimension mismatch")
    require(model.coef_.shape == (1, matrix.shape[1]) and model.intercept_.shape == (1,), "Model dimensions differ")
    out = root / "outputs/models/text/classifier"
    require(not out.exists() or not any(out.iterdir()), "Step-19 output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out / "text_vectorizer.joblib")
    joblib.dump(model, out / "text_model.joblib")
    indices = audit_indices(dev.record_id.tolist())
    audit_matrix = vectorizer.transform([texts[i] for i in indices])
    scores = model.decision_function(audit_matrix)
    predictions = model.predict(audit_matrix)
    with (out / "audit_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "label", "prediction", "decision_score"])
        for i, prediction, score in zip(indices, predictions, scores, strict=True):
            writer.writerow([dev.record_id.iat[i], dev.label.iat[i], prediction, format(float(score), ".17g")])
    outputs = {name: {"path": f"outputs/models/text/classifier/{name}", "sha256": sha256(out / name)} for name in ("text_vectorizer.joblib", "text_model.joblib", "audit_predictions.csv")}
    model_semantic_payload = {
        "contract_version": "artifact-provenance-v1:text-model-v1",
        "inputs": PROTECTED, "selection_semantic_sha256": selection["semantic_sha256"],
        "winner_config_id": winner["config_id"], "view": winner["view"],
        "tfidf_parameters": winner["tfidf_parameters"], "model_parameters": model.get_params(deep=False),
        "feature_dimension": matrix.shape[1], "model_classes": classes,
    }
    _model_run_id, model_semantic_sha = content_run_id("text-model", model_semantic_payload)
    manifest = {
        "artifact_type": "base_text_model", "artifact_version": 1, "step": 19,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split_run_id": RUN_ID, "benchmark_run_id": winner["benchmark_run_id"], "winner_config_id": winner["config_id"],
        "selection_provenance": {"manifest_path": "outputs/benchmarks/text/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]},
        "winner": {"path": "outputs/benchmarks/text/winner.json", "sha256": sha256(root / "outputs/benchmarks/text/winner.json")},
        "development": {"path": "outputs/features/text/development_inputs.csv", "sha256": DEV_SHA, "count": len(dev), "label_counts": dict(Counter(dev.label))},
        "view": winner["view"], "view_definition": "Current validated model-selection view from Step-17 build_text_views",
        "tfidf_parameters": winner["tfidf_parameters"], "model_parameters": model.get_params(deep=False),
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(), "random_state": model.random_state,
        "vocabulary_size": len(vectorizer.vocabulary_), "feature_dimension": matrix.shape[1], "model_classes": classes,
        "positive_semantic_class": "phishing", "decision_function_positive_class": classes[1],
        "coefficient_shape": list(model.coef_.shape), "intercept_shape": list(model.intercept_.shape), "intercept": model.intercept_.tolist(),
        "n_iter": np.asarray(model.n_iter_).tolist(),
        "audit_rule": "256 smallest SHA-256 hex digests of UTF-8 record_id; record_id tie-breaker; output in that order; label and prediction independent selection",
        "outputs": outputs, "flags": FLAGS,
    }
    manifest["contract_version"] = model_semantic_payload["contract_version"]
    manifest["semantic_sha256"] = model_semantic_sha
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs["manifest.json"] = {"path": "outputs/models/text/classifier/manifest.json", "sha256": sha256(out / "manifest.json")}
    report = {"status": "runner_checks_passed", "validation_status": "independent_validator_result_in_handoff", "created_at": manifest["created_at"],
              "inputs": {"development": manifest["development"], "winner": manifest["winner"], "holdout_integrity_only": {"path": "outputs/features/text/holdout_inputs.csv", "sha256": HOLDOUT_SHA}},
              "protected_sha256": PROTECTED, "split_run_id": RUN_ID, "benchmark_run_id": winner["benchmark_run_id"], "winner_config_id": winner["config_id"], "selection_provenance": manifest["selection_provenance"],
              "view": manifest["view"], "tfidf_parameters": manifest["tfidf_parameters"], "model_parameters": manifest["model_parameters"],
              "duration_seconds": time.perf_counter() - start, "outputs": outputs, "flags": FLAGS}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "duration_seconds": result["duration_seconds"], "outputs": result["outputs"]}, indent=2))
