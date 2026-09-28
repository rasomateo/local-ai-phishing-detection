from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import LinearSVC

from src.prepare_text_inputs import read_csv, sha256
from src.text_model_selection import current_input_identity, current_protected_paths
from src.provenance import content_run_id
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
RUN = ""
OOF_COLUMNS = ["benchmark_run_id", "split_run_id", "record_id", "label", "cv_fold", "prediction", "model_score", "score_type"]
OUTPUT_COLUMNS = ["calibration_run_id", "split_run_id", "benchmark_run_id", "record_id", "label", "cv_fold", "raw_model_score", "calibrated_probability_phishing"]
PARAMETERS = {"penalty": None, "solver": "lbfgs", "class_weight": None, "fit_intercept": True, "tol": 1e-10, "max_iter": 1000}
PROTECTED = {}


def refresh_active_provenance(root: Path) -> None:
    global RUN, PROTECTED
    identity = current_input_identity(root)
    PROTECTED = current_protected_paths(root)
    RUN = identity["split_run_id"]
FLAGS = {key: False for key in ("holdout_accessed_for_metrics", "external_data_used", "rq2_data_used", "url_data_used", "threshold_tuned", "base_model_modified", "model_selection_performed", "fusion_performed")}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_hashes(root: Path, protected: dict[str, str] = PROTECTED) -> None:
    for name, expected in protected.items():
        require(sha256(root / name) == expected, f"Protected artifact hash mismatch: {name}")


def validate_oof(oof: list[dict[str, str]], development: list[dict[str, str]],
                 splits: list[dict[str, str]], folds: list[dict[str, str]],
                 reserved_ids: set[str], *, benchmark_run_id: str, expected_count: int = 201016) -> None:
    require(len(oof) == len(development) == expected_count, "Development/OOF row count mismatch")
    dev = {r["record_id"]: r for r in development}
    split = {r["record_id"]: r for r in splits}
    fold = {r["record_id"]: r for r in folds}
    ids = [r["record_id"] for r in oof]
    require(all(ids) and len(set(ids)) == len(ids) == len(dev) == len(fold), "Duplicate or missing Development/OOF ID")
    require(set(ids) == set(dev) == set(fold) == {rid for rid, r in split.items() if r["partition"] == "development"}, "Canonical Development ID coverage mismatch")
    require(not set(ids) & reserved_ids, "Reserved ID in OOF")
    require(not set(ids) & {rid for rid, r in split.items() if r["partition"] == "final_internal_holdout"}, "Holdout ID in OOF")
    require(Counter(r["label"] for r in oof) == {"legitimate": 198800, "phishing": 2216} if expected_count == 201016 else {"legitimate", "phishing"} == {r["label"] for r in oof}, "Invalid label counts or classes")
    require({r["cv_fold"] for r in oof} == set("01234"), "Missing or invalid fold")
    for r in oof:
        rid, label, k = r["record_id"], r["label"], r["cv_fold"]
        require(r["benchmark_run_id"] == benchmark_run_id and r["split_run_id"] == RUN and dev[rid]["split_run_id"] == RUN, "Run ID mismatch")
        require(label in {"legitimate", "phishing"} and all(source[rid]["label"] == label for source in (dev, split, fold)), "Invalid label or canonical label mismatch")
        require(k in set("01234") and dev[rid]["cv_fold"] == fold[rid]["cv_fold"] == k, "Canonical fold mismatch")
        require(r["score_type"] == "decision_function", "OOF score is not raw decision_function")
        score = float(r["model_score"])
        require(math.isfinite(score), "Non-finite raw score")
        require(r["prediction"] == ("phishing" if score > 0 else "legitimate") and score != 0, "Raw score orientation/prediction mismatch")


def load_inputs(root: Path = REPOSITORY_ROOT) -> tuple[list[dict[str, str]], dict]:
    from scripts.validate_text_model_selection import validate as validate_selection

    refresh_active_provenance(root)
    selection = validate_selection(root)
    benchmark_run_id = selection["benchmark_run_id"]
    base = root / f"outputs/splits/runs/{RUN}"
    oof = read_csv(root / "outputs/benchmarks/text/winner_oof_predictions.csv", OOF_COLUMNS)
    development = read_csv(root / "outputs/features/text/development_inputs.csv", ["split_run_id", "record_id", "label", "cv_fold", "subject", "body"])
    splits = read_csv(base / "split_assignments.csv", ["record_id", "label", "partition"])
    folds = read_csv(base / "cv_fold_assignments.csv", ["record_id", "label", "cv_fold"])
    winner = json.loads((root / "outputs/benchmarks/text/winner.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "outputs/models/text/classifier/manifest.json").read_text(encoding="utf-8"))
    require(winner["benchmark_run_id"] == benchmark_run_id and winner["split_run_id"] == RUN and selection["winner"] == winner["config_id"], "Winner provenance mismatch")
    require(manifest["benchmark_run_id"] == benchmark_run_id and manifest["split_run_id"] == RUN and manifest["winner_config_id"] == winner["config_id"] and manifest["model_classes"] == ["legitimate", "phishing"] and manifest["decision_function_positive_class"] == "phishing", "Base model provenance/orientation mismatch")
    require(manifest.get("selection_provenance", {}).get("semantic_sha256") == selection["semantic_sha256"], "Base model selection provenance mismatch")
    for name, spec in manifest["outputs"].items():
        require(sha256(root / spec["path"]) == spec["sha256"], f"Base model output hash mismatch: {name}")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    require(type(model) is LinearSVC and list(model.classes_) == ["legitimate", "phishing"], "Frozen model positive-score orientation mismatch")
    reserved_ids = set()
    for path in (root / "outputs/dataset_roles").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "record_id" in reader.fieldnames:
                reserved_ids.update(row["record_id"] for row in reader)
    validate_oof(oof, development, splits, folds, reserved_ids, benchmark_run_id=benchmark_run_id,
                 expected_count=current_contract(root).expected("text_development_count", 201016))
    return sorted(oof, key=lambda r: r["record_id"]), selection


def fit_sigmoid(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    require(scores.ndim == 1 and labels.ndim == 1 and len(scores) == len(labels) and set(np.unique(labels)) == {0, 1} and np.isfinite(scores).all(), "Invalid sigmoid fit input")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(**PARAMETERS).fit(scores.reshape(-1, 1), labels)
    require(list(model.classes_) == [0, 1] and model.coef_.shape == (1, 1) and float(model.coef_[0, 0]) > 0, "Non-positive sigmoid slope or wrong class orientation")
    return model


def probabilities(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    p = model.predict_proba(scores.reshape(-1, 1))[:, 1]
    require(np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), "Invalid probability")
    return p


def run(root: Path = REPOSITORY_ROOT) -> dict:
    start = time.perf_counter()
    rows, selection = load_inputs(root)
    benchmark_run_id = selection["benchmark_run_id"]
    base_manifest_path = root / "outputs/models/text/classifier/manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_model_provenance = {
        "manifest_path": "outputs/models/text/classifier/manifest.json",
        "manifest_sha256": sha256(base_manifest_path),
        "output_sha256": {name: spec["sha256"] for name, spec in sorted(base_manifest["outputs"].items())},
    }
    scores = np.array([float(r["model_score"]) for r in rows], dtype=np.float64)
    labels = np.array([int(r["label"] == "phishing") for r in rows], dtype=np.int8)
    fold_ids = np.array([int(r["cv_fold"]) for r in rows], dtype=np.int8)
    config = {"method": "sigmoid_logistic_regression", "parameters": PARAMETERS, "folds": [0, 1, 2, 3, 4]}
    run_payload = {"inputs": PROTECTED, "selection_semantic_sha256": selection["semantic_sha256"], "calibration": config}
    calibration_run_id, calibration_semantic_sha = content_run_id("text-calibration", run_payload)
    crossfit = np.empty(len(rows), dtype=np.float64)
    fold_models = []
    for k in range(5):
        target = fold_ids == k
        model = fit_sigmoid(scores[~target], labels[~target])
        crossfit[target] = probabilities(model, scores[target])
        fold_models.append({"target_fold": k, "training_folds": [j for j in range(5) if j != k], "training_rows": int((~target).sum()), "target_rows": int(target.sum()), "slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])})
    final = fit_sigmoid(scores, labels)
    metrics = {"development_prevalence": float(labels.mean()), "crossfit_brier_score": float(brier_score_loss(labels, crossfit)), "crossfit_log_loss": float(log_loss(labels, crossfit, labels=[0, 1]))}
    out = root / "outputs/models/text/calibration"
    require(not out.exists() or not any(out.iterdir()), "Step-20 output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "text_calibrator.joblib")
    with (out / "development_crossfit_calibrated_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OUTPUT_COLUMNS)
        for r, p in zip(rows, crossfit, strict=True):
            writer.writerow([calibration_run_id, RUN, benchmark_run_id, r["record_id"], r["label"], r["cv_fold"], r["model_score"], format(float(p), ".17g")])
    with (out / "calibration_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["calibration_run_id", "scope", "metric", "value"])
        for name, value in metrics.items():
            writer.writerow([calibration_run_id, "development_crossfit", name, format(value, ".17g")])
    outputs = {name: {"path": f"outputs/models/text/calibration/{name}", "sha256": sha256(out / name)} for name in ("text_calibrator.joblib", "development_crossfit_calibrated_scores.csv", "calibration_diagnostics.csv")}
    manifest = {"artifact_type": "text_probability_calibration", "step": 20, "calibration_run_id": calibration_run_id, "split_run_id": RUN, "benchmark_run_id": benchmark_run_id, "winner_config_id": base_manifest["winner_config_id"], "selection_provenance": {"manifest_path": "outputs/benchmarks/text/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]}, "base_model_provenance": base_model_provenance,
        "protected_inputs": {path: {"path": path, "sha256": digest} for path, digest in PROTECTED.items()}, "base_model_classes": ["legitimate", "phishing"], "raw_score_positive_class": "phishing",
        "calibration": config, "fold_calibrators": fold_models, "final_calibrator": {"fit_rows": len(rows), "slope": float(final.coef_[0, 0]), "intercept": float(final.intercept_[0]), "classes": list(map(int, final.classes_))},
        "development_output_semantics": "Cross-fitted from Step-18 OOF raw scores: each row transformed by a sigmoid fitted on the other four Development folds only; sole eligible text probability input for later Development fusion selection.",
        "final_calibrator_semantics": "Fitted on all Development OOF scores/labels for future unseen-data inference; not used for Development crossfit output.",
        "base_prediction_rule": "Frozen Step-19 LinearSVC predict/decision_function sign; no probability decision threshold", "metrics": metrics, "outputs": outputs, "flags": FLAGS}
    manifest["contract_version"] = "artifact-provenance-v1:text-calibration-v1"
    manifest["semantic_sha256"] = calibration_semantic_sha
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_outputs = {**outputs, "manifest.json": {"path": "outputs/models/text/calibration/manifest.json", "sha256": sha256(out / "manifest.json")}}
    report = {"status": "runner_checks_passed", "calibration_run_id": calibration_run_id, "split_run_id": RUN, "benchmark_run_id": benchmark_run_id, "selection_provenance": manifest["selection_provenance"], "base_model_provenance": base_model_provenance,
        "counts": {"total": len(rows), "labels": dict(Counter(r["label"] for r in rows)), "folds": dict(Counter(r["cv_fold"] for r in rows))}, "input_validation": "passed", "protected_sha256": PROTECTED,
        "calibration": config, "fold_calibrators": fold_models, "final_calibrator": manifest["final_calibrator"], "metrics": metrics,
        "diagnostic_scope": "Development cross-fitted calibration diagnostics; not independent Holdout performance", "duration_seconds": time.perf_counter() - start,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "outputs": report_outputs, "flags": FLAGS}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    check_hashes(root)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "calibration_run_id": result["calibration_run_id"], "metrics": result["metrics"]}, indent=2))
