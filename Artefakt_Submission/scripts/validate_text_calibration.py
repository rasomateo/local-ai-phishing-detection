from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import LinearSVC

from src.utils import REPOSITORY_ROOT
from src.text_model_selection import current_input_identity
from src.provenance import content_run_id
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
RUN = ""
PARAMETERS = {"penalty": None, "solver": "lbfgs", "class_weight": None, "fit_intercept": True, "tol": 1e-10, "max_iter": 1000}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == fields, f"Unexpected schema: {path}")
        return list(reader)


def fit(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(**PARAMETERS).fit(scores.reshape(-1, 1), labels)
    check(list(model.classes_) == [0, 1] and float(model.coef_[0, 0]) > 0, "Independent refit orientation")
    return model


def close(actual: float, expected: float, name: str, *, atol: float = 1e-8) -> None:
    check(math.isfinite(actual) and math.isclose(actual, expected, abs_tol=atol, rel_tol=1e-8), f"{name} mismatch")


def validate(root: Path = REPOSITORY_ROOT) -> dict:
    global RUN
    RUN = current_input_identity(root)["split_run_id"]
    from scripts.validate_text_model import validate as validate_base_model
    from scripts.validate_text_model_selection import validate as validate_selection

    out = root / "outputs/models/text/calibration"
    selection = validate_selection(root)
    validate_base_model(root)
    benchmark = selection["benchmark_run_id"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    protected = manifest["protected_inputs"]
    reference = json.loads((root / "outputs/models/text/classifier/report.json").read_text(encoding="utf-8"))
    expected = dict(reference["protected_sha256"])
    check(protected == {name: {"path": name, "sha256": value} for name, value in expected.items()} and report["protected_sha256"] == expected, "Protected inventory mismatch")
    for name, value in expected.items():
        check(digest(root / name) == value, f"Protected hash mismatch: {name}")
    selection_provenance = {"manifest_path": "outputs/benchmarks/text/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]}
    check(manifest["selection_provenance"] == report["selection_provenance"] == selection_provenance, "Selection provenance mismatch")
    base_manifest_path = root / "outputs/models/text/classifier/manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_provenance = {"manifest_path": "outputs/models/text/classifier/manifest.json", "manifest_sha256": digest(base_manifest_path), "output_sha256": {name: spec["sha256"] for name, spec in sorted(base_manifest["outputs"].items())}}
    check(manifest["base_model_provenance"] == report["base_model_provenance"] == base_provenance, "Base model provenance mismatch")
    check(manifest["split_run_id"] == report["split_run_id"] == RUN and manifest["benchmark_run_id"] == report["benchmark_run_id"] == benchmark, "Run provenance mismatch")
    check(manifest["calibration"] == report["calibration"] == {"method": "sigmoid_logistic_regression", "parameters": PARAMETERS, "folds": [0, 1, 2, 3, 4]}, "Calibration configuration mismatch")
    run_id, semantic_sha = content_run_id("text-calibration", {"inputs": expected, "selection_semantic_sha256": selection["semantic_sha256"], "calibration": manifest["calibration"]})
    check(manifest["calibration_run_id"] == report["calibration_run_id"] == run_id, "Calibration run ID mismatch")
    check(manifest["semantic_sha256"] == semantic_sha, "Calibration semantic fingerprint mismatch")
    check(manifest["base_model_classes"] == ["legitimate", "phishing"] and manifest["raw_score_positive_class"] == "phishing", "Base orientation mismatch")
    base_model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    check(type(base_model) is LinearSVC and list(base_model.classes_) == ["legitimate", "phishing"], "Persisted base orientation mismatch")
    check(all(value is False for value in manifest["flags"].values()) and manifest["flags"] == report["flags"], "Method flags mismatch")
    for name in ("text_calibrator.joblib", "development_crossfit_calibrated_scores.csv", "calibration_diagnostics.csv"):
        spec = {"path": f"outputs/models/text/calibration/{name}", "sha256": digest(out / name)}
        check(manifest["outputs"][name] == report["outputs"][name] == spec, f"Output hash mismatch: {name}")
    check(report["outputs"]["manifest.json"] == {"path": "outputs/models/text/calibration/manifest.json", "sha256": digest(out / "manifest.json")}, "Manifest hash mismatch")
    oof = rows(root / "outputs/benchmarks/text/winner_oof_predictions.csv", ["benchmark_run_id", "split_run_id", "record_id", "label", "cv_fold", "prediction", "model_score", "score_type"])
    dev = rows(root / "outputs/features/text/development_inputs.csv", ["split_run_id", "record_id", "label", "cv_fold", "subject", "body"])
    split = rows(root / f"outputs/splits/runs/{RUN}/split_assignments.csv", ["record_id", "label", "partition"])
    folds = rows(root / f"outputs/splits/runs/{RUN}/cv_fold_assignments.csv", ["record_id", "label", "cv_fold"])
    by_dev = {r["record_id"]: r for r in dev}
    by_split = {r["record_id"]: r for r in split}
    by_fold = {r["record_id"]: r for r in folds}
    ids = [r["record_id"] for r in oof]
    expected_total = current_contract(root).expected("text_development_count", 201016)
    check(len(oof) == len(dev) == expected_total and len(set(ids)) == len(by_dev) == len(by_fold) == expected_total, "OOF/Development count or duplicate IDs")
    check(set(ids) == set(by_dev) == set(by_fold) == {rid for rid, r in by_split.items() if r["partition"] == "development"}, "Canonical ID coverage")
    reserved = set()
    for path in (root / "outputs/dataset_roles").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "record_id" in reader.fieldnames:
                reserved.update(r["record_id"] for r in reader)
    check(not set(ids) & reserved and not set(ids) & {rid for rid, r in by_split.items() if r["partition"] == "final_internal_holdout"}, "Reserved or Holdout ID")
    check(Counter(r["label"] for r in oof) == Counter(current_contract(root).expected("text_development_labels", {"legitimate": 198800, "phishing": 2216})) and {r["cv_fold"] for r in oof} == set("01234"), "Labels/folds mismatch")
    oof = sorted(oof, key=lambda r: r["record_id"])
    for r in oof:
        rid, label, fold_id = r["record_id"], r["label"], r["cv_fold"]
        check(r["benchmark_run_id"] == benchmark and r["split_run_id"] == by_dev[rid]["split_run_id"] == RUN, "OOF run mismatch")
        check(label == by_dev[rid]["label"] == by_split[rid]["label"] == by_fold[rid]["label"] and fold_id == by_dev[rid]["cv_fold"] == by_fold[rid]["cv_fold"], "Canonical label/fold mismatch")
        score = float(r["model_score"])
        check(r["score_type"] == "decision_function" and math.isfinite(score) and score != 0 and r["prediction"] == ("phishing" if score > 0 else "legitimate"), "Raw score semantics mismatch")
    observed = rows(out / "development_crossfit_calibrated_scores.csv", ["calibration_run_id", "split_run_id", "benchmark_run_id", "record_id", "label", "cv_fold", "raw_model_score", "calibrated_probability_phishing"])
    check(len(observed) == len(oof) and [r["record_id"] for r in observed] == [r["record_id"] for r in oof], "Crossfit output ID/order mismatch")
    scores = np.array([float(r["model_score"]) for r in oof], dtype=np.float64)
    labels = np.array([int(r["label"] == "phishing") for r in oof], dtype=np.int8)
    fold_array = np.array([int(r["cv_fold"]) for r in oof], dtype=np.int8)
    recreated = np.empty(len(oof), dtype=np.float64)
    check(len(manifest["fold_calibrators"]) == len(report["fold_calibrators"]) == 5, "Fold calibrator count")
    for k in range(5):
        target = fold_array == k
        model = fit(scores[~target], labels[~target])
        recreated[target] = model.predict_proba(scores[target].reshape(-1, 1))[:, 1]
        info = manifest["fold_calibrators"][k]
        check(info == report["fold_calibrators"][k] and info["target_fold"] == k and info["training_folds"] == [j for j in range(5) if j != k] and info["training_rows"] == int((~target).sum()) and info["target_rows"] == int(target.sum()), "Fold exclusion/provenance mismatch")
        close(info["slope"], float(model.coef_[0, 0]), f"Fold {k} slope")
        close(info["intercept"], float(model.intercept_[0]), f"Fold {k} intercept")
    for source, row, p in zip(oof, observed, recreated, strict=True):
        check(row["calibration_run_id"] == run_id and row["split_run_id"] == RUN and row["benchmark_run_id"] == benchmark and row["label"] == source["label"] and row["cv_fold"] == source["cv_fold"], "Crossfit row provenance mismatch")
        close(float(row["raw_model_score"]), float(source["model_score"]), "Raw score", atol=1e-12)
        close(float(row["calibrated_probability_phishing"]), float(p), "Crossfit probability", atol=1e-12)
    check(np.isfinite(recreated).all() and ((0 <= recreated) & (recreated <= 1)).all(), "Probability bounds")
    metrics = {"development_prevalence": float(labels.mean()), "crossfit_brier_score": float(brier_score_loss(labels, recreated)), "crossfit_log_loss": float(log_loss(labels, recreated, labels=[0, 1]))}
    diagnostics = rows(out / "calibration_diagnostics.csv", ["calibration_run_id", "scope", "metric", "value"])
    check(len(diagnostics) == 3 and {r["metric"] for r in diagnostics} == set(metrics), "Diagnostics shape")
    for row in diagnostics:
        check(row["calibration_run_id"] == run_id and row["scope"] == "development_crossfit", "Diagnostics provenance")
        close(float(row["value"]), metrics[row["metric"]], row["metric"], atol=1e-12)
    for name, value in metrics.items():
        close(manifest["metrics"][name], value, name, atol=1e-12)
        close(report["metrics"][name], value, name, atol=1e-12)
    final = joblib.load(out / "text_calibrator.joblib")
    check(type(final) is LogisticRegression and final.get_params(deep=False) == LogisticRegression(**PARAMETERS).get_params(deep=False), "Final calibrator type/parameters")
    independent = fit(scores, labels)
    check(list(final.classes_) == [0, 1] and final.coef_.shape == (1, 1) and final.intercept_.shape == (1,), "Final calibrator shape/classes")
    close(float(final.coef_[0, 0]), float(independent.coef_[0, 0]), "Final slope")
    close(float(final.intercept_[0]), float(independent.intercept_[0]), "Final intercept")
    check(manifest["final_calibrator"] == report["final_calibrator"] and manifest["final_calibrator"]["fit_rows"] == len(oof), "Final provenance")
    close(manifest["final_calibrator"]["slope"], float(final.coef_[0, 0]), "Manifest final slope")
    close(manifest["final_calibrator"]["intercept"], float(final.intercept_[0]), "Manifest final intercept")
    check(all(np.all(np.diff(recreated[fold_array == k][np.argsort(scores[fold_array == k])]) >= 0) for k in range(5)), "Crossfit monotonicity")
    check(np.max(np.abs(recreated - final.predict_proba(scores.reshape(-1, 1))[:, 1])) > 1e-8, "Crossfit probabilities appear to come from final calibrator")
    return {"status": "validated", "calibration_run_id": run_id, "metrics": metrics,
            "selection_semantic_sha256": selection["semantic_sha256"],
            "report_sha256": digest(out / "report.json")}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
