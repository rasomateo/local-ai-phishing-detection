from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import LinearSVC

from src.utils import REPOSITORY_ROOT
from src.provenance import CONTRACT_VERSION, content_run_id
from src.execution_contract import current_contract

URL_RUN_ID = ""
FOLDS = tuple(range(5))
CALIBRATION_PARAMETERS = {"penalty": None, "solver": "lbfgs", "class_weight": None, "fit_intercept": True, "tol": 1e-10, "max_iter": 1000}
STEP21B = {}
CROSSFIT_LIMITATION = (
    "Development-only calibration diagnostics, not independent end-to-end generalization: each target-fold calibrator excludes that fold's OOF rows, "
    "but the CV base models that produced OOF scores in the other four folds may have trained on records from the target fold. Independent evaluation "
    "of the final URL branch is deferred to the sealed URL Holdout."
)
FLAGS = {
    "full_development_fit": True, "development_only_calibration": True, "model_selection_performed": False,
    "holdout_used_for_fit": False, "holdout_vectorized": False, "holdout_predicted": False, "holdout_metrics_computed": False,
    "threshold_tuned": False, "probability_threshold_applied": False, "fusion_performed": False, "url_aggregation_performed": False,
    "external_data_used": False, "rq2_data_used": False, "step24_started": False,
}
OOF_COLUMNS = ["benchmark_run_id", "url_run_id", "canonical_url_id", "label", "cv_fold", "prediction", "model_score", "score_type"]
CROSSFIT_COLUMNS = ["artifact_run_id", "url_run_id", "benchmark_run_id", "canonical_url_id", "label", "cv_fold", "raw_oof_decision_score", "crossfit_probability_phishing"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, f"Unexpected schema: {path}")
        return list(reader)


def safe(value):
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if value is np.float32:
        return "float32"
    if isinstance(value, np.generic):
        return value.item()
    return value


def protected_hashes(root: Path) -> dict[str, str]:
    global URL_RUN_ID, STEP21B
    prior = json.loads((root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8"))
    URL_RUN_ID = prior["url_run_id"]
    STEP21B = {name: spec["sha256"] for name, spec in prior["outputs"].items()}
    STEP21B["report.json"] = sha256(root / "outputs/datasets/url/canonical/report.json")
    protected = {f"outputs/datasets/url/canonical/{name}": digest for name, digest in STEP21B.items()}
    protected.update(prior["protected_sha256"])
    return protected


def check_hashes(root: Path, protected: dict[str, str]) -> None:
    for relative, expected in protected.items():
        require(sha256(root / relative) == expected, f"Protected artifact hash mismatch: {relative}")


def validate_winner(root: Path) -> tuple[dict, dict]:
    from scripts.validate_url_model_selection import validate as validate_selection

    selection = validate_selection(root)
    winner = json.loads((root / "outputs/benchmarks/url/winner.json").read_text(encoding="utf-8"))
    require(winner["url_run_id"] == URL_RUN_ID and winner["benchmark_run_id"] == selection["benchmark_run_id"], "Winner run provenance mismatch")
    require(winner["representation"] == "normalized_url_char" and winner["model_family"] == "LinearSVC",
            "Current winner is incompatible with the persisted single-vectorizer calibrated URL bundle")
    require(selection["winner"] == winner["config_id"], "Validated selection/winner mismatch")
    return winner, selection


def load_inputs(root: Path = REPOSITORY_ROOT) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str], dict, dict]:
    contract = current_contract(root)
    protected = protected_hashes(root)
    check_hashes(root, protected)
    winner, selection = validate_winner(root)
    base = root / "outputs/datasets/url/canonical"
    splits = read_rows(base / "split_assignments.csv", ["url_run_id", "canonical_url_id", "label", "leakage_group", "split"])
    folds = read_rows(base / "cv_assignments.csv", ["url_run_id", "canonical_url_id", "label", "leakage_group", "cv_fold"])
    split_by_id = {row["canonical_url_id"]: row for row in splits}
    fold_by_id = {row["canonical_url_id"]: row for row in folds}
    require(len(splits) == len(split_by_id) == contract.expected("url_canonical_count", 4883) and all(row["url_run_id"] == URL_RUN_ID for row in splits), "Canonical split mismatch")
    development_ids = {rid for rid, row in split_by_id.items() if row["split"] == "development"}
    holdout_ids = {rid for rid, row in split_by_id.items() if row["split"] == "holdout"}
    expected_dev = (sum(row["split"] == "development" for row in splits) if contract.is_fixture
                    else contract.expected("url_development_count", 4001))
    expected_holdout = (sum(row["split"] == "holdout" for row in splits) if contract.is_fixture
                        else contract.expected("url_holdout_count", 882))
    require(len(development_ids) == expected_dev and len(holdout_ids) == expected_holdout and not development_ids & holdout_ids, "Development/Holdout isolation mismatch")
    require(len(folds) == len(fold_by_id) == expected_dev and set(fold_by_id) == development_ids, "CV coverage mismatch")
    development = []
    dataset_fields = ["url_run_id", "canonical_url_id", "normalized_url", "representative_raw_url", "label", "leakage_group", "source", "source_role", "label_provenance", "source_record_count"]
    seen = set()
    with (base / "url_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == dataset_fields, "Canonical URL dataset schema mismatch")
        for row in reader:
            rid = row["canonical_url_id"]
            require(rid and rid not in seen, "Duplicate canonical URL ID")
            seen.add(rid)
            if rid not in development_ids:
                continue
            assignment, fold = split_by_id[rid], fold_by_id[rid]
            require(row["url_run_id"] == URL_RUN_ID and row["source"] == "phishload" and row["source_role"] == "url_only", "Invalid Development source")
            require((row["label"], row["leakage_group"]) == (assignment["label"], assignment["leakage_group"]) == (fold["label"], fold["leakage_group"]), "Development assignment mismatch")
            development.append({"canonical_url_id": rid, "normalized_url": row["normalized_url"], "label": row["label"], "cv_fold": fold["cv_fold"]})
    require(seen == set(split_by_id) and len(development) == expected_dev and len({row["normalized_url"] for row in development}) == expected_dev, "Dataset coverage mismatch")
    if not contract.is_fixture:
        require(Counter(row["label"] for row in development) == Counter(contract.expected("url_development_labels", {"legitimate": 948, "phishing": 3053})), "Development label counts mismatch")
    development.sort(key=lambda row: row["canonical_url_id"])
    oof = read_rows(root / "outputs/benchmarks/url/winner_oof_predictions.csv", OOF_COLUMNS)
    validate_oof(oof, development, holdout_ids, benchmark_run_id=winner["benchmark_run_id"], expected_count=expected_dev)
    oof.sort(key=lambda row: row["canonical_url_id"])
    return development, oof, protected, winner, selection


def validate_oof(oof: list[dict[str, str]], development: list[dict[str, str]], holdout_ids: set[str], *, benchmark_run_id: str, expected_count: int = 4001) -> None:
    dev = {row["canonical_url_id"]: row for row in development}
    ids = [row["canonical_url_id"] for row in oof]
    require(len(oof) == len(ids) == len(set(ids)) == len(dev) == expected_count and set(ids) == set(dev), "OOF Development coverage mismatch")
    require(not set(ids) & holdout_ids, "Holdout ID in OOF")
    require({row["cv_fold"] for row in oof} == set("01234"), "OOF fold mismatch")
    for row in oof:
        canonical = dev[row["canonical_url_id"]]
        require(row["benchmark_run_id"] == benchmark_run_id and row["url_run_id"] == URL_RUN_ID, "OOF run mismatch")
        require((row["label"], row["cv_fold"]) == (canonical["label"], canonical["cv_fold"]), "OOF assignment mismatch")
        require(row["score_type"] == "decision_function", "OOF score is not a raw decision_function")
        score = float(row["model_score"])
        require(math.isfinite(score) and score != 0, "Non-finite or zero OOF score")
        require(row["prediction"] == ("phishing" if score > 0 else "legitimate"), "OOF score direction mismatch")


def fit_sigmoid(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    require(scores.ndim == labels.ndim == 1 and len(scores) == len(labels) and set(np.unique(labels)) == {0, 1} and np.isfinite(scores).all(), "Invalid sigmoid input")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibrator = LogisticRegression(**CALIBRATION_PARAMETERS).fit(scores.reshape(-1, 1), labels)
    require(list(calibrator.classes_) == [0, 1] and calibrator.coef_.shape == (1, 1) and float(calibrator.coef_[0, 0]) > 0, "Calibrator class 1 must be P(phishing) with positive slope")
    return calibrator


def probability_phishing(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    require(list(calibrator.classes_) == [0, 1], "Probability column 1 is not phishing=1")
    probabilities = np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)
    require(np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all(), "Invalid P(phishing)")
    return probabilities


def run(root: Path = REPOSITORY_ROOT) -> dict:
    started = time.perf_counter()
    development, oof, protected, winner, selection = load_inputs(root)
    benchmark_run_id = winner["benchmark_run_id"]
    urls = np.asarray([row["normalized_url"] for row in development], dtype=object)
    labels = np.asarray([row["label"] for row in development], dtype=object)
    scores = np.asarray([float(row["model_score"]) for row in oof], dtype=np.float64)
    binary_labels = np.asarray([int(row["label"] == "phishing") for row in oof], dtype=np.int8)
    fold_ids = np.asarray([int(row["cv_fold"]) for row in oof], dtype=np.int8)
    calibration = {"method": "sigmoid_logistic_regression", "parameters": CALIBRATION_PARAMETERS, "effective_parameters": safe(LogisticRegression(**CALIBRATION_PARAMETERS).get_params(deep=False)), "folds": list(FOLDS), "positive_target": {"legitimate": 0, "phishing": 1}, "probability_column_class": 1}
    semantic_payload = {
        "contract_version": f"{CONTRACT_VERSION}:url-model-v1",
        "url_dataset_fingerprint": json.loads(
            (root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8")
        )["semantic_sha256"],
        "benchmark_run": benchmark_run_id,
        "selection_fingerprint": selection["semantic_sha256"],
        "canonical_artifact_sha256": {
            path: digest for path, digest in protected.items()
            if path.startswith("outputs/datasets/url/canonical/") and not path.endswith("/report.json")
        },
        "winner": winner["config_id"],
        "representation": winner["representation_specification"],
        "model_parameters": winner["model_parameters"],
        "calibration": calibration,
    }
    artifact_run_id, semantic_sha = content_run_id("url-model", semantic_payload)

    calibration_started = time.perf_counter()
    crossfit = np.empty(len(oof), dtype=np.float64)
    fold_calibrators = []
    for target_fold in FOLDS:
        target = fold_ids == target_fold
        calibrator = fit_sigmoid(scores[~target], binary_labels[~target])
        crossfit[target] = probability_phishing(calibrator, scores[target])
        fold_calibrators.append({"target_fold": target_fold, "training_folds": [fold for fold in FOLDS if fold != target_fold], "training_rows": int((~target).sum()), "target_rows": int(target.sum()), "slope": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0]), "classes": [0, 1]})
    final_calibrator = fit_sigmoid(scores, binary_labels)
    final_oof_probabilities = probability_phishing(final_calibrator, scores)
    calibration_seconds = time.perf_counter() - calibration_started

    fit_started = time.perf_counter()
    tfidf_parameters = winner["representation_specification"]
    model_parameters = winner["model_parameters"]
    vectorizer = TfidfVectorizer(**{**tfidf_parameters, "ngram_range": tuple(tfidf_parameters["ngram_range"]), "dtype": np.float32})
    matrix = vectorizer.fit_transform(urls)
    model = LinearSVC(**model_parameters).fit(matrix, labels)
    fit_seconds = time.perf_counter() - fit_started
    require(matrix.shape == (len(development), len(vectorizer.vocabulary_)) and matrix.shape[1] <= 50000, "Final feature dimension mismatch")
    require(list(model.classes_) == ["legitimate", "phishing"] and model.coef_.shape == (1, matrix.shape[1]) and model.intercept_.shape == (1,), "Final model class or shape mismatch")
    require(np.isfinite(vectorizer.idf_).all() and np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), "Non-finite final model value")

    out = root / "outputs/models/url"
    require(not out.exists() or not any(out.iterdir()), "Step-23 output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out / "url_vectorizer.joblib")
    joblib.dump(model, out / "url_model.joblib")
    joblib.dump(final_calibrator, out / "url_calibrator.joblib")
    with (out / "development_crossfit_calibrated_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CROSSFIT_COLUMNS)
        for row, probability in zip(oof, crossfit, strict=True):
            writer.writerow([artifact_run_id, URL_RUN_ID, benchmark_run_id, row["canonical_url_id"], row["label"], row["cv_fold"], row["model_score"], format(float(probability), ".17g")])
    crossfit_metrics = {"brier_score": float(brier_score_loss(binary_labels, crossfit)), "log_loss": float(log_loss(binary_labels, crossfit, labels=[0, 1]))}
    diagnostics = [
        ("development_oof_raw_scores", "score_min", float(scores.min())), ("development_oof_raw_scores", "score_max", float(scores.max())), ("development_oof_raw_scores", "nonfinite_count", int((~np.isfinite(scores)).sum())),
        ("development_crossfit_calibration", "probability_min", float(crossfit.min())), ("development_crossfit_calibration", "probability_max", float(crossfit.max())), ("development_crossfit_calibration", "nonfinite_count", int((~np.isfinite(crossfit)).sum())),
        ("development_crossfit_calibration", "out_of_bounds_count", int(((crossfit < 0) | (crossfit > 1)).sum())), ("development_crossfit_calibration", "brier_score", crossfit_metrics["brier_score"]), ("development_crossfit_calibration", "log_loss", crossfit_metrics["log_loss"]),
        ("final_all_oof_calibrator_fit_integrity", "probability_min", float(final_oof_probabilities.min())), ("final_all_oof_calibrator_fit_integrity", "probability_max", float(final_oof_probabilities.max())), ("final_all_oof_calibrator_fit_integrity", "nonfinite_count", int((~np.isfinite(final_oof_probabilities)).sum())),
        ("final_all_oof_calibrator_fit_integrity", "slope", float(final_calibrator.coef_[0, 0])), ("final_all_oof_calibrator_fit_integrity", "intercept", float(final_calibrator.intercept_[0])), ("final_all_oof_calibrator_fit_integrity", "n_iter", int(final_calibrator.n_iter_[0])),
    ]
    with (out / "calibration_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["artifact_run_id", "scope", "metric", "value", "interpretation"])
        for scope, metric, value in diagnostics:
            interpretation = CROSSFIT_LIMITATION if scope == "development_crossfit_calibration" else ("Raw decision score integrity; not a probability metric." if scope == "development_oof_raw_scores" else "All-OOF calibrator fit integrity only; not independent performance.")
            writer.writerow([artifact_run_id, scope, metric, format(float(value), ".17g"), interpretation])

    output_names = ("url_vectorizer.joblib", "url_model.joblib", "url_calibrator.joblib", "development_crossfit_calibrated_scores.csv", "calibration_diagnostics.csv")
    outputs = {name: {"path": f"outputs/models/url/{name}", "sha256": sha256(out / name)} for name in output_names}
    final_spec = {"fit_rows": 4001, "slope": float(final_calibrator.coef_[0, 0]), "intercept": float(final_calibrator.intercept_[0]), "classes": [0, 1], "n_iter": int(final_calibrator.n_iter_[0])}
    counts = {"development": {"total": 4001, "labels": dict(Counter(labels)), "folds": dict(Counter(str(value) for value in fold_ids))}, "holdout_metadata_only": {"total": 882, "labels": {"legitimate": 228, "phishing": 654}}}
    manifest = {
        "artifact_type": "url_model_probability_bundle", "artifact_version": 1, "step": 23, "created_at": datetime.now(timezone.utc).isoformat(), "artifact_run_id": artifact_run_id,
        "contract_version": f"{CONTRACT_VERSION}:url-model-v1", "semantic_sha256": semantic_sha,
        "url_run_id": URL_RUN_ID, "benchmark_run_id": benchmark_run_id, "winner_config_id": winner["config_id"],
        "selection_provenance": {"manifest_path": "outputs/benchmarks/url/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]},
        "winner": {"path": "outputs/benchmarks/url/winner.json", "sha256": sha256(root / "outputs/benchmarks/url/winner.json")},
        "protected_inputs": {path: {"path": path, "sha256": digest} for path, digest in protected.items()}, "counts": counts,
        "representation": winner["representation"], "input_semantics": "Already canonical-normalized URL string; no normalization is performed by the bundle scorer.", "tfidf_parameters": tfidf_parameters, "tfidf_effective_parameters": safe(vectorizer.get_params(deep=False)), "model_parameters": model_parameters,
        "feature_dimension": matrix.shape[1], "vocabulary_size": len(vectorizer.vocabulary_), "model_classes": ["legitimate", "phishing"], "decision_function_positive_class": "phishing", "coefficient_shape": list(model.coef_.shape), "intercept_shape": list(model.intercept_.shape), "intercept": safe(model.intercept_), "n_iter": safe(model.n_iter_),
        "calibration": calibration, "fold_calibrators": fold_calibrators, "final_calibrator": final_spec, "probability_semantics": "predict_proba column for calibrator class 1, where class 1 means phishing: P(phishing). No threshold decision is applied.",
        "development_crossfit_semantics": "Each row is calibrated by a sigmoid fitted on OOF scores from the other four Development folds only.", "crossfit_diagnostic_limitation": {"scope": "development_only_calibration_diagnostics", "independent_end_to_end_generalization": False, "statement": CROSSFIT_LIMITATION},
        "final_calibrator_semantics": "Fitted on all 4,001 Development OOF scores for future unseen normalized URLs; not used to create the Development crossfit output.", "crossfit_metrics": crossfit_metrics,
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "outputs": outputs, "flags": FLAGS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_outputs = {**outputs, "manifest.json": {"path": "outputs/models/url/manifest.json", "sha256": sha256(out / "manifest.json")}}
    report = {
        "status": "runner_checks_passed", "artifact_run_id": artifact_run_id, "url_run_id": URL_RUN_ID, "benchmark_run_id": benchmark_run_id, "winner_config_id": winner["config_id"], "selection_provenance": manifest["selection_provenance"],
        "contract_version": manifest["contract_version"], "semantic_sha256": semantic_sha,
        "protected_sha256": protected, "counts": counts,
        "tfidf_parameters": tfidf_parameters, "tfidf_effective_parameters": manifest["tfidf_effective_parameters"], "model_parameters": model_parameters, "feature_dimension": matrix.shape[1], "model_classes": manifest["model_classes"], "decision_function_positive_class": "phishing",
        "calibration": calibration, "fold_calibrators": fold_calibrators, "final_calibrator": final_spec, "probability_semantics": manifest["probability_semantics"], "crossfit_metrics": crossfit_metrics,
        "diagnostic_scope": "Development-only calibration diagnostics; not URL Holdout performance.", "crossfit_diagnostic_limitation": manifest["crossfit_diagnostic_limitation"],
        "fit_seconds": fit_seconds, "calibration_seconds": calibration_seconds, "duration_seconds": time.perf_counter() - started, "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "outputs": report_outputs, "flags": FLAGS,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    check_hashes(root, protected)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "artifact_run_id": result["artifact_run_id"], "feature_dimension": result["feature_dimension"], "crossfit_metrics": result["crossfit_metrics"]}, indent=2))
