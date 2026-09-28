from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.svm import LinearSVC
from src.execution_contract import current_contract

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PARAMETERS = {"penalty": None, "solver": "lbfgs", "class_weight": None, "fit_intercept": True, "tol": 1e-10, "max_iter": 1000}
LIMITATION_TERMS = ("Development-only", "not independent end-to-end generalization", "target-fold", "sealed URL Holdout")
OOF_FIELDS = ["benchmark_run_id", "url_run_id", "canonical_url_id", "label", "cv_fold", "prediction", "model_score", "score_type"]
CROSSFIT_FIELDS = ["artifact_run_id", "url_run_id", "benchmark_run_id", "canonical_url_id", "label", "cv_fold", "raw_oof_decision_score", "crossfit_probability_phishing"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path, fields: list[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is not None:
            check(reader.fieldnames == fields, f"Unexpected schema: {path}")
        return list(reader)


def compare(left, right, name: str, *, atol: float = 1e-8, rtol: float = 1e-8) -> float:
    a, b = np.asarray(left), np.asarray(right)
    check(a.shape == b.shape, f"{name} shape mismatch")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    check(np.allclose(a, b, atol=atol, rtol=rtol), f"{name} numerical mismatch: {delta}")
    return delta


def fit_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(**CALIBRATION_PARAMETERS).fit(scores.reshape(-1, 1), labels)
    check(list(model.classes_) == [0, 1] and float(model.coef_[0, 0]) > 0, "Independent calibrator orientation mismatch")
    return model


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if value is np.float32:
        return "float32"
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate(root: Path = ROOT) -> dict:
    contract = current_contract(root)
    from scripts.validate_url_model_selection import validate as validate_selection
    from src.provenance import CONTRACT_VERSION, content_run_id

    out = root / "outputs/models/url"
    selection = validate_selection(root)
    benchmark_run_id = selection["benchmark_run_id"]
    prior = json.loads((root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8"))
    URL_RUN_ID = prior["url_run_id"]
    STEP21B = {name: spec["sha256"] for name, spec in prior["outputs"].items()}
    STEP21B["report.json"] = digest(root / "outputs/datasets/url/canonical/report.json")
    protected = {f"outputs/datasets/url/canonical/{name}": value for name, value in STEP21B.items()}
    protected.update(prior["protected_sha256"])
    for relative, expected in protected.items():
        check(digest(root / relative) == expected, f"Protected hash mismatch: {relative}")

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    winner = json.loads((root / "outputs/benchmarks/url/winner.json").read_text(encoding="utf-8"))
    check(report["status"] == "runner_checks_passed", "Runner status mismatch")
    check(manifest["step"] == 23 and manifest["artifact_type"] == "url_model_probability_bundle", "Manifest identity mismatch")
    check(manifest["artifact_run_id"] == report["artifact_run_id"], "Artifact run mismatch")
    check(manifest["url_run_id"] == report["url_run_id"] == winner["url_run_id"] == URL_RUN_ID, "URL run mismatch")
    check(manifest["benchmark_run_id"] == report["benchmark_run_id"] == winner["benchmark_run_id"] == benchmark_run_id, "Benchmark run mismatch")
    expected_selection = {"manifest_path": "outputs/benchmarks/url/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]}
    check(manifest["selection_provenance"] == report["selection_provenance"] == expected_selection, "Selection provenance mismatch")
    check(manifest["winner_config_id"] == report["winner_config_id"] == winner["config_id"] == selection["winner"], "Winner mismatch")
    check(winner["representation"] == "normalized_url_char" and winner["model_family"] == "LinearSVC",
          "Current winner is incompatible with the persisted calibrated URL bundle")
    tfidf_parameters, model_parameters = winner["representation_specification"], winner["model_parameters"]
    check(manifest["tfidf_parameters"] == report["tfidf_parameters"] == tfidf_parameters, "TF-IDF specification mismatch")
    check(manifest["model_parameters"] == report["model_parameters"] == model_parameters, "Model specification mismatch")
    check(manifest["protected_inputs"] == {path: {"path": path, "sha256": value} for path, value in protected.items()} and report["protected_sha256"] == protected, "Protected inventory mismatch")
    limitation = manifest["crossfit_diagnostic_limitation"]
    check(limitation == report["crossfit_diagnostic_limitation"] and limitation["scope"] == "development_only_calibration_diagnostics" and limitation["independent_end_to_end_generalization"] is False, "Diagnostic limitation structure mismatch")
    check(all(term.lower() in limitation["statement"].lower() for term in LIMITATION_TERMS), "Diagnostic limitation text incomplete")
    check("Development-only" in report["diagnostic_scope"] and "not URL Holdout performance" in report["diagnostic_scope"], "Diagnostic scope missing")
    true_flags = ("full_development_fit", "development_only_calibration")
    false_flags = ("model_selection_performed", "holdout_used_for_fit", "holdout_vectorized", "holdout_predicted", "holdout_metrics_computed", "threshold_tuned", "probability_threshold_applied", "fusion_performed", "url_aggregation_performed", "external_data_used", "rq2_data_used", "step24_started")
    check(all(manifest["flags"][key] is True for key in true_flags) and all(manifest["flags"][key] is False for key in false_flags) and manifest["flags"] == report["flags"], "Method flags mismatch")
    calibration = {"method": "sigmoid_logistic_regression", "parameters": CALIBRATION_PARAMETERS, "effective_parameters": json_safe(LogisticRegression(**CALIBRATION_PARAMETERS).get_params(deep=False)), "folds": [0, 1, 2, 3, 4], "positive_target": {"legitimate": 0, "phishing": 1}, "probability_column_class": 1}
    check(manifest["calibration"] == report["calibration"] == calibration, "Calibration configuration mismatch")
    semantic_payload = {
        "contract_version": f"{CONTRACT_VERSION}:url-model-v1",
        "url_dataset_fingerprint": prior["semantic_sha256"],
        "benchmark_run": benchmark_run_id,
        "selection_fingerprint": selection["semantic_sha256"],
        "canonical_artifact_sha256": {
            path: value for path, value in protected.items()
            if path.startswith("outputs/datasets/url/canonical/") and not path.endswith("/report.json")
        },
        "winner": winner["config_id"],
        "representation": winner["representation_specification"],
        "model_parameters": winner["model_parameters"],
        "calibration": calibration,
    }
    expected_run_id, expected_semantic = content_run_id("url-model", semantic_payload)
    check(manifest["contract_version"] == report["contract_version"] == f"{CONTRACT_VERSION}:url-model-v1",
          "Provenance contract mismatch")
    check(manifest["artifact_run_id"] == report["artifact_run_id"] == expected_run_id and
          manifest["semantic_sha256"] == report["semantic_sha256"] == expected_semantic,
          "Semantic URL-model identity mismatch")
    check("P(phishing)" in manifest["probability_semantics"] and "No threshold" in manifest["probability_semantics"], "Probability semantics mismatch")

    for name, spec in report["outputs"].items():
        check(spec == {"path": f"outputs/models/url/{name}", "sha256": digest(out / name)}, f"Output hash mismatch: {name}")
    check(manifest["outputs"] == {name: report["outputs"][name] for name in ("url_vectorizer.joblib", "url_model.joblib", "url_calibrator.joblib", "development_crossfit_calibrated_scores.csv", "calibration_diagnostics.csv")}, "Manifest outputs mismatch")

    split = rows(root / "outputs/datasets/url/canonical/split_assignments.csv")
    folds = rows(root / "outputs/datasets/url/canonical/cv_assignments.csv")
    split_by_id = {row["canonical_url_id"]: row for row in split}
    fold_by_id = {row["canonical_url_id"]: row for row in folds}
    dev_ids = {rid for rid, row in split_by_id.items() if row["split"] == "development"}
    holdout_ids = {rid for rid, row in split_by_id.items() if row["split"] == "holdout"}
    expected_total = contract.expected("url_canonical_count", 4883)
    expected_dev = len(dev_ids) if contract.is_fixture else contract.expected("url_development_count", 4001)
    expected_holdout = len(holdout_ids) if contract.is_fixture else contract.expected("url_holdout_count", 882)
    check(len(split) == len(split_by_id) == expected_total and len(dev_ids) == expected_dev and len(holdout_ids) == expected_holdout and not dev_ids & holdout_ids, "Partition identity mismatch")
    check(len(folds) == len(fold_by_id) == expected_dev and set(fold_by_id) == dev_ids, "Fold coverage mismatch")
    group_folds = defaultdict(set)
    for row in folds:
        check((row["label"], row["leakage_group"]) == (split_by_id[row["canonical_url_id"]]["label"], split_by_id[row["canonical_url_id"]]["leakage_group"]), "Fold assignment mismatch")
        group_folds[row["leakage_group"]].add(row["cv_fold"])
    check(all(len(value) == 1 for value in group_folds.values()), "Leakage group crosses folds")

    dataset = rows(root / "outputs/datasets/url/canonical/url_dataset.csv")
    dev_rows = sorted((row for row in dataset if row["canonical_url_id"] in dev_ids), key=lambda row: row["canonical_url_id"])
    check(len(dataset) == expected_total and len(dev_rows) == expected_dev and not {row["canonical_url_id"] for row in dev_rows} & holdout_ids, "Development materialization mismatch")
    if not contract.is_fixture:
        check(Counter(row["label"] for row in dev_rows) == Counter(contract.expected("url_development_labels", {"legitimate": 948, "phishing": 3053})), "Development counts mismatch")
    oof = sorted(rows(root / "outputs/benchmarks/url/winner_oof_predictions.csv", OOF_FIELDS), key=lambda row: row["canonical_url_id"])
    check(len(oof) == len({row["canonical_url_id"] for row in oof}) == expected_dev and {row["canonical_url_id"] for row in oof} == dev_ids and not dev_ids & holdout_ids, "OOF isolation mismatch")
    for row in oof:
        rid, score = row["canonical_url_id"], float(row["model_score"])
        check(row["label"] == split_by_id[rid]["label"] and row["cv_fold"] == fold_by_id[rid]["cv_fold"], "OOF canonical assignment mismatch")
        check(row["score_type"] == "decision_function" and math.isfinite(score) and row["prediction"] == ("phishing" if score > 0 else "legitimate"), "OOF direction mismatch")

    vectorizer = joblib.load(out / "url_vectorizer.joblib")
    model = joblib.load(out / "url_model.joblib")
    calibrator = joblib.load(out / "url_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and type(calibrator) is LogisticRegression, "Bundle type mismatch")
    actual_tfidf = vectorizer.get_params(deep=False)
    check(all(actual_tfidf[key] == (tuple(value) if key == "ngram_range" else np.float32 if key == "dtype" else value) for key, value in tfidf_parameters.items()), "Persisted TF-IDF parameters mismatch")
    check(manifest["tfidf_effective_parameters"] == report["tfidf_effective_parameters"] == json_safe(actual_tfidf), "Effective TF-IDF parameters mismatch")
    check(model.get_params(deep=False) == model_parameters and list(model.classes_) == ["legitimate", "phishing"], "Persisted model parameters or direction mismatch")
    check(calibrator.get_params(deep=False) == LogisticRegression(**CALIBRATION_PARAMETERS).get_params(deep=False) and list(calibrator.classes_) == [0, 1] and float(calibrator.coef_[0, 0]) > 0, "Persisted calibrator mismatch")
    dimensions = len(vectorizer.vocabulary_)
    check(manifest["feature_dimension"] == report["feature_dimension"] == manifest["vocabulary_size"] == dimensions and model.coef_.shape == (1, dimensions), "Feature dimension mismatch")

    stored_crossfit = rows(out / "development_crossfit_calibrated_scores.csv", CROSSFIT_FIELDS)
    check(len(stored_crossfit) == expected_dev and [row["canonical_url_id"] for row in stored_crossfit] == [row["canonical_url_id"] for row in oof], "Crossfit row ordering/coverage mismatch")
    scores = np.asarray([float(row["model_score"]) for row in oof], dtype=np.float64)
    labels = np.asarray([int(row["label"] == "phishing") for row in oof], dtype=np.int8)
    fold_ids = np.asarray([int(row["cv_fold"]) for row in oof], dtype=np.int8)
    recreated = np.empty(len(oof), dtype=np.float64)
    recreated_fold_specs = []
    for target_fold in range(5):
        target = fold_ids == target_fold
        item = fit_calibrator(scores[~target], labels[~target])
        recreated[target] = item.predict_proba(scores[target].reshape(-1, 1))[:, 1]
        recreated_fold_specs.append({"target_fold": target_fold, "training_folds": [fold for fold in range(5) if fold != target_fold], "training_rows": int((~target).sum()), "target_rows": int(target.sum()), "slope": float(item.coef_[0, 0]), "intercept": float(item.intercept_[0]), "classes": [0, 1]})
    stored_probabilities = np.asarray([float(row["crossfit_probability_phishing"]) for row in stored_crossfit])
    compare(recreated, stored_probabilities, "Crossfit probabilities", atol=1e-12)
    for actual, expected in zip(manifest["fold_calibrators"], recreated_fold_specs, strict=True):
        check({key: actual[key] for key in ("target_fold", "training_folds", "training_rows", "target_rows", "classes")} == {key: expected[key] for key in ("target_fold", "training_folds", "training_rows", "target_rows", "classes")}, "Fold calibrator boundary mismatch")
        compare([actual["slope"], actual["intercept"]], [expected["slope"], expected["intercept"]], "Fold calibrator parameters")
    final = fit_calibrator(scores, labels)
    compare(final.coef_, calibrator.coef_, "Final calibrator slope")
    compare(final.intercept_, calibrator.intercept_, "Final calibrator intercept")
    check(manifest["final_calibrator"] == report["final_calibrator"] and manifest["final_calibrator"]["classes"] == [0, 1], "Final calibrator metadata mismatch")
    compare([manifest["final_calibrator"]["slope"], manifest["final_calibrator"]["intercept"]], [float(final.coef_[0, 0]), float(final.intercept_[0])], "Final calibrator metadata")
    final_probabilities = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    check(float(np.max(np.abs(recreated - final_probabilities))) > 1e-8, "Crossfit output appears to use final calibrator")
    metrics = {"brier_score": float(brier_score_loss(labels, recreated)), "log_loss": float(log_loss(labels, recreated, labels=[0, 1]))}
    for name, value in metrics.items():
        check(math.isclose(manifest["crossfit_metrics"][name], value, rel_tol=1e-8, abs_tol=1e-12) and math.isclose(report["crossfit_metrics"][name], value, rel_tol=1e-8, abs_tol=1e-12), f"Calibration metric mismatch: {name}")
    check(np.isfinite(stored_probabilities).all() and ((0 <= stored_probabilities) & (stored_probabilities <= 1)).all(), "Invalid stored probabilities")

    diagnostics = rows(out / "calibration_diagnostics.csv", ["artifact_run_id", "scope", "metric", "value", "interpretation"])
    diagnostics_by_key = {(row["scope"], row["metric"]): row for row in diagnostics}
    expected_diagnostics = {
        ("development_oof_raw_scores", "score_min"): scores.min(), ("development_oof_raw_scores", "score_max"): scores.max(), ("development_oof_raw_scores", "nonfinite_count"): 0,
        ("development_crossfit_calibration", "probability_min"): recreated.min(), ("development_crossfit_calibration", "probability_max"): recreated.max(), ("development_crossfit_calibration", "nonfinite_count"): 0,
        ("development_crossfit_calibration", "out_of_bounds_count"): 0, ("development_crossfit_calibration", "brier_score"): metrics["brier_score"], ("development_crossfit_calibration", "log_loss"): metrics["log_loss"],
        ("final_all_oof_calibrator_fit_integrity", "probability_min"): final_probabilities.min(), ("final_all_oof_calibrator_fit_integrity", "probability_max"): final_probabilities.max(), ("final_all_oof_calibrator_fit_integrity", "nonfinite_count"): 0,
        ("final_all_oof_calibrator_fit_integrity", "slope"): float(final.coef_[0, 0]), ("final_all_oof_calibrator_fit_integrity", "intercept"): float(final.intercept_[0]), ("final_all_oof_calibrator_fit_integrity", "n_iter"): int(final.n_iter_[0]),
    }
    check(set(diagnostics_by_key) == set(expected_diagnostics), "Diagnostic inventory mismatch")
    for key, expected in expected_diagnostics.items():
        check(math.isclose(float(diagnostics_by_key[key]["value"]), float(expected), rel_tol=1e-8, abs_tol=1e-12), f"Diagnostic mismatch: {key}")
    check(all("not independent end-to-end generalization" in row["interpretation"] for row in diagnostics if row["scope"] == "development_crossfit_calibration"), "Crossfit diagnostic warning missing")

    urls = np.asarray([row["normalized_url"] for row in dev_rows], dtype=object)
    dev_labels = np.asarray([row["label"] for row in dev_rows], dtype=object)
    refit_vectorizer = TfidfVectorizer(**{**tfidf_parameters, "ngram_range": tuple(tfidf_parameters["ngram_range"]), "dtype": np.float32})
    refit_matrix = refit_vectorizer.fit_transform(urls)
    check(refit_vectorizer.vocabulary_ == vectorizer.vocabulary_, "Independent vocabulary mismatch")
    idf_delta = compare(refit_vectorizer.idf_, vectorizer.idf_, "IDF", atol=1e-5, rtol=1e-5)
    refit_model = LinearSVC(**model_parameters).fit(refit_matrix, dev_labels)
    coefficient_delta = compare(refit_model.coef_, model.coef_, "Model coefficients", atol=1e-5, rtol=1e-5)
    intercept_delta = compare(refit_model.intercept_, model.intercept_, "Model intercept", atol=1e-5, rtol=1e-5)
    indices = sorted(range(len(dev_rows)), key=lambda index: (hashlib.sha256(dev_rows[index]["canonical_url_id"].encode()).hexdigest(), dev_rows[index]["canonical_url_id"]))[:256]
    audit_urls = urls[indices]
    loaded_scores = model.decision_function(vectorizer.transform(audit_urls))
    refit_scores = refit_model.decision_function(refit_vectorizer.transform(audit_urls))
    audit_delta = compare(refit_scores, loaded_scores, "Audit decision scores", atol=1e-5, rtol=1e-5)
    bundle_probabilities = calibrator.predict_proba(np.asarray(loaded_scores).reshape(-1, 1))[:, 1]
    check(np.isfinite(bundle_probabilities).all() and ((0 <= bundle_probabilities) & (bundle_probabilities <= 1)).all(), "Bundle P(phishing) invalid")
    return {"status": "validated", "artifact_run_id": report["artifact_run_id"],
            "selection_semantic_sha256": selection["semantic_sha256"],
            "feature_dimension": dimensions, "crossfit_metrics": metrics, "max_abs_differences": {"idf": idf_delta, "coefficients": coefficient_delta, "intercept": intercept_delta, "audit_scores": audit_delta}, "report_sha256": digest(out / "report.json")}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
