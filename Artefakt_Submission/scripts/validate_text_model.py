from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.features import build_text_views
from src.load_data import NORMALIZED_COLUMNS
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract
from src.text_model_selection import current_input_identity, current_protected_paths

csv.field_size_limit(sys.maxsize)
RUN = ""
PROTECTED = {}
FLAGS = {"full_development_fit": True, "model_selection_performed": False, "holdout_used_for_fit": False, "holdout_used_for_metrics": False, "holdout_predictions_generated": False, "calibration_performed": False, "threshold_tuned": False, "url_data_used": False, "fusion_performed": False, "external_data_used": False, "rq2_data_used": False}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path, fields: list[str] | None = None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if fields is not None:
            check(reader.fieldnames == fields, f"Unexpected CSV schema: {path}")
        return list(reader)


def score_classes(model) -> list[str]:
    classes = [str(value) for value in model.classes_]
    check(classes == ["legitimate", "phishing"] and classes[1] == "phishing", "Positive decision score does not mean phishing")
    return classes


def compare_arrays(left, right, name: str) -> float:
    a, b = np.asarray(left), np.asarray(right)
    check(a.shape == b.shape, f"{name} shape mismatch")
    delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    check(np.allclose(a, b, atol=1e-5, rtol=1e-5), f"{name} numerical mismatch (max abs {delta})")
    return delta


def validate(root: Path = REPOSITORY_ROOT) -> dict:
    contract = current_contract(root)
    global RUN, PROTECTED
    identity = current_input_identity(root)
    PROTECTED = current_protected_paths(root)
    RUN = identity["split_run_id"]
    from scripts.validate_text_model_selection import validate as validate_selection

    out = root / "outputs/models/text/classifier"
    selection = validate_selection(root)
    benchmark_run = selection["benchmark_run_id"]
    for relative, expected in PROTECTED.items():
        check(digest(root / relative) == expected, f"Protected artifact changed: {relative}")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    winner = json.loads((root / "outputs/benchmarks/text/winner.json").read_text(encoding="utf-8"))
    check(winner["benchmark_run_id"] == benchmark_run and winner["split_run_id"] == RUN and selection["winner"] == winner["config_id"], "Winner provenance mismatch")
    check(winner["model_family"] == "LinearSVC", "Current winner is incompatible with the calibrated explainable text bundle")
    tfidf, model_parameters = winner["tfidf_parameters"], winner["model_parameters"]
    check(manifest["winner"] == {"path": "outputs/benchmarks/text/winner.json", "sha256": digest(root / "outputs/benchmarks/text/winner.json")}, "Manifest winner mismatch")
    check(manifest["selection_provenance"] == report["selection_provenance"] == {"manifest_path": "outputs/benchmarks/text/selection_manifest.json", "manifest_sha256": selection["selection_manifest_sha256"], "semantic_sha256": selection["semantic_sha256"]}, "Selection provenance mismatch")
    check(manifest["split_run_id"] == report["split_run_id"] == RUN and manifest["benchmark_run_id"] == report["benchmark_run_id"] == benchmark_run, "Run IDs differ")
    check(manifest["winner_config_id"] == report["winner_config_id"] == winner["config_id"], "Config ID differs")
    check(manifest["view"] == report["view"] == winner["view"] and manifest["tfidf_parameters"] == report["tfidf_parameters"] == tfidf, "TF-IDF manifest differs")
    check(manifest["model_parameters"] == report["model_parameters"] == model_parameters, "Model manifest differs")
    check(manifest["flags"] == report["flags"] == FLAGS and report["protected_sha256"] == PROTECTED, "Method or protected flags differ")
    check(report["status"] == "runner_checks_passed", "Runner status differs")
    check(report["inputs"]["development"] == manifest["development"] and manifest["development"]["sha256"] == PROTECTED["outputs/features/text/development_inputs.csv"], "Development hash differs")
    check(report["inputs"]["holdout_integrity_only"]["sha256"] == PROTECTED["outputs/features/text/holdout_inputs.csv"], "Holdout hash differs")
    check(set(report["outputs"]) == {"text_vectorizer.joblib", "text_model.joblib", "audit_predictions.csv", "manifest.json"}, "Output list differs")
    for name, spec in report["outputs"].items():
        check(spec == {"path": f"outputs/models/text/classifier/{name}", "sha256": digest(out / name)}, f"Output hash mismatch: {name}")
    check(manifest["outputs"] == {key: report["outputs"][key] for key in ("text_vectorizer.joblib", "text_model.joblib", "audit_predictions.csv")}, "Manifest output hashes differ")

    dev = rows(root / "outputs/features/text/development_inputs.csv", ["split_run_id", "record_id", "label", "cv_fold", "subject", "body"])
    ids = [row["record_id"] for row in dev]
    id_set = set(ids)
    check(len(dev) == contract.expected("text_development_count", 201016) and len(id_set) == len(dev) and all(ids), "Development count or IDs differ")
    check({row["split_run_id"] for row in dev} == {RUN} and Counter(row["label"] for row in dev) == Counter(contract.expected("text_development_labels", {"legitimate": 198800, "phishing": 2216})), "Development run or labels differ")
    check(manifest["development"]["count"] == len(dev) and manifest["development"]["label_counts"] == dict(Counter(row["label"] for row in dev)), "Development manifest counts differ")
    split = rows(root / f"outputs/splits/runs/{RUN}/split_assignments.csv", ["record_id", "label", "partition"])
    canonical = {row["record_id"]: row["label"] for row in split if row["partition"] == "development"}
    holdout_ids = {row["record_id"] for row in split if row["partition"] == "final_internal_holdout"}
    check(len(canonical) == len(dev) and id_set == set(canonical) and not id_set & holdout_ids, "Development/Holdout identity mismatch")
    check(all(canonical[row["record_id"]] == row["label"] for row in dev), "Development split label mismatch")
    folds = rows(root / f"outputs/splits/runs/{RUN}/cv_fold_assignments.csv", ["record_id", "label", "cv_fold"])
    fold_by_id = {row["record_id"]: (row["label"], row["cv_fold"]) for row in folds}
    check(len(fold_by_id) == len(dev) and all(fold_by_id.get(row["record_id"]) == (row["label"], row["cv_fold"]) for row in dev), "CV assignment mismatch")
    preprocessed = rows(root / "outputs/preprocessed_dataset.csv", NORMALIZED_COLUMNS)
    role_by_id = {row["record_id"]: row["source_role"] for row in preprocessed}
    check(len(role_by_id) == len(preprocessed) and all(role_by_id.get(rid) == "text_development" for rid in id_set), "Development role mismatch")
    for path in (root / "outputs/dataset_roles").glob("*.csv"):
        reserved = rows(path)
        if reserved and "record_id" in reserved[0]:
            check(not any(row["record_id"] in id_set for row in reserved), f"Reserved ID in Development: {path.name}")

    vectorizer = joblib.load(out / "text_vectorizer.joblib")
    model = joblib.load(out / "text_model.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC, "Bundle classes differ")
    actual_params = vectorizer.get_params(deep=False)
    check(all(actual_params[key] == (tuple(value) if key == "ngram_range" else np.float32 if key == "dtype" else value) for key, value in tfidf.items()), "Persisted TF-IDF parameters differ")
    check(model.get_params(deep=False) == model_parameters, "Persisted model parameters differ")
    classes = score_classes(model)
    check(manifest["model_classes"] == classes and manifest["positive_semantic_class"] == manifest["decision_function_positive_class"] == classes[1] == "phishing", "Positive score metadata differs")
    dimensions = len(vectorizer.vocabulary_)
    check(manifest["vocabulary_size"] == manifest["feature_dimension"] == dimensions, "Vocabulary dimension mismatch")
    check(model.coef_.shape == (1, dimensions) and model.intercept_.shape == (1,), "Model dimension mismatch")
    check(manifest["coefficient_shape"] == list(model.coef_.shape) and manifest["intercept_shape"] == list(model.intercept_.shape), "Model shape manifest differs")
    compare_arrays(manifest["intercept"], model.intercept_, "Persisted intercept")
    check(manifest["n_iter"] == np.asarray(model.n_iter_).tolist(), "Iteration count differs")

    digest_ids = lambda rid: hashlib.sha256(rid.encode("utf-8")).hexdigest()
    ordered = sorted(dev, key=lambda row: (digest_ids(row["record_id"]), row["record_id"]))[:256]
    audit = rows(out / "audit_predictions.csv", ["record_id", "label", "prediction", "decision_score"])
    check(len(audit) == 256 and [(row["record_id"], row["label"]) for row in audit] == [(row["record_id"], row["label"]) for row in ordered], "Audit selection differs")
    import pandas as pd
    audit_text = build_text_views(pd.DataFrame(ordered)[["subject", "body"]])[winner["view"]]
    audit_matrix = vectorizer.transform(audit_text)
    stored_scores = model.decision_function(audit_matrix)
    check(list(model.predict(audit_matrix)) == [row["prediction"] for row in audit], "Audit predictions differ")
    audit_score_delta = compare_arrays(stored_scores, [float(row["decision_score"]) for row in audit], "Audit decision scores")

    refit_vectorizer = TfidfVectorizer(**{**tfidf, "ngram_range": tuple(tfidf["ngram_range"]), "dtype": np.float32})
    full_text = build_text_views(pd.DataFrame(dev)[["subject", "body"]])[winner["view"]]
    refit_matrix = refit_vectorizer.fit_transform(full_text)
    check(refit_matrix.shape == (len(dev), dimensions), "Refit feature dimension differs")
    check(refit_vectorizer.vocabulary_ == vectorizer.vocabulary_, "Full vocabulary mapping differs")
    idf_delta = compare_arrays(refit_vectorizer.idf_, vectorizer.idf_, "Full IDF vector")
    refit_model = LinearSVC(**model_parameters).fit(refit_matrix, np.asarray([row["label"] for row in dev]))
    check(score_classes(refit_model) == classes, "Refit classes differ")
    coefficient_delta = compare_arrays(refit_model.coef_, model.coef_, "Coefficients")
    intercept_delta = compare_arrays(refit_model.intercept_, model.intercept_, "Intercept")
    refit_audit = refit_vectorizer.transform(audit_text)
    check(list(refit_model.predict(refit_audit)) == [row["prediction"] for row in audit], "Refit audit predictions differ")
    refit_score_delta = compare_arrays(refit_model.decision_function(refit_audit), stored_scores, "Refit audit scores")
    return {"status": "validated", "split_run_id": RUN, "winner_config_id": winner["config_id"], "full_vocabulary_exact": True,
            "max_abs_differences": {"idf": idf_delta, "coefficients": coefficient_delta, "intercept": intercept_delta, "stored_audit_score": audit_score_delta, "refit_audit_score": refit_score_delta},
            "report_sha256": digest(out / "report.json")}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
