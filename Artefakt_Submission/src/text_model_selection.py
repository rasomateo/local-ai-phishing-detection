from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.features import build_text_views
from src.prepare_text_inputs import DEVELOPMENT_COLUMNS, sha256
from src.provenance import validate_artifact_inventory, validate_predecessor
from src.selection_contract import build_selection_manifest, build_semantic_payload
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
OUT = REPOSITORY_ROOT / "outputs" / "benchmarks" / "text"
DEV = REPOSITORY_ROOT / "outputs" / "features" / "text" / "development_inputs.csv"
HOLDOUT = DEV.with_name("holdout_inputs.csv")
STEP17_REPORT = DEV.with_name("report.json")
FOLDS = tuple(range(5))
VIEWS = ("subject", "body", "subject_body")
TFIDF = {"TFIDF_A": (1, 1), "TFIDF_B": (1, 2)}
FOLD_COLUMNS = ["benchmark_run_id", "split_run_id", "config_id", "view", "tfidf_config_id", "model_family", "model_config_id", "fold", "train_n", "validation_n", "train_phishing_n", "validation_phishing_n", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "fit_seconds", "predict_seconds"]
SUMMARY_COLUMNS = ["config_id", "view", "tfidf_config_id", "model_family", "model_config_id", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "std_f1", "total_tp", "total_tn", "total_fp", "total_fn", "rank", "selected"]
OOF_COLUMNS = ["benchmark_run_id", "split_run_id", "record_id", "label", "cv_fold", "prediction", "model_score", "score_type"]


def current_input_identity(root: Path = REPOSITORY_ROOT) -> dict:
    from scripts.validate_leakage_audit import main as validate_leakage
    from scripts.validate_text_inputs import validate as validate_text_inputs

    validate_leakage()
    validate_text_inputs(root)
    report = json.loads((root / "outputs/features/text/report.json").read_text(encoding="utf-8"))
    split_manifest = validate_predecessor(root, report["predecessor"])
    validate_artifact_inventory(root, report["artifacts"])
    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    if leakage.get("canonical_split_run_id") != split_manifest["run_id"] or not leakage.get("training_ready_after_step16"):
        raise ValueError("Reserved-source audit does not authorize the canonical split")
    return {
        "split_run_id": split_manifest["run_id"],
        "split_semantic_sha256": split_manifest["semantic_sha256"],
        "text_inputs_semantic_sha256": report["semantic_sha256"],
        "development_sha256": sha256(root / "outputs/features/text/development_inputs.csv"),
        "holdout_integrity_sha256": sha256(root / "outputs/features/text/holdout_inputs.csv"),
        "leakage_semantic_sha256": leakage["semantic_sha256"],
    }


def current_protected_paths(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    split_path = root / "outputs/splits/manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    text_report_path = root / "outputs/features/text/report.json"
    text_report = json.loads(text_report_path.read_text(encoding="utf-8"))
    protected = {
        "outputs/splits/manifest.json": sha256(split_path),
        f"{split['run_directory']}/manifest.json": sha256(root / split["run_directory"] / "manifest.json"),
        "outputs/leakage/reserved_sources/summary.json": sha256(root / "outputs/leakage/reserved_sources/summary.json"),
        "outputs/features/text/report.json": sha256(text_report_path),
    }
    protected.update({spec["path"]: spec["sha256"] for spec in split["artifacts"].values()})
    protected.update({spec["path"]: spec["sha256"] for spec in text_report["artifacts"].values()})
    return protected


def tfidf_params(config_id: str) -> dict:
    return {"analyzer": "word", "ngram_range": TFIDF[config_id], "token_pattern": r"(?u)\b\w\w+\b", "lowercase": False, "sublinear_tf": True, "min_df": 2, "max_df": 0.95, "max_features": 30000, "dtype": np.float32}


def model_specs() -> list[tuple[str, str, dict]]:
    result = []
    for c in (0.5, 1.0, 2.0):
        for weight in (None, "balanced"):
            result.append(("LinearSVC", f"C={c:g};class_weight={weight or 'none'}", {"C": c, "class_weight": weight, "random_state": 42}))
    for alpha in (0.1, 0.5, 1.0):
        for prior in (True, False):
            result.append(("MultinomialNB", f"alpha={alpha:g};fit_prior={str(prior).lower()}", {"alpha": alpha, "fit_prior": prior}))
    for depth in (50, None):
        for weight in (None, "balanced"):
            result.append(("DecisionTreeClassifier", f"max_depth={depth or 'none'};class_weight={weight or 'none'}", {"max_depth": depth, "class_weight": weight, "random_state": 42}))
    for depth in (50, None):
        for weight in (None, "balanced_subsample"):
            result.append(("RandomForestClassifier", f"max_depth={depth or 'none'};class_weight={weight or 'none'}", {"n_estimators": 300, "max_depth": depth, "class_weight": weight, "random_state": 42, "n_jobs": -1}))
    return result


def make_model(family: str, params: dict):
    return {"LinearSVC": LinearSVC, "MultinomialNB": MultinomialNB, "DecisionTreeClassifier": DecisionTreeClassifier, "RandomForestClassifier": RandomForestClassifier}[family](**params)


def _make_runtime_model(family: str, params: dict):
    runtime_params = dict(params)
    if family == "RandomForestClassifier":
        runtime_params["n_jobs"] = 1
    return make_model(family, runtime_params)


def config_id(view: str, tfidf_id: str, family: str, model_id: str) -> str:
    return f"{view}|{tfidf_id}|{family}|{model_id}"


def rank_key(row: dict) -> tuple:
    return (-float(row["mean_f1"]), -float(row["mean_recall"]), -float(row["mean_precision"]), float(row["std_f1"]), row["config_id"])


def counts(y_true, y_pred) -> dict:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    tp = int(np.sum((true == "phishing") & (pred == "phishing")))
    tn = int(np.sum((true == "legitimate") & (pred == "legitimate")))
    fp = int(np.sum((true == "legitimate") & (pred == "phishing")))
    fn = int(np.sum((true == "phishing") & (pred == "legitimate")))
    n = len(true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def summarize(rows: list[dict], cid: str, view: str, tid: str, family: str, mid: str) -> dict:
    result = {"config_id": cid, "view": view, "tfidf_config_id": tid, "model_family": family, "model_config_id": mid}
    for metric in ("accuracy", "precision", "recall", "f1"):
        result[f"mean_{metric}"] = float(np.mean([r[metric] for r in rows]))
    result["std_f1"] = float(np.std([r["f1"] for r in rows], ddof=0))
    for metric in ("tp", "tn", "fp", "fn"):
        result[f"total_{metric}"] = sum(r[metric] for r in rows)
    return result


def vectorize_fold(texts: np.ndarray, valid_mask: np.ndarray, tfidf_id: str):
    vectorizer = TfidfVectorizer(**tfidf_params(tfidf_id))
    train = vectorizer.fit_transform(texts[~valid_mask])
    valid = vectorizer.transform(texts[valid_mask])
    return train, valid, vectorizer


def select_complete_candidate(leader: dict | None, leader_oof, candidate: dict, candidate_oof, fold_rows: list[dict]):
    if len(fold_rows) != 5 or {int(row["fold"]) for row in fold_rows} != set(FOLDS):
        raise ValueError("Only a complete five-fold candidate may lead")
    if leader is None or rank_key(candidate) < rank_key(leader):
        return candidate, candidate_oof
    return leader, leader_oof


def _read_ids(path: Path, expected: list[str]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected schema: {path}")
        return list(reader)


def load_development(root: Path = REPOSITORY_ROOT) -> pd.DataFrame:
    contract = current_contract(root)
    expected_total = contract.expected("text_development_count", 201016)
    expected_labels = contract.expected("text_development_labels", {"legitimate": 198800, "phishing": 2216})
    expected_folds = contract.expected("text_fold_counts", {
        "0": [40204, 444], "1": [40203, 443], "2": [40203, 443],
        "3": [40203, 443], "4": [40203, 443],
    })
    dev_path = root / "outputs/features/text/development_inputs.csv"
    holdout_path = root / "outputs/features/text/holdout_inputs.csv"
    report_path = root / "outputs/features/text/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = current_input_identity(root)
    run_id = identity["split_run_id"]
    if report["split_run_id"] != run_id:
        raise ValueError("Step-17 run/hash mismatch")
    if (report["artifacts"]["development_inputs.csv"]["sha256"] != identity["development_sha256"] or
            report["artifacts"]["holdout_inputs.csv"]["sha256"] != identity["holdout_integrity_sha256"]):
        raise ValueError("Step-17 report hash mismatch")
    dev = pd.DataFrame(_read_ids(dev_path, DEVELOPMENT_COLUMNS))
    if len(dev) != expected_total or dev["record_id"].duplicated().any() or set(dev["split_run_id"]) != {run_id} or set(dev["label"]) != {"legitimate", "phishing"} or set(dev["cv_fold"]) != set("01234"):
        raise ValueError("Invalid Development identity, label, fold or count")
    if Counter(dev["label"]) != Counter(expected_labels):
        raise ValueError("Development label counts differ")
    for fold, (total, pos) in expected_folds.items():
        group = dev[dev.cv_fold == fold]
        if len(group) != total or sum(group.label == "phishing") != pos:
            raise ValueError(f"Fold {fold} count mismatch")
    split_path = root / f"outputs/splits/runs/{run_id}/split_assignments.csv"
    splits = _read_ids(split_path, ["record_id", "label", "partition"])
    canonical_dev = {r["record_id"]: r["label"] for r in splits if r["partition"] == "development"}
    if len(canonical_dev) != expected_total or set(dev.record_id) != set(canonical_dev) or any(canonical_dev[r.record_id] != r.label for r in dev.itertuples()):
        raise ValueError("Development differs from canonical split or intersects holdout")
    folds = _read_ids(root / f"outputs/splits/runs/{run_id}/cv_fold_assignments.csv", ["record_id", "label", "cv_fold"])
    canonical_folds = {r["record_id"]: (r["label"], r["cv_fold"]) for r in folds}
    if len(canonical_folds) != len(dev) or any(canonical_folds.get(r.record_id) != (r.label, r.cv_fold) for r in dev.itertuples()):
        raise ValueError("Development differs from canonical CV")
    for path in (root / "outputs/dataset_roles").glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames and {"record_id", "source_role"}.issubset(reader.fieldnames):
                if any(r["record_id"] in canonical_dev for r in reader if r["source_role"] in {"temporal_evaluation", "external_evaluation", "rq2_evaluation"}):
                    raise ValueError(f"Reserved ID in Development: {path}")
    return dev.sort_values("record_id").reset_index(drop=True)


def _score(model, matrix):
    classes = list(model.classes_)
    if hasattr(model, "decision_function"):
        value = np.asarray(model.decision_function(matrix))
        if value.ndim == 2:
            value = value[:, classes.index("phishing")]
        elif classes[1] != "phishing":
            value = -value
        return value, "decision_function"
    return np.asarray(model.predict_proba(matrix))[:, classes.index("phishing")], "positive_class_probability"


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_candidate(family, mid, params, x_train, x_valid, y_train, y_valid):
    with threadpool_limits(limits=1):
        model = _make_runtime_model(family, params)
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        pred = model.predict(x_valid)
        score, score_type = _score(model, x_valid)
        predict_seconds = time.perf_counter() - predict_start
    return {
        "family": family,
        "model_id": mid,
        "prediction_is_phishing": np.asarray(pred == "phishing", dtype=np.bool_),
        "model_score": np.asarray(score, dtype=np.float64),
        "score_type": score_type,
        "metrics": counts(y_valid, pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def build_current_semantic_payload(root: Path = REPOSITORY_ROOT) -> dict:
    out = root / "outputs/benchmarks/text"
    fold_rows = _read_ids(out / "fold_results.csv", FOLD_COLUMNS)
    summaries = _read_ids(out / "config_summary.csv", SUMMARY_COLUMNS)
    winner_oof = _read_ids(out / "winner_oof_predictions.csv", OOF_COLUMNS)
    winner = json.loads((out / "winner.json").read_text(encoding="utf-8"))
    search = {
        "views": list(VIEWS),
        "tfidf": {key: tfidf_params(key) for key in TFIDF},
        "models": model_specs(),
        "folds": list(FOLDS),
        "selection_rule": winner["selection_rule"],
    }
    identity = current_input_identity(root)
    return build_semantic_payload(
        selection_type="text",
        benchmark_run_id=winner["benchmark_run_id"],
        input_identity=identity,
        method_identity=search,
        fold_rows=fold_rows,
        summaries=summaries,
        winner=winner,
        winner_oof=winner_oof,
    )


def run(root: Path = REPOSITORY_ROOT, *, jobs: int = 1) -> dict:
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    start = time.perf_counter()
    dev = load_development(root)
    identity = current_input_identity(root)
    split_run_id = identity["split_run_id"]
    RUN_ID = split_run_id
    DEV_HASH = identity["development_sha256"]
    HOLDOUT_HASH = identity["holdout_integrity_sha256"]
    PROTECTED = dict(identity)
    specs = model_specs()
    if len(specs) != 20:
        raise AssertionError("Unexpected model search space")
    search = {"views": VIEWS, "tfidf": {k: tfidf_params(k) for k in TFIDF}, "models": specs}
    def safe(value):
        if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [safe(v) for v in value]
        if value is np.float32: return "float32"
        return value
    search = safe(search)
    run_id = hashlib.sha256(json.dumps({"inputs": identity, "search": search}, sort_keys=True).encode()).hexdigest()[:32]
    views = build_text_views(dev[["subject", "body"]])
    fold_rows, summaries = [], []
    leader, leader_oof = None, None
    with parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
        with Parallel(max_nbytes="1M", mmap_mode="r") as parallel:
            for view in VIEWS:
                texts = np.asarray(views[view], dtype=object)
                for tid in TFIDF:
                    candidate_rows = {(family, mid): [] for family, mid, _ in specs}
                    candidate_oof = {(family, mid): (np.empty(len(dev), dtype=np.bool_), np.empty(len(dev), dtype=np.float64)) for family, mid, _ in specs}
                    for fold in FOLDS:
                        valid_mask = (dev.cv_fold == str(fold)).to_numpy()
                        train_mask = ~valid_mask
                        x_train, x_valid, _ = vectorize_fold(texts, valid_mask, tid)
                        y_train = dev.label.to_numpy()[train_mask]
                        y_valid = dev.label.to_numpy()[valid_mask]
                        results = parallel(
                            delayed(_evaluate_candidate)(family, mid, params, x_train, x_valid, y_train, y_valid)
                            for family, mid, params in specs
                        )
                        for result in results:
                            family, mid = result["family"], result["model_id"]
                            cid = config_id(view, tid, family, mid)
                            metrics = result["metrics"]
                            row = {"benchmark_run_id": run_id, "split_run_id": RUN_ID, "config_id": cid, "view": view, "tfidf_config_id": tid, "model_family": family, "model_config_id": mid, "fold": fold, "train_n": len(y_train), "validation_n": len(y_valid), "train_phishing_n": int(sum(y_train == "phishing")), "validation_phishing_n": int(sum(y_valid == "phishing")), **metrics, "fit_seconds": result["fit_seconds"], "predict_seconds": result["predict_seconds"]}
                            candidate_rows[(family, mid)].append(row)
                            candidate_oof[(family, mid)][0][valid_mask] = result["prediction_is_phishing"]
                            candidate_oof[(family, mid)][1][valid_mask] = result["model_score"]
                            print(f"{view} {tid} {family} {mid} fold={fold} f1={metrics['f1']:.6f}", flush=True)
                    for family, mid, _ in specs:
                        cid = config_id(view, tid, family, mid)
                        rows = candidate_rows[(family, mid)]
                        if len(rows) != 5 or {int(r["fold"]) for r in rows} != set(FOLDS):
                            raise ValueError("Incomplete candidate")
                        summary = summarize(rows, cid, view, tid, family, mid)
                        fold_rows.extend(rows)
                        summaries.append(summary)
                        if leader is None or rank_key(summary) < rank_key(leader):
                            flags, scores = candidate_oof[(family, mid)]
                            score_type = "decision_function" if family == "LinearSVC" else "positive_class_probability"
                            candidate_prediction_rows = [{"benchmark_run_id": run_id, "split_run_id": RUN_ID, "record_id": str(rid), "label": str(label), "cv_fold": int(fold), "prediction": "phishing" if flag else "legitimate", "model_score": float(score), "score_type": score_type} for rid, label, fold, flag, score in zip(dev.record_id, dev.label, dev.cv_fold, flags, scores, strict=True)]
                            leader, leader_oof = select_complete_candidate(leader, leader_oof, summary, candidate_prediction_rows, rows)
                    del candidate_oof, candidate_rows, x_train, x_valid
    summaries.sort(key=rank_key)
    for i, summary in enumerate(summaries, 1):
        summary["rank"] = i
        summary["selected"] = i == 1
    if summaries[0]["config_id"] != leader["config_id"] or len(leader_oof) != len(dev):
        raise AssertionError("Winner OOF or ranking mismatch")
    method = current_contract(root).payload["method"]
    if len(summaries) != method["text_candidate_count"] or len(fold_rows) != method["text_fold_fit_count"]:
        raise ValueError("Protected full text-selection search space was not executed")
    leader_oof.sort(key=lambda r: r["record_id"])
    out = root / "outputs/benchmarks/text"
    out.mkdir(parents=True, exist_ok=True)
    fold_rows.sort(key=lambda r: (r["config_id"], r["fold"]))
    _write_csv(out / "fold_results.csv", FOLD_COLUMNS, fold_rows)
    _write_csv(out / "config_summary.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(out / "winner_oof_predictions.csv", OOF_COLUMNS, leader_oof)
    family, mid = leader["model_family"], leader["model_config_id"]
    model_params = next(params for fam, ident, params in specs if (fam, ident) == (family, mid))
    winner = {"benchmark_run_id": run_id, "split_run_id": RUN_ID, "config_id": leader["config_id"], "view": leader["view"], "tfidf_config_id": leader["tfidf_config_id"], "tfidf_parameters": safe(tfidf_params(leader["tfidf_config_id"])), "model_family": family, "model_config_id": mid, "model_parameters": safe(make_model(family, model_params).get_params(deep=False)), "cv_metrics": leader, "selection_rule": "descending mean F1, descending mean recall, descending mean precision, ascending population std F1, lexicographic config_id", "holdout_evaluated": False, "status": "Development-CV selection; not final test performance"}
    (out / "winner.json").write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
    semantic_payload = build_current_semantic_payload(root)
    selection_manifest = build_selection_manifest(
        payload=semantic_payload,
        input_paths={"development": "outputs/features/text/development_inputs.csv", "folds": f"outputs/splits/runs/{RUN_ID}/cv_fold_assignments.csv"},
        output_paths={"fold_results": "outputs/benchmarks/text/fold_results.csv", "config_summary": "outputs/benchmarks/text/config_summary.csv", "winner": "outputs/benchmarks/text/winner.json", "winner_oof": "outputs/benchmarks/text/winner_oof_predictions.csv"},
    )
    (out / "selection_manifest.json").write_text(json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "benchmark_run_id": run_id, "split_run_id": RUN_ID, "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "inputs": {"development": {"path": str(DEV.relative_to(REPOSITORY_ROOT)), "sha256": DEV_HASH}, "holdout_integrity_only": {"path": str(HOLDOUT.relative_to(REPOSITORY_ROOT)), "sha256": HOLDOUT_HASH}, "step17_report": {"path": str(STEP17_REPORT.relative_to(REPOSITORY_ROOT)), "sha256": sha256(STEP17_REPORT)}}, "development_counts": {"total": len(dev), "by_label": dict(Counter(dev.label))}, "fold_counts": {str(f): {"total": int(sum(dev.cv_fold == str(f))), "by_label": dict(Counter(dev.loc[dev.cv_fold == str(f), "label"]))} for f in FOLDS}, "text_views": {"subject": "subject", "body": "body", "subject_body": "Step-17 build_text_views: strip fields; join two nonempty fields with two newlines"}, "search_space": search, "candidate_count": len(summaries), "model_fit_count": len(fold_rows), "selection_rule": winner["selection_rule"], "winner": winner, "duration_seconds": time.perf_counter() - start, "parallelism": {"backend": "serial" if jobs == 1 else "loky", "jobs": jobs, "inner_threads": 1, "random_forest_runtime_n_jobs": 1}, "outputs": {name: {"path": f"outputs/benchmarks/text/{name}", "sha256": sha256(out / name)} for name in ("fold_results.csv", "config_summary.csv", "winner.json", "winner_oof_predictions.csv", "selection_manifest.json")}, "protected_sha256": PROTECTED, "holdout_accessed_for_metrics": False, "external_data_used": False, "threshold_tuned": False, "calibration_performed": False, "model_artifact_frozen": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Development-only text model selection")
    parser.add_argument("--jobs", type=int, default=1, help="Number of outer candidate workers (positive integer)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    result = run(jobs=args.jobs)
    print(json.dumps({"benchmark_run_id": result["benchmark_run_id"], "winner": result["winner"]["config_id"], "duration_seconds": result["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
