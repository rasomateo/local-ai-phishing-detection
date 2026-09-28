from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src.selection_contract import build_selection_manifest, build_semantic_payload
from src.execution_contract import current_contract

ROOT = Path(__file__).resolve().parents[1]
URL_RUN_ID = ""
STEP21B = {}
REPRESENTATIONS = ("normalized_url_char", "url_component_text", "combined_url")
FOLDS = tuple(range(5))
CHAR_PARAMS = {"analyzer": "char", "ngram_range": (3, 5), "lowercase": False, "sublinear_tf": True, "min_df": 2, "max_df": 1.0, "max_features": 50000, "dtype": np.float32}
COMPONENT_PARAMS = {"analyzer": "word", "token_pattern": r"(?u)\S+", "ngram_range": (1, 2), "lowercase": False, "sublinear_tf": True, "min_df": 2, "max_df": 1.0, "max_features": 30000, "dtype": np.float32}
COMPONENT_SPEC = "urlsplit(normalized_url); scheme, full hostname, each dot-separated hostname label, explicit port, nonempty slash-separated path segments, nonempty raw query keys and values split on & then first =; preserve order/repetitions; no decoding. Escape literal % as %25 and each whitespace codepoint as %HEX (uppercase, at least two digits); prefix each nonempty component by type; join tokens with one ASCII space."
FOLD_COLUMNS = ["benchmark_run_id", "url_run_id", "representation", "model_family", "model_config_id", "config_id", "fold", "train_n", "validation_n", "train_legitimate_n", "train_phishing_n", "validation_legitimate_n", "validation_phishing_n", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "fit_seconds", "predict_seconds"]
SUMMARY_COLUMNS = ["representation", "model_family", "model_config_id", "config_id", "model_parameters", "mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "std_f1", "total_tn", "total_fp", "total_fn", "total_tp", "rank", "selected"]
OOF_COLUMNS = ["benchmark_run_id", "url_run_id", "canonical_url_id", "label", "cv_fold", "prediction", "model_score", "score_type"]


def canonical_selection_input_identity(report: dict, artifacts: dict[str, str]) -> dict:
    return {
        "url_run_id": report["url_run_id"],
        "canonical_semantic_sha256": report["semantic_sha256"],
        "canonical_artifact_sha256": {
            name: digest for name, digest in artifacts.items() if name != "report.json"
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_protected(root: Path = ROOT) -> dict[str, str]:
    global URL_RUN_ID, STEP21B
    base = root / "outputs/datasets/url/canonical"
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))
    URL_RUN_ID = report["url_run_id"]
    STEP21B = {name: spec["sha256"] for name, spec in report["outputs"].items()}
    STEP21B["report.json"] = sha256(base / "report.json")
    for name, expected in STEP21B.items():
        if sha256(base / name) != expected:
            raise ValueError(f"Canonical URL artifact hash mismatch: {name}")
    protected = dict(report["protected_sha256"])
    for relative, expected in protected.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Protected hash mismatch: {relative}")
    return {**{f"outputs/datasets/url/canonical/{k}": v for k, v in STEP21B.items()}, **protected}


def read_rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected schema: {path}")
        return list(reader)


def load_development(root: Path = ROOT) -> tuple[list[dict[str, str]], dict]:
    contract = current_contract(root)
    check_protected(root)
    base = root / "outputs/datasets/url/canonical"
    splits = read_rows(base / "split_assignments.csv", ["url_run_id", "canonical_url_id", "label", "leakage_group", "split"])
    cv = read_rows(base / "cv_assignments.csv", ["url_run_id", "canonical_url_id", "label", "leakage_group", "cv_fold"])
    expected_total = contract.expected("url_canonical_count", 4883)
    if len(splits) != expected_total or len({r["canonical_url_id"] for r in splits}) != expected_total or any(r["url_run_id"] != URL_RUN_ID for r in splits):
        raise ValueError("Invalid canonical splits")
    dev = {r["canonical_url_id"]: r for r in splits if r["split"] == "development"}
    holdout = {r["canonical_url_id"]: r for r in splits if r["split"] == "holdout"}
    expected_dev = (sum(r["split"] == "development" for r in splits) if contract.is_fixture
                    else contract.expected("url_development_count", 4001))
    expected_holdout = (sum(r["split"] == "holdout" for r in splits) if contract.is_fixture
                        else contract.expected("url_holdout_count", 882))
    if len(dev) != expected_dev or len(holdout) != expected_holdout or set(dev) & set(holdout) or len(dev) + len(holdout) != len(splits):
        raise ValueError("Development/Holdout partition mismatch")
    expected_dev_labels = (Counter(r["label"] for r in dev.values()) if contract.is_fixture else Counter(contract.expected("url_development_labels", {"legitimate": 948, "phishing": 3053})))
    expected_holdout_labels = (Counter(r["label"] for r in holdout.values()) if contract.is_fixture else Counter(contract.expected("url_holdout_labels", {"legitimate": 228, "phishing": 654})))
    if Counter(r["label"] for r in dev.values()) != expected_dev_labels or Counter(r["label"] for r in holdout.values()) != expected_holdout_labels:
        raise ValueError("Split label counts mismatch")
    cv_by_id = {r["canonical_url_id"]: r for r in cv}
    if len(cv) != len(cv_by_id) or set(cv_by_id) != set(dev) or any(r["url_run_id"] != URL_RUN_ID or r["cv_fold"] not in {"0", "1", "2", "3", "4"} or (r["label"], r["leakage_group"]) != (dev[r["canonical_url_id"]]["label"], dev[r["canonical_url_id"]]["leakage_group"]) for r in cv):
        raise ValueError("Canonical CV mismatch")
    expected_folds = ({fold: (sum(r["cv_fold"] == str(fold) for r in cv),
                              sum(r["cv_fold"] == str(fold) and r["label"] == "legitimate" for r in cv),
                              sum(r["cv_fold"] == str(fold) and r["label"] == "phishing" for r in cv))
                       for fold in FOLDS} if contract.is_fixture else
                      {int(key): tuple(value) for key, value in contract.expected("url_fold_counts", {"0": [953, 157, 796], "1": [762, 187, 575], "2": [717, 200, 517], "3": [771, 223, 548], "4": [798, 181, 617]}).items()})
    fold_counts = {}
    for fold, expected in expected_folds.items():
        rows = [r for r in cv if r["cv_fold"] == str(fold)]
        actual = (len(rows), sum(r["label"] == "legitimate" for r in rows), sum(r["label"] == "phishing" for r in rows))
        if actual != expected:
            raise ValueError(f"Fold counts mismatch: {fold}")
        fold_counts[str(fold)] = {"total": actual[0], "legitimate": actual[1], "phishing": actual[2]}
    if set(r["leakage_group"] for r in dev.values()) & set(r["leakage_group"] for r in holdout.values()):
        raise ValueError("Split leakage group crossing")
    group_folds = defaultdict(set)
    for r in cv:
        group_folds[r["leakage_group"]].add(r["cv_fold"])
    if any(len(folds) != 1 for folds in group_folds.values()):
        raise ValueError("CV leakage group crossing")
    dataset_fields = ["url_run_id", "canonical_url_id", "normalized_url", "representative_raw_url", "label", "leakage_group", "source", "source_role", "label_provenance", "source_record_count"]
    selected = []
    all_ids = set()
    with (base / "url_dataset.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != dataset_fields:
            raise ValueError("Canonical URL dataset schema mismatch")
        for row in reader:
            rid = row["canonical_url_id"]
            if rid in all_ids:
                raise ValueError("Duplicate canonical URL ID")
            all_ids.add(rid)
            if rid not in dev:
                continue
            assignment = dev[rid]
            if row["url_run_id"] != URL_RUN_ID or row["source"] != "phishload" or row["source_role"] != "url_only" or (row["label"], row["leakage_group"]) != (assignment["label"], assignment["leakage_group"]):
                raise ValueError("Development dataset assignment mismatch")
            selected.append({"canonical_url_id": rid, "normalized_url": row["normalized_url"], "label": row["label"], "cv_fold": cv_by_id[rid]["cv_fold"]})
    if all_ids != {r["canonical_url_id"] for r in splits} or len(selected) != expected_dev or len({r["normalized_url"] for r in selected}) != expected_dev:
        raise ValueError("Canonical ID/URL uniqueness or coverage mismatch")
    selected.sort(key=lambda row: row["canonical_url_id"])
    return selected, {"development": {"total": expected_dev, **dict(Counter(r["label"] for r in dev.values()))}, "holdout_metadata_only": {"total": expected_holdout, **dict(Counter(r["label"] for r in holdout.values()))}, "folds": fold_counts}


def escape_component(value: str) -> str:
    return "".join("%25" if char == "%" else f"%{ord(char):02X}" if char.isspace() else char for char in value)


def component_text(normalized_url: str) -> str:
    parsed = urlsplit(normalized_url)
    host = parsed.hostname or ""
    parts = [("scheme", parsed.scheme), ("host", host)]
    parts += [("hostlabel", label) for label in host.split(".")]
    if parsed.port is not None:
        parts.append(("port", str(parsed.port)))
    parts += [("path", segment) for segment in parsed.path.split("/")]
    for pair in parsed.query.split("&") if parsed.query else ():
        key, _, value = pair.partition("=")
        parts.extend((("querykey", key), ("queryvalue", value)))
    return " ".join(f"{prefix}_{escape_component(value)}" for prefix, value in parts if value)


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


def config_id(representation: str, family: str, model_id: str) -> str:
    return f"{representation}|{family}|{model_id}"


def rank_key(row: dict) -> tuple:
    return (-float(row["mean_f1"]), -float(row["mean_recall"]), -float(row["mean_precision"]), float(row["std_f1"]), row["config_id"])


def select_complete_candidate(leader, leader_oof, candidate, candidate_oof, rows):
    expected_oof = sum(int(row["validation_n"]) for row in rows)
    if (len(rows) != 5 or {int(r["fold"]) for r in rows} != set(FOLDS)
            or len(candidate_oof) != expected_oof):
        raise ValueError("Only a complete five-fold candidate may lead")
    return (candidate, candidate_oof) if leader is None or rank_key(candidate) < rank_key(leader) else (leader, leader_oof)


def metrics(y_true, y_pred) -> dict:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((true == "phishing") & (pred == "phishing")))
    tn = int(np.sum((true == "legitimate") & (pred == "legitimate")))
    fp = int(np.sum((true == "legitimate") & (pred == "phishing")))
    fn = int(np.sum((true == "phishing") & (pred == "legitimate")))
    return {"accuracy": (tp + tn) / len(true), "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def score(model, matrix):
    classes = list(model.classes_)
    if isinstance(model, LinearSVC):
        values = np.asarray(model.decision_function(matrix))
        return (-values if classes[1] != "phishing" else values), "decision_function"
    return np.asarray(model.predict_proba(matrix))[:, classes.index("phishing")], "positive_class_probability"


def vectorize_fold(urls, component_docs, valid_mask):
    train_mask = ~valid_mask
    char_vec = TfidfVectorizer(**CHAR_PARAMS)
    comp_vec = TfidfVectorizer(**COMPONENT_PARAMS)
    char_train = char_vec.fit_transform(urls[train_mask])
    char_valid = char_vec.transform(urls[valid_mask])
    comp_train = comp_vec.fit_transform(component_docs[train_mask])
    comp_valid = comp_vec.transform(component_docs[valid_mask])
    return {"normalized_url_char": (char_train, char_valid), "url_component_text": (comp_train, comp_valid), "combined_url": (sparse.hstack((char_train, comp_train), format="csr", dtype=np.float32), sparse.hstack((char_valid, comp_valid), format="csr", dtype=np.float32))}, (char_vec, comp_vec)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_candidate(family, mid, params, x_train, x_valid, y_train, y_valid):
    with threadpool_limits(limits=1):
        model = _make_runtime_model(family, params)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict(x_valid)
        values, score_type = score(model, x_valid)
        predict_seconds = time.perf_counter() - started
    return {
        "family": family,
        "model_id": mid,
        "prediction": np.asarray(prediction, dtype=object),
        "model_score": np.asarray(values, dtype=np.float64),
        "score_type": score_type,
        "metrics": metrics(y_valid, prediction),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def build_current_semantic_payload(root: Path = ROOT) -> dict:
    check_protected(root)
    out = root / "outputs/benchmarks/url"
    canonical_report = json.loads(
        (root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8")
    )
    fold_rows = read_rows(out / "fold_results.csv", FOLD_COLUMNS)
    summaries = read_rows(out / "config_summary.csv", SUMMARY_COLUMNS)
    winner_oof = read_rows(out / "winner_oof_predictions.csv", OOF_COLUMNS)
    winner = json.loads((out / "winner.json").read_text(encoding="utf-8"))
    representation_specs = {
        "normalized_url_char": safe_value(CHAR_PARAMS),
        "url_component_text": safe_value(COMPONENT_PARAMS),
        "combined_url": {"operation": "sparse horizontal concatenation", "parts": ["normalized_url_char", "url_component_text"]},
    }
    return build_semantic_payload(
        selection_type="url",
        benchmark_run_id=winner["benchmark_run_id"],
        input_identity=canonical_selection_input_identity(canonical_report, STEP21B),
        method_identity={
            "representations": representation_specs,
            "component_transformation": COMPONENT_SPEC,
            "models": model_specs(),
            "folds": list(FOLDS),
            "ranking_rule": winner["ranking_rule"],
        },
        fold_rows=fold_rows,
        summaries=summaries,
        winner=winner,
        winner_oof=winner_oof,
    )


def safe_value(value):
    if isinstance(value, dict):
        return {key: safe_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_value(item) for item in value]
    if value is np.float32:
        return "float32"
    return value


def run(root: Path = ROOT, *, jobs: int = 1) -> dict:
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    start = time.perf_counter()
    dev, counts = load_development(root)
    protected = check_protected(root)
    specs = model_specs()
    if len(specs) != 20:
        raise ValueError("Unexpected search space")
    def safe(value):
        if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)): return [safe(v) for v in value]
        if value is np.float32: return "float32"
        return value
    representation_specs = {"normalized_url_char": safe(CHAR_PARAMS), "url_component_text": safe(COMPONENT_PARAMS), "combined_url": {"operation": "sparse horizontal concatenation", "parts": ["normalized_url_char", "url_component_text"]}}
    search = {"representations": representation_specs, "component_transformation": COMPONENT_SPEC, "models": [(family, mid, params) for family, mid, params in specs]}
    canonical_report = json.loads(
        (root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8")
    )
    semantic_inputs = canonical_selection_input_identity(canonical_report, STEP21B)
    run_id = hashlib.sha256(json.dumps({"url_run_id": URL_RUN_ID, "inputs": semantic_inputs, "search": search}, sort_keys=True, default=str).encode()).hexdigest()[:32]
    urls = np.asarray([r["normalized_url"] for r in dev], dtype=object)
    components = np.asarray([component_text(url) for url in urls], dtype=object)
    labels = np.asarray([r["label"] for r in dev], dtype=object)
    folds = np.asarray([int(r["cv_fold"]) for r in dev])
    rows_by_config = defaultdict(list)
    oof_by_config = defaultdict(dict)
    family_seconds = defaultdict(float)
    representation_seconds = defaultdict(float)
    with parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
        with Parallel(max_nbytes="1M", mmap_mode="r") as parallel:
            for fold in FOLDS:
                valid_mask = folds == fold
                train_mask = ~valid_mask
                matrices, _ = vectorize_fold(urls, components, valid_mask)
                for representation in REPRESENTATIONS:
                    x_train, x_valid = matrices[representation]
                    if x_train.nnz and float(x_train.data.min()) < 0:
                        raise ValueError("Negative TF-IDF feature")
                    candidate_results = parallel(
                        delayed(_evaluate_candidate)(family, mid, params, x_train, x_valid, labels[train_mask], labels[valid_mask])
                        for family, mid, params in specs
                    )
                    for candidate in candidate_results:
                        family, mid = candidate["family"], candidate["model_id"]
                        cid = config_id(representation, family, mid)
                        elapsed = candidate["fit_seconds"] + candidate["predict_seconds"]
                        family_seconds[family] += elapsed
                        representation_seconds[representation] += elapsed
                        result = {"benchmark_run_id": run_id, "url_run_id": URL_RUN_ID, "representation": representation, "model_family": family, "model_config_id": mid, "config_id": cid, "fold": fold, "train_n": int(train_mask.sum()), "validation_n": int(valid_mask.sum()), "train_legitimate_n": int(np.sum(labels[train_mask] == "legitimate")), "train_phishing_n": int(np.sum(labels[train_mask] == "phishing")), "validation_legitimate_n": int(np.sum(labels[valid_mask] == "legitimate")), "validation_phishing_n": int(np.sum(labels[valid_mask] == "phishing")), **candidate["metrics"], "fit_seconds": candidate["fit_seconds"], "predict_seconds": candidate["predict_seconds"]}
                        rows_by_config[cid].append(result)
                        for index, prediction, value in zip(np.flatnonzero(valid_mask), candidate["prediction"], candidate["model_score"], strict=True):
                            oof_by_config[cid][int(index)] = {"benchmark_run_id": run_id, "url_run_id": URL_RUN_ID, "canonical_url_id": dev[int(index)]["canonical_url_id"], "label": dev[int(index)]["label"], "cv_fold": fold, "prediction": str(prediction), "model_score": float(value), "score_type": candidate["score_type"]}
                        print(f"fold={fold} {cid} f1={result['f1']:.6f}", flush=True)
                del matrices
    summaries = []
    leader, leader_oof = None, None
    for representation in REPRESENTATIONS:
        for family, mid, params in specs:
            cid = config_id(representation, family, mid)
            fold_rows = rows_by_config[cid]
            if len(fold_rows) != 5:
                raise ValueError("Incomplete candidate")
            summary = {"representation": representation, "model_family": family, "model_config_id": mid, "config_id": cid, "model_parameters": json.dumps(safe(make_model(family, params).get_params(deep=False)), sort_keys=True)}
            for key in ("accuracy", "precision", "recall", "f1"):
                summary[f"mean_{key}"] = float(np.mean([r[key] for r in fold_rows]))
            summary["std_f1"] = float(np.std([r["f1"] for r in fold_rows], ddof=0))
            for key in ("tn", "fp", "fn", "tp"):
                summary[f"total_{key}"] = sum(r[key] for r in fold_rows)
            summaries.append(summary)
            candidate_oof = [oof_by_config[cid][index] for index in range(len(dev))]
            leader, leader_oof = select_complete_candidate(leader, leader_oof, summary, candidate_oof, fold_rows)
            del oof_by_config[cid]
    summaries.sort(key=rank_key)
    for rank, summary in enumerate(summaries, 1):
        summary["rank"] = rank
        summary["selected"] = rank == 1
    if summaries[0]["config_id"] != leader["config_id"] or len(leader_oof) != len(dev):
        raise ValueError("Winner/OOF mismatch")
    fold_rows = sorted((r for group in rows_by_config.values() for r in group), key=lambda r: (r["config_id"], r["fold"]))
    method = current_contract(root).payload["method"]
    if len(summaries) != method["url_candidate_count"] or len(fold_rows) != method["url_fold_fit_count"]:
        raise ValueError("Incomplete benchmark")
    leader_oof.sort(key=lambda r: r["canonical_url_id"])
    chosen_params = next(params for family, mid, params in specs if family == leader["model_family"] and mid == leader["model_config_id"])
    winner = {"benchmark_run_id": run_id, "url_run_id": URL_RUN_ID, "config_id": leader["config_id"], "representation": leader["representation"], "representation_specification": representation_specs[leader["representation"]], "component_transformation": COMPONENT_SPEC, "model_family": leader["model_family"], "model_config_id": leader["model_config_id"], "model_parameters": safe(make_model(leader["model_family"], chosen_params).get_params(deep=False)), "cv_metrics": {key: leader[key] for key in ("mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "std_f1")}, "oof_confusion": {key: leader[f"total_{key}"] for key in ("tn", "fp", "fn", "tp")}, "ranking_rule": "descending unrounded mean F1, recall, precision; ascending population std F1; lexicographic config_id", "status": "Development CV only; URL Holdout not evaluated; no final model fit; no calibration; no threshold tuning; no fusion"}
    out = root / "outputs/benchmarks/url"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "fold_results.csv", FOLD_COLUMNS, fold_rows)
    write_csv(out / "config_summary.csv", SUMMARY_COLUMNS, summaries)
    write_csv(out / "winner_oof_predictions.csv", OOF_COLUMNS, leader_oof)
    (out / "winner.json").write_text(json.dumps(winner, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    semantic_payload = build_current_semantic_payload(root)
    selection_manifest = build_selection_manifest(
        payload=semantic_payload,
        input_paths={"canonical_dataset": "outputs/datasets/url/canonical/url_dataset.csv", "folds": "outputs/datasets/url/canonical/cv_assignments.csv"},
        output_paths={"fold_results": "outputs/benchmarks/url/fold_results.csv", "config_summary": "outputs/benchmarks/url/config_summary.csv", "winner": "outputs/benchmarks/url/winner.json", "winner_oof": "outputs/benchmarks/url/winner_oof_predictions.csv"},
    )
    (out / "selection_manifest.json").write_text(json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    flags = {"development_only_selection": True, "holdout_used_for_fit": False, "holdout_vectorized": False, "holdout_predicted": False, "holdout_metrics_computed": False, "final_model_fitted": False, "calibration_performed": False, "threshold_tuned": False, "fusion_performed": False, "external_data_used": False, "rq2_data_used": False, "source_metadata_used_as_model_feature": False, "representative_raw_url_used_as_model_feature": False, "outer_parallelism_used": jobs > 1, "random_forest_internal_parallelism": False, "step23_authorized": False}
    report = {"status": "completed", "benchmark_run_id": run_id, "url_run_id": URL_RUN_ID, "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "input_sha256": protected, "counts": counts, "representation_specifications": representation_specs, "component_transformation": COMPONENT_SPEC, "search_space": safe(search), "candidate_count": 60, "model_fit_count": 300, "ranking_rule": winner["ranking_rule"], "winner": winner, "cpu_count": os.cpu_count(), "parallelism": {"outer": "serial" if jobs == 1 else "loky", "jobs": jobs, "inner_threads": 1, "random_forest_runtime_n_jobs": 1}, "duration_seconds": time.perf_counter() - start, "model_family_seconds": dict(family_seconds), "representation_seconds": dict(representation_seconds), "outputs": {name: {"path": f"outputs/benchmarks/url/{name}", "sha256": sha256(out / name)} for name in ("fold_results.csv", "config_summary.csv", "winner_oof_predictions.csv", "winner.json", "selection_manifest.json")}, **flags}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    check_protected(root)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Development-only URL model selection")
    parser.add_argument("--jobs", type=int, default=1, help="Number of outer candidate workers (positive integer)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    result = run(jobs=args.jobs)
    print(json.dumps({"benchmark_run_id": result["benchmark_run_id"], "winner": result["winner"]["config_id"], "duration_seconds": result["duration_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
