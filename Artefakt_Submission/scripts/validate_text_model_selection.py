from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.utils import REPOSITORY_ROOT
from src.selection_contract import read_selection_manifest, validate_selection_manifest
from src.text_model_selection import build_current_semantic_payload, current_input_identity
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
NAMES = ("fold_results.csv", "config_summary.csv", "winner.json", "winner_oof_predictions.csv", "selection_manifest.json")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def metrics(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    return {"accuracy": (tn + tp) / n, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def key(row):
    return (-float(row["mean_f1"]), -float(row["mean_recall"]), -float(row["mean_precision"]), float(row["std_f1"]), row["config_id"])


def expected_configs():
    result = set()
    specs = []
    for c in (0.5, 1.0, 2.0):
        for weight in ("none", "balanced"):
            specs.append(("LinearSVC", f"C={c:g};class_weight={weight}"))
    for alpha in (0.1, 0.5, 1.0):
        for prior in ("true", "false"):
            specs.append(("MultinomialNB", f"alpha={alpha:g};fit_prior={prior}"))
    for depth in ("50", "none"):
        for weight in ("none", "balanced"):
            specs.append(("DecisionTreeClassifier", f"max_depth={depth};class_weight={weight}"))
    for depth in ("50", "none"):
        for weight in ("none", "balanced_subsample"):
            specs.append(("RandomForestClassifier", f"max_depth={depth};class_weight={weight}"))
    for view in ("subject", "body", "subject_body"):
        for tfidf in ("TFIDF_A", "TFIDF_B"):
            for family, model_id in specs:
                result.add((f"{view}|{tfidf}|{family}|{model_id}", view, tfidf, family, model_id))
    return result


def validate(root: Path = REPOSITORY_ROOT):
    contract = current_contract(root)
    expected_total = contract.expected("text_development_count", 201016)
    expected_labels = Counter(contract.expected("text_development_labels", {"legitimate": 198800, "phishing": 2216}))
    expected_fold_counts = {str(key): tuple(value) for key, value in contract.expected("text_fold_counts", {"0": [40204, 444], "1": [40203, 443], "2": [40203, 443], "3": [40203, 443], "4": [40203, 443]}).items()}
    identity = current_input_identity(root)
    RUN = identity["split_run_id"]
    DEV_SHA = identity["development_sha256"]
    HOLD_SHA = identity["holdout_integrity_sha256"]
    PROTECTED = dict(identity)
    out = root / "outputs/benchmarks/text"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    winner = json.loads((out / "winner.json").read_text(encoding="utf-8"))
    check(report["protected_sha256"] == PROTECTED, "Protected manifest differs")
    check(report["split_run_id"] == winner["split_run_id"] == RUN, "Split run mismatch")
    check(report["inputs"]["development"]["sha256"] == DEV_SHA and report["inputs"]["holdout_integrity_only"]["sha256"] == HOLD_SHA, "Input report hash mismatch")
    check(
        report["inputs"]["step17_report"]["sha256"]
        == sha(root / "outputs/features/text/report.json"),
        "Step-17 report mismatch",
    )
    for name in NAMES:
        check(report["outputs"][name] == {"path": f"outputs/benchmarks/text/{name}", "sha256": sha(out / name)}, f"Output hash mismatch: {name}")
    check(set(report["outputs"]) == set(NAMES), "Unexpected output manifest")
    for flag in ("holdout_accessed_for_metrics", "external_data_used", "threshold_tuned", "calibration_performed", "model_artifact_frozen"):
        check(report[flag] is False, f"Unexpected methodological flag: {flag}")
    dev = rows(root / "outputs/features/text/development_inputs.csv")
    check(len(dev) == expected_total, "Development count mismatch")
    by_id = {r["record_id"]: r for r in dev}
    check(len(by_id) == len(dev), "Duplicate Development IDs")
    check(Counter(r["label"] for r in dev) == expected_labels, "Development label counts")
    check({r["cv_fold"] for r in dev} == set("01234"), "Invalid Development folds")
    for fold, (n, pos) in expected_fold_counts.items():
        group = [r for r in dev if r["cv_fold"] == fold]
        check(len(group) == n and sum(r["label"] == "phishing" for r in group) == pos, f"Fold {fold} counts")
    splits = rows(root / f"outputs/splits/runs/{RUN}/split_assignments.csv")
    holdout_ids = {r["record_id"] for r in splits if r["partition"] == "final_internal_holdout"}
    check(not set(by_id) & holdout_ids, "Holdout ID in Development")
    canonical_dev = {r["record_id"]: r["label"] for r in splits if r["partition"] == "development"}
    check(set(by_id) == set(canonical_dev) and all(r["label"] == canonical_dev[rid] for rid, r in by_id.items()), "Canonical Development mismatch")
    folds = rows(root / f"outputs/splits/runs/{RUN}/cv_fold_assignments.csv")
    canonical_folds = {r["record_id"]: (r["label"], r["cv_fold"]) for r in folds}
    check(len(canonical_folds) == len(dev) and all((r["label"], r["cv_fold"]) == canonical_folds[rid] for rid, r in by_id.items()), "Canonical fold mismatch")
    fold_results = rows(out / "fold_results.csv")
    summary = rows(out / "config_summary.csv")
    oof = rows(out / "winner_oof_predictions.csv")
    configs = expected_configs()
    check(len(configs) == 120 and len(summary) == 120 and len(fold_results) == 600, "Candidate/fold count mismatch")
    check({tuple(r[field] for field in ("config_id", "view", "tfidf_config_id", "model_family", "model_config_id")) for r in summary} == configs, "Candidate search space differs")
    groups = defaultdict(list)
    seen = set()
    run_id = report["benchmark_run_id"]
    for row in fold_results:
        cid = row["config_id"]
        fold = row["fold"]
        check((cid, row["view"], row["tfidf_config_id"], row["model_family"], row["model_config_id"]) in configs, "Unexpected fold config")
        check(fold in set("01234") and (cid, fold) not in seen, "Duplicate/invalid config fold")
        seen.add((cid, fold))
        check(row["benchmark_run_id"] == run_id and row["split_run_id"] == RUN, "Fold provenance mismatch")
        expected_n, expected_pos = expected_fold_counts[fold]
        check(int(row["validation_n"]) == expected_n and int(row["validation_phishing_n"]) == expected_pos and int(row["train_n"]) == expected_total - expected_n and int(row["train_phishing_n"]) == expected_labels["phishing"] - expected_pos, "Fold support mismatch")
        tn, fp, fn, tp = (int(row[k]) for k in ("tn", "fp", "fn", "tp"))
        check(min(tn, fp, fn, tp) >= 0 and tn + fp == expected_n - expected_pos and fn + tp == expected_pos, "Fold confusion counts mismatch")
        for metric, value in metrics(tn, fp, fn, tp).items():
            check(near(row[metric], value), f"Fold {metric} mismatch")
        check(float(row["fit_seconds"]) >= 0 and float(row["predict_seconds"]) >= 0, "Invalid timings")
        groups[cid].append(row)
    check(all(len(group) == 5 for group in groups.values()) and len(groups) == 120, "Incomplete config folds")
    summary_by_id = {r["config_id"]: r for r in summary}
    check(len(summary_by_id) == 120, "Duplicate summary config")
    for cid, group in groups.items():
        row = summary_by_id[cid]
        for metric in ("accuracy", "precision", "recall", "f1"):
            check(near(row[f"mean_{metric}"], sum(float(r[metric]) for r in group) / 5), f"Summary {metric} mismatch")
        mean = sum(float(r["f1"]) for r in group) / 5
        std = math.sqrt(sum((float(r["f1"]) - mean) ** 2 for r in group) / 5)
        check(near(row["std_f1"], std), "Summary std mismatch")
        for metric in ("tp", "tn", "fp", "fn"):
            check(int(row[f"total_{metric}"]) == sum(int(r[metric]) for r in group), f"Summary {metric} mismatch")
    ordered = sorted(summary, key=key)
    for rank, row in enumerate(ordered, 1):
        check(int(row["rank"]) == rank and row["selected"] == str(rank == 1), "Ranking mismatch")
    best = ordered[0]
    check(winner["config_id"] == best["config_id"] and winner["cv_metrics"]["config_id"] == best["config_id"], "Winner differs")
    check(report["winner"] == winner and report["candidate_count"] == 120 and report["model_fit_count"] == 600, "Report winner/search mismatch")
    parallelism = report.get("parallelism", {})
    check(parallelism.get("backend") in {"serial", "loky"} and int(parallelism.get("jobs", 0)) >= 1
          and parallelism.get("inner_threads") == 1 and parallelism.get("random_forest_runtime_n_jobs") == 1,
          "Parallelism contract mismatch")
    check(tuple(report["search_space"]["views"]) == ("subject", "body", "subject_body"), "View search differs")
    check(set(report["search_space"]["tfidf"]) == {"TFIDF_A", "TFIDF_B"}, "TF-IDF search differs")
    for tid, ngrams in (("TFIDF_A", [1, 1]), ("TFIDF_B", [1, 2])):
        params = report["search_space"]["tfidf"][tid]
        check(params == {"analyzer": "word", "ngram_range": ngrams, "token_pattern": r"(?u)\b\w\w+\b", "lowercase": False, "sublinear_tf": True, "min_df": 2, "max_df": 0.95, "max_features": 30000, "dtype": "float32"}, "TF-IDF parameters differ")
    check(len(report["search_space"]["models"]) == 20, "Model search differs")
    check(winner["view"] == best["view"] and winner["tfidf_config_id"] == best["tfidf_config_id"] and winner["model_family"] == best["model_family"] and winner["model_config_id"] == best["model_config_id"], "Winner identity mismatch")
    check(winner["tfidf_parameters"] == report["search_space"]["tfidf"][winner["tfidf_config_id"]], "Winner TF-IDF mismatch")
    chosen_models = [params for family, mid, params in report["search_space"]["models"] if (family, mid) == (winner["model_family"], winner["model_config_id"])]
    check(len(chosen_models) == 1 and all(winner["model_parameters"][name] == value for name, value in chosen_models[0].items()), "Winner model parameters mismatch")
    for metric in ("accuracy", "precision", "recall", "f1"):
        check(near(winner["cv_metrics"][f"mean_{metric}"], best[f"mean_{metric}"]), "Winner metric mismatch")
    check(len(oof) == len(dev), "OOF count mismatch")
    oof_ids = [r["record_id"] for r in oof]
    check(len(set(oof_ids)) == len(dev) and set(oof_ids) == set(by_id), "OOF IDs incomplete/duplicated")
    confusion = defaultdict(lambda: Counter())
    for row in oof:
        source = by_id[row["record_id"]]
        check(row["benchmark_run_id"] == run_id and row["split_run_id"] == RUN and row["label"] == source["label"] and row["cv_fold"] == source["cv_fold"], "OOF source/fold mismatch")
        check(row["prediction"] in {"legitimate", "phishing"} and math.isfinite(float(row["model_score"])), "OOF prediction/score invalid")
        expected_type = "decision_function" if best["model_family"] == "LinearSVC" else "positive_class_probability"
        check(row["score_type"] == expected_type, "OOF score type mismatch")
        if expected_type == "positive_class_probability":
            check(0 <= float(row["model_score"]) <= 1, "OOF probability out of range")
        confusion[row["cv_fold"]][(row["label"], row["prediction"])] += 1
    for fold in "01234":
        fold_result = next(r for r in groups[best["config_id"]] if r["fold"] == fold)
        for name, pair in {"tn": ("legitimate", "legitimate"), "fp": ("legitimate", "phishing"), "fn": ("phishing", "legitimate"), "tp": ("phishing", "phishing")}.items():
            check(confusion[fold][pair] == int(fold_result[name]), "OOF confusion differs from winner fold")
    manifest = read_selection_manifest(out / "selection_manifest.json")
    check(manifest["input_paths"] == {"development": "outputs/features/text/development_inputs.csv",
                                      "folds": f"outputs/splits/runs/{RUN}/cv_fold_assignments.csv"},
          "Selection manifest input paths differ")
    check(manifest["output_paths"] == {"config_summary": "outputs/benchmarks/text/config_summary.csv",
                                       "fold_results": "outputs/benchmarks/text/fold_results.csv",
                                       "winner": "outputs/benchmarks/text/winner.json",
                                       "winner_oof": "outputs/benchmarks/text/winner_oof_predictions.csv"},
          "Selection manifest output paths differ")
    semantic = validate_selection_manifest(manifest, build_current_semantic_payload(root))
    return {"status": "validated", "benchmark_run_id": run_id, "winner": best["config_id"], "semantic_sha256": semantic, "selection_manifest_sha256": sha(out / "selection_manifest.json"), "report_sha256": sha(out / "report.json"), "oof_confusion": {name: sum(int(r[name]) for r in groups[best["config_id"]]) for name in ("tn", "fp", "fn", "tp")}}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
