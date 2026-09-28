from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from src.selection_contract import read_selection_manifest, validate_selection_manifest
from src.url_model_selection import build_current_semantic_payload
from src.execution_contract import current_contract

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("normalized_url_char", "url_component_text", "combined_url")
EXPECTED_FOLDS = {"0": (953, 157, 796), "1": (762, 187, 575), "2": (717, 200, 517), "3": (771, 223, 548), "4": (798, 181, 617)}


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)


def scores(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    return {"accuracy": (tn + tp) / n, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def ranking(row):
    return (-float(row["mean_f1"]), -float(row["mean_recall"]), -float(row["mean_precision"]), float(row["std_f1"]), row["config_id"])


def expected_configs():
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
    return {f"{view}|{family}|{mid}" for view in VIEWS for family, mid in specs}


def validate(root: Path = ROOT):
    contract = current_contract(root)
    canonical_report = json.loads((root / "outputs/datasets/url/canonical/report.json").read_text(encoding="utf-8"))
    RUN_ID = canonical_report["url_run_id"]
    STEP21B = {name: spec["sha256"] for name, spec in canonical_report["outputs"].items()}
    STEP21B["report.json"] = sha(root / "outputs/datasets/url/canonical/report.json")
    base = root / "outputs/datasets/url/canonical"
    out = root / "outputs/benchmarks/url"
    for name, digest in STEP21B.items():
        check(sha(base / name) == digest, f"Step-21B hash: {name}")
    prior = json.loads((base / "report.json").read_text(encoding="utf-8"))
    protected = {f"outputs/datasets/url/canonical/{name}": digest for name, digest in STEP21B.items()}
    protected.update(prior["protected_sha256"])
    for name, digest in protected.items():
        check(sha(root / name) == digest, f"Protected hash: {name}")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    winner = json.loads((out / "winner.json").read_text(encoding="utf-8"))
    check(report["status"] == "completed" and report["url_run_id"] == winner["url_run_id"] == RUN_ID, "Run/status")
    check(report["input_sha256"] == protected, "Input manifest")
    for name in ("fold_results.csv", "config_summary.csv", "winner_oof_predictions.csv", "winner.json", "selection_manifest.json"):
        check(report["outputs"][name] == {"path": f"outputs/benchmarks/url/{name}", "sha256": sha(out / name)}, f"Output hash: {name}")
    check(report["development_only_selection"] is True and report["random_forest_internal_parallelism"] is False, "Method flags")
    false_flags = ("holdout_used_for_fit", "holdout_vectorized", "holdout_predicted", "holdout_metrics_computed", "final_model_fitted", "calibration_performed", "threshold_tuned", "fusion_performed", "external_data_used", "rq2_data_used", "source_metadata_used_as_model_feature", "representative_raw_url_used_as_model_feature", "step23_authorized")
    check(all(report[k] is False for k in false_flags), "Method flags")
    parallelism = report["parallelism"]
    check(parallelism["outer"] in {"serial", "loky"} and int(parallelism["jobs"]) >= 1
          and parallelism["inner_threads"] == 1 and parallelism["random_forest_runtime_n_jobs"] == 1,
          "Parallelism")
    check(report["outer_parallelism_used"] is (parallelism["jobs"] > 1), "Outer parallelism flag")
    check(set(report["representation_specifications"]) == set(VIEWS), "Representations")
    check(len(report["search_space"]["models"]) == 20, "Model search count")
    reported_models = {(family, mid): params for family, mid, params in report["search_space"]["models"]}
    check(len(reported_models) == 20 and {f"{view}|{family}|{mid}" for view in VIEWS for family, mid in reported_models} == expected_configs(), "Model search identities")
    for (family, mid), params in reported_models.items():
        fields = dict(part.split("=", 1) for part in mid.split(";"))
        if family == "LinearSVC":
            check(params == {"C": float(fields["C"]), "class_weight": None if fields["class_weight"] == "none" else "balanced", "random_state": 42}, "LinearSVC search parameters")
        elif family == "MultinomialNB":
            check(params == {"alpha": float(fields["alpha"]), "fit_prior": fields["fit_prior"] == "true"}, "NB search parameters")
        elif family == "DecisionTreeClassifier":
            check(params == {"max_depth": None if fields["max_depth"] == "none" else 50, "class_weight": None if fields["class_weight"] == "none" else "balanced", "random_state": 42}, "Tree search parameters")
        else:
            check(params == {"n_estimators": 300, "max_depth": None if fields["max_depth"] == "none" else 50, "class_weight": None if fields["class_weight"] == "none" else "balanced_subsample", "random_state": 42, "n_jobs": -1}, "Forest search parameters")
    spec_text = json.dumps(report["representation_specifications"]).lower() + report["component_transformation"].lower()
    for forbidden in ("urlbasedomain", "phishtank", "isphish", "leakage_group", "representative_raw_url", "source_id", "scanned", "created", "parent"):
        check(forbidden not in spec_text, f"Source feature: {forbidden}")
    split = rows(base / "split_assignments.csv")
    cv = rows(base / "cv_assignments.csv")
    expected_total = contract.expected("url_canonical_count", 4883)
    check(len(split) == len({r["canonical_url_id"] for r in split}) == expected_total, "Split IDs")
    dev = {r["canonical_url_id"]: r for r in split if r["split"] == "development"}
    hold = {r["canonical_url_id"]: r for r in split if r["split"] == "holdout"}
    expected_dev = len(dev) if contract.is_fixture else contract.expected("url_development_count", 4001)
    expected_hold = len(hold) if contract.is_fixture else contract.expected("url_holdout_count", 882)
    expected_dev_labels = Counter(r["label"] for r in dev.values()) if contract.is_fixture else Counter(contract.expected("url_development_labels", {"legitimate": 948, "phishing": 3053}))
    expected_hold_labels = Counter(r["label"] for r in hold.values()) if contract.is_fixture else Counter(contract.expected("url_holdout_labels", {"legitimate": 228, "phishing": 654}))
    check(len(dev) == expected_dev and len(hold) == expected_hold and not set(dev) & set(hold), "Partitions")
    check(Counter(r["label"] for r in dev.values()) == expected_dev_labels, "Development labels")
    check(Counter(r["label"] for r in hold.values()) == expected_hold_labels, "Holdout labels")
    cv_map = {r["canonical_url_id"]: r for r in cv}
    check(len(cv) == len(cv_map) == expected_dev and set(cv_map) == set(dev), "CV IDs")
    expected_folds = ({fold: (sum(r["cv_fold"] == fold for r in cv), sum(r["cv_fold"] == fold and r["label"] == "legitimate" for r in cv), sum(r["cv_fold"] == fold and r["label"] == "phishing" for r in cv)) for fold in "01234"} if contract.is_fixture else EXPECTED_FOLDS)
    for fold, (total, legitimate, phishing) in expected_folds.items():
        fold_rows = [r for r in cv if r["cv_fold"] == fold]
        check((len(fold_rows), sum(r["label"] == "legitimate" for r in fold_rows), sum(r["label"] == "phishing" for r in fold_rows)) == (total, legitimate, phishing), f"Fold {fold}")
        training = {r["canonical_url_id"] for r in cv if r["cv_fold"] != fold}
        validation = {r["canonical_url_id"] for r in fold_rows}
        check(not training & validation and len(training) + len(validation) == expected_dev, f"Fold isolation: {fold}")
    group_folds = defaultdict(set)
    for r in cv:
        check((r["label"], r["leakage_group"]) == (dev[r["canonical_url_id"]]["label"], dev[r["canonical_url_id"]]["leakage_group"]), "CV assignments")
        group_folds[r["leakage_group"]].add(r["cv_fold"])
    check(all(len(value) == 1 for value in group_folds.values()), "Group crossing")
    summary = rows(out / "config_summary.csv")
    folds = rows(out / "fold_results.csv")
    oof = rows(out / "winner_oof_predictions.csv")
    configs = expected_configs()
    check(len(configs) == len(summary) == 60 and len(folds) == 300, "Candidate/fit count")
    check({r["config_id"] for r in summary} == configs and len({r["config_id"] for r in summary}) == 60, "Search space")
    by_config = defaultdict(list)
    seen = set()
    for row in folds:
        cid, fold = row["config_id"], row["fold"]
        check(cid in configs and fold in expected_folds and (cid, fold) not in seen, "Duplicate/unexpected fold row")
        seen.add((cid, fold))
        check(row["benchmark_run_id"] == report["benchmark_run_id"] and row["url_run_id"] == RUN_ID, "Fold run IDs")
        check(cid == f"{row['representation']}|{row['model_family']}|{row['model_config_id']}", "Config fields")
        n, legitimate, phishing = expected_folds[fold]
        check((int(row["validation_n"]), int(row["validation_legitimate_n"]), int(row["validation_phishing_n"])) == (n, legitimate, phishing), "Validation support")
        check((int(row["train_n"]), int(row["train_legitimate_n"]), int(row["train_phishing_n"])) == (expected_dev - n, expected_dev_labels["legitimate"] - legitimate, expected_dev_labels["phishing"] - phishing), "Training support")
        tn, fp, fn, tp = (int(row[key]) for key in ("tn", "fp", "fn", "tp"))
        check(min(tn, fp, fn, tp) >= 0 and tn + fp == legitimate and fn + tp == phishing, "Confusion support")
        for key, expected in scores(tn, fp, fn, tp).items():
            check(close(row[key], expected), f"Fold metric: {key}")
        check(float(row["fit_seconds"]) >= 0 and float(row["predict_seconds"]) >= 0, "Timings")
        by_config[cid].append(row)
    by_summary = {r["config_id"]: r for r in summary}
    for cid, group in by_config.items():
        check(len(group) == 5 and {r["fold"] for r in group} == set(expected_folds), "Incomplete candidate")
        item = by_summary[cid]
        check(item["config_id"] == f"{item['representation']}|{item['model_family']}|{item['model_config_id']}", "Summary config")
        params = json.loads(item["model_parameters"])
        check(all(params[key] == value for key, value in reported_models[(item["model_family"], item["model_config_id"])].items()), "Summary model parameters")
        if item["model_family"] == "RandomForestClassifier":
            check(params["n_jobs"] == -1 and params["n_estimators"] == 300, "Forest parameters")
        for key in ("accuracy", "precision", "recall", "f1"):
            check(close(item[f"mean_{key}"], sum(float(r[key]) for r in group) / 5), f"Mean {key}")
        mean = sum(float(r["f1"]) for r in group) / 5
        check(close(item["std_f1"], math.sqrt(sum((float(r["f1"]) - mean) ** 2 for r in group) / 5)), "F1 std")
        for key in ("tn", "fp", "fn", "tp"):
            check(int(item[f"total_{key}"]) == sum(int(r[key]) for r in group), f"Total {key}")
    ordered = sorted(summary, key=ranking)
    for index, item in enumerate(ordered, 1):
        check(int(item["rank"]) == index and item["selected"] == str(index == 1), "Rank/selection")
    best = ordered[0]
    check(winner["config_id"] == best["config_id"] and winner["representation"] == best["representation"] and winner["model_family"] == best["model_family"] and winner["model_config_id"] == best["model_config_id"], "Winner identity")
    check(winner["model_parameters"] == json.loads(best["model_parameters"]), "Winner parameters")
    for key in ("mean_accuracy", "mean_precision", "mean_recall", "mean_f1", "std_f1"):
        check(close(winner["cv_metrics"][key], float(best[key])), f"Winner {key}")
    check(len(oof) == len({r["canonical_url_id"] for r in oof}) == expected_dev and {r["canonical_url_id"] for r in oof} == set(dev), "OOF coverage/Holdout isolation")
    confusion = defaultdict(Counter)
    for row in oof:
        rid = row["canonical_url_id"]
        check(row["benchmark_run_id"] == report["benchmark_run_id"] and row["url_run_id"] == RUN_ID and row["label"] == dev[rid]["label"] and row["cv_fold"] == cv_map[rid]["cv_fold"], "OOF assignments")
        check(row["prediction"] in ("legitimate", "phishing") and math.isfinite(float(row["model_score"])), "OOF values")
        expected_type = "decision_function" if best["model_family"] == "LinearSVC" else "positive_class_probability"
        check(row["score_type"] == expected_type, "OOF score type")
        confusion[row["cv_fold"]][(row["label"], row["prediction"])] += 1
    totals = Counter()
    for fold in expected_folds:
        c = confusion[fold]
        values = {"tn": c[("legitimate", "legitimate")], "fp": c[("legitimate", "phishing")], "fn": c[("phishing", "legitimate")], "tp": c[("phishing", "phishing")]}
        expected = next(r for r in by_config[best["config_id"]] if r["fold"] == fold)
        check(all(values[k] == int(expected[k]) for k in values), f"OOF fold confusion: {fold}")
        totals.update(values)
    check(dict(totals) == winner["oof_confusion"], "OOF total confusion")
    check(report["winner"] == winner and report["candidate_count"] == 60 and report["model_fit_count"] == 300, "Report winner/counts")
    manifest = read_selection_manifest(out / "selection_manifest.json")
    check(manifest["input_paths"] == {"canonical_dataset": "outputs/datasets/url/canonical/url_dataset.csv",
                                      "folds": "outputs/datasets/url/canonical/cv_assignments.csv"},
          "Selection manifest input paths differ")
    check(manifest["output_paths"] == {"config_summary": "outputs/benchmarks/url/config_summary.csv",
                                       "fold_results": "outputs/benchmarks/url/fold_results.csv",
                                       "winner": "outputs/benchmarks/url/winner.json",
                                       "winner_oof": "outputs/benchmarks/url/winner_oof_predictions.csv"},
          "Selection manifest output paths differ")
    semantic = validate_selection_manifest(manifest, build_current_semantic_payload(root))
    return {"status": "validated", "benchmark_run_id": report["benchmark_run_id"], "winner": best["config_id"], "semantic_sha256": semantic, "selection_manifest_sha256": sha(out / "selection_manifest.json")}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
