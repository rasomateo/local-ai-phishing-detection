
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from src.utils import REPOSITORY_ROOT
from src.provenance import content_run_id, validate_artifact_inventory, validate_predecessor
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
DEV_COLUMNS = ["split_run_id", "record_id", "label", "cv_fold", "subject", "body"]
HOLDOUT_COLUMNS = ["split_run_id", "record_id", "label", "subject", "body"]
PREPROCESSING_COLUMNS = ["record_id", "dataset", "source_subset", "source_file", "source_record", "source_role", "split", "subject", "body", "label", "raw_label", "label_provenance", "collection_period", "inclusion_reason"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == columns, f"Unexpected CSV schema: {path}")
        return list(reader)


def validate(root: Path = REPOSITORY_ROOT) -> dict:
    contract = current_contract(root)
    out = root / "outputs" / "features" / "text"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    manifest_path = root / "outputs" / "splits" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    check(report["schema_version"] == 2 and report["status"] == "prepared", "Invalid report status/schema")
    check(report["split_run_id"] == run_id, "Split run mismatch")
    check(manifest["status"] == "validated", "Canonical split is not validated")
    predecessor = validate_predecessor(root, report["predecessor"])
    check(predecessor["run_id"] == run_id and predecessor["semantic_sha256"] == manifest["semantic_sha256"],
          "Text-input predecessor mismatch")
    check(report["vectorizer_fit"].startswith("none;"), "Vectorizer fit claim is invalid")
    check(report["legacy_tfidf_config"]["independent_precommitment_established"] is False,
          "Historical TF-IDF configuration was promoted without evidence")
    expected_paths = {
        "preprocessed": "outputs/preprocessed_dataset.csv",
        "split_manifest": "outputs/splits/manifest.json",
        "split_assignments": f"outputs/splits/runs/{run_id}/split_assignments.csv",
        "cv_fold_assignments": f"outputs/splits/runs/{run_id}/cv_fold_assignments.csv",
    }
    check(set(report["inputs"]) == set(expected_paths), "Input manifest incomplete")
    for key, relative in expected_paths.items():
        check(report["inputs"][key] == {"path": relative, "sha256": digest(root / relative)},
              f"Input provenance differs: {key}")
    check(manifest["inputs"]["text_development"]["sha256"] == digest(root / expected_paths["preprocessed"]),
          "Preprocessing differs from canonical split")
    for name, key in (("split_assignments.csv", "split_assignments"), ("cv_fold_assignments.csv", "cv_fold_assignments")):
        check(manifest["artifacts"][name]["sha256"] == report["inputs"][key]["sha256"],
              f"Canonical hash differs: {name}")
    preprocessed = rows(root / expected_paths["preprocessed"], PREPROCESSING_COLUMNS)
    splits = rows(root / expected_paths["split_assignments"], ["record_id", "label", "partition"])
    folds = rows(root / expected_paths["cv_fold_assignments"], ["record_id", "label", "cv_fold"])
    dev = rows(out / "development_inputs.csv", DEV_COLUMNS)
    holdout = rows(out / "holdout_inputs.csv", HOLDOUT_COLUMNS)
    for collection, label in ((preprocessed, "preprocessed"), (splits, "split"), (folds, "CV"), (dev, "development"), (holdout, "holdout")):
        identifiers = [row["record_id"] for row in collection]
        check(len(identifiers) == len(set(identifiers)), f"Duplicate {label} record_id")
    source = {row["record_id"]: row for row in preprocessed}
    split = {row["record_id"]: row for row in splits}
    fold = {row["record_id"]: row for row in folds}
    check(set(source) == set(split), "Split does not cover preprocessing")
    check(all(row["source_role"] == "text_development" for row in preprocessed), "Reserved role in preprocessing")
    dev_ids = {key for key, value in split.items() if value["partition"] == "development"}
    holdout_ids = {key for key, value in split.items() if value["partition"] == "final_internal_holdout"}
    check(dev_ids | holdout_ids == set(split) and not dev_ids & holdout_ids, "Invalid canonical partitions")
    check(set(fold) == dev_ids and {value["cv_fold"] for value in folds} == set("01234"), "Invalid canonical CV")
    check([row["record_id"] for row in dev] == sorted(dev_ids), "Development IDs/order differ")
    check([row["record_id"] for row in holdout] == sorted(holdout_ids), "Holdout IDs/order differ")
    for row in dev + holdout:
        record_id = row["record_id"]
        check(row["split_run_id"] == run_id, f"Run mismatch: {record_id}")
        check(row["label"] == split[record_id]["label"] == source[record_id]["label"], f"Label mismatch: {record_id}")
        check((row["subject"], row["body"]) == (source[record_id]["subject"], source[record_id]["body"]),
              f"Text mismatch: {record_id}")
        if "cv_fold" in row:
            check(row["cv_fold"] == fold[record_id]["cv_fold"] and row["label"] == fold[record_id]["label"],
                  f"CV mismatch: {record_id}")
    for path in (root / "outputs" / "dataset_roles").glob("*.csv"):
        if path.name in {"dideco_annotations.csv", "phishload_url_only.csv", "trec_excluded.csv"}:
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "record_id" in reader.fieldnames and "source_role" in reader.fieldnames:
                check(not any(row["record_id"] in split for row in reader if row["source_role"] in
                              {"temporal_evaluation", "external_evaluation", "rq2_evaluation"}),
                      f"Reserved ID in Step-17 input: {path}")
    check(len(dev) == contract.expected("text_development_count", 201016) and
          len(holdout) == contract.expected("text_holdout_count", 50255), "Partition count differs")
    expected_counts = {
        "development": len(dev), "final_internal_holdout": len(holdout),
        "development_by_label": dict(sorted(Counter(row["label"] for row in dev).items())),
        "holdout_by_label": dict(sorted(Counter(row["label"] for row in holdout).items())),
        "cv_validation_folds": {str(i): dict(sorted(Counter(row["label"] for row in dev if row["cv_fold"] == str(i)).items())) for i in range(5)},
    }
    check(report["counts"] == expected_counts, "Report count mismatch")
    check(set(report["artifacts"]) == {"development_inputs.csv", "holdout_inputs.csv"}, "Artifact list mismatch")
    for name, spec in report["artifacts"].items():
        check(spec == {"path": f"outputs/features/text/{name}", "sha256": digest(out / name)},
              f"Artifact hash/path mismatch: {name}")
    validate_artifact_inventory(root, report["artifacts"])
    semantic_payload = {
        "contract_version": "artifact-provenance-v1:text-inputs-v1",
        "split_semantic_sha256": manifest["semantic_sha256"],
        "normalization": "Step-14 normalize_text_field",
        "views": report["views"],
        "counts": {"development": len(dev), "final_internal_holdout": len(holdout)},
    }
    _identity, expected_semantic = content_run_id("text-inputs", semantic_payload)
    check(report["semantic_sha256"] == expected_semantic, "Text-input semantic fingerprint mismatch")
    return {"split_run_id": run_id, "counts": expected_counts, "status": "validated"}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
