
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src.split_data import (
    EXCLUSION_COLUMNS, FOLD_COLUMNS, HOLDOUT_FRACTION, LABELS, SPLIT_CONTRACT_VERSION,
    SEED, SPLIT_COLUMNS, counts, make_assignments, scan_inputs, sha256_file,
    split_semantic_payload,
)
from src.provenance import content_run_id
from src.utils import (
    NAZARIO_2025_TEMPORAL_OUTPUT_PATH, PREPROCESSED_OUTPUT_PATH, REPOSITORY_ROOT,
    SPLITS_MANIFEST_OUTPUT_PATH, SPLITS_OUTPUT_DIR,
)

csv.field_size_limit(sys.maxsize)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_assignment(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == columns, f"Unexpected assignment schema in {path}")
        return list(reader)


def validate_run(
    run_dir: Path,
    manifest: dict,
    *,
    development_path: Path = PREPROCESSED_OUTPUT_PATH,
    temporal_path: Path = NAZARIO_2025_TEMPORAL_OUTPUT_PATH,
    check_input_hashes: bool = True,
) -> dict:
    require(manifest.get("schema_version") == 3 and manifest.get("status") == "validated", "Invalid manifest status/schema")
    require(manifest.get("contract_version") == SPLIT_CONTRACT_VERSION, "Invalid split contract version")
    parameters = manifest.get("parameters", {})
    require(parameters == {
        "seed": SEED, "holdout_fraction": HOLDOUT_FRACTION,
        "stratification_field": "label", "cv_fold_count": 5,
        "cv_shuffle": True, "input_order": "record_id_ascending",
    }, "Split/CV parameters differ from Step-15 contract")
    pending = manifest.get("pending_external_leakage_checks")
    require(isinstance(pending, list) and len(pending) == 3 and
            all(any(name in item and "exact" in item for item in pending)
                for name in ("DIDECO", "E-PhishLLM", "PhishingPot")), "Missing pending external exact checks")
    require(all(manifest.get("invariants", {}).values()), "Manifest contains a failed invariant")
    require(set(manifest.get("artifacts", {})) == {
        "split_assignments.csv", "cv_fold_assignments.csv", "temporal_exclusions.csv"
    }, "Incomplete artifact list")
    if check_input_hashes:
        require(manifest["inputs"]["text_development"]["path"] == "outputs/preprocessed_dataset.csv", "Development input path mismatch")
        require(manifest["inputs"]["nazario_2025_temporal_evaluation"]["path"] == "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv", "Temporal input path mismatch")
        require(manifest["inputs"]["text_development"]["sha256"] == sha256_file(development_path), "Development input hash mismatch")
        require(manifest["inputs"]["nazario_2025_temporal_evaluation"]["sha256"] == sha256_file(temporal_path), "Temporal input hash mismatch")
    for name, spec in manifest["artifacts"].items():
        require(spec["sha256"] == sha256_file(run_dir / name), f"Artifact hash mismatch: {name}")
        require(spec["path"] == f"{manifest['run_directory']}/{name}", f"Artifact path mismatch: {name}")

    splits = read_assignment(run_dir / "split_assignments.csv", SPLIT_COLUMNS)
    folds = read_assignment(run_dir / "cv_fold_assignments.csv", FOLD_COLUMNS)
    exclusions = read_assignment(run_dir / "temporal_exclusions.csv", EXCLUSION_COLUMNS)
    eligible, expected_exclusions, audit = scan_inputs(development_path, temporal_path)
    expected_splits, expected_folds, assignment_counts = make_assignments(eligible)
    require(splits == expected_splits, "Split assignments differ from deterministic eligible-record split")
    require(folds == expected_folds, "CV assignments differ from deterministic Development-only folds")
    require(exclusions == expected_exclusions, "Temporal exclusions differ from exact overlap audit")
    require(manifest.get("counts") == {**audit, **assignment_counts}, "Manifest counts disagree with CSV/input counts")
    payload = split_semantic_payload(
        {
            "text_development": sha256_file(development_path),
            "nazario_2025_temporal_evaluation": sha256_file(temporal_path),
        }, audit, assignment_counts,
    )
    expected_run_id, expected_fingerprint = content_run_id("text-split", payload)
    require(manifest.get("semantic_sha256") == expected_fingerprint, "Split semantic fingerprint mismatch")
    require(manifest.get("run_id") == expected_run_id, "Split run ID is not content-addressed")

    eligible_ids = {row["record_id"] for row in eligible}
    split_ids = {row["record_id"] for row in splits}
    development_ids = {row["record_id"] for row in splits if row["partition"] == "development"}
    holdout_ids = {row["record_id"] for row in splits if row["partition"] == "final_internal_holdout"}
    fold_ids = {row["record_id"] for row in folds}
    require(len(split_ids) == len(splits) == len(eligible_ids), "Split IDs are not unique or complete")
    require(split_ids == eligible_ids and not development_ids & holdout_ids, "Split coverage/disjointness failed")
    require(len(fold_ids) == len(folds) == len(development_ids) and fold_ids == development_ids, "CV coverage or holdout isolation failed")
    require({row["cv_fold"] for row in folds} == {"0", "1", "2", "3", "4"}, "CV fold set is incomplete")
    require({row["label"] for row in splits} == LABELS, "Split labels are incomplete")
    for fold in range(5):
        fold_rows = [row for row in folds if row["cv_fold"] == str(fold)]
        require(all(counts(fold_rows)["by_label"][label] > 0 for label in LABELS), f"Fold {fold} lacks a class")
    for label in LABELS:
        per_fold = [counts([row for row in folds if row["cv_fold"] == str(fold)])["by_label"][label] for fold in range(5)]
        require(max(per_fold) - min(per_fold) <= 1, f"Fold {label} distribution is not stratified")
    chain = manifest["counts"]
    require(
        chain["original_text_development"]["total"] - chain["temporal_exclusions"]["development_records"]
        == chain["split_eligible_corpus"]["total"]
        == chain["development"]["total"] + chain["final_internal_holdout"]["total"],
        "Count chain does not reconcile",
    )
    return {"run_id": manifest["run_id"], "status": "passed", "split_records": len(splits), "development_records": len(folds), "temporal_excluded_records": chain["temporal_exclusions"]["development_records"]}


def main() -> dict:
    manifest = json.loads(SPLITS_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    run_rel = Path(manifest["run_directory"])
    run_dir = (REPOSITORY_ROOT / run_rel).resolve()
    runs_root = (SPLITS_OUTPUT_DIR / "runs").resolve()
    require(run_dir.parent == runs_root and run_dir.name == manifest["run_id"], "Canonical manifest references an invalid run directory")
    require(json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) == manifest, "Run manifest differs from canonical manifest")
    return validate_run(run_dir, manifest)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2), flush=True)
