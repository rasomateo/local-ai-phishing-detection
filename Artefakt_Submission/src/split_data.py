
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import normalize_text_field
from src.provenance import CONTRACT_VERSION, content_run_id, semantic_fingerprint
from src.utils import (
    NAZARIO_2025_TEMPORAL_OUTPUT_PATH,
    PREPROCESSED_OUTPUT_PATH,
    PREPROCESSING_REPORT_OUTPUT_PATH,
    REPOSITORY_ROOT,
    SPLITS_MANIFEST_OUTPUT_PATH,
    SPLITS_OUTPUT_DIR,
)

csv.field_size_limit(sys.maxsize)
SEED = 42
HOLDOUT_FRACTION = 0.20
FOLD_COUNT = 5
LABELS = {"legitimate", "phishing"}
YEARS = {str(year) for year in range(2015, 2025)}
ID_PATTERN = re.compile(r"[a-z0-9_]+:[0-9a-f]{64}\Z")
SPLIT_COLUMNS = ["record_id", "label", "partition"]
FOLD_COLUMNS = ["record_id", "label", "cv_fold"]
EXCLUSION_COLUMNS = [
    "record_id", "development_source", "development_label", "temporal_record_id",
    "temporal_label", "exclusion_reason", "label_conflict",
]
PENDING_CHECKS = [
    "DIDECO cross-source exact leakage audit",
    "E-PhishLLM cross-source exact leakage audit",
    "PhishingPot cross-source exact leakage audit",
]
SPLIT_CONTRACT_VERSION = f"{CONTRACT_VERSION}:text-split-v1"


def split_semantic_payload(input_hashes: dict, audit: dict, assignment_counts: dict, *,
                           exact_exclusions_sha256: str | None = None,
                           based_on_semantic_sha256: str | None = None) -> dict:
    payload = {
        "contract_version": SPLIT_CONTRACT_VERSION,
        "inputs": dict(sorted(input_hashes.items())),
        "parameters": {
            "seed": SEED, "holdout_fraction": HOLDOUT_FRACTION,
            "stratification_field": "label", "cv_fold_count": FOLD_COUNT,
            "cv_shuffle": True, "input_order": "record_id_ascending",
        },
        "counts": {**audit, **assignment_counts},
    }
    if exact_exclusions_sha256 is not None:
        payload["exact_exclusions_sha256"] = exact_exclusions_sha256
        payload["based_on_semantic_sha256"] = based_on_semantic_sha256
    return payload


def relative(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != NORMALIZED_COLUMNS:
            raise ValueError(f"Unexpected input schema in {path}: {reader.fieldnames!r}")
        yield from reader


def _text_pair(row: dict[str, str]) -> tuple[str, str]:
    subject, body = row["subject"], row["body"]
    normalized = (normalize_text_field(subject), normalize_text_field(body))
    if normalized != (subject, body):
        raise ValueError(f"Input text is not Step-14-normalized: {row['record_id']}")
    return normalized


def scan_inputs(development_path: Path, temporal_path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    temporal_pairs: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    temporal_ids: set[str] = set()
    for row in csv_rows(temporal_path):
        record_id = row["record_id"]
        if not ID_PATTERN.fullmatch(record_id) or record_id in temporal_ids:
            raise ValueError(f"Invalid or duplicate temporal record_id: {record_id!r}")
        temporal_ids.add(record_id)
        if (row["dataset"], row["source_role"], row["collection_period"], row["label"]) != (
            "nazario", "temporal_evaluation", "2025", "phishing"
        ):
            raise ValueError(f"Unexpected temporal role/source/label: {record_id}")
        temporal_pairs[_text_pair(row)].append(row)

    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    eligible: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    original_counts: Counter[str] = Counter()
    exclusion_sources: Counter[str] = Counter()
    conflict_sources: Counter[str] = Counter()
    for row in csv_rows(development_path):
        record_id = row["record_id"]
        if not ID_PATTERN.fullmatch(record_id) or record_id in seen_ids or record_id in temporal_ids:
            raise ValueError(f"Invalid, duplicate, or cross-role record_id: {record_id!r}")
        seen_ids.add(record_id)
        source, label = row["dataset"], row["label"]
        if row["source_role"] != "text_development" or not (
            (source == "enron" and label == "legitimate")
            or (source == "nazario" and label == "phishing" and row["collection_period"] in YEARS)
        ):
            raise ValueError(f"Unexpected development role/source/label: {record_id}")
        pair = _text_pair(row)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate normalized (subject, body) in text_development: {record_id}")
        seen_pairs.add(pair)
        original_counts[label] += 1
        temporal_matches = temporal_pairs.get(pair, [])
        if temporal_matches:
            exclusion_sources[source] += 1
            for match in temporal_matches:
                conflict = label != match["label"]
                if conflict:
                    conflict_sources[source] += 1
                exclusions.append({
                    "record_id": record_id,
                    "development_source": source,
                    "development_label": label,
                    "temporal_record_id": match["record_id"],
                    "temporal_label": match["label"],
                    "exclusion_reason": "exact_normalized_subject_body_overlap_with_nazario_2025",
                    "label_conflict": str(conflict).lower(),
                })
        else:
            eligible.append({"record_id": record_id, "label": label, "dataset": source})

    eligible.sort(key=lambda row: row["record_id"])
    exclusions.sort(key=lambda row: (row["record_id"], row["temporal_record_id"]))
    if set(original_counts) != LABELS or any(original_counts[label] < FOLD_COUNT + 1 for label in LABELS):
        raise ValueError("Development input must contain both classes with enough rows for split and CV")
    audit = {
        "original_text_development": {"total": len(seen_ids), "by_label": dict(sorted(original_counts.items()))},
        "temporal_evaluation_rows": len(temporal_ids),
        "temporal_exclusions": {
            "development_records": len(seen_ids) - len(eligible),
            "match_rows": len(exclusions),
            "by_development_source": dict(sorted(exclusion_sources.items())),
            "label_conflict_match_rows": sum(conflict_sources.values()),
            "label_conflicts_by_development_source": dict(sorted(conflict_sources.items())),
        },
        "split_eligible_corpus": counts(eligible),
    }
    return eligible, exclusions, audit


def counts(rows: list[dict[str, str]]) -> dict:
    by_label = Counter(row["label"] for row in rows)
    return {"total": len(rows), "by_label": {label: by_label[label] for label in sorted(LABELS)}}


def make_assignments(eligible: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    eligible = sorted(eligible, key=lambda row: row["record_id"])
    ids = [row["record_id"] for row in eligible]
    labels = [row["label"] for row in eligible]
    train_ids, holdout_ids = train_test_split(
        ids, test_size=HOLDOUT_FRACTION, random_state=SEED, shuffle=True, stratify=labels
    )
    development_ids = set(train_ids)
    holdout_ids = set(holdout_ids)
    splits = [
        {"record_id": row["record_id"], "label": row["label"],
         "partition": "development" if row["record_id"] in development_ids else "final_internal_holdout"}
        for row in eligible
    ]
    development = [row for row in eligible if row["record_id"] in development_ids]
    holdout = [row for row in eligible if row["record_id"] in holdout_ids]
    if len(development) + len(holdout) != len(eligible):
        raise ValueError("Split does not cover eligible records")
    if any(counts(development)["by_label"][label] < FOLD_COUNT for label in LABELS) or any(
        counts(holdout)["by_label"][label] < 1 for label in LABELS
    ):
        raise ValueError("Development needs five examples per class and Holdout needs both classes")
    development_labels = [row["label"] for row in development]
    fold_by_id: dict[str, int] = {}
    splitter = StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    for fold, (_train_positions, validation_positions) in enumerate(splitter.split(development, development_labels)):
        for position in validation_positions:
            fold_by_id[development[int(position)]["record_id"]] = fold
    folds = [
        {"record_id": row["record_id"], "label": row["label"], "cv_fold": str(fold_by_id[row["record_id"]])}
        for row in development
    ]
    fold_counts = {str(fold): counts([row for row in folds if row["cv_fold"] == str(fold)]) for fold in range(FOLD_COUNT)}
    return splits, folds, {"development": counts(development), "final_internal_holdout": counts(holdout), "cv_validation_folds": fold_counts}


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(run_id: str, semantic_sha256: str, audit: dict, assignment_counts: dict, input_hashes: dict, run_dir: Path) -> dict:
    return {
        "schema_version": 3,
        "contract_version": SPLIT_CONTRACT_VERSION,
        "semantic_sha256": semantic_sha256,
        "status": "validated",
        "run_id": run_id,
        "run_directory": relative(run_dir),
        "inputs": {
            "text_development": {"path": relative(PREPROCESSED_OUTPUT_PATH), "sha256": input_hashes["text_development"]},
            "nazario_2025_temporal_evaluation": {"path": relative(NAZARIO_2025_TEMPORAL_OUTPUT_PATH), "sha256": input_hashes["nazario_2025_temporal_evaluation"]},
        },
        "parameters": {"seed": SEED, "holdout_fraction": HOLDOUT_FRACTION, "stratification_field": "label", "cv_fold_count": FOLD_COUNT, "cv_shuffle": True, "input_order": "record_id_ascending"},
        "counts": {**audit, **assignment_counts},
        "artifacts": {},
        "invariants": {
            "unique_record_ids": True, "unique_normalized_development_text_pairs": True,
            "approved_sources_roles_and_labels_only": True, "complete_disjoint_partitions": True,
            "both_classes_in_both_partitions_and_all_folds": True, "development_only_cv": True,
            "temporal_exact_overlaps_excluded": True, "manifest_csv_reconciled": True,
        },
        "pending_external_leakage_checks": PENDING_CHECKS,
        "holdout_use_policy": "Use once for internal evaluation only after final text configuration is frozen; never use for feature selection, CV, model selection, hyperparameters, calibration, thresholds, or fusion policy.",
    }


def exact_assignments(eligible: list[dict[str, str]]):
    if len({row["record_id"] for row in eligible}) != len(eligible):
        raise ValueError("Duplicate eligible record ID")
    return make_assignments(eligible)


def publish_exact_split(
    eligible: list[dict[str, str]],
    exclusion_path: Path,
    exact_report_path: Path,
    based_on_run_id: str,
) -> dict:
    if not eligible or not exclusion_path.is_file() or not exact_report_path.is_file():
        raise ValueError("Missing exact-exclusion split inputs")
    split_rows, fold_rows, counts = exact_assignments(eligible)
    input_hashes = {
        "exact_overlaps": sha256_file(exact_report_path),
        "exact_exclusions": sha256_file(exclusion_path),
    }
    payload = split_semantic_payload(
        input_hashes, {"eligible": len(eligible)}, counts,
        exact_exclusions_sha256=input_hashes["exact_exclusions"],
        based_on_semantic_sha256=json.loads(SPLITS_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))["semantic_sha256"],
    )
    run_id, fingerprint = content_run_id("text-exact-split", payload)
    final_dir = SPLITS_OUTPUT_DIR / "runs" / run_id
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".exact-split-", dir=SPLITS_OUTPUT_DIR) as directory:
        stage = Path(directory)
        write_csv(stage / "split_assignments.csv", SPLIT_COLUMNS, split_rows)
        write_csv(stage / "cv_fold_assignments.csv", FOLD_COLUMNS, fold_rows)
        (stage / "exact_exclusions.csv").write_bytes(exclusion_path.read_bytes())
        manifest = {
            "schema_version": 3,
            "contract_version": SPLIT_CONTRACT_VERSION,
            "semantic_sha256": fingerprint,
            "status": "validated",
            "run_id": run_id,
            "run_directory": final_dir.relative_to(REPOSITORY_ROOT).as_posix(),
            "based_on_step15_run_id": based_on_run_id,
            "based_on_semantic_sha256": payload["based_on_semantic_sha256"],
            "parameters": {
                "assignment": "record_id_ascending_label_stratified_seed_42",
                "holdout_fraction": 0.2,
                "cv_folds": 5,
            },
            "inputs": {
                "exact_overlaps": {
                    "path": exact_report_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "sha256": sha256_file(exact_report_path),
                },
                "exact_exclusions": {
                    "path": exclusion_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "sha256": sha256_file(exclusion_path),
                },
            },
            "counts": {
                "eligible": len(eligible),
                "development": counts["development"]["total"],
                "final_internal_holdout": counts["final_internal_holdout"]["total"],
                "cv_validation_folds": counts["cv_validation_folds"],
            },
            "artifacts": {},
        }
        for name in ("split_assignments.csv", "cv_fold_assignments.csv", "exact_exclusions.csv"):
            manifest["artifacts"][name] = {
                "path": f"{manifest['run_directory']}/{name}",
                "sha256": sha256_file(stage / name),
            }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        from scripts.validate_leakage_audit import validate_exact_split_run

        validate_exact_split_run(stage, manifest, eligible)
        stage.rename(final_dir)
    temporary = SPLITS_OUTPUT_DIR / f".manifest-exact-split-{run_id}.json"
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, SPLITS_MANIFEST_OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> dict:
    from scripts.validate_data_splits import validate_run

    if not json.loads(PREPROCESSING_REPORT_OUTPUT_PATH.read_text(encoding="utf-8")).get(
        "text_development_ready_for_supervised_training", False
    ):
        raise ValueError("Step-14 text_development readiness gate is false")
    SPLITS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_dir = SPLITS_OUTPUT_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    input_hashes = {
        "text_development": sha256_file(PREPROCESSED_OUTPUT_PATH),
        "nazario_2025_temporal_evaluation": sha256_file(NAZARIO_2025_TEMPORAL_OUTPUT_PATH),
    }
    eligible, exclusions, audit = scan_inputs(PREPROCESSED_OUTPUT_PATH, NAZARIO_2025_TEMPORAL_OUTPUT_PATH)
    splits, folds, assignment_counts = make_assignments(eligible)
    payload = split_semantic_payload(input_hashes, audit, assignment_counts)
    run_id, fingerprint = content_run_id("text-split", payload)
    final_run_dir = runs_dir / run_id
    if final_run_dir.exists():
        raise ValueError(f"Deterministic split output already exists: {final_run_dir}")
    with tempfile.TemporaryDirectory(prefix=".step15-", dir=SPLITS_OUTPUT_DIR) as temporary:
        staging_dir = Path(temporary)
        for name, columns, rows in (
            ("split_assignments.csv", SPLIT_COLUMNS, splits),
            ("cv_fold_assignments.csv", FOLD_COLUMNS, folds),
            ("temporal_exclusions.csv", EXCLUSION_COLUMNS, exclusions),
        ):
            write_csv(staging_dir / name, columns, rows)
        manifest = build_manifest(run_id, fingerprint, audit, assignment_counts, input_hashes, final_run_dir)
        manifest["artifacts"] = {
            name: {"path": relative(final_run_dir / name), "sha256": sha256_file(staging_dir / name)}
            for name in ("split_assignments.csv", "cv_fold_assignments.csv", "temporal_exclusions.csv")
        }
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validate_run(staging_dir, manifest, check_input_hashes=True)
        if input_hashes != {
            "text_development": sha256_file(PREPROCESSED_OUTPUT_PATH),
            "nazario_2025_temporal_evaluation": sha256_file(NAZARIO_2025_TEMPORAL_OUTPUT_PATH),
        }:
            raise ValueError("An input changed during the Step-15 run")
        staging_dir.rename(final_run_dir)
    manifest_temp = SPLITS_OUTPUT_DIR / f".manifest-{run_id}.json"
    try:
        manifest_temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(manifest_temp, SPLITS_MANIFEST_OUTPUT_PATH)
    finally:
        manifest_temp.unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    result = main()
    print(json.dumps({"run_id": result["run_id"], "counts": result["counts"]}, indent=2), flush=True)
