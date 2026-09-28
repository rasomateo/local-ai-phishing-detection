
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.load_data import NORMALIZED_COLUMNS, build_record_id
from src.prepare_reserved_evaluation import ANNOTATION_COLUMNS, sha256_file
from src.split_data import (FOLD_COLUMNS, SPLIT_COLUMNS, SPLIT_CONTRACT_VERSION,
                            make_assignments, split_semantic_payload)
from src.provenance import content_run_id, validate_artifact_inventory, validate_predecessor
from src.reserved_source_audit import AUDIT_DIR, EXCLUSION_COLUMNS, PAIR_COLUMNS, SOURCE_PATHS
from src.utils import (NAZARIO_2025_TEMPORAL_OUTPUT_PATH, PREPROCESSED_OUTPUT_PATH,
                       REPOSITORY_ROOT, SPLITS_MANIFEST_OUTPUT_PATH)

csv.field_size_limit(sys.maxsize)
def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == columns, f"Invalid columns: {path}")
        return list(reader)


def validate_exact_split_run(run_dir: Path, manifest: dict, eligible: list[dict[str, str]]) -> dict:
    require(manifest.get("schema_version") == 3 and manifest.get("contract_version") == SPLIT_CONTRACT_VERSION
            and manifest.get("status") == "validated", "Invalid replacement split manifest")
    splits = read(run_dir / "split_assignments.csv", SPLIT_COLUMNS)
    folds = read(run_dir / "cv_fold_assignments.csv", FOLD_COLUMNS)
    expected_splits, expected_folds, counts = make_assignments(eligible)
    require(splits == expected_splits and folds == expected_folds, "Replacement assignments differ from deterministic record-ID split")
    require({row["record_id"] for row in folds} == {row["record_id"] for row in splits if row["partition"] == "development"},
            "Replacement CV is not Development-only")
    require(manifest["counts"]["eligible"] == len(eligible) and
            manifest["counts"]["development"] == counts["development"]["total"], "Replacement split count mismatch")
    for name, spec in manifest["artifacts"].items():
        require(spec["sha256"] == sha256_file(run_dir / name), f"Replacement artifact hash mismatch: {name}")
    for spec in manifest["inputs"].values():
        require(spec["sha256"] == sha256_file(REPOSITORY_ROOT / spec["path"]), "Replacement input hash mismatch")
    input_hashes = {name: spec["sha256"] for name, spec in manifest["inputs"].items()}
    payload = split_semantic_payload(
        input_hashes, {"eligible": len(eligible)}, counts,
        exact_exclusions_sha256=input_hashes["exact_exclusions"],
        based_on_semantic_sha256=manifest["based_on_semantic_sha256"],
    )
    expected_run, expected_semantic = content_run_id("text-exact-split", payload)
    require(manifest["run_id"] == expected_run and manifest["semantic_sha256"] == expected_semantic,
            "Replacement split semantic identity mismatch")
    return {"run_id": manifest["run_id"], "split_records": len(splits), "development_records": len(folds)}


def exact_data_quality(rows: list[dict[str, str]]) -> dict[str, int]:
    groups = Counter((row["subject"], row["body"]) for row in rows)
    duplicates = [amount for amount in groups.values() if amount > 1]
    return {"records": len(rows), "duplicate_groups": len(duplicates),
            "affected_records": sum(duplicates), "excess_records": sum(n - 1 for n in duplicates),
            "duplicate_pairs": sum(n * (n - 1) // 2 for n in duplicates)}


def expected_pairs(rows: list[dict[str, str]], assignment: dict[str, str]) -> tuple[list[dict[str, str]], dict[str, int]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subject"], row["body"])].append(row)
    result = []
    scopes = Counter()
    for group in groups.values():
        for left, right in combinations(sorted(group, key=lambda item: item["record_id"]), 2):
            l_internal = left["source_role"] == "text_development"
            r_internal = right["source_role"] == "text_development"
            if not (l_internal or r_internal or left["dataset"] != right["dataset"]):
                continue
            mixed = l_internal != r_internal
            result.append({
                "record_id_left": left["record_id"], "source_left": left["dataset"],
                "role_left": left["source_role"], "label_left": left["label"],
                "record_id_right": right["record_id"], "source_right": right["dataset"],
                "role_right": right["source_role"], "label_right": right["label"],
                "overlap_type": "exact", "label_conflict": str(left["label"] != right["label"]).lower(),
                "remediation": ("exclude_internal_copy" if mixed else
                                "block_internal_overlap" if l_internal else "report_reserved_overlap"),
            })
            if l_internal and r_internal:
                scope = ("development|final_internal_holdout" if assignment[left["record_id"]] != assignment[right["record_id"]]
                         else "internal|same_partition")
            elif mixed:
                other = right if l_internal else left
                source = "nazario_2025" if other["source_role"] == "temporal_evaluation" else other["dataset"]
                scope = f"internal|{source}"
            else:
                names = ["nazario_2025" if item["source_role"] == "temporal_evaluation" else item["dataset"]
                         for item in (left, right)]
                scope = "reserved|" + "|".join(sorted(names))
            scopes[scope] += 1
    result.sort(key=lambda pair: (pair["record_id_left"], pair["record_id_right"]))
    return result, dict(scopes)


def main() -> dict:
    from scripts.validate_data_splits import main as validate_step15

    report = json.loads((AUDIT_DIR / "summary.json").read_text(encoding="utf-8"))
    require(report["schema_version"] == 2 and report["exact_duplicate_method"] ==
            "equality of Step-14-normalized (subject, body)", "Invalid exact methodology")
    predecessor = validate_predecessor(REPOSITORY_ROOT, report["predecessor"])
    validate_artifact_inventory(REPOSITORY_ROOT, report["artifacts"])
    sources = {"nazario_2025": read(NAZARIO_2025_TEMPORAL_OUTPUT_PATH, NORMALIZED_COLUMNS)}
    sources.update({name: read(path, NORMALIZED_COLUMNS) for name, path in SOURCE_PATHS.items()})
    internal = read(PREPROCESSED_OUTPUT_PATH, NORMALIZED_COLUMNS)
    annotation_path = REPOSITORY_ROOT / "outputs/dataset_roles/dideco_annotations.csv"
    annotations = read(annotation_path, ANNOTATION_COLUMNS)
    all_rows = internal + [row for group in sources.values() for row in group]
    by_id = {row["record_id"]: row for row in all_rows}
    require(len(by_id) == len(all_rows), "Cross-source record ID collision")
    require(all(row["source_role"] == "text_development" for row in internal), "Internal role mismatch")
    require(all(row["source_role"] == "temporal_evaluation" and row["label"] == "phishing" for row in sources["nazario_2025"]),
            "Temporal role or label mismatch")
    require(all(row["source_role"] == "external_evaluation" and row["source_subset"] == "english" and
                row["label"] in {"legitimate", "phishing"} for row in sources["ephishllm"]), "E-PhishLLM role/language/label mismatch")
    require(all(row["source_role"] == "external_evaluation" and row["label"] == "phishing" for row in sources["phishingpot"]),
            "PhishingPot role/label mismatch")
    require(all(row["source_role"] == "rq2_evaluation" and row["label"] == "phishing" and not row["subject"] and
                "corpus" in row["label_provenance"] for row in sources["dideco"]), "DIDECO role/provenance mismatch")
    require(len(annotations) == report["reserved_source_counts"]["dideco"]["raw_annotation_rows"], "DIDECO annotation count mismatch")
    require(sum(row["displayed_text_empty"] == "true" for row in annotations) ==
            report["reserved_source_counts"]["dideco"]["empty_displayed_text_annotation_rows"], "DIDECO empty count mismatch")
    require(all(not row["message_record_id"] for row in annotations if row["displayed_text_empty"] == "true"),
            "Empty DIDECO annotation created a message")
    require(len({row["record_id"] for row in sources["dideco"]}) == len(sources["dideco"]), "DIDECO message ID collision")
    for name, group in sources.items():
        expected_count = (report["reserved_source_counts"][name].get("retained") or
                          report["reserved_source_counts"][name].get("retained_messages"))
        require(len(group) == expected_count, f"{name} count mismatch")
        for row in group:
            require(row["record_id"] == build_record_id(dataset=row["dataset"], source_subset=row["source_subset"],
                    source_file=row["source_file"], source_record=row["source_record"]), "Unstable reserved record ID")
    require(report["reserved_source_counts"]["nazario_2025"]["source_sha256"] ==
            sha256_file(NAZARIO_2025_TEMPORAL_OUTPUT_PATH), "Temporal hash mismatch")
    require(report["intra_source_exact_data_quality"] == {name: exact_data_quality(rows) for name, rows in sources.items()},
            "Intra-source exact counts mismatch")

    manifest = json.loads(SPLITS_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    splits = read(REPOSITORY_ROOT / manifest["run_directory"] / "split_assignments.csv", SPLIT_COLUMNS)
    folds = read(REPOSITORY_ROOT / manifest["run_directory"] / "cv_fold_assignments.csv", FOLD_COLUMNS)
    assignment = {row["record_id"]: row["partition"] for row in splits}
    require(len(assignment) == len(splits), "Duplicate split record ID")
    require({row["record_id"] for row in folds} == {row["record_id"] for row in splits if row["partition"] == "development"},
            "Holdout entered CV")
    require({row["cv_fold"] for row in folds} == {"0", "1", "2", "3", "4"}, "Incomplete CV folds")
    require(set(assignment).isdisjoint({row["record_id"] for group in sources.values() for row in group}),
            "Reserved source entered internal split")
    based_on_run_id = report["based_on_step15_run_id"]
    old_dir = REPOSITORY_ROOT / "outputs/splits/runs" / based_on_run_id
    old_manifest = json.loads((old_dir / "manifest.json").read_text(encoding="utf-8"))
    require(old_manifest["semantic_sha256"] == predecessor["semantic_sha256"], "Audit predecessor changed")
    old_rows = read(old_dir / "split_assignments.csv", SPLIT_COLUMNS)
    scope_assignment = {row["record_id"]: row["partition"] for row in old_rows}
    if not report["split_regenerated"]:
        require(manifest["run_id"] == based_on_run_id == report["canonical_split_run_id"], "Canonical split changed")
        validate_step15()
    require(set(scope_assignment) == {row["record_id"] for row in internal}, "Original split coverage mismatch")

    persisted = read(AUDIT_DIR / "exact_overlaps.csv", PAIR_COLUMNS)
    expected, counts = expected_pairs(all_rows, scope_assignment)
    require(persisted == expected and len(persisted) == report["exact_overlap_count"], "Exact overlap report mismatch")
    require(all(report["exact_overlaps_by_scope"].get(name, 0) == count for name, count in counts.items()) and
            sum(report["exact_overlaps_by_scope"].values()) == len(persisted), "Exact scope counts mismatch")
    conflicts = sum(pair["label_conflict"] == "true" for pair in persisted)
    require(conflicts == report["label_conflict_count"], "Label conflict count mismatch")
    exclusions = read(AUDIT_DIR / "internal_exclusions.csv", EXCLUSION_COLUMNS)
    expected_exclusion_keys = {(pair["record_id_left"] if pair["role_left"] == "text_development" else pair["record_id_right"],
                                pair["record_id_right"] if pair["role_left"] == "text_development" else pair["record_id_left"])
                               for pair in persisted if pair["remediation"] == "exclude_internal_copy"}
    require({(row["record_id"], row["against_record_id"]) for row in exclusions} == expected_exclusion_keys,
            "Exact exclusion pairs mismatch")
    excluded_ids = {row["record_id"] for row in exclusions}
    require(len(excluded_ids) == report["internal_exact_exclusions"] and len(exclusions) == report["exclusion_match_rows"],
            "Exact exclusion counts mismatch")
    require(report["split_regenerated"] == bool(excluded_ids), "Split regeneration gate mismatch")
    if excluded_ids:
        require(manifest["run_id"] == report["canonical_split_run_id"] and
                manifest["based_on_step15_run_id"] == based_on_run_id, "Replacement split reference mismatch")
        eligible = [row for row in internal if row["record_id"] not in excluded_ids]
        validate_exact_split_run(REPOSITORY_ROOT / manifest["run_directory"], manifest, eligible)
        require(set(assignment) == {row["record_id"] for row in eligible}, "Excluded record remains in split")
    ready = not persisted and not excluded_ids and conflicts == 0 and report["source_role_isolation_passed"]
    require(report["training_ready_after_step16"] == ready and report["status"] == ("completed" if ready else "blocked"),
            "Readiness does not follow exact gates")
    semantic_payload = {
        "contract_version": report["contract_version"],
        "split_semantic_sha256": predecessor["semantic_sha256"],
        "exact_duplicate_method": report["exact_duplicate_method"],
        "exact_overlap_count": len(persisted), "exact_overlaps_by_scope": report["exact_overlaps_by_scope"],
        "label_conflict_count": conflicts, "internal_exact_exclusions": len(excluded_ids),
        "exclusion_match_rows": len(exclusions), "source_counts": report["reserved_source_counts"],
        "source_role_isolation_passed": report["source_role_isolation_passed"],
    }
    _audit_id, expected_fingerprint = content_run_id("reserved-audit", semantic_payload)
    require(report["semantic_sha256"] == expected_fingerprint, "Reserved-audit semantic fingerprint mismatch")
    return {"status": "passed", "exact_overlap_count": len(persisted),
            "label_conflict_count": conflicts, "internal_exact_exclusions": len(excluded_ids),
            "canonical_split_run_id": manifest["run_id"], "training_ready_after_step16": ready}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2), flush=True)
