
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from src.load_data import NORMALIZED_COLUMNS
from src.prepare_reserved_evaluation import (
    ANNOTATION_COLUMNS, load_dideco, load_ephish, load_phishingpot,
    sha256_file,
)
from src.split_data import csv_rows, publish_exact_split, write_csv
from src.provenance import artifact_inventory, content_run_id, predecessor_reference
from src.utils import (
    DATASET_ROLES_OUTPUT_DIR, NAZARIO_2025_TEMPORAL_OUTPUT_PATH, OUTPUTS_DIR,
    PREPROCESSED_OUTPUT_PATH, REPOSITORY_ROOT, SPLITS_MANIFEST_OUTPUT_PATH,
)

csv.field_size_limit(sys.maxsize)
AUDIT_DIR = OUTPUTS_DIR / "leakage" / "reserved_sources"
PAIR_COLUMNS = [
    "record_id_left", "source_left", "role_left", "label_left",
    "record_id_right", "source_right", "role_right", "label_right",
    "overlap_type", "label_conflict", "remediation",
]
EXCLUSION_COLUMNS = [
    "record_id", "dataset", "label", "reason", "against_record_id",
    "against_source", "overlap_type", "label_conflict",
]
SOURCE_PATHS = {
    "ephishllm": DATASET_ROLES_OUTPUT_DIR / "ephishllm_english_external_evaluation.csv",
    "phishingpot": DATASET_ROLES_OUTPUT_DIR / "phishingpot_external_evaluation.csv",
    "dideco": DATASET_ROLES_OUTPUT_DIR / "dideco_intent_evaluation.csv",
}
RESERVED_SOURCES = ("nazario_2025", "ephishllm", "phishingpot", "dideco")


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv_rows(path))


def is_internal(row: dict[str, str]) -> bool:
    return row["source_role"] == "text_development"


def protected(left: dict[str, str], right: dict[str, str]) -> bool:
    if is_internal(left) or is_internal(right):
        return True
    return left["dataset"] != right["dataset"]


def pair_row(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    if left["record_id"] > right["record_id"]:
        left, right = right, left
    mixed = is_internal(left) != is_internal(right)
    return {
        "record_id_left": left["record_id"], "source_left": left["dataset"],
        "role_left": left["source_role"], "label_left": left["label"],
        "record_id_right": right["record_id"], "source_right": right["dataset"],
        "role_right": right["source_role"], "label_right": right["label"],
        "overlap_type": "exact", "label_conflict": str(left["label"] != right["label"]).lower(),
        "remediation": ("exclude_internal_copy" if mixed else
                        "block_internal_overlap" if is_internal(left) else "report_reserved_overlap"),
    }


def exact_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_text: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_text[(row["subject"], row["body"])].append(row)
    matches = []
    for bucket in by_text.values():
        for left, right in combinations(sorted(bucket, key=lambda row: row["record_id"]), 2):
            if protected(left, right):
                matches.append(pair_row(left, right))
    return sorted(matches, key=lambda row: (row["record_id_left"], row["record_id_right"]))


def exclusion_rows(pairs: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for pair in pairs:
        if pair["remediation"] != "exclude_internal_copy":
            continue
        side, other = (("left", "right") if pair["role_left"] == "text_development"
                       else ("right", "left"))
        rows.append({
            "record_id": pair[f"record_id_{side}"], "dataset": pair[f"source_{side}"],
            "label": pair[f"label_{side}"], "reason": "exact_overlap_with_reserved",
            "against_record_id": pair[f"record_id_{other}"],
            "against_source": pair[f"source_{other}"], "overlap_type": "exact",
            "label_conflict": pair["label_conflict"],
        })
    return sorted(rows, key=lambda row: (row["record_id"], row["against_record_id"]))


def exact_scope(pair: dict[str, str], assignment: dict[str, str]) -> str:
    a, b = pair["record_id_left"], pair["record_id_right"]
    left_internal = pair["role_left"] == "text_development"
    right_internal = pair["role_right"] == "text_development"
    if left_internal and right_internal:
        return ("development|final_internal_holdout" if assignment[a] != assignment[b]
                else "internal|same_partition")
    if left_internal != right_internal:
        other = "right" if left_internal else "left"
        source = ("nazario_2025" if pair[f"role_{other}"] == "temporal_evaluation"
                  else pair[f"source_{other}"])
        return f"internal|{source}"
    names = [("nazario_2025" if pair[f"role_{side}"] == "temporal_evaluation"
              else pair[f"source_{side}"]) for side in ("left", "right")]
    return "reserved|" + "|".join(sorted(names))


def scope_counts(pairs: list[dict[str, str]], assignment: dict[str, str]) -> dict[str, int]:
    expected = ["development|final_internal_holdout", "internal|same_partition"]
    expected += [f"internal|{source}" for source in RESERVED_SOURCES]
    expected += ["reserved|" + "|".join(sorted(sources))
                 for sources in combinations(RESERVED_SOURCES, 2)]
    counts = Counter(exact_scope(pair, assignment) for pair in pairs)
    return {scope: counts[scope] for scope in sorted(expected)}


def intra_source_exact(rows: list[dict[str, str]]) -> dict[str, int]:
    groups = Counter((row["subject"], row["body"]) for row in rows)
    repeated = [count for count in groups.values() if count > 1]
    return {"records": len(rows), "duplicate_groups": len(repeated),
            "affected_records": sum(repeated), "excess_records": sum(n - 1 for n in repeated),
            "duplicate_pairs": sum(n * (n - 1) // 2 for n in repeated)}


def exact_readiness(pairs: list[dict[str, str]], role_isolation_passed: bool) -> bool:
    return role_isolation_passed and not pairs


def read_assignment(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["record_id", "label", "partition"]:
            raise ValueError("Unexpected split assignment schema")
        return list(reader)


def main() -> dict:
    from scripts.validate_data_splits import main as validate_step15

    DATASET_ROLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    previous_manifest = json.loads(SPLITS_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    validate_step15()
    old_assignment_rows = read_assignment(REPOSITORY_ROOT / previous_manifest["run_directory"] / "split_assignments.csv")
    assignment = {row["record_id"]: row["partition"] for row in old_assignment_rows}

    ephish, ephish_meta = load_ephish()
    pot, pot_meta = load_phishingpot()
    dideco, annotations, dideco_meta = load_dideco()
    nazario = read_rows(NAZARIO_2025_TEMPORAL_OUTPUT_PATH)
    internal = read_rows(PREPROCESSED_OUTPUT_PATH)
    sources = {"nazario_2025": nazario, "ephishllm": ephish, "phishingpot": pot, "dideco": dideco}
    all_rows = internal + nazario + ephish + pot + dideco
    ids = [row["record_id"] for row in all_rows]
    if len(ids) != len(set(ids)) or set(assignment) != {row["record_id"] for row in internal}:
        raise ValueError("Record ID uniqueness or split coverage failed")
    role_ok = (all(row["source_role"] == "text_development" for row in internal) and
               all(row["source_role"] == "temporal_evaluation" for row in nazario) and
               all(row["source_role"] == "external_evaluation" for row in ephish + pot) and
               all(row["source_role"] == "rq2_evaluation" for row in dideco))
    if not role_ok:
        raise ValueError("Source-role isolation failed")

    for source, records in (("ephishllm", ephish), ("phishingpot", pot), ("dideco", dideco)):
        write_csv(SOURCE_PATHS[source], NORMALIZED_COLUMNS, records)
    write_csv(DATASET_ROLES_OUTPUT_DIR / "dideco_annotations.csv", ANNOTATION_COLUMNS, annotations)

    exact = exact_pairs(all_rows)
    exclusions = exclusion_rows(exact)
    write_csv(AUDIT_DIR / "exact_overlaps.csv", PAIR_COLUMNS, exact)
    exclusion_path = AUDIT_DIR / "internal_exclusions.csv"
    write_csv(exclusion_path, EXCLUSION_COLUMNS, exclusions)
    excluded_ids = {row["record_id"] for row in exclusions}
    scopes = scope_counts(exact, assignment)
    conflicts = sum(row["label_conflict"] == "true" for row in exact)
    source_counts = {"nazario_2025": {"retained": len(nazario), "source_sha256": sha256_file(NAZARIO_2025_TEMPORAL_OUTPUT_PATH)},
                     "ephishllm": ephish_meta, "phishingpot": pot_meta, "dideco": dideco_meta}
    split_regenerated = False
    canonical_run = previous_manifest["run_id"]
    if excluded_ids:
        new_manifest = publish_exact_split([row for row in internal if row["record_id"] not in excluded_ids],
                                           exclusion_path, AUDIT_DIR / "exact_overlaps.csv", previous_manifest["run_id"])
        canonical_run = new_manifest["run_id"]
        split_regenerated = True
    blockers = []
    if exact:
        blockers.append("Protected-boundary exact overlap detected; inspect reported pairs and any exclusions")
    if conflicts:
        blockers.append("Exact-overlap label conflict detected")
    semantic_payload = {
        "contract_version": "artifact-provenance-v1:reserved-source-audit-v1",
        "split_semantic_sha256": previous_manifest["semantic_sha256"],
        "exact_duplicate_method": "equality of Step-14-normalized (subject, body)",
        "exact_overlap_count": len(exact), "exact_overlaps_by_scope": scopes,
        "label_conflict_count": conflicts, "internal_exact_exclusions": len(excluded_ids),
        "exclusion_match_rows": len(exclusions), "source_counts": source_counts,
        "source_role_isolation_passed": role_ok,
    }
    _audit_run_id, audit_fingerprint = content_run_id("reserved-audit", semantic_payload)
    report = {
        "schema_version": 2, "status": "blocked" if blockers else "completed",
        "contract_version": semantic_payload["contract_version"],
        "semantic_sha256": audit_fingerprint,
        "predecessor": predecessor_reference(
            REPOSITORY_ROOT,
            REPOSITORY_ROOT / previous_manifest["run_directory"] / "manifest.json",
        ),
        "exact_duplicate_method": "equality of Step-14-normalized (subject, body)",
        "exact_overlap_count": len(exact), "exact_overlaps_by_scope": scopes,
        "label_conflict_count": conflicts, "internal_exact_exclusions": len(excluded_ids),
        "exclusion_match_rows": len(exclusions), "original_internal_count": len(internal),
        "eligible_internal_count": len(internal) - len(excluded_ids),
        "reserved_source_counts": source_counts,
        "intra_source_exact_data_quality": {name: intra_source_exact(records) for name, records in sources.items()},
        "source_role_isolation_passed": role_ok,
        "canonical_split_run_id": canonical_run,
        "based_on_step15_run_id": previous_manifest["run_id"],
        "split_regenerated": split_regenerated,
        "training_ready_after_step16": exact_readiness(exact, role_ok),
        "blockers": blockers,
        "limitation": "Non-exact near-duplicate or template-level overlap is outside the scope of the implemented leakage-control procedure and is therefore not systematically excluded.",
    }
    report["artifacts"] = artifact_inventory(
        AUDIT_DIR, ["exact_overlaps.csv", "internal_exclusions.csv"],
        "outputs/leakage/reserved_sources",
    )
    (AUDIT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2), flush=True)
