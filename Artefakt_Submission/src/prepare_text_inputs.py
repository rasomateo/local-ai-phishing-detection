
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import normalize_text_field
from src.utils import PREPROCESSED_OUTPUT_PATH, REPOSITORY_ROOT, SPLITS_MANIFEST_OUTPUT_PATH
from src.provenance import artifact_inventory, content_run_id, predecessor_reference
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
OUTPUT_DIR = REPOSITORY_ROOT / "outputs" / "features" / "text"
DEVELOPMENT_COLUMNS = ["split_run_id", "record_id", "label", "cv_fold", "subject", "body"]
HOLDOUT_COLUMNS = ["split_run_id", "record_id", "label", "subject", "body"]
PARTITIONS = {"development", "final_internal_holdout"}
LABELS = {"legitimate", "phishing"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected schema: {path}")
        return list(reader)


def build_inputs(
    preprocessed: list[dict[str, str]],
    splits: list[dict[str, str]],
    folds: list[dict[str, str]],
    *,
    run_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not run_id:
        raise ValueError("Missing canonical split run")
    ids = [row["record_id"] for row in preprocessed]
    split_ids = [row["record_id"] for row in splits]
    fold_ids = [row["record_id"] for row in folds]
    if (len(ids) != len(set(ids)) or len(split_ids) != len(set(split_ids))
            or len(fold_ids) != len(set(fold_ids)) or set(ids) != set(split_ids)):
        raise ValueError("Duplicate or incomplete record_id coverage")
    if any(row["source_role"] != "text_development" for row in preprocessed):
        raise ValueError("Reserved source in preprocessing input")
    if any(row["label"] not in LABELS or row["partition"] not in PARTITIONS for row in splits):
        raise ValueError("Unexpected label or partition")
    if any(row["cv_fold"] not in {"0", "1", "2", "3", "4"} for row in folds):
        raise ValueError("Unexpected CV fold")
    development_ids = {row["record_id"] for row in splits if row["partition"] == "development"}
    if set(fold_ids) != development_ids or {row["cv_fold"] for row in folds} != {"0", "1", "2", "3", "4"}:
        raise ValueError("CV does not cover Development exactly")
    by_id = {row["record_id"]: row for row in preprocessed}
    fold_by_id = {row["record_id"]: row for row in folds}
    development, holdout = [], []
    for split in sorted(splits, key=lambda row: row["record_id"]):
        record_id = split["record_id"]
        source = by_id[record_id]
        if source["label"] != split["label"]:
            raise ValueError(f"Split label mismatch: {record_id}")
        if any(normalize_text_field(source[field]) != source[field] for field in ("subject", "body")):
            raise ValueError(f"Text is not Step-14-normalized: {record_id}")
        base = {"split_run_id": run_id, "record_id": record_id, "label": split["label"]}
        if split["partition"] == "development":
            if fold_by_id[record_id]["label"] != split["label"]:
                raise ValueError(f"CV label mismatch: {record_id}")
            development.append({**base, "cv_fold": fold_by_id[record_id]["cv_fold"],
                                "subject": source["subject"], "body": source["body"]})
        else:
            holdout.append({**base, "subject": source["subject"], "body": source["body"]})
    return development, holdout


def fold_record_ids(development: list[dict[str, str]], fold: int) -> tuple[list[str], list[str]]:
    if fold not in range(5):
        raise ValueError("CV fold must be 0 through 4")
    train = [row["record_id"] for row in development if row["cv_fold"] != str(fold)]
    validation = [row["record_id"] for row in development if row["cv_fold"] == str(fold)]
    if not validation or set(train) & set(validation):
        raise ValueError("Invalid fold partition")
    return train, validation


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> dict:
    contract = current_contract(REPOSITORY_ROOT)
    manifest = json.loads(SPLITS_MANIFEST_OUTPUT_PATH.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    if not run_id or manifest.get("status") != "validated":
        raise ValueError("Canonical Step-15 manifest is not validated")
    run_dir = REPOSITORY_ROOT / manifest["run_directory"]
    if (run_dir.resolve() != (REPOSITORY_ROOT / "outputs" / "splits" / "runs" / run_id).resolve()
            or json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) != manifest):
        raise ValueError("Split run directory/manifest mismatch")
    if manifest["inputs"]["text_development"]["sha256"] != sha256(PREPROCESSED_OUTPUT_PATH):
        raise ValueError("Preprocessing input hash differs from canonical split")
    for name in ("split_assignments.csv", "cv_fold_assignments.csv"):
        if manifest["artifacts"][name]["sha256"] != sha256(run_dir / name):
            raise ValueError(f"Canonical assignment hash mismatch: {name}")
    preprocessed = read_csv(PREPROCESSED_OUTPUT_PATH, NORMALIZED_COLUMNS)
    splits = read_csv(run_dir / "split_assignments.csv", ["record_id", "label", "partition"])
    folds = read_csv(run_dir / "cv_fold_assignments.csv", ["record_id", "label", "cv_fold"])
    development, holdout = build_inputs(preprocessed, splits, folds, run_id=run_id)
    if (len(development) != contract.expected("text_development_count", 201016)
            or len(holdout) != contract.expected("text_holdout_count", 50255)):
        raise ValueError("Canonical partition counts differ from Step 16C")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "development_inputs.csv", DEVELOPMENT_COLUMNS, development)
    write_csv(OUTPUT_DIR / "holdout_inputs.csv", HOLDOUT_COLUMNS, holdout)
    semantic_payload = {
        "contract_version": "artifact-provenance-v1:text-inputs-v1",
        "split_semantic_sha256": manifest["semantic_sha256"],
        "normalization": "Step-14 normalize_text_field",
        "views": {"subject": "subject", "body": "body", "subject_body": "subject.strip() + '\\n\\n' + body.strip() when both nonempty, otherwise the nonempty field"},
        "counts": {"development": len(development), "final_internal_holdout": len(holdout)},
    }
    _inputs_id, fingerprint = content_run_id("text-inputs", semantic_payload)
    report = {
        "schema_version": 2, "split_run_id": run_id, "status": "prepared",
        "contract_version": semantic_payload["contract_version"], "semantic_sha256": fingerprint,
        "predecessor": predecessor_reference(REPOSITORY_ROOT, SPLITS_MANIFEST_OUTPUT_PATH),
        "inputs": {
            "preprocessed": {"path": "outputs/preprocessed_dataset.csv", "sha256": sha256(PREPROCESSED_OUTPUT_PATH)},
            "split_manifest": {"path": "outputs/splits/manifest.json", "sha256": sha256(SPLITS_MANIFEST_OUTPUT_PATH)},
            "split_assignments": {"path": f"outputs/splits/runs/{run_id}/split_assignments.csv", "sha256": sha256(run_dir / "split_assignments.csv")},
            "cv_fold_assignments": {"path": f"outputs/splits/runs/{run_id}/cv_fold_assignments.csv", "sha256": sha256(run_dir / "cv_fold_assignments.csv")},
        },
        "views": {"subject": "subject", "body": "body", "subject_body": "subject.strip() + '\\n\\n' + body.strip() when both nonempty, otherwise the nonempty field"},
        "normalization": "Step-14 normalize_text_field, already applied to subject and body",
        "vectorizer_fit": "none; later fit separately on each Development fold train IDs",
        "legacy_tfidf_config": {
            "parameters": {"analyzer": "word", "ngram_range": [1, 2], "token_pattern": r"(?u)\b\w\w+\b", "lowercase": False, "sublinear_tf": True, "min_df": 2, "max_df": 0.95, "max_features": 30000, "dtype": "float32"},
            "origin": "src/features.py, introduced in initial repository history; no separate experiment configuration file",
            "independent_precommitment_established": False,
            "status": "historical reference only; not canonical or selected",
        },
        "counts": {"development": len(development), "final_internal_holdout": len(holdout),
                   "development_by_label": dict(sorted(Counter(row["label"] for row in development).items())),
                   "holdout_by_label": dict(sorted(Counter(row["label"] for row in holdout).items())),
                   "cv_validation_folds": {str(i): dict(sorted(Counter(row["label"] for row in development if row["cv_fold"] == str(i)).items())) for i in range(5)}},
        "artifacts": artifact_inventory(OUTPUT_DIR, ["development_inputs.csv", "holdout_inputs.csv"], "outputs/features/text"),
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({"split_run_id": result["split_run_id"], "counts": result["counts"]}, indent=2))
