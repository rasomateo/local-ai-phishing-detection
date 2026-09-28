
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)
OUTPUTS = REPOSITORY_ROOT / "outputs"
YEARS = {str(year) for year in range(2015, 2025)}
ID_PATTERN = re.compile(r"[a-z0-9_]+:[0-9a-f]{64}\Z")


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def check_identity(row: dict[str, str], seen: set[str]) -> None:
    record_id = row["record_id"]
    assert ID_PATTERN.fullmatch(record_id), f"Invalid record_id: {record_id!r}"
    assert record_id not in seen, f"Duplicate record_id: {record_id}"
    seen.add(record_id)
    source_file = row["source_file"]
    assert source_file and not Path(source_file).is_absolute() and not re.match(r"^[A-Za-z]:[\\/]", source_file)
    assert all(row[field] for field in ("source_subset", "source_record", "source_role", "label_provenance", "collection_period", "inclusion_reason"))


def main() -> None:
    contract = current_contract(REPOSITORY_ROOT)
    report = json.loads((OUTPUTS / "preprocessing_report.json").read_text(encoding="utf-8"))
    raw_ids: set[str] = set()
    raw_counts: Counter[tuple[str, str, str, str]] = Counter()
    raw_nazario_ids: set[str] = set()
    for row in rows(OUTPUTS / "merged_raw_dataset.csv"):
        check_identity(row, raw_ids)
        raw_counts[(row["dataset"], row["source_subset"], row["source_role"], row["label"])] += 1
        if row["dataset"] == "nazario" and row["collection_period"] in YEARS:
            assert row["source_role"] == "text_development" and row["label"] == "phishing"
            raw_nazario_ids.add(row["record_id"])

    development_ids: set[str] = set()
    development_counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows(OUTPUTS / "preprocessed_dataset.csv"):
        check_identity(row, development_ids)
        assert row["record_id"] in raw_ids
        assert row["source_role"] == "text_development"
        assert (row["dataset"] == "enron" and row["label"] == "legitimate") or (
            row["dataset"] == "nazario" and row["collection_period"] in YEARS and row["label"] == "phishing"
        )
        development_counts[(row["dataset"], row["collection_period"], row["label"])] += 1

    role_counts: dict[str, int] = {}
    for name, expected_dataset, expected_role in (
        ("nazario_2025_temporal_evaluation", "nazario", "temporal_evaluation"),
        ("phishload_url_only", "phishload", "url_only"),
        ("trec_excluded", "trec", "excluded"),
    ):
        seen: set[str] = set()
        for row in rows(OUTPUTS / "dataset_roles" / f"{name}.csv"):
            check_identity(row, seen)
            assert row["record_id"] in raw_ids
            assert row["dataset"] == expected_dataset and row["source_role"] == expected_role
            if expected_dataset == "nazario":
                assert row["collection_period"] == "2025" and row["label"] == "phishing"
            if expected_dataset == "trec":
                assert row["label"] == "" and row["raw_label"] in {"0", "1"}
        role_counts[name] = len(seen)

    url_counts: Counter[str] = Counter()
    url_record_ids: set[str] = set()
    for row in rows(OUTPUTS / "extracted_urls.csv"):
        assert row["record_id"] in development_ids
        assert row["dataset"] in {"enron", "nazario"}
        url_counts[row["dataset"]] += 1
        url_record_ids.add(row["record_id"])

    raw_nazario_count = len(raw_nazario_ids)
    retained_nazario_count = len(raw_nazario_ids & development_ids)
    assert raw_nazario_count == contract.expected("nazario_historical_count", 2985)
    assert sum(raw_counts.values()) == report["counts"]["raw"]["rows"]
    assert len(development_ids) == report["counts"]["processed"]["text_development"]["rows"]
    assert sum(url_counts.values()) == report["url_summary"]["development_extracted_url_count"]
    assert report["text_development_ready_for_supervised_training"] is True
    assert all(all(checks.values()) for checks in report["invariants"]["record_identity"].values())
    assert all(value for key, value in report["invariants"].items() if key != "record_identity")
    assert {label for _, _, label in development_counts} == {"legitimate", "phishing"}
    print(json.dumps({
        "raw_rows": sum(raw_counts.values()),
        "raw_by_source_role_label": [{"dataset": d, "subset": s, "role": r, "label": l, "rows": count} for (d, s, r, l), count in sorted(raw_counts.items())],
        "development_rows": len(development_ids),
        "development_by_source_year_label": [{"dataset": d, "year": y, "label": l, "rows": count} for (d, y, l), count in sorted(development_counts.items())],
        "nazario_raw": raw_nazario_count,
        "nazario_retained": retained_nazario_count,
        "nazario_removed_by_deduplication": raw_nazario_count - retained_nazario_count,
        "role_artifact_rows": role_counts,
        "url_occurrences_by_source": dict(url_counts),
        "development_records_with_urls": len(url_record_ids),
        "readiness": report["text_development_ready_for_supervised_training"],
    }, indent=2))


if __name__ == "__main__":
    main()
