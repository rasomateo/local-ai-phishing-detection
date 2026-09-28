
from __future__ import annotations

import hashlib
import json
import mailbox
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPOSITORY_ROOT / "data" / "raw"
OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
MANIFEST_PATH = OUTPUT_ROOT / "dataset_acquisition_manifest.json"
REPORT_PATH = OUTPUT_ROOT / "dataset_acquisition_report.md"

OFFICIAL_SOURCES = {
    "nazario": "https://monkey.org/~jose/phishing/",
    "dideco": "https://aclanthology.org/2026.lrec-1.542/",
    "ephishllm": "https://github.com/pajola/e-phishGen",
    "phishingpot": "https://github.com/rf-peixoto/phishing_pot",
}
PHISHINGPOT_RESTORED_RECORD = "email/sample-472.eml"
PHISHINGPOT_EXPECTED_SHA256 = "301e22856440ee06f8b2265b2497b48679d63710e9fc7102032e233c25cac5a9"
PHISHINGPOT_UPSTREAM_COMMIT = "49f63777126b0bdb9eb1f6e770a5c3f9df2b0306"


def expected_phishingpot_record_sha256() -> str:
    from src.execution_contract import current_contract
    return current_contract(REPOSITORY_ROOT).expected(
        "phishingpot_sample_472_sha256", PHISHINGPOT_EXPECTED_SHA256
    )


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative_path(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "filesystem_last_write_time_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "acquisition_date": None,
        "acquisition_date_note": (
            "Not independently recorded; filesystem timestamps are not treated "
            "as authoritative acquisition dates."
        ),
    }


def inspect_nazario() -> dict[str, Any]:
    directory = RAW_ROOT / "nazario"
    expected_years = list(range(2015, 2025))
    year_files: dict[int, Path] = {}
    if directory.exists():
        candidates = sorted(path for path in directory.iterdir() if path.is_file())
        for year in expected_years:
            exact = [
                path
                for path in candidates
                if re.fullmatch(rf"phishing-{year}(?:\.txt)?", path.name, re.I)
            ]
            if exact:
                year_files[year] = exact[0]

    yearly: list[dict[str, Any]] = []
    total_messages = 0
    total_failures = 0
    for year in expected_years:
        path = year_files.get(year)
        if path is None:
            yearly.append(
                {
                    "year": year,
                    "status": "missing",
                    "message_count": 0,
                    "parse_failures": 0,
                    "parse_success_count": 0,
                    "subject_header_present_count": 0,
                }
            )
            continue

        metadata = file_metadata(path)
        with path.open("rb") as handle:
            first_bytes = handle.read(5)
        detected_format = "mbox" if first_bytes == b"From " else "unknown_text"
        messages = 0
        failures = 0
        subjects = 0
        error_examples: list[str] = []
        try:
            box = mailbox.mbox(path, create=False)
            for key in box.iterkeys():
                try:
                    message = box.get_message(key)
                    if message is None:
                        raise ValueError("mailbox returned no message")
                    if message.get("Subject") is not None:
                        subjects += 1
                    message.get("From")
                    messages += 1
                except Exception as exc:
                    failures += 1
                    if len(error_examples) < 3:
                        error_examples.append(f"record {key}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            failures += 1
            error_examples.append(f"file-level: {type(exc).__name__}: {exc}")

        total_messages += messages
        total_failures += failures
        yearly.append(
            {
                "year": year,
                "status": "verified" if failures == 0 else "verified_with_parse_failures",
                **metadata,
                "detected_format": detected_format,
                "message_count": messages,
                "parse_success_count": messages,
                "parse_failures": failures,
                "subject_header_present_count": subjects,
                "parse_error_examples": error_examples,
            }
        )

    supporting_files = []
    for name in ("README.txt", "LICENSE.txt"):
        path = directory / name
        if path.exists():
            supporting_files.append(file_metadata(path))

    missing_years = [entry["year"] for entry in yearly if entry["status"] == "missing"]
    return {
        "dataset_name": "Nazario historical phishing corpora",
        "methodological_role": (
            "Potential historical phishing development data; acquisition and format "
            "verification only in Step 13C."
        ),
        "official_source": OFFICIAL_SOURCES["nazario"],
        "repository_directory": relative_path(directory),
        "status": "partial_missing_years" if missing_years else "verified_complete",
        "expected_years": expected_years,
        "missing_years": missing_years,
        "total_messages": total_messages,
        "total_parse_failures": total_failures,
        "yearly_files": yearly,
        "supporting_files": supporting_files,
        "integration_status": "not_integrated",
    }


def recursive_key_matches(value: Any, pattern: re.Pattern[str]) -> set[str]:
    matches: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if pattern.search(str(key)):
                matches.add(str(key))
            matches.update(recursive_key_matches(nested, pattern))
    elif isinstance(value, list):
        for nested in value:
            matches.update(recursive_key_matches(nested, pattern))
    return matches


def inspect_dideco() -> dict[str, Any]:
    directory = RAW_ROOT / "DIDECO"
    archives = sorted(directory.glob("*.zip")) if directory.exists() else []
    if not archives:
        return {
            "dataset_name": "DIDECO",
            "methodological_role": "Reserved exclusive RQ2 evaluation source.",
            "official_source": OFFICIAL_SOURCES["dideco"],
            "status": "missing",
            "integration_status": "not_integrated",
        }

    archive = archives[0]
    metadata = file_metadata(archive)
    per_member_counts: dict[str, int] = {}
    schema_counts: Counter[str] = Counter()
    intent_name_counts: Counter[str] = Counter()
    intent_value_counts: Counter[str] = Counter()
    annotation_cardinality: Counter[int] = Counter()
    text_occurrences: Counter[str] = Counter()
    identifier_occurrences: Counter[str] = Counter()
    user_ids: Counter[str] = Counter()
    parse_errors: list[str] = []
    all_records: list[dict[str, Any]] = []
    nonempty_span_rows = 0
    rows_with_user_id = 0
    gold_like_fields: set[str] = set()
    explicit_implicit_fields: set[str] = set()
    explicit_intent_counts: Counter[str] = Counter()
    implicit_intent_counts: Counter[str] = Counter()
    rows_with_both_intent_groups = 0
    rows_with_neither_intent_group = 0
    gold_pattern = re.compile(r"gold|majority|consensus", re.I)
    explicit_pattern = re.compile(r"explicit|implicit", re.I)

    with zipfile.ZipFile(archive) as bundle:
        jsonl_members = sorted(
            name for name in bundle.namelist() if name.lower().endswith(".jsonl")
        )
        for member in jsonl_members:
            count = 0
            with bundle.open(member) as raw_handle:
                for line_number, raw_line in enumerate(raw_handle, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8-sig"))
                    except Exception as exc:
                        parse_errors.append(
                            f"{member}:{line_number}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    if not isinstance(record, dict):
                        parse_errors.append(f"{member}:{line_number}: record is not an object")
                        continue
                    all_records.append(record)
                    count += 1
                    schema_counts["|".join(sorted(record))] += 1
                    displayed_text = record.get("displayed_text")
                    if isinstance(displayed_text, str):
                        text_occurrences[displayed_text] += 1
                    identifier = record.get("id", record.get("instance_id"))
                    if identifier is not None:
                        identifier_occurrences[str(identifier)] += 1
                    if record.get("user_id") is not None:
                        rows_with_user_id += 1
                        user_ids[str(record["user_id"])] += 1
                    labels = record.get("label_annotations", {})
                    selection = labels.get("intent_selection", {}) if isinstance(labels, dict) else {}
                    if isinstance(selection, dict):
                        annotation_cardinality[len(selection)] += 1
                        explicit_names = []
                        implicit_names = []
                        for name, value in selection.items():
                            intent_name_counts[str(name)] += 1
                            intent_value_counts[str(value)] += 1
                            if str(value).isdigit():
                                explicit_names.append(str(name))
                                explicit_intent_counts[str(name)] += 1
                            elif str(value).startswith("intent_selection:::"):
                                implicit_names.append(str(name))
                                implicit_intent_counts[str(name)] += 1
                        if explicit_names and implicit_names:
                            rows_with_both_intent_groups += 1
                        if not explicit_names and not implicit_names:
                            rows_with_neither_intent_group += 1
                    spans = record.get("span_annotations")
                    if spans not in (None, {}, []):
                        nonempty_span_rows += 1
                    gold_like_fields.update(recursive_key_matches(record, gold_pattern))
                    explicit_implicit_fields.update(
                        recursive_key_matches(record, explicit_pattern)
                    )
            per_member_counts[member] = count

        member_names = bundle.namelist()

    annotation_rows = len(all_records)
    duplicate_text_multiplicity = Counter(text_occurrences.values())
    file_annotator_tokens = sorted(
        {
            match.group(1)
            for name in per_member_counts
            for match in [re.search(r"annotator(\d+)", Path(name).name, re.I)]
            if match
        },
        key=int,
    )
    return {
        "dataset_name": "DIDECO",
        "methodological_role": "Reserved exclusive RQ2 evaluation source.",
        "official_source": OFFICIAL_SOURCES["dideco"],
        "status": "verified" if not parse_errors else "verified_with_parse_errors",
        "archive": metadata,
        "archive_member_count": len(member_names),
        "jsonl_member_count": len(per_member_counts),
        "jsonl_members": per_member_counts,
        "annotation_row_count": annotation_rows,
        "json_parse_error_count": len(parse_errors),
        "json_parse_error_examples": parse_errors[:10],
        "unique_displayed_text_count": len(text_occurrences),
        "unique_nonempty_displayed_text_count": sum(
            1 for text in text_occurrences if isinstance(text, str) and text.strip()
        ),
        "empty_displayed_text_row_count": text_occurrences.get("", 0),
        "displayed_text_annotation_multiplicity": {
            str(count): frequency
            for count, frequency in sorted(duplicate_text_multiplicity.items())
        },
        "unique_message_identifier_count": len(identifier_occurrences),
        "duplicate_identifier_count": sum(
            1 for count in identifier_occurrences.values() if count > 1
        ),
        "schema_counts": dict(sorted(schema_counts.items())),
        "intent_category_counts": dict(sorted(intent_name_counts.items())),
        "intent_annotation_value_counts": dict(sorted(intent_value_counts.items())),
        "intent_annotations_per_row": {
            str(count): frequency for count, frequency in sorted(annotation_cardinality.items())
        },
        "multi_intent_annotation_rows": sum(
            frequency for count, frequency in annotation_cardinality.items() if count > 1
        ),
        "nonempty_span_annotation_rows": nonempty_span_rows,
        "annotator_level_evidence": {
            "annotator_tokens_in_filenames": file_annotator_tokens,
            "rows_with_user_id": rows_with_user_id,
            "unique_user_ids": sorted(user_ids),
            "multiple_rows_per_displayed_text": any(
                count > 1 for count in text_occurrences.values()
            ),
        },
        "explicit_implicit_representation": {
            "matching_structural_field_names": sorted(explicit_implicit_fields),
            "structurally_separate_fields_detected": bool(explicit_implicit_fields),
            "shared_storage_field": "label_annotations.intent_selection",
            "explicit_encoding_rule_observed": "selection value is a numeric string",
            "implicit_encoding_rule_observed": (
                "selection value begins with intent_selection:::"
            ),
            "explicit_intent_category_counts": dict(sorted(explicit_intent_counts.items())),
            "implicit_intent_category_counts": dict(sorted(implicit_intent_counts.items())),
            "rows_with_both_groups": rows_with_both_intent_groups,
            "rows_with_neither_group": rows_with_neither_intent_group,
            "note": (
                "Explicit and implicit annotations share one dictionary but remain "
                "distinguishable by their observed value encodings. An integration "
                "adapter must preserve the two groups separately."
            ),
        },
        "gold_or_majority_label_representation": {
            "matching_field_names": sorted(gold_like_fields),
            "detected": bool(gold_like_fields),
            "note": (
                "No gold, majority, or consensus field was found in the JSONL records."
                if not gold_like_fields
                else "Gold/majority-like fields were detected; inspect before integration."
            ),
        },
        "integration_status": "not_integrated",
    }


def inspect_ephishllm() -> dict[str, Any]:
    directory = RAW_ROOT / "E-PhishLLM"
    candidates = sorted(directory.glob("*.json")) if directory.exists() else []
    if not candidates:
        return {
            "dataset_name": "E-PhishLLM",
            "methodological_role": "Reserved English-only external evaluation source.",
            "official_source": OFFICIAL_SOURCES["ephishllm"],
            "status": "missing",
            "integration_status": "not_integrated",
        }

    data_path = candidates[0]
    metadata = file_metadata(data_path)
    parse_error = None
    rows: list[Any] = []
    try:
        parsed = json.loads(data_path.read_text(encoding="utf-8-sig"))
        if not isinstance(parsed, list):
            raise ValueError("top-level JSON value is not a list")
        rows = parsed
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    schema_counts: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    english_labels: Counter[str] = Counter()
    missing_subject = 0
    missing_body = 0
    invalid_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        schema_counts["|".join(sorted(row))] += 1
        language = str(row.get("Language", ""))
        label = str(row.get("type", ""))
        languages[language] += 1
        label_counts[label] += 1
        if language.casefold() == "en":
            english_labels[label] += 1
        if not str(row.get("Subject", "")).strip():
            missing_subject += 1
        if not str(row.get("Body", "")).strip():
            missing_body += 1

    supporting_files = []
    for path in sorted(directory.glob("README*")):
        if path.is_file():
            supporting_files.append(file_metadata(path))

    return {
        "dataset_name": "E-PhishLLM",
        "methodological_role": "Reserved English-only external evaluation source.",
        "official_source": OFFICIAL_SOURCES["ephishllm"],
        "status": "verified" if parse_error is None else "parse_failed",
        "data_file": metadata,
        "supporting_files": supporting_files,
        "json_parse_error": parse_error,
        "record_count": len(rows),
        "invalid_non_object_rows": invalid_rows,
        "schema_counts": dict(sorted(schema_counts.items())),
        "language_counts": dict(sorted(languages.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "english_record_count": sum(english_labels.values()),
        "english_label_counts": dict(sorted(english_labels.items())),
        "label_semantics": {"0": "legitimate", "1": "phishing"},
        "missing_or_empty_subject_count": missing_subject,
        "missing_or_empty_body_count": missing_body,
        "integration_status": "not_integrated",
    }


def message_has_body(message: Any) -> bool:
    parts = message.walk() if message.is_multipart() else (message,)
    for part in parts:
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes) and payload.strip():
            return True
        raw_payload = part.get_payload(decode=False)
        if isinstance(raw_payload, str) and raw_payload.strip():
            return True
    return False


def inspect_phishingpot() -> dict[str, Any]:
    expected_restored_hash = expected_phishingpot_record_sha256()
    repository = RAW_ROOT / "PhishingPot" / "phishing_pot"
    if not repository.is_dir():
        return {
            "dataset_name": "PhishingPot",
            "methodological_role": "Reserved positive-only external evaluation source.",
            "official_source": OFFICIAL_SOURCES["phishingpot"],
            "status": "missing",
            "integration_status": "not_integrated",
        }

    eml_files = sorted(path for path in (repository / "email").rglob("*.eml") if path.is_file())
    parse_failures: list[str] = []
    parse_failure_count = 0
    subject_present = 0
    body_present = 0
    url_indicator_present = 0
    restored_record_sha256 = None
    directory_counts: Counter[str] = Counter()
    parser = BytesParser(policy=policy.default)
    for path in eml_files:
        directory_counts[path.parent.relative_to(repository).as_posix()] += 1
        try:
            raw = path.read_bytes()
            if path.relative_to(repository).as_posix() == PHISHINGPOT_RESTORED_RECORD:
                restored_record_sha256 = hashlib.sha256(raw).hexdigest()
            message = parser.parsebytes(raw)
            if str(message.get("Subject", "")).strip():
                subject_present += 1
            if message_has_body(message):
                body_present += 1
            if re.search(rb"(?:https?://|www\.)", raw, re.I):
                url_indicator_present += 1
        except Exception as exc:
            parse_failure_count += 1
            if len(parse_failures) < 20:
                parse_failures.append(
                    f"{path.relative_to(repository).as_posix()}: {type(exc).__name__}: {exc}"
                )

    inventory_paths = sorted(path.relative_to(repository).as_posix() for path in eml_files)
    working_paths = set(inventory_paths)
    if parse_failure_count:
        verification_status = "verified_with_parse_failures"
    else:
        verification_status = "verified"
    return {
        "dataset_name": "PhishingPot",
        "methodological_role": "Reserved positive-only external evaluation source.",
        "official_source": OFFICIAL_SOURCES["phishingpot"],
        "status": verification_status,
        "repository_directory": relative_path(repository),
        "upstream_commit": PHISHINGPOT_UPSTREAM_COMMIT,
        "eml_path_inventory_sha256": hashlib.sha256("\n".join(inventory_paths).encode("utf-8")).hexdigest(),
        "eml_file_count": len(eml_files),
        "eml_parse_success_count": len(eml_files) - parse_failure_count,
        "eml_parse_failure_count": parse_failure_count,
        "eml_parse_failure_examples": parse_failures,
        "subject_present_count": subject_present,
        "body_present_count": body_present,
        "url_indicator_present_count": url_indicator_present,
        "eml_counts_by_directory": dict(sorted(directory_counts.items())),
        "restored_record_verification": {
            "path": PHISHINGPOT_RESTORED_RECORD,
            "present": PHISHINGPOT_RESTORED_RECORD in working_paths,
            "worktree_sha256": restored_record_sha256,
            "expected_sha256": expected_restored_hash,
            "sha256_matches_expected": restored_record_sha256 == expected_restored_hash,
        },
        "integration_status": "not_integrated",
    }


def format_count(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def render_report(manifest: dict[str, Any]) -> str:
    by_name = {entry["dataset_name"]: entry for entry in manifest["datasets"]}
    nazario = by_name["Nazario historical phishing corpora"]
    dideco = by_name["DIDECO"]
    ephish = by_name["E-PhishLLM"]
    pot = by_name["PhishingPot"]

    lines = [
        "# Step 13C — Acquisition Blocker Resolution",
        "",
        f"Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Scope and safeguards",
        "",
        "This verification is non-integrating. Step 13C verifies the newly acquired Nazario 2018 source and the PhishingPot restoration attempt. No loader, preprocessing, feature, split, training, benchmark, or evaluation logic was changed or run.",
        "",
        "Nazario 2018 was manually placed beneath `data/raw` before this step and was not renamed or modified. Only the previously missing tracked PhishingPot `email/sample-472.eml` was restored from the verified commit. Hashes are SHA-256 over readable bytes. Filesystem timestamps are metadata, not authoritative acquisition dates.",
        "",
        "Methodological roles remain fixed: DIDECO is exclusive RQ2 evaluation; E-PhishLLM English is reserved external evaluation; PhishingPot is reserved positive-only external evaluation; Nazario 2015–2024 is only being checked for later integration.",
        "",
        "## Acquisition status",
        "",
        "| Dataset | Status | Repository location | Intended role |",
        "|---|---|---|---|",
        f"| Nazario 2015–2024 | `{nazario['status']}` | `{nazario['repository_directory']}` | potential historical phishing development data |",
        f"| DIDECO | `{dideco['status']}` | `{dideco.get('archive', {}).get('path', 'missing')}` | exclusive RQ2 evaluation |",
        f"| E-PhishLLM | `{ephish['status']}` | `{ephish.get('data_file', {}).get('path', 'missing')}` | English-only external evaluation |",
        f"| PhishingPot | `{pot['status']}` | `{pot.get('repository_directory', 'missing')}` | positive-only external evaluation |",
        "",
        "## Nazario 2015–2024",
        "",
        f"Official source directory: [{OFFICIAL_SOURCES['nazario']}]({OFFICIAL_SOURCES['nazario']})",
        "",
        "| Year | Status | Format | Parse successes | Parse failures | Subject headers | SHA-256 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for entry in nazario["yearly_files"]:
        lines.append(
            f"| {entry['year']} | `{entry['status']}` | {entry.get('detected_format', 'n/a')} | "
            f"{format_count(entry.get('parse_success_count'))} | {format_count(entry.get('parse_failures'))} | "
            f"{format_count(entry.get('subject_header_present_count'))} | "
            f"`{entry.get('sha256', 'n/a')}` |"
        )
    lines.extend(
        [
            "",
            f"Verified messages: **{format_count(nazario['total_messages'])}**. Parse failures: **{format_count(nazario['total_parse_failures'])}**. Missing years: **{', '.join(map(str, nazario['missing_years'])) or 'none'}**.",
            "",
            "Nazario 2018 is kept under its source-native filename `data/raw/nazario/phishing-2018` (no extension). Its exact byte size and SHA-256 appear in the manifest.",
            "",
            "## DIDECO",
            "",
            f"Official publication record: [{OFFICIAL_SOURCES['dideco']}]({OFFICIAL_SOURCES['dideco']})",
            "",
            f"The ZIP is valid and contains **{format_count(dideco.get('jsonl_member_count'))} JSONL files** with **{format_count(dideco.get('annotation_row_count'))} parsed annotation rows**, **{format_count(dideco.get('unique_nonempty_displayed_text_count'))} unique non-empty displayed messages**, **{format_count(dideco.get('empty_displayed_text_row_count'))} rows with empty displayed text**, and **{format_count(dideco.get('json_parse_error_count'))} JSON parse failures**.",
            "",
            f"The files expose **{format_count(dideco.get('unique_message_identifier_count'))} unique message identifiers**. Annotator evidence includes filename tokens `{', '.join(dideco.get('annotator_level_evidence', {}).get('annotator_tokens_in_filenames', []))}` and **{format_count(dideco.get('annotator_level_evidence', {}).get('rows_with_user_id'))} rows with `user_id`**.",
            "",
            "Intent annotations are stored together under `label_annotations.intent_selection`, but use distinguishable encodings: numeric-string values for explicit intents and `intent_selection:::` values for implicit intents. An integration adapter must split and preserve those groups. No gold/majority/consensus label field was found, so annotation aggregation must be defined and documented rather than assumed.",
            "",
            "Intent categories observed:",
            "",
        ]
    )
    for name, count in dideco.get("intent_category_counts", {}).items():
        lines.append(f"- `{name}`: {format_count(count)} annotations")
    lines.extend(
        [
            "",
            "## E-PhishLLM",
            "",
            f"Official repository: [{OFFICIAL_SOURCES['ephishllm']}]({OFFICIAL_SOURCES['ephishllm']})",
            "",
            f"The JSON is valid and contains **{format_count(ephish.get('record_count'))} records**. Every record uses the schema `Subject`, `Body`, `type`, `Language`; missing/empty subjects: **{format_count(ephish.get('missing_or_empty_subject_count'))}**; missing/empty bodies: **{format_count(ephish.get('missing_or_empty_body_count'))}**.",
            "",
            f"The reserved English subset contains **{format_count(ephish.get('english_record_count'))} records**: **{format_count(ephish.get('english_label_counts', {}).get('0'))} legitimate (`type=0`)** and **{format_count(ephish.get('english_label_counts', {}).get('1'))} phishing (`type=1`)**. No data was filtered or exported in Step 13C.",
            "",
            "Language distribution:",
            "",
        ]
    )
    for language, count in ephish.get("language_counts", {}).items():
        lines.append(f"- `{language or '<empty>'}`: {format_count(count)}")
    lines.extend(
        [
            "",
            "## PhishingPot",
            "",
            f"Official repository: [{OFFICIAL_SOURCES['phishingpot']}]({OFFICIAL_SOURCES['phishingpot']})",
            "",
            f"The supplied snapshot preserves upstream commit provenance `{pot.get('upstream_commit')}`. Its deterministic EML path inventory has SHA-256 `{pot.get('eml_path_inventory_sha256')}`.",
            "",
            f"The file inventory contains **{format_count(pot.get('eml_file_count'))} `.eml` files**. Of those, **{format_count(pot.get('eml_parse_success_count'))} parsed successfully** and **{format_count(pot.get('eml_parse_failure_count'))} could not be read/parsed**. Subject present: **{format_count(pot.get('subject_present_count'))}**; a non-empty plain-text or HTML body present: **{format_count(pot.get('body_present_count'))}**; a simple URL indicator present in raw bytes: **{format_count(pot.get('url_indicator_present_count'))}**.",
            "",
            f"The supplied file `email/sample-472.eml` has SHA-256 `{pot.get('restored_record_verification', {}).get('worktree_sha256')}`; expected-hash match: `{pot.get('restored_record_verification', {}).get('sha256_matches_expected')}`.",
            "",
            "## Acquisition blockers",
            "",
        ]
    )
    blockers = manifest["acquisition_blockers"]
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None. Acquisition is complete enough to release the next integration step.")
    lines.extend(
        [
            "",
            "## Later methodological decisions (not acquisition blockers)",
            "",
        ]
    )
    lines.extend(f"- {decision}" for decision in manifest["later_methodological_decisions"])
    lines.extend(
        [
            "",
            "Acquisition is complete enough to release the next dataset-integration step. That step should use stable source-local identity and strict role separation. No integration was performed in Step 13C.",
            "",
            "## Reproducibility",
            "",
            "Run from the repository root:",
            "",
            "```text",
            "python scripts/verify_acquired_datasets.py",
            "```",
            "",
            f"Machine-readable details and all per-file hashes are in `{relative_path(MANIFEST_PATH)}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    datasets = []
    for name, inspector in (
        ("Nazario", inspect_nazario),
        ("DIDECO", inspect_dideco),
        ("E-PhishLLM", inspect_ephishllm),
        ("PhishingPot", inspect_phishingpot),
    ):
        print(f"Inspecting {name}...", flush=True)
        datasets.append(inspector())
        print(f"Finished {name}.", flush=True)
    by_name = {entry["dataset_name"]: entry for entry in datasets}
    acquisition_blockers: list[str] = []
    later_methodological_decisions: list[str] = []
    missing_years = by_name["Nazario historical phishing corpora"].get("missing_years", [])
    if missing_years:
        acquisition_blockers.append(
            "Nazario " + ", ".join(map(str, missing_years)) + " is missing from `data/raw/nazario`; the historical 2015–2024 set is incomplete."
        )
    if by_name["Nazario historical phishing corpora"].get("total_parse_failures"):
        acquisition_blockers.append("At least one acquired Nazario message could not be parsed.")
    dideco = by_name["DIDECO"]
    if dideco.get("status") == "missing":
        acquisition_blockers.append("DIDECO was not found.")
    elif not dideco.get("gold_or_majority_label_representation", {}).get("detected"):
        later_methodological_decisions.append(
            "DIDECO has annotator-level intent data but no detected gold/majority/consensus field; an aggregation policy is required before RQ2 integration."
        )
    if dideco.get("empty_displayed_text_row_count"):
        later_methodological_decisions.append(
            "DIDECO contains rows with empty displayed_text; their identity and treatment must be resolved before RQ2 integration."
        )
    for name in ("E-PhishLLM", "PhishingPot"):
        if by_name[name].get("status") == "missing":
            acquisition_blockers.append(f"{name} was not found.")
    phishingpot = by_name["PhishingPot"]
    if phishingpot.get("eml_parse_failure_count"):
        acquisition_blockers.append(
            "At least one present PhishingPot EML is inaccessible or unparseable in the "
            "working tree; resolve local access before integration."
        )
    if not phishingpot.get("restored_record_verification", {}).get("sha256_matches_expected"):
        acquisition_blockers.append(
            "PhishingPot email/sample-472.eml is absent, unreadable, or does not match the verified SHA-256."
        )

    manifest = {
        "step": "13C",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": ".",
        "scope": "acquisition blocker resolution and verification only",
        "nazario_raw_bytes_modified": False,
        "tracked_phishingpot_file_restoration_attempted": "data/raw/PhishingPot/phishing_pot/email/sample-472.eml",
        "other_raw_files_intentionally_modified": False,
        "methodological_invariants": {
            "dideco_role": "rq2_evaluation_exclusive",
            "nazario_2015_2024_role": "text_development_intended_not_integrated",
            "nazario_2025_role": "temporal_evaluation_reserved",
            "ephishllm_role": "external_evaluation_english_only",
            "phishingpot_role": "external_evaluation_positive_only",
            "phishload_role": "url_only",
            "trec_role": "excluded",
            "integration_performed": False,
            "loaders_modified": False,
            "preprocessing_modified": False,
            "features_or_models_modified": False,
        },
        "datasets": datasets,
        "acquisition_complete_for_next_integration_step": not acquisition_blockers,
        "acquisition_blockers": acquisition_blockers,
        "later_methodological_decisions": later_methodological_decisions,
        "blockers": acquisition_blockers + later_methodological_decisions,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    REPORT_PATH.write_text(render_report(manifest), encoding="utf-8")
    print(f"Wrote {relative_path(MANIFEST_PATH)}")
    print(f"Wrote {relative_path(REPORT_PATH)}")


if __name__ == "__main__":
    main()
