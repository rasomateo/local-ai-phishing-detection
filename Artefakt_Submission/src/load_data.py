
from __future__ import annotations

import csv
import hashlib
import io
import mailbox
import re
import sys
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Sequence
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import (
    DIDECO_DIR,
    ENRON_ARCHIVE,
    MERGED_OUTPUT_PATH,
    NAZARIO_2025_MBOX,
    NAZARIO_DIR,
    PHISHLOAD_ARCHIVE,
    REPOSITORY_ROOT,
    TREC_ARCHIVE,
    ensure_directory,
    safe_text as _safe_text,
)


def merge_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()
    return pd.concat(valid_frames, ignore_index=True)


NORMALIZED_COLUMNS = [
    "record_id",
    "dataset",
    "source_subset",
    "source_file",
    "source_record",
    "source_role",
    "split",
    "subject",
    "body",
    "label",
    "raw_label",
    "label_provenance",
    "collection_period",
    "inclusion_reason",
]

SOURCE_ROLES = {
    "text_development",
    "temporal_evaluation",
    "url_only",
    "excluded",
    "external_evaluation",
    "rq2_evaluation",
}
NAZARIO_HISTORICAL_FILENAME_PATTERN = re.compile(r"phishing-(20(?:1[5-9]|2[0-4]))(?:\.txt|\.mbox)?", re.I)
DIDECO_SOURCE_DEFINITION = {
    "dataset": "dideco",
    "source_role": "rq2_evaluation",
    "expected_messages": 220,
    "expected_intent_categories": 20,
    "annotation_type": "multi_label_explicit_and_implicit_separate",
    "future_text_mapping": {"subject": "", "body": "displayed_text"},
    "parser_status": "format_verified_parser_reserved_for_later_rq2_step",
    "prohibited_uses": [
        "training",
        "cross_validation",
        "model_selection",
        "calibration",
        "threshold_selection",
        "fusion_policy_selection",
    ],
}

PHISHLOAD_WEBSITE_COLUMNS = [
    "id",
    "alexarank",
    "isPhish",
    "parent",
    "parentCount",
    "url",
    "urlHash",
    "urlBasedomain",
    "finalUrl",
    "finalUrlBasedomain",
    "name",
    "quality",
    "scanned",
    "rescan",
    "statusCode",
    "htmlContent",
    "loadtime",
    "phishtank_id",
    "phishtank_detailurl",
    "phishtank_submissiontime",
    "phishtank_verified",
    "phishtank_verifiedtime",
    "phishtank_isonline",
    "phishtank_targetname",
    "state",
    "duplicatedFrom",
    "created",
]

PHISHLOAD_PHISHING_STATES = {
    "phishonly",
    "phishing website",
    "malwaredownload",
}
PHISHLOAD_LEGITIMATE_STATES = {
    "original back",
}

CSV_WRITER_KWARGS = {
    "quoting": csv.QUOTE_ALL,
    "lineterminator": "\n",
}


def _repository_relative_posix(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def build_record_id(*, dataset: str, source_subset: str, source_file: str, source_record: str) -> str:
    source_path = Path(source_file)
    if source_path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", source_file):
        raise ValueError("record_id source_file must be repository-relative, not absolute")
    identity = "\x1f".join((dataset, source_subset, source_file.replace("\\", "/"), source_record))
    return f"{dataset}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _record_metadata(
    *,
    dataset: str,
    source_subset: str,
    source_file: str,
    source_record: str,
    source_role: str,
    label_provenance: str,
    collection_period: str,
    inclusion_reason: str,
) -> dict[str, str]:
    if source_role not in SOURCE_ROLES:
        raise ValueError(f"Unsupported source role: {source_role!r}")
    return {
        "record_id": build_record_id(
            dataset=dataset,
            source_subset=source_subset,
            source_file=source_file,
            source_record=source_record,
        ),
        "dataset": dataset,
        "source_subset": source_subset,
        "source_file": source_file,
        "source_record": source_record,
        "source_role": source_role,
        "label_provenance": label_provenance,
        "collection_period": collection_period,
        "inclusion_reason": inclusion_reason,
    }


def discover_historical_nazario_files(nazario_dir: Path = NAZARIO_DIR) -> list[tuple[int, Path]]:
    if not nazario_dir.exists():
        return []
    discovered: list[tuple[int, Path]] = []
    for path in nazario_dir.iterdir():
        if not path.is_file():
            continue
        match = NAZARIO_HISTORICAL_FILENAME_PATTERN.fullmatch(path.name)
        if match:
            discovered.append((int(match.group(1)), path))
    return sorted(discovered, key=lambda item: (item[0], item[1].as_posix().lower()))


def get_dataset_input_status(
    *,
    nazario_dir: Path = NAZARIO_DIR,
    nazario_2025_path: Path = NAZARIO_2025_MBOX,
    dideco_dir: Path = DIDECO_DIR,
) -> dict[str, Any]:
    historical = discover_historical_nazario_files(nazario_dir)
    missing_training: list[str] = []
    if not historical:
        missing_training.append("historical Nazario development files for 2015-2024")

    missing_evaluation: list[str] = []
    if not nazario_2025_path.exists():
        missing_evaluation.append("Nazario 2025 temporal evaluation source")

    dideco_files = sorted(
        (_repository_relative_posix(path) for path in dideco_dir.rglob("*") if path.is_file()),
        key=str.lower,
    ) if dideco_dir.exists() else []
    if not dideco_files:
        missing_evaluation.append("DIDECO RQ2 evaluation source")

    return {
        "historical_nazario_files": [_repository_relative_posix(path) for _, path in historical],
        "nazario_2025_file": _repository_relative_posix(nazario_2025_path) if nazario_2025_path.exists() else "",
        "dideco": {**DIDECO_SOURCE_DEFINITION, "files": dideco_files},
        "missing_training_inputs": missing_training,
        "missing_evaluation_inputs": missing_evaluation,
        "unsupported_sources": (
            [{"source": path, "reason": "verified acquisition; RQ2 parser and aggregation not implemented in Step 14"} for path in dideco_files]
        ),
    }


def _finalize_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    frame = frame.reindex(columns=NORMALIZED_COLUMNS, fill_value="")
    for column in NORMALIZED_COLUMNS:
        frame[column] = frame[column].map(_safe_text)
    return frame


def _empty_storage() -> dict[str, list[str]]:
    return {column: [] for column in NORMALIZED_COLUMNS}


def _append_record(storage: dict[str, list[str]], **values: Any) -> None:
    for column in NORMALIZED_COLUMNS:
        storage[column].append(_safe_text(values.get(column, "")))


def _finalize_storage(storage: dict[str, list[str]]) -> pd.DataFrame:
    if not any(storage.values()):
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.DataFrame(storage, columns=NORMALIZED_COLUMNS)


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _decode_payload(part: Message) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except Exception:
        pass

    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        if isinstance(raw_payload, str):
            return raw_payload
        return ""

    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _extract_message_body(message: Message) -> str:
    if message.is_multipart():
        text_parts: list[str] = []
        html_parts: list[str] = []

        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_filename():
                continue
            payload_text = _decode_payload(part)
            if not payload_text:
                continue

            content_type = part.get_content_type()
            if content_type == "text/plain":
                text_parts.append(payload_text)
            elif content_type == "text/html":
                html_parts.append(payload_text)

        combined = "\n\n".join(text_parts or html_parts)
        return combined.strip()

    return _decode_payload(message).strip()


def _parse_email_bytes(email_bytes: bytes) -> tuple[str, str]:
    message = BytesParser(policy=policy.default).parsebytes(email_bytes)
    subject = _decode_header_value(message.get("Subject"))
    body = _extract_message_body(message)
    return subject, body


def _normalize_simple_label(raw_label: str) -> str:
    value = raw_label.strip().lower()
    if not value:
        return ""

    explicit_phishing = {
        "phishing",
        "phish",
        "malicious",
        "spam-like phishing",
    }
    explicit_legitimate = {
        "legitimate",
        "ham",
        "benign",
    }
    if value in explicit_phishing or "phish" in value:
        return "phishing"
    if value in explicit_legitimate:
        return "legitimate"
    return ""


def load_enron_dataset() -> pd.DataFrame:
    return _finalize_frame(iter_enron_records())


def iter_enron_records() -> Iterable[dict[str, str]]:
    source_file = _repository_relative_posix(ENRON_ARCHIVE)
    with tarfile.open(ENRON_ARCHIVE, "r:gz") as archive:
        for archive_index, member in enumerate(archive, start=1):
            if not member.isfile():
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue

            subject, body = _parse_email_bytes(extracted.read())
            source_record = f"member={member.name};archive_index={archive_index}"
            yield {
                **_record_metadata(
                    dataset="enron",
                    source_subset="enron_mail_20150507",
                    source_file=source_file,
                    source_record=source_record,
                    source_role="text_development",
                    label_provenance="project_proxy_assignment:enron_benign_email",
                    collection_period="not_encoded_in_source",
                    inclusion_reason="benign-email proxy for supervised text development",
                ),
                "split": "",
                "subject": subject,
                "body": body,
                "label": "legitimate",
                "raw_label": "legitimate",
            }


def load_nazario_dataset() -> pd.DataFrame:
    return _finalize_frame(iter_nazario_records())


def iter_nazario_file_records(path: Path, *, year: int, source_file: str | None = None) -> Iterable[dict[str, str]]:
    source_file = source_file or _repository_relative_posix(path)
    source_subset = f"nazario_{year}"
    if 2015 <= year <= 2024:
        source_role = "text_development"
        inclusion_reason = "historical phishing email data eligible for supervised text development"
    elif year == 2025:
        source_role = "temporal_evaluation"
        inclusion_reason = "reserved positive-only temporal evaluation; never training or model selection"
    else:
        raise ValueError(f"Unsupported Nazario collection year: {year}")

    mailbox_source = mailbox.mbox(str(path), create=False)

    try:
        for index, message in enumerate(mailbox_source, start=1):
            subject = _decode_header_value(message.get("Subject"))
            body = _extract_message_body(message)
            source_record = f"message={index}"
            yield {
                **_record_metadata(
                    dataset="nazario",
                    source_subset=source_subset,
                    source_file=source_file,
                    source_record=source_record,
                    source_role=source_role,
                    label_provenance="source_corpus_membership:nazario_phishing",
                    collection_period=str(year),
                    inclusion_reason=inclusion_reason,
                ),
                "split": "",
                "subject": subject,
                "body": body,
                "label": "phishing",
                "raw_label": "phishing",
            }
    finally:
        mailbox_source.close()


def iter_historical_nazario_records() -> Iterable[dict[str, str]]:
    for year, path in discover_historical_nazario_files():
        yield from iter_nazario_file_records(path, year=year)


def iter_nazario_2025_records() -> Iterable[dict[str, str]]:
    if NAZARIO_2025_MBOX.exists():
        yield from iter_nazario_file_records(NAZARIO_2025_MBOX, year=2025)


def iter_nazario_records() -> Iterable[dict[str, str]]:
    yield from iter_historical_nazario_records()
    yield from iter_nazario_2025_records()


def _parse_sql_insert_rows(values_text: str) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    current_row: list[str | None] = []
    current_field: list[str] = []
    in_string = False
    escape_next = False
    quoted_field = False
    inside_row = False

    def flush_field() -> None:
        nonlocal current_field, quoted_field, current_row
        token = "".join(current_field)
        if quoted_field:
            value: str | None = token
        else:
            stripped = token.strip()
            value = None if stripped.upper() == "NULL" or stripped == "" else stripped
        current_row.append(value)
        current_field = []
        quoted_field = False

    for character in values_text:
        if in_string:
            if escape_next:
                current_field.append(character)
                escape_next = False
            elif character == "\\":
                escape_next = True
            elif character == "'":
                in_string = False
            else:
                current_field.append(character)
            continue

        if character == "'":
            in_string = True
            quoted_field = True
            continue

        if character == "(":
            inside_row = True
            current_row = []
            current_field = []
            quoted_field = False
            continue

        if not inside_row:
            continue

        if character == ",":
            flush_field()
            continue

        if character == ")":
            flush_field()
            rows.append(current_row)
            inside_row = False
            current_row = []
            current_field = []
            quoted_field = False
            continue

        current_field.append(character)

    return rows


def _extract_insert_values(sql_text: str, table_name: str) -> str:
    marker = f"INSERT INTO `{table_name}` VALUES "
    start = sql_text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find INSERT statement for table '{table_name}'.")

    start += len(marker)
    end_marker = f"\n/*!40000 ALTER TABLE `{table_name}` ENABLE KEYS */;"
    end = sql_text.find(end_marker, start)
    if end < 0:
        end = sql_text.find("\nUNLOCK TABLES;", start)
    if end < 0:
        raise ValueError(f"Could not locate end of INSERT statement for table '{table_name}'.")

    values_text = sql_text[start:end].rstrip()
    if values_text.endswith(";"):
        values_text = values_text[:-1]
    return values_text


def _load_phishload_state_lookup(sql_text: str) -> dict[str, str]:
    state_rows = _parse_sql_insert_rows(_extract_insert_values(sql_text, "states"))
    lookup: dict[str, str] = {}
    for row in state_rows:
        if len(row) < 2:
            continue
        state_id = _safe_text(row[0])
        lookup[state_id] = _safe_text(row[1])
    return lookup


def _build_phishload_raw_label(is_phish: str, state: str, state_name: str) -> str:
    parts = [f"isPhish={is_phish or ''}", f"state={state or ''}"]
    if state_name:
        parts.append(f"state_name={state_name}")
    return ";".join(parts)


def _normalize_phishload_label(is_phish: str, state_name: str) -> str:
    normalized_state = state_name.strip().lower()
    if normalized_state in PHISHLOAD_PHISHING_STATES:
        return "phishing"
    if normalized_state in PHISHLOAD_LEGITIMATE_STATES:
        return "legitimate"
    if is_phish == "0":
        return "legitimate"
    return ""


def load_phishload_dataset() -> pd.DataFrame:
    return _finalize_frame(iter_phishload_records())


def iter_phishload_records() -> Iterable[dict[str, str]]:
    with zipfile.ZipFile(PHISHLOAD_ARCHIVE) as archive:
        sql_name = "phishload_120322.sql"
        sql_text = archive.read(sql_name).decode("utf-8", errors="replace")

    archive_prefix = _repository_relative_posix(PHISHLOAD_ARCHIVE)
    state_lookup = _load_phishload_state_lookup(sql_text)
    website_rows = _parse_sql_insert_rows(_extract_insert_values(sql_text, "websites"))

    for row_index, row in enumerate(website_rows, start=1):
        row_map = dict(zip(PHISHLOAD_WEBSITE_COLUMNS, row))
        website_id = _safe_text(row_map.get("id"))
        is_phish = _safe_text(row_map.get("isPhish"))
        state = _safe_text(row_map.get("state"))
        state_name = state_lookup.get(state, "")
        body = _safe_text(row_map.get("htmlContent")) or _safe_text(row_map.get("finalUrl")) or _safe_text(row_map.get("url"))

        source_record = f"member={sql_name};website_id={website_id};row={row_index}"
        yield {
            **_record_metadata(
                dataset="phishload",
                source_subset="phishload_120322",
                source_file=archive_prefix,
                source_record=source_record,
                source_role="url_only",
                label_provenance="source_metadata:state_then_isPhish",
                collection_period="snapshot_2022-03-12",
                inclusion_reason="website/URL branch source; excluded from email-text development",
            ),
            "split": "",
            "subject": "",
            "body": body,
            "label": _normalize_phishload_label(is_phish=is_phish, state_name=state_name),
            "raw_label": _build_phishload_raw_label(is_phish=is_phish, state=state, state_name=state_name),
        }


def _build_trec_record(row: dict[str, str], member_name: str, index: int, archive_prefix: str) -> dict[str, str]:
    raw_label = _safe_text(row.get("label"))
    return {
        **_record_metadata(
            dataset="trec",
            source_subset="trec_processed_data",
            source_file=archive_prefix,
            source_record=f"member={member_name};row={index}",
            source_role="excluded",
            label_provenance="source_label_preserved_only:spam_ham_not_remapped",
            collection_period="not_encoded_in_source",
            inclusion_reason="spam/ham labels are not phishing/legitimate ground truth",
        ),
        "split": "",
        "subject": _safe_text(row.get("subject")),
        "body": _safe_text(row.get("message")),
        "label": "",
        "raw_label": raw_label,
    }


def load_trec_dataset() -> pd.DataFrame:
    return _finalize_frame(iter_trec_records())


def iter_trec_records() -> Iterable[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    archive_prefix = _repository_relative_posix(TREC_ARCHIVE)
    with zipfile.ZipFile(TREC_ARCHIVE) as archive:
        member_name = "processed_data.csv"
        with archive.open(member_name) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(text_handle)
            for index, row in enumerate(reader, start=1):
                yield _build_trec_record(
                    row=row,
                    member_name=member_name,
                    index=index,
                    archive_prefix=archive_prefix,
                )


def get_dataset_loaders() -> list[tuple[str, Callable[[], pd.DataFrame]]]:
    return [
        ("enron", load_enron_dataset),
        ("nazario", load_nazario_dataset),
        ("phishload", load_phishload_dataset),
        ("trec", load_trec_dataset),
    ]


def get_dataset_record_iterators() -> list[tuple[str, Callable[[], Iterable[dict[str, str]]]]]:
    return [
        ("enron", iter_enron_records),
        ("nazario_historical", iter_historical_nazario_records),
        ("nazario_2025", iter_nazario_2025_records),
        ("phishload", iter_phishload_records),
        ("trec", iter_trec_records),
    ]


def load_all_datasets() -> pd.DataFrame:
    frames = [loader() for _, loader in get_dataset_loaders()]
    merged_frame = merge_frames(frames)
    if merged_frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return merged_frame.reindex(columns=NORMALIZED_COLUMNS)


def write_merged_raw_dataset(return_merged_frame: bool = False) -> pd.DataFrame | None:
    ensure_directory(MERGED_OUTPUT_PATH.parent)

    accumulated_records: list[dict[str, str]] = []

    with MERGED_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COLUMNS, **CSV_WRITER_KWARGS)
        writer.writeheader()

        for dataset_name, iterator_factory in get_dataset_record_iterators():
            for record in iterator_factory():
                normalized_record = {column: _safe_text(record.get(column, "")) for column in NORMALIZED_COLUMNS}
                writer.writerow(normalized_record)

                if return_merged_frame:
                    accumulated_records.append(normalized_record)

    if not return_merged_frame:
        return None
    return _finalize_frame(accumulated_records)


def main() -> None:
    write_merged_raw_dataset(return_merged_frame=False)


if __name__ == "__main__":
    main()
