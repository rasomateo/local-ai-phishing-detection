
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from urlextract import URLExtract

from src.load_data import CSV_WRITER_KWARGS, NORMALIZED_COLUMNS, get_dataset_input_status, write_merged_raw_dataset
from src.utils import (
    DATASET_ROLES_OUTPUT_DIR,
    EXTRACTED_URLS_OUTPUT_PATH,
    MERGED_OUTPUT_PATH,
    NAZARIO_2025_TEMPORAL_OUTPUT_PATH,
    PHISHLOAD_URL_ONLY_OUTPUT_PATH,
    PREPROCESSED_OUTPUT_PATH,
    PREPROCESSING_REPORT_OUTPUT_PATH,
    TREC_EXCLUDED_OUTPUT_PATH,
    ensure_directory,
    json_safe as _json_safe,
    relative_to_repository as _relative_to_repository,
    safe_text as _safe_text,
)


TRAILING_URL_PUNCTUATION = ".,;:!?)]}>"
WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
LINE_BREAK_PATTERN = re.compile(r"\r\n?|\n")
REJECTED_BARE_DOMAIN_TLDS = {"pdf", "docx", "png", "zip", "txt", "csv", "py", "css", "js", "md"}
URL_EXTRACTION_PARSE_EXCEPTIONS = (UnicodeError, ValueError)
EXTRACTED_URL_COLUMNS = ["record_id", "dataset", "source_file", "field", "url", "url_index"]
DUPLICATE_RULE = "Exact duplicates are removed after normalization using the pair (subject, body), keeping the first occurrence in original row order."
COMPATIBILITY_NOTE = (
    "The existing merged raw CSV may contain malformed quoting in large text fields. "
    "If the file cannot be parsed, the preprocessing pipeline regenerates it once with the hardened raw writer "
    "and then continues from outputs/merged_raw_dataset.csv."
)


def normalize_text_field(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""

    text = LINE_BREAK_PATTERN.sub("\n", text)
    lines = [WHITESPACE_PATTERN.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip().lower()


def _read_normalized_csv(path: Path) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty.") from exc

        if header != NORMALIZED_COLUMNS:
            raise ValueError(f"{path} has an unexpected header: {header!r}")

        records: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(NORMALIZED_COLUMNS):
                raise ValueError(
                    f"{path} contains a malformed row at line {row_number}: expected {len(NORMALIZED_COLUMNS)} "
                    f"fields, found {len(row)}."
                )
            records.append(dict(zip(NORMALIZED_COLUMNS, row)))

    if not records:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.DataFrame.from_records(records, columns=NORMALIZED_COLUMNS)


def load_merged_raw_dataset() -> pd.DataFrame:
    ensure_directory(MERGED_OUTPUT_PATH.parent)

    if not MERGED_OUTPUT_PATH.exists():
        write_merged_raw_dataset(return_merged_frame=False)

    try:
        return _read_normalized_csv(MERGED_OUTPUT_PATH)
    except ValueError:
        write_merged_raw_dataset(return_merged_frame=False)
        return _read_normalized_csv(MERGED_OUTPUT_PATH)


def _build_summary_sentence(metrics: dict[str, Any], extracted_url_count: int) -> str:
    return (
        "According to the persisted preprocessing report, the merged raw dataset is reduced "
        f"from {metrics['input_rows']:,} rows to {metrics['output_rows']:,} rows, which means "
        f"that {metrics['duplicate_rows_removed']:,} exact duplicates are removed. In the same "
        f"step, {extracted_url_count:,} URLs are extracted and stored as a separate artifact."
    )


def preprocess_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_frame = frame.reindex(columns=NORMALIZED_COLUMNS, fill_value="").copy()
    for column in NORMALIZED_COLUMNS:
        working_frame[column] = working_frame[column].map(_safe_text)

    raw_subject = working_frame["subject"].copy()
    raw_body = working_frame["body"].copy()

    normalized_subject = raw_subject.map(normalize_text_field)
    normalized_body = raw_body.map(normalize_text_field)

    working_frame["subject"] = normalized_subject
    working_frame["body"] = normalized_body

    duplicate_mask = working_frame.duplicated(subset=["subject", "body"], keep="first")
    preprocessed_frame = working_frame.loc[~duplicate_mask].reset_index(drop=True)

    metrics: dict[str, Any] = {
        "input_rows": int(len(working_frame)),
        "output_rows": int(len(preprocessed_frame)),
        "rows_changed_during_normalization": int(((raw_subject != normalized_subject) | (raw_body != normalized_body)).sum()),
        "subject_changed_rows": int((raw_subject != normalized_subject).sum()),
        "body_changed_rows": int((raw_body != normalized_body).sum()),
        "duplicate_rows_removed": int(duplicate_mask.sum()),
        "subject_originally_missing_or_empty": int(raw_subject.eq("").sum()),
        "body_originally_missing_or_empty": int(raw_body.eq("").sum()),
        "subject_empty_after_normalization": int(normalized_subject.eq("").sum()),
        "body_empty_after_normalization": int(normalized_body.eq("").sum()),
        "all_empty_subject_body_after_normalization": int((normalized_subject.eq("") & normalized_body.eq("")).sum()),
        "duplicate_rule": DUPLICATE_RULE,
        "compatibility_note": COMPATIBILITY_NOTE,
    }
    return preprocessed_frame, metrics


def _count_by(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    counts = frame.groupby(columns, dropna=False, sort=True).size().reset_index(name="count")
    records: list[dict[str, Any]] = []
    for record in counts.to_dict(orient="records"):
        records.append({key: (int(value) if key == "count" else _safe_text(value)) for key, value in record.items()})
    return records


def _is_absolute_source_identifier(value: Any) -> bool:
    text = _safe_text(value)
    return Path(text).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", text))


def _record_identity_checks(frame: pd.DataFrame) -> dict[str, bool]:
    record_ids = frame["record_id"].map(_safe_text) if "record_id" in frame else pd.Series(dtype=str)
    source_files = frame["source_file"].map(_safe_text) if "source_file" in frame else pd.Series(dtype=str)
    return {
        "record_ids_nonempty": bool(not record_ids.eq("").any()),
        "record_ids_unique": bool(not record_ids.duplicated().any()),
        "record_ids_have_hash_only_format": bool(record_ids.str.fullmatch(r"[a-z0-9_]+:[0-9a-f]{64}").all()),
        "source_files_are_repository_relative": bool(not source_files.map(_is_absolute_source_identifier).any()),
    }


def _partition_raw_frames(raw_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    nazario_mask = raw_frame["dataset"].eq("nazario")
    return {
        "text_development": raw_frame.loc[raw_frame["source_role"].eq("text_development")].copy(),
        "nazario_2025_temporal_evaluation": raw_frame.loc[
            nazario_mask & raw_frame["collection_period"].eq("2025")
        ].copy(),
        "phishload_url_only": raw_frame.loc[raw_frame["dataset"].eq("phishload")].copy(),
        "trec_excluded": raw_frame.loc[raw_frame["dataset"].eq("trec")].copy(),
        "dideco_intent_evaluation": raw_frame.loc[raw_frame["dataset"].eq("dideco")].copy(),
    }


def build_preprocessing_report(
    *,
    raw_frame: pd.DataFrame,
    raw_artifacts: dict[str, pd.DataFrame],
    processed_artifacts: dict[str, pd.DataFrame],
    artifact_metrics: dict[str, dict[str, Any]],
    urls_frame: pd.DataFrame,
    input_status: dict[str, Any],
) -> dict[str, Any]:
    development = processed_artifacts["text_development"]
    development_labels = set(development.loc[development["label"].ne(""), "label"].tolist())
    has_both_classes = {"phishing", "legitimate"}.issubset(development_labels)

    all_record_checks = {"merged_raw_dataset": _record_identity_checks(raw_frame)}
    all_record_checks.update({name: _record_identity_checks(frame) for name, frame in processed_artifacts.items()})

    invariants = {
        "record_identity": all_record_checks,
        "all_raw_rows_assigned_to_one_role_artifact": sum(len(frame) for frame in raw_artifacts.values()) == len(raw_frame),
        "development_contains_only_text_development_role": set(development["source_role"].tolist()) <= {"text_development"},
        "phishload_absent_from_text_development": not development["dataset"].eq("phishload").any(),
        "phishload_absent_from_development_extracted_urls": not urls_frame["dataset"].eq("phishload").any(),
        "development_extracted_urls_reference_only_development_records": set(urls_frame["record_id"]).issubset(set(development["record_id"])),
        "reserved_external_sources_absent_from_text_development": not development["dataset"].isin(["ephishllm", "phishingpot"]).any(),
        "historical_nazario_2015_2024_role_is_text_development": set(
            raw_frame.loc[
                raw_frame["dataset"].eq("nazario")
                & raw_frame["collection_period"].isin([str(year) for year in range(2015, 2025)]),
                "source_role",
            ].tolist()
        ) <= {"text_development"},
        "nazario_2025_absent_from_text_development": not (
            development["dataset"].eq("nazario") & development["collection_period"].eq("2025")
        ).any(),
        "nazario_2025_role_is_temporal_evaluation": set(
            processed_artifacts["nazario_2025_temporal_evaluation"]["source_role"].tolist()
        ) <= {"temporal_evaluation"},
        "trec_normalized_labels_are_empty": processed_artifacts["trec_excluded"]["label"].eq("").all(),
        "dideco_role_is_rq2_evaluation": set(
            processed_artifacts["dideco_intent_evaluation"]["source_role"].tolist()
        ) <= {"rq2_evaluation"},
        "dideco_absent_from_text_development": not development["dataset"].eq("dideco").any(),
    }

    approved_development_sources = bool(
        (
            (development["dataset"].eq("enron") & development["label"].eq("legitimate"))
            | (
                development["dataset"].eq("nazario")
                   & development["collection_period"].isin([str(year) for year in range(2015, 2025)])
                   & development["label"].eq("phishing"))
        ).all()
    )
    invariants["development_contains_only_approved_sources_and_labels"] = approved_development_sources
    ready_for_training = has_both_classes and approved_development_sources and all(
        value for name, value in invariants.items()
        if name != "record_identity" and isinstance(value, bool)
    ) and all(all(checks.values()) for checks in all_record_checks.values())
    invariants["text_development_ready_for_supervised_training"] = ready_for_training

    training_blockers = [f"Missing training input: {source}" for source in input_status["missing_training_inputs"]]
    if not has_both_classes:
        training_blockers.append(
            "Text development does not contain both phishing and legitimate classes; historical Nazario phishing examples are absent."
        )
    if has_both_classes and not ready_for_training:
        training_blockers.append("Text development failed a source-role or record-identity invariant.")
    missing_evaluation_inputs = [
        f"Missing evaluation input: {source}" for source in input_status["missing_evaluation_inputs"]
    ]
    missing_evaluation_inputs.extend(
        f"Unsupported evaluation input {item['source']}: {item['reason']}"
        for item in input_status["unsupported_sources"]
    )

    processed_counts = {
        name: {
            "rows": int(len(frame)),
            "by_source_role_label": _count_by(frame, ["dataset", "source_subset", "source_role", "label"]),
            "by_source_raw_label": _count_by(frame, ["dataset", "raw_label"]),
        }
        for name, frame in processed_artifacts.items()
    }
    input_rows = int(len(raw_frame))
    output_rows = sum(len(frame) for frame in processed_artifacts.values())
    duplicate_rows_removed = sum(int(metrics["duplicate_rows_removed"]) for metrics in artifact_metrics.values())

    output_files = {
        "preprocessed_dataset": _relative_to_repository(PREPROCESSED_OUTPUT_PATH),
        "nazario_2025_temporal_evaluation": _relative_to_repository(NAZARIO_2025_TEMPORAL_OUTPUT_PATH),
        "phishload_url_only": _relative_to_repository(PHISHLOAD_URL_ONLY_OUTPUT_PATH),
        "trec_excluded": _relative_to_repository(TREC_EXCLUDED_OUTPUT_PATH),
        "extracted_urls": _relative_to_repository(EXTRACTED_URLS_OUTPUT_PATH),
        "preprocessing_report": _relative_to_repository(PREPROCESSING_REPORT_OUTPUT_PATH),
    }
    return {
        "pipeline": "src.preprocess",
        "step": "14",
        "input_files": {"merged_raw_dataset": _relative_to_repository(MERGED_OUTPUT_PATH)},
        "input_status": input_status,
        "row_summary": {
            "input_rows": input_rows,
            "output_rows_across_role_artifacts": int(output_rows),
            "rows_removed_total": input_rows - output_rows,
            "exact_duplicate_rows_removed": duplicate_rows_removed,
        },
        "counts": {
            "raw": {
                "rows": input_rows,
                "by_source_role_label": _count_by(raw_frame, ["dataset", "source_subset", "source_role", "label"]),
                "by_source_raw_label": _count_by(raw_frame, ["dataset", "raw_label"]),
            },
            "processed": processed_counts,
        },
        "normalization_by_artifact": {
            name: {key: value for key, value in metrics.items() if key != "compatibility_note"}
            for name, metrics in artifact_metrics.items()
        },
        "url_summary": {"development_extracted_url_count": int(len(urls_frame))},
        "invariants": invariants,
        "text_development_ready_for_supervised_training": ready_for_training,
        "training_readiness": {
            "text_development_ready_for_supervised_training": ready_for_training,
            "blockers": training_blockers,
        },
        "missing_future_evaluation_inputs": missing_evaluation_inputs,
        "step_14_blockers": training_blockers,
        "output_files": output_files,
    }


def _normalize_url_match(value: str) -> str:
    return value.rstrip(TRAILING_URL_PUNCTUATION)


@lru_cache(maxsize=1)
def _url_extractor() -> URLExtract:
    cache_dir = tempfile.mkdtemp(prefix=f"artefakt-urlextract-{os.getpid()}-")
    return URLExtract(extract_email=False, extract_localhost=False, cache_dns=False, cache_dir=cache_dir)


def _candidate_host(candidate: str) -> str:
    parse_target = candidate if "://" in candidate else f"http://{candidate}"
    try:
        return (urlsplit(parse_target).hostname or "").lower()
    except ValueError:
        return ""


def _is_scorable_url_candidate(candidate: str, text: str, start: int) -> bool:
    if not candidate:
        return False

    lowered = candidate.lower()
    if lowered.startswith("mailto:") or "@" in candidate:
        return False
    if start > 0 and text[start - 1] == "@":
        return False

    host = _candidate_host(candidate)
    if not host or host == "localhost" or "." not in host:
        return False

    labels = host.split(".")
    if any(not label for label in labels):
        return False

    tld = labels[-1]
    if tld.isdigit():
        return False
    if "://" not in candidate and not lowered.startswith("www.") and tld in REJECTED_BARE_DOMAIN_TLDS:
        return False

    return True


def _iter_url_extractor_matches(text: str) -> list[tuple[str, int]]:
    return [
        (raw_url, start)
        for raw_url, (start, _end) in _url_extractor().find_urls(text, get_indices=True, check_dns=False)
    ]


def iter_extracted_urls(text: str) -> list[tuple[int, str]]:
    safe_text = _safe_text(text)
    if not safe_text:
        return []

    extracted: list[tuple[int, str]] = []
    try:
        matches = _iter_url_extractor_matches(safe_text)
    except URL_EXTRACTION_PARSE_EXCEPTIONS:
        matches = []
        for chunk_match in re.finditer(r"\S+", safe_text):
            chunk = chunk_match.group(0)
            chunk_start = chunk_match.start()
            try:
                chunk_matches = _iter_url_extractor_matches(chunk)
            except URL_EXTRACTION_PARSE_EXCEPTIONS:
                continue
            matches.extend((raw_url, chunk_start + start) for raw_url, start in chunk_matches)

    for raw_url, start in matches:
        url = _normalize_url_match(raw_url)
        if not _is_scorable_url_candidate(url, safe_text, start):
            continue
        extracted.append((start, url))

    return sorted(extracted, key=lambda item: item[0])


def extract_urls(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        for field_name in ("subject", "body"):
            text = _safe_text(getattr(row, field_name))
            if not text:
                continue

            url_index = 0
            for _start, url in iter_extracted_urls(text):
                records.append(
                    {
                        "record_id": _safe_text(getattr(row, "record_id")),
                        "dataset": _safe_text(getattr(row, "dataset")),
                        "source_file": _safe_text(getattr(row, "source_file")),
                        "field": field_name,
                        "url": url,
                        "url_index": url_index,
                    }
                )
                url_index += 1

    if not records:
        return pd.DataFrame(columns=EXTRACTED_URL_COLUMNS)
    return pd.DataFrame.from_records(records, columns=EXTRACTED_URL_COLUMNS)


def write_preprocessed_outputs(
    *,
    raw_frame: pd.DataFrame,
    raw_artifacts: dict[str, pd.DataFrame],
    processed_artifacts: dict[str, pd.DataFrame],
    artifact_metrics: dict[str, dict[str, Any]],
    urls_frame: pd.DataFrame,
    input_status: dict[str, Any],
) -> dict[str, Any]:
    ensure_directory(PREPROCESSED_OUTPUT_PATH.parent)
    ensure_directory(DATASET_ROLES_OUTPUT_DIR)

    to_csv_kwargs = {
        "index": False,
        "quoting": CSV_WRITER_KWARGS["quoting"],
        "lineterminator": CSV_WRITER_KWARGS["lineterminator"],
    }
    processed_artifacts["text_development"].to_csv(PREPROCESSED_OUTPUT_PATH, **to_csv_kwargs)
    processed_artifacts["nazario_2025_temporal_evaluation"].to_csv(
        NAZARIO_2025_TEMPORAL_OUTPUT_PATH, **to_csv_kwargs
    )
    processed_artifacts["phishload_url_only"].to_csv(PHISHLOAD_URL_ONLY_OUTPUT_PATH, **to_csv_kwargs)
    processed_artifacts["trec_excluded"].to_csv(TREC_EXCLUDED_OUTPUT_PATH, **to_csv_kwargs)
    urls_frame.to_csv(EXTRACTED_URLS_OUTPUT_PATH, **to_csv_kwargs)
    report = build_preprocessing_report(
        raw_frame=raw_frame,
        raw_artifacts=raw_artifacts,
        processed_artifacts=processed_artifacts,
        artifact_metrics=artifact_metrics,
        urls_frame=urls_frame,
        input_status=input_status,
    )
    PREPROCESSING_REPORT_OUTPUT_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def main() -> dict[str, Any]:
    raw_frame = load_merged_raw_dataset()
    raw_artifacts = _partition_raw_frames(raw_frame)
    processed_artifacts: dict[str, pd.DataFrame] = {}
    artifact_metrics: dict[str, dict[str, Any]] = {}
    for name, artifact_frame in raw_artifacts.items():
        processed_artifacts[name], artifact_metrics[name] = preprocess_dataset(artifact_frame)

    urls_frame = extract_urls(processed_artifacts["text_development"])
    return write_preprocessed_outputs(
        raw_frame=raw_frame,
        raw_artifacts=raw_artifacts,
        processed_artifacts=processed_artifacts,
        artifact_metrics=artifact_metrics,
        urls_frame=urls_frame,
        input_status=get_dataset_input_status(),
    )


if __name__ == "__main__":
    main()
    print(_relative_to_repository(PREPROCESSING_REPORT_OUTPUT_PATH), flush=True)
