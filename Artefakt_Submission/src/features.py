
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.load_data import CSV_WRITER_KWARGS, NORMALIZED_COLUMNS
from src.utils import (
    EXTRACTED_URLS_OUTPUT_PATH,
    FEATURES_OUTPUT_DIR,
    LABELED_DATASET_INDEX_OUTPUT_PATH,
    PREPROCESSED_OUTPUT_PATH,
    TEXT_BODY_TFIDF_OUTPUT_PATH,
    TEXT_SUBJECT_BODY_TFIDF_OUTPUT_PATH,
    TEXT_SUBJECT_TFIDF_OUTPUT_PATH,
    TRAIN_TEST_SPLIT_OUTPUT_PATH,
    URL_FEATURE_TABLE_OUTPUT_PATH,
    ensure_directory,
    json_safe as _json_safe,
    read_csv_records as _read_csv_records,
    safe_text as _safe_text,
)


EXTRACTED_URL_COLUMNS = ["record_id", "dataset", "source_file", "field", "url", "url_index"]
EXPLICIT_LABELS = {"phishing", "legitimate"}
DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_TFIDF_CONFIG: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "token_pattern": r"(?u)\b\w\w+\b",
    "lowercase": False,
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.95,
    "max_features": 30000,
    "dtype": np.float32,
}
TEXT_VIEW_DEFINITIONS = {
    "subject": {
        "label": "subject only",
        "columns": ["subject"],
        "artifact_path": TEXT_SUBJECT_TFIDF_OUTPUT_PATH,
    },
    "body": {
        "label": "body only",
        "columns": ["body"],
        "artifact_path": TEXT_BODY_TFIDF_OUTPUT_PATH,
    },
    "subject_body": {
        "label": "subject + body combined",
        "columns": ["subject", "body"],
        "artifact_path": TEXT_SUBJECT_BODY_TFIDF_OUTPUT_PATH,
    },
}
IPV4_HOST_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def load_preprocessed_dataset() -> pd.DataFrame:
    return _read_csv_records(PREPROCESSED_OUTPUT_PATH, NORMALIZED_COLUMNS)


def load_extracted_urls() -> pd.DataFrame:
    return _read_csv_records(EXTRACTED_URLS_OUTPUT_PATH, EXTRACTED_URL_COLUMNS)


def build_labeled_dataset_index(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working_frame = frame.reindex(columns=NORMALIZED_COLUMNS, fill_value="").copy()
    for column in NORMALIZED_COLUMNS:
        working_frame[column] = working_frame[column].map(_safe_text)

    working_frame.insert(0, "row_index", range(len(working_frame)))
    explicit_label_mask = working_frame["label"].isin(EXPLICIT_LABELS)
    working_frame["has_explicit_label"] = explicit_label_mask
    working_frame["included_in_split"] = explicit_label_mask
    working_frame["exclusion_reason"] = np.where(explicit_label_mask, "", "missing_normalized_label")

    if working_frame["record_id"].duplicated().any():
        duplicate_rows = working_frame.loc[
            working_frame["record_id"].duplicated(keep=False),
            ["record_id"],
        ].head(5)
        raise ValueError(
            "Expected unique record_id values in the preprocessed dataset. "
            f"Found duplicates such as: {duplicate_rows.to_dict(orient='records')}"
        )

    labeled_frame = working_frame.loc[explicit_label_mask].reset_index(drop=True)
    return working_frame, labeled_frame


def _combine_subject_body(subject: str, body: str) -> str:
    subject = subject.strip()
    body = body.strip()
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


def build_text_views(frame: pd.DataFrame) -> dict[str, list[str]]:
    subject_texts = frame["subject"].map(_safe_text).tolist()
    body_texts = frame["body"].map(_safe_text).tolist()
    return {
        "subject": subject_texts,
        "body": body_texts,
        "subject_body": [
            _combine_subject_body(subject, body)
            for subject, body in zip(subject_texts, body_texts, strict=True)
        ],
    }


def build_tfidf_features(
    texts: list[str],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    tfidf_config = dict(DEFAULT_TFIDF_CONFIG if config is None else config)
    vectorizer = TfidfVectorizer(**tfidf_config)
    matrix = vectorizer.fit_transform(texts)

    metadata = {
        "vectorizer": "sklearn.feature_extraction.text.TfidfVectorizer",
        "parameters": _json_safe(tfidf_config),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "fit_scope": "full_labeled_corpus",
        "fit_scope_description": (
            "Fit on the full labeled corpus for milestone artifact generation. "
            "This is not a train-only benchmark fit."
        ),
        "fit_input_row_filter": "label in {'phishing', 'legitimate'}",
    }
    return matrix.tocsr(), metadata


def _dataset_identity_key(record_id: str) -> str:
    return record_id


def _extract_host(url: str) -> str:
    if url.startswith("mailto:"):
        return ""
    normalized_url = url if "://" in url else f"http://{url}" if url.startswith("www.") else url
    try:
        parsed = urlparse(normalized_url)
    except ValueError:
        return ""
    return (parsed.hostname or "").lower()


def _parse_url_safely(url: str) -> Any:
    normalized_url = url if "://" in url else f"http://{url}" if url.startswith("www.") else url
    try:
        return urlparse(normalized_url)
    except ValueError:
        return urlparse(normalized_url.replace("[", "%5B").replace("]", "%5D"))


def _is_non_default_port(parsed_result: Any) -> bool:
    try:
        port = parsed_result.port
    except ValueError:
        return True

    if port is None:
        return False
    default_ports = {"http": 80, "https": 443}
    return default_ports.get(parsed_result.scheme, None) != port


def build_url_feature_table(dataset_frame: pd.DataFrame, urls_frame: pd.DataFrame) -> pd.DataFrame:
    row_lookup = {
        _dataset_identity_key(row.record_id): int(row.row_index)
        for row in dataset_frame.itertuples(index=False)
    }
    base_row_lookup = {
        int(row.row_index): row
        for row in dataset_frame.itertuples(index=False)
    }

    aggregates: dict[int, dict[str, Any]] = {}
    missing_keys: list[str] = []

    for url_row in urls_frame.itertuples(index=False):
        key = _dataset_identity_key(_safe_text(url_row.record_id))
        row_index = row_lookup.get(key)
        if row_index is None:
            missing_keys.append(key)
            if len(missing_keys) >= 5:
                break
            continue

        raw_url = _safe_text(url_row.url)
        field = _safe_text(url_row.field)
        parsed = _parse_url_safely(raw_url)
        host = _extract_host(raw_url)

        bucket = aggregates.setdefault(
            row_index,
            {
                "_unique_urls": set(),
                "_unique_domains": set(),
                "url_count_total": 0,
                "url_count_subject": 0,
                "url_count_body": 0,
                "has_any_url": False,
                "has_mailto_url": False,
                "has_http_url": False,
                "has_https_url": False,
                "has_www_prefix_url": False,
                "max_url_length": 0,
                "url_length_sum": 0,
                "count_ip_host_urls": 0,
                "count_at_symbol_urls": 0,
                "count_hyphenated_domain_urls": 0,
                "count_query_urls": 0,
                "count_fragment_urls": 0,
                "count_non_default_port_urls": 0,
            },
        )

        bucket["url_count_total"] += 1
        bucket["has_any_url"] = True
        bucket["max_url_length"] = max(bucket["max_url_length"], len(raw_url))
        bucket["url_length_sum"] += len(raw_url)
        bucket["_unique_urls"].add(raw_url)
        if field == "subject":
            bucket["url_count_subject"] += 1
        elif field == "body":
            bucket["url_count_body"] += 1

        scheme = parsed.scheme.lower()
        if scheme == "mailto":
            bucket["has_mailto_url"] = True
        if scheme == "http":
            bucket["has_http_url"] = True
        if scheme == "https":
            bucket["has_https_url"] = True
        if raw_url.startswith("www."):
            bucket["has_www_prefix_url"] = True
        if "@" in raw_url:
            bucket["count_at_symbol_urls"] += 1
        if parsed.query:
            bucket["count_query_urls"] += 1
        if parsed.fragment:
            bucket["count_fragment_urls"] += 1
        if _is_non_default_port(parsed):
            bucket["count_non_default_port_urls"] += 1
        if host:
            bucket["_unique_domains"].add(host)
            if IPV4_HOST_PATTERN.match(host):
                bucket["count_ip_host_urls"] += 1
            if "-" in host:
                bucket["count_hyphenated_domain_urls"] += 1

    if missing_keys:
        raise ValueError(
            "Found extracted URL rows that do not map back to the preprocessed dataset: "
            f"{missing_keys!r}"
        )

    records: list[dict[str, Any]] = []
    for row_index in dataset_frame["row_index"].tolist():
        row = base_row_lookup[int(row_index)]
        aggregate = aggregates.get(int(row_index))
        if aggregate is None:
            records.append(
                {
                    "row_index": int(row_index),
                    "dataset": row.dataset,
                    "source_file": row.source_file,
                    "label": row.label,
                    "has_explicit_label": bool(row.has_explicit_label),
                    "included_in_split": bool(row.included_in_split),
                    "url_count_total": 0,
                    "url_count_subject": 0,
                    "url_count_body": 0,
                    "unique_url_count": 0,
                    "unique_domain_count": 0,
                    "has_any_url": False,
                    "has_mailto_url": False,
                    "has_http_url": False,
                    "has_https_url": False,
                    "has_www_prefix_url": False,
                    "max_url_length": 0,
                    "mean_url_length": 0.0,
                    "count_ip_host_urls": 0,
                    "count_at_symbol_urls": 0,
                    "count_hyphenated_domain_urls": 0,
                    "count_query_urls": 0,
                    "count_fragment_urls": 0,
                    "count_non_default_port_urls": 0,
                }
            )
            continue

        url_count_total = int(aggregate["url_count_total"])
        records.append(
            {
                "row_index": int(row_index),
                "dataset": row.dataset,
                "source_file": row.source_file,
                "label": row.label,
                "has_explicit_label": bool(row.has_explicit_label),
                "included_in_split": bool(row.included_in_split),
                "url_count_total": url_count_total,
                "url_count_subject": int(aggregate["url_count_subject"]),
                "url_count_body": int(aggregate["url_count_body"]),
                "unique_url_count": int(len(aggregate["_unique_urls"])),
                "unique_domain_count": int(len(aggregate["_unique_domains"])),
                "has_any_url": bool(aggregate["has_any_url"]),
                "has_mailto_url": bool(aggregate["has_mailto_url"]),
                "has_http_url": bool(aggregate["has_http_url"]),
                "has_https_url": bool(aggregate["has_https_url"]),
                "has_www_prefix_url": bool(aggregate["has_www_prefix_url"]),
                "max_url_length": int(aggregate["max_url_length"]),
                "mean_url_length": round(aggregate["url_length_sum"] / url_count_total, 6),
                "count_ip_host_urls": int(aggregate["count_ip_host_urls"]),
                "count_at_symbol_urls": int(aggregate["count_at_symbol_urls"]),
                "count_hyphenated_domain_urls": int(aggregate["count_hyphenated_domain_urls"]),
                "count_query_urls": int(aggregate["count_query_urls"]),
                "count_fragment_urls": int(aggregate["count_fragment_urls"]),
                "count_non_default_port_urls": int(aggregate["count_non_default_port_urls"]),
            }
        )

    return pd.DataFrame.from_records(records)


def create_train_test_split(
    labeled_index_frame: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    if labeled_index_frame.empty:
        raise ValueError("Cannot create a train/test split from an empty labeled dataset.")

    train_rows, test_rows = train_test_split(
        labeled_index_frame[["row_index", "label"]],
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=labeled_index_frame["label"],
    )

    train_split = train_rows.assign(split_set="train")
    test_split = test_rows.assign(split_set="test")
    split_frame = pd.concat([train_split, test_split], ignore_index=True)
    return split_frame.sort_values("row_index").reset_index(drop=True)


def _log(message: str) -> None:
    print(message, flush=True)


def write_feature_outputs(
    *,
    dataset_index_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    url_feature_table: pd.DataFrame,
    text_matrices: dict[str, sparse.csr_matrix],
) -> None:
    ensure_directory(FEATURES_OUTPUT_DIR)

    for view_name, matrix in text_matrices.items():
        sparse.save_npz(TEXT_VIEW_DEFINITIONS[view_name]["artifact_path"], matrix)

    csv_kwargs = {
        "index": False,
        "quoting": CSV_WRITER_KWARGS["quoting"],
        "lineterminator": CSV_WRITER_KWARGS["lineterminator"],
    }
    dataset_index_frame.to_csv(LABELED_DATASET_INDEX_OUTPUT_PATH, **csv_kwargs)
    split_frame.to_csv(TRAIN_TEST_SPLIT_OUTPUT_PATH, **csv_kwargs)
    url_feature_table.to_csv(URL_FEATURE_TABLE_OUTPUT_PATH, **csv_kwargs)


def main() -> None:
    from src.prepare_text_inputs import prepare

    report = prepare()
    _log(f"Step-17 text inputs prepared for {report['split_run_id']}.")


if __name__ == "__main__":
    main()
