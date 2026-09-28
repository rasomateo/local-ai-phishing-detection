
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
OUTPUTS_DIR = REPOSITORY_ROOT / "outputs"
FEATURES_OUTPUT_DIR = OUTPUTS_DIR / "features"
BENCHMARK_OUTPUT_DIR = OUTPUTS_DIR / "benchmark"
ARTIFACT_OUTPUT_DIR = OUTPUTS_DIR / "artifact"
ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "analysis"
URL_TEXT_OUTPUT_DIR = OUTPUTS_DIR / "url_text"
DATASET_ROLES_OUTPUT_DIR = OUTPUTS_DIR / "dataset_roles"
SPLITS_OUTPUT_DIR = OUTPUTS_DIR / "splits"
SPLITS_MANIFEST_OUTPUT_PATH = SPLITS_OUTPUT_DIR / "manifest.json"
URL_BENCHMARK_OUTPUT_DIR = OUTPUTS_DIR / "url_benchmark"
URL_ARTIFACT_OUTPUT_DIR = OUTPUTS_DIR / "url_artifact"
URL_EXPLANATIONS_OUTPUT_DIR = OUTPUTS_DIR / "url_explanations"
CALIBRATED_TEXT_ARTIFACT_OUTPUT_DIR = OUTPUTS_DIR / "calibrated_text_artifact"
CALIBRATED_URL_ARTIFACT_OUTPUT_DIR = OUTPUTS_DIR / "calibrated_url_artifact"
CALIBRATION_OUTPUT_DIR = OUTPUTS_DIR / "calibration"
UNIFIED_ARTIFACT_OUTPUT_DIR = OUTPUTS_DIR / "unified_artifact"
DOCS_DIR = REPOSITORY_ROOT / "docs"

ENRON_ARCHIVE = DATA_RAW_DIR / "enron" / "enron_mail_20150507.tar.gz"
NAZARIO_DIR = DATA_RAW_DIR / "nazario"
NAZARIO_2025_MBOX = NAZARIO_DIR / "nazario2025.txt"
DIDECO_DIR = DATA_RAW_DIR / "DIDECO"
NAZARIO_MBOX = NAZARIO_2025_MBOX
PHISHLOAD_ARCHIVE = DATA_RAW_DIR / "phishload" / "current.zip"
TREC_ARCHIVE = DATA_RAW_DIR / "trec" / "archive.zip"

MERGED_OUTPUT_PATH = OUTPUTS_DIR / "merged_raw_dataset.csv"
PREPROCESSED_OUTPUT_PATH = OUTPUTS_DIR / "preprocessed_dataset.csv"
EXTRACTED_URLS_OUTPUT_PATH = OUTPUTS_DIR / "extracted_urls.csv"
PREPROCESSING_REPORT_OUTPUT_PATH = OUTPUTS_DIR / "preprocessing_report.json"
NAZARIO_2025_TEMPORAL_OUTPUT_PATH = DATASET_ROLES_OUTPUT_DIR / "nazario_2025_temporal_evaluation.csv"
PHISHLOAD_URL_ONLY_OUTPUT_PATH = DATASET_ROLES_OUTPUT_DIR / "phishload_url_only.csv"
TREC_EXCLUDED_OUTPUT_PATH = DATASET_ROLES_OUTPUT_DIR / "trec_excluded.csv"
TEXT_SUBJECT_TFIDF_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "text_subject_tfidf.npz"
TEXT_BODY_TFIDF_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "text_body_tfidf.npz"
TEXT_SUBJECT_BODY_TFIDF_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "text_subject_body_tfidf.npz"
LABELED_DATASET_INDEX_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "labeled_dataset_index.csv"
TRAIN_TEST_SPLIT_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "train_test_split.csv"
URL_FEATURE_TABLE_OUTPUT_PATH = FEATURES_OUTPUT_DIR / "url_feature_table.csv"
MODEL_RESULTS_OUTPUT_PATH = BENCHMARK_OUTPUT_DIR / "model_results.csv"
MODEL_RESULTS_DETAILED_OUTPUT_PATH = BENCHMARK_OUTPUT_DIR / "model_results_detailed.csv"
BEST_CONFIGURATION_OUTPUT_PATH = BENCHMARK_OUTPUT_DIR / "best_configuration.json"
CONFUSION_MATRIX_BEST_OUTPUT_PATH = BENCHMARK_OUTPUT_DIR / "confusion_matrix_best.csv"
ARTIFACT_MODEL_OUTPUT_PATH = ARTIFACT_OUTPUT_DIR / "model.joblib"
ARTIFACT_VECTORIZER_OUTPUT_PATH = ARTIFACT_OUTPUT_DIR / "vectorizer.joblib"
ARTIFACT_MANIFEST_OUTPUT_PATH = ARTIFACT_OUTPUT_DIR / "artifact_manifest.json"
FALSE_NEGATIVES_OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "false_negatives.csv"
FALSE_NEGATIVE_CATEGORY_COUNTS_OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "false_negative_category_counts.csv"

URL_DOCUMENT_TABLE_OUTPUT_PATH = URL_TEXT_OUTPUT_DIR / "url_document_table.csv"
URL_TEXT_MANIFEST_OUTPUT_PATH = URL_TEXT_OUTPUT_DIR / "url_text_manifest.json"
URL_TEXT_PREPROCESSING_REPORT_PATH = DOCS_DIR / "url_text_preprocessing_report.md"
THESIS_URL_TEXT_PREPROCESSING_NOTE_PATH = DOCS_DIR / "thesis_url_text_preprocessing_note.md"
DEFENSIO_URL_TEXT_PREPROCESSING_NOTE_PATH = DOCS_DIR / "defensio_url_text_preprocessing_note.md"

URL_MODEL_RESULTS_OUTPUT_PATH = URL_BENCHMARK_OUTPUT_DIR / "model_results.csv"
URL_MODEL_RESULTS_DETAILED_OUTPUT_PATH = URL_BENCHMARK_OUTPUT_DIR / "model_results_detailed.csv"
URL_BEST_CONFIGURATION_OUTPUT_PATH = URL_BENCHMARK_OUTPUT_DIR / "best_configuration.json"
URL_CONFUSION_MATRIX_BEST_OUTPUT_PATH = URL_BENCHMARK_OUTPUT_DIR / "confusion_matrix_best.csv"
URL_BENCHMARK_MANIFEST_OUTPUT_PATH = URL_BENCHMARK_OUTPUT_DIR / "url_benchmark_manifest.json"
URL_ARTIFACT_MODEL_OUTPUT_PATH = URL_ARTIFACT_OUTPUT_DIR / "model.joblib"
URL_ARTIFACT_VECTORIZER_OUTPUT_PATH = URL_ARTIFACT_OUTPUT_DIR / "vectorizer.joblib"
URL_ARTIFACT_MANIFEST_OUTPUT_PATH = URL_ARTIFACT_OUTPUT_DIR / "url_artifact_manifest.json"
URL_SAMPLE_EXPLANATIONS_JSON_OUTPUT_PATH = URL_EXPLANATIONS_OUTPUT_DIR / "sample_url_explanations.json"
URL_SAMPLE_EXPLANATIONS_MARKDOWN_OUTPUT_PATH = URL_EXPLANATIONS_OUTPUT_DIR / "sample_url_explanations.md"
URL_EXPLANATION_MANIFEST_OUTPUT_PATH = URL_EXPLANATIONS_OUTPUT_DIR / "url_explanation_manifest.json"
URL_BENCHMARK_REPORT_PATH = DOCS_DIR / "url_benchmark_report.md"
THESIS_URL_BENCHMARK_NOTE_PATH = DOCS_DIR / "thesis_url_benchmark_note.md"
DEFENSIO_URL_BENCHMARK_NOTE_PATH = DOCS_DIR / "defensio_url_benchmark_note.md"
URL_EXPLANATION_REPORT_PATH = DOCS_DIR / "url_explanation_report.md"
THESIS_URL_EXPLANATION_NOTE_PATH = DOCS_DIR / "thesis_url_explanation_note.md"
DEFENSIO_URL_EXPLANATION_NOTE_PATH = DOCS_DIR / "defensio_url_explanation_note.md"

CALIBRATED_TEXT_MODEL_OUTPUT_PATH = CALIBRATED_TEXT_ARTIFACT_OUTPUT_DIR / "calibrated_model.joblib"
CALIBRATED_TEXT_RAW_MODEL_OUTPUT_PATH = CALIBRATED_TEXT_ARTIFACT_OUTPUT_DIR / "raw_model.joblib"
CALIBRATED_TEXT_VECTORIZER_OUTPUT_PATH = CALIBRATED_TEXT_ARTIFACT_OUTPUT_DIR / "vectorizer.joblib"
CALIBRATED_TEXT_MANIFEST_OUTPUT_PATH = CALIBRATED_TEXT_ARTIFACT_OUTPUT_DIR / "calibrated_artifact_manifest.json"
CALIBRATED_URL_MODEL_OUTPUT_PATH = CALIBRATED_URL_ARTIFACT_OUTPUT_DIR / "calibrated_model.joblib"
CALIBRATED_URL_RAW_MODEL_OUTPUT_PATH = CALIBRATED_URL_ARTIFACT_OUTPUT_DIR / "raw_model.joblib"
CALIBRATED_URL_VECTORIZER_OUTPUT_PATH = CALIBRATED_URL_ARTIFACT_OUTPUT_DIR / "vectorizer.joblib"
CALIBRATED_URL_MANIFEST_OUTPUT_PATH = CALIBRATED_URL_ARTIFACT_OUTPUT_DIR / "calibrated_artifact_manifest.json"
TEXT_PROBABILITY_EXAMPLES_OUTPUT_PATH = CALIBRATION_OUTPUT_DIR / "text_probability_examples.json"
URL_PROBABILITY_EXAMPLES_OUTPUT_PATH = CALIBRATION_OUTPUT_DIR / "url_probability_examples.json"
CALIBRATION_SUMMARY_OUTPUT_PATH = CALIBRATION_OUTPUT_DIR / "calibration_summary.json"
PROBABILITY_CALIBRATION_REPORT_PATH = DOCS_DIR / "probability_calibration_report.md"
THESIS_PROBABILITY_CALIBRATION_NOTE_PATH = DOCS_DIR / "thesis_probability_calibration_note.md"
DEFENSIO_PROBABILITY_CALIBRATION_NOTE_PATH = DOCS_DIR / "defensio_probability_calibration_note.md"
UNIFIED_ARTIFACT_MANIFEST_OUTPUT_PATH = UNIFIED_ARTIFACT_OUTPUT_DIR / "unified_artifact_manifest.json"
UNIFIED_PREDICTION_EXAMPLES_OUTPUT_PATH = UNIFIED_ARTIFACT_OUTPUT_DIR / "unified_prediction_examples.json"
UNIFIED_ARTIFACT_REPORT_PATH = DOCS_DIR / "unified_artifact_report.md"
THESIS_UNIFIED_ARTIFACT_NOTE_PATH = DOCS_DIR / "thesis_unified_artifact_note.md"
DEFENSIO_UNIFIED_ARTIFACT_NOTE_PATH = DOCS_DIR / "defensio_unified_artifact_note.md"


def ensure_directory(path: Path) -> Path:

    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_repository(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def read_csv_records(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{path} has an unexpected header: {list(frame.columns)!r}")
    return frame.reindex(columns=expected_columns, fill_value="")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def parse_bool(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        normalized = ""
    else:
        normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, type):
        try:
            return value.tolist()
        except (ValueError, TypeError):
            pass
    if value.__class__.__module__.startswith("numpy"):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return str(value)
    return value
