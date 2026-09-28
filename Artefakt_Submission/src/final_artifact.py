from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.preprocess import iter_extracted_urls, normalize_text_field
from src.url_model_bundle import load_bundle as load_url_bundle
from src.url_model_bundle import normalize_url_model_input
from src.url_model_bundle import score_normalized_urls
from src.utils import REPOSITORY_ROOT

DEFAULT_ARTIFACT_DIR = REPOSITORY_ROOT / "outputs/models/final_artifact"
TEXT_VECTORIZER = REPOSITORY_ROOT / "outputs/models/text/classifier/text_vectorizer.joblib"
TEXT_MODEL = REPOSITORY_ROOT / "outputs/models/text/classifier/text_model.joblib"
TEXT_CALIBRATOR = REPOSITORY_ROOT / "outputs/models/text/calibration/text_calibrator.joblib"
URL_BUNDLE_DIR = REPOSITORY_ROOT / "outputs/models/url"
DECISION_THRESHOLD = 0.5
ARCHITECTURE_NAME = "step25c_text_led_email_classifier_with_auxiliary_url_evidence"
SCHEMA_VERSION = 2
URL_MODEL_SCOPE = "Frozen Step-23 model trained and calibrated on the Phishload URL dataset."
URL_VALIDATION_SCOPE = "Not independently validated on URLs extracted from emails."
URL_INTERPRETATION_WARNING = (
    "A high URL model score is auxiliary model evidence only; it does not prove that an individual URL "
    "is malicious, and a low score does not prove that it is harmless."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combine_subject_body(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    return f"{subject}\n\n{body}" if subject and body else subject or body


def _extract_and_normalize_urls(subject: str, body: str) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for field, value in (("subject", subject), ("body", body)):
        for index, (_start, extracted) in enumerate(iter_extracted_urls(value)):
            try:
                normalized = normalize_url_model_input(extracted)
                status, reason = "valid", ""
            except ValueError:
                normalized = ""
                status, reason = "invalid", "invalid_step21_url"
            occurrences.append({
                "field": field,
                "url_index": index,
                "extracted_url": extracted,
                "normalized_url": normalized,
                "normalization_status": status,
                "exclusion_reason": reason,
            })
    return occurrences


def load_final_artifact_components(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    architecture = json.loads((artifact_dir / "architecture.json").read_text(encoding="utf-8"))
    protected = manifest["protected_inputs"]
    required = {
        "outputs/models/text/classifier/text_vectorizer.joblib": TEXT_VECTORIZER,
        "outputs/models/text/classifier/text_model.joblib": TEXT_MODEL,
        "outputs/models/text/calibration/text_calibrator.joblib": TEXT_CALIBRATOR,
        "outputs/models/url/url_vectorizer.joblib": URL_BUNDLE_DIR / "url_vectorizer.joblib",
        "outputs/models/url/url_model.joblib": URL_BUNDLE_DIR / "url_model.joblib",
        "outputs/models/url/url_calibrator.joblib": URL_BUNDLE_DIR / "url_calibrator.joblib",
    }
    for relative, path in required.items():
        if protected.get(relative) != sha256(path):
            raise ValueError(f"Frozen component hash mismatch: {relative}")
    if manifest["output_sha256"].get("architecture.json") != sha256(artifact_dir / "architecture.json"):
        raise ValueError("Step-25C architecture hash mismatch")
    if architecture.get("architecture_name") != ARCHITECTURE_NAME:
        raise ValueError("Unexpected Step-25C architecture")
    if architecture.get("decision_threshold") != DECISION_THRESHOLD or architecture.get("fusion_applied") is not False:
        raise ValueError("Invalid Step-25C decision contract")

    text_vectorizer = joblib.load(TEXT_VECTORIZER)
    text_model = joblib.load(TEXT_MODEL)
    text_calibrator = joblib.load(TEXT_CALIBRATOR)
    if type(text_vectorizer) is not TfidfVectorizer or type(text_model) is not LinearSVC:
        raise ValueError("Unexpected frozen text bundle type")
    if type(text_calibrator) is not LogisticRegression:
        raise ValueError("Unexpected frozen text calibrator type")
    if list(text_model.classes_) != ["legitimate", "phishing"]:
        raise ValueError("Text decision score positive class is not phishing")
    if (list(text_calibrator.classes_) != [0, 1] or text_calibrator.coef_.shape != (1, 1)
            or float(text_calibrator.coef_[0, 0]) <= 0):
        raise ValueError("Text calibrator class 1 is not P(phishing)")
    return {
        "text_vectorizer": text_vectorizer,
        "text_model": text_model,
        "text_calibrator": text_calibrator,
        "url_bundle": load_url_bundle(URL_BUNDLE_DIR),
        "architecture": architecture,
        "manifest": manifest,
    }


def predict_final_artifact(
    subject: str,
    body: str,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(subject, str) or not isinstance(body, str):
        raise TypeError("subject and body must be strings")
    if not subject.strip() and not body.strip():
        raise ValueError("Provide a non-empty subject or body")
    loaded = components or load_final_artifact_components()

    occurrences = _extract_and_normalize_urls(subject, body)
    canonical_subject = normalize_text_field(subject)
    canonical_body = normalize_text_field(body)
    matrix = loaded["text_vectorizer"].transform([_combine_subject_body(canonical_subject, canonical_body)])
    text_raw = float(loaded["text_model"].decision_function(matrix)[0])
    text_probability = float(
        loaded["text_calibrator"].predict_proba(np.asarray([[text_raw]], dtype=np.float64))[0, 1]
    )
    if not np.isfinite(text_probability) or not 0.0 <= text_probability <= 1.0:
        raise ValueError("Invalid calibrated text probability")
    email_prediction = "phishing" if text_probability >= DECISION_THRESHOLD else "legitimate"

    unique_urls = list(dict.fromkeys(row["normalized_url"] for row in occurrences if row["normalized_url"]))
    score_by_url: dict[str, tuple[float, float]] = {}
    if unique_urls:
        result = score_normalized_urls(unique_urls, loaded["url_bundle"])
        score_by_url = {
            url: (float(raw), float(score))
            for url, raw, score in zip(
                unique_urls, result["decision_score"], result["probability_phishing"], strict=True
            )
        }
    for row in occurrences:
        scored = score_by_url.get(row["normalized_url"])
        row["scored"] = scored is not None
        row["raw_url_decision_score"] = None if scored is None else scored[0]
        row["url_model_score_phishing"] = None if scored is None else scored[1]

    highest_url = max(score_by_url, key=lambda url: score_by_url[url][1]) if score_by_url else None
    maximum_score = None if highest_url is None else score_by_url[highest_url][1]
    return {
        "architecture": {
            "name": ARCHITECTURE_NAME,
            "schema_version": SCHEMA_VERSION,
            "fusion_applied": False,
        },
        "email_classification": {
            "predicted_label": email_prediction,
            "text_probability_phishing": text_probability,
            "text_raw_decision_score": text_raw,
            "decision_threshold": DECISION_THRESHOLD,
            "decision_rule": "phishing iff text_probability_phishing >= 0.5; otherwise legitimate",
            "decision_source": "Frozen Step-19 text model plus frozen Step-20 inference calibrator only",
        },
        "url_model_evidence": {
            "available": bool(score_by_url),
            "extracted_url_count": len(occurrences),
            "valid_url_occurrence_count": sum(bool(row["normalized_url"]) for row in occurrences),
            "scored_unique_url_count": len(unique_urls),
            "maximum_url_model_score_phishing": maximum_score,
            "highest_scoring_normalized_url": highest_url,
            "affects_email_classification": False,
            "model_scope": URL_MODEL_SCOPE,
            "validation_scope": URL_VALIDATION_SCOPE,
            "interpretation_warning": URL_INTERPRETATION_WARNING,
            "occurrences": occurrences,
        },
    }


__all__ = ["load_final_artifact_components", "predict_final_artifact"]
