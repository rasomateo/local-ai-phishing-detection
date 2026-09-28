from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.phishload_dataset import normalize_url as normalize_phishload_url
from src.utils import REPOSITORY_ROOT

DEFAULT_BUNDLE_DIR = REPOSITORY_ROOT / "outputs/models/url"
REJECTED_BARE_DOMAIN_TLDS = {"pdf", "docx", "png", "zip", "txt", "csv", "py", "css", "js", "md"}


def normalize_url_model_input(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("invalid_url")
    value = raw.strip()
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("invalid_url")

    if "://" in value:
        return normalize_phishload_url(value)
    if value.startswith("//") or "@" in value:
        raise ValueError("invalid_url")

    try:
        parts = urlsplit(f"//{value}")
        if parts.scheme or not parts.netloc or not parts.hostname:
            raise ValueError("invalid_host")
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or "." not in host or any(char.isspace() for char in host):
            raise ValueError("invalid_host")
        labels = host.split(".")
        if not all(label and len(label) <= 63 and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                   for label in labels):
            raise ValueError("invalid_host")
        if not value.lower().startswith("www.") and labels[-1] in REJECTED_BARE_DOMAIN_TLDS:
            raise ValueError("file_like_token")
        port = parts.port
        normalized_host = f"[{host}]" if ":" in host else host
        netloc = normalized_host + (f":{port}" if port is not None else "")
        suffix = urlunsplit(("", "", parts.path, parts.query, ""))
        return f"{netloc}{suffix}"
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_url") from exc


def load_bundle(bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> tuple[TfidfVectorizer, LinearSVC, LogisticRegression]:
    vectorizer = joblib.load(bundle_dir / "url_vectorizer.joblib")
    model = joblib.load(bundle_dir / "url_model.joblib")
    calibrator = joblib.load(bundle_dir / "url_calibrator.joblib")
    if type(vectorizer) is not TfidfVectorizer or type(model) is not LinearSVC or type(calibrator) is not LogisticRegression:
        raise ValueError("Unexpected Step-23 bundle object type")
    if list(model.classes_) != ["legitimate", "phishing"]:
        raise ValueError("Positive LinearSVC decision score must mean phishing")
    if list(calibrator.classes_) != [0, 1] or calibrator.coef_.shape != (1, 1) or float(calibrator.coef_[0, 0]) <= 0:
        raise ValueError("Calibrator class 1 must represent P(phishing) with positive slope")
    return vectorizer, model, calibrator


def score_normalized_urls(
    normalized_urls: Iterable[str],
    bundle: tuple[TfidfVectorizer, LinearSVC, LogisticRegression] | None = None,
) -> dict[str, np.ndarray]:
    urls = list(normalized_urls)
    if any(not isinstance(value, str) or not value for value in urls):
        raise ValueError("Inputs must be non-empty, already normalized URL strings")
    vectorizer, model, calibrator = bundle or load_bundle()
    matrix = vectorizer.transform(urls)
    scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
    probabilities = np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)
    if scores.shape != (len(urls),) or probabilities.shape != (len(urls),):
        raise ValueError("Unexpected Step-23 score shape")
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError("Non-finite score or invalid P(phishing)")
    return {"decision_score": scores, "probability_phishing": probabilities}
