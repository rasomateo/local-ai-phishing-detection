from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.preprocess import normalize_text_field
from src.final_artifact import DECISION_THRESHOLD, load_final_artifact_components

DEFAULT_TOP_K = 10
RECONSTRUCTION_ABS_TOL = 1e-10
RECONSTRUCTION_REL_TOL = 1e-12
MAX_OCCURRENCES_PER_FEATURE = 20
MAX_POSITION_OCCURRENCES = 200


def combine_subject_body(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    return f"{subject}\n\n{body}" if subject and body else subject or body


def _validate_components(components: dict[str, Any]) -> tuple[TfidfVectorizer, LinearSVC, LogisticRegression]:
    vectorizer = components["text_vectorizer"]
    model = components["text_model"]
    calibrator = components["text_calibrator"]
    if type(vectorizer) is not TfidfVectorizer:
        raise TypeError("Step-31 explanation requires the frozen TfidfVectorizer")
    if type(model) is not LinearSVC:
        raise TypeError("Step-31 explanation requires the frozen LinearSVC")
    if type(calibrator) is not LogisticRegression:
        raise TypeError("Step-31 explanation requires the frozen LogisticRegression calibrator")
    classes = [str(value) for value in model.classes_]
    if classes != ["legitimate", "phishing"] or model.coef_.shape != (1, len(vectorizer.vocabulary_)):
        raise ValueError("Frozen LinearSVC orientation or dimensions changed")
    if list(calibrator.classes_) != [0, 1] or calibrator.coef_.shape != (1, 1):
        raise ValueError("Frozen calibrator orientation changed")
    if float(calibrator.coef_[0, 0]) <= 0.0:
        raise ValueError("Frozen calibrator is not monotonic toward phishing")
    return vectorizer, model, calibrator


def _validate_word_analyzer(vectorizer: TfidfVectorizer) -> re.Pattern[str]:
    params = vectorizer.get_params(deep=False)
    expected = {"analyzer": "word", "ngram_range": (1, 2), "lowercase": False,
                "preprocessor": None, "tokenizer": None, "stop_words": None, "strip_accents": None}
    for name, value in expected.items():
        if params.get(name) != value:
            raise ValueError(f"Unsupported frozen analyzer parameter {name}: {params.get(name)!r}")
    pattern = params.get("token_pattern")
    if not isinstance(pattern, str):
        raise ValueError("Frozen vectorizer has no reconstructible token pattern")
    return re.compile(pattern)


def _feature_occurrences(canonical_text: str, vectorizer: TfidfVectorizer,
                         displayed_features: list[dict[str, Any]]) -> dict[str, Any]:
    token_pattern = _validate_word_analyzer(vectorizer)
    tokens = [(match.group(0), match.start(), match.end()) for match in token_pattern.finditer(canonical_text)]
    requested = {int(item["feature_index"]): item for item in displayed_features}
    candidates: list[dict[str, Any]] = []
    total_by_feature: dict[int, int] = {index: 0 for index in requested}
    feature_names = vectorizer.get_feature_names_out()
    for feature_index in sorted(requested):
        feature = str(feature_names[feature_index])
        parts = feature.split(" ")
        if len(parts) not in {1, 2}:
            continue
        for token_index in range(0, len(tokens) - len(parts) + 1):
            window = tokens[token_index:token_index + len(parts)]
            if [item[0] for item in window] != parts:
                continue
            total_by_feature[feature_index] += 1
            if total_by_feature[feature_index] <= MAX_OCCURRENCES_PER_FEATURE:
                candidates.append({
                    "feature_index": feature_index, "feature": feature, "ngram_order": len(parts),
                    "start": int(window[0][1]), "end": int(window[-1][2]),
                    "occurrence_index": total_by_feature[feature_index] - 1,
                    "direction": "phishing" if float(requested[feature_index]["signed_contribution"]) > 0
                    else "legitimate",
                })
    candidates.sort(key=lambda item: (item["start"], item["end"], item["feature_index"],
                                      item["occurrence_index"]))
    returned = candidates[:MAX_POSITION_OCCURRENCES]
    total = sum(total_by_feature.values())
    return {
        "canonical_model_text": canonical_text,
        "scope": "positions_in_canonical_normalized_model_text_only",
        "attribution_scope": "position_only; signed contributions belong to complete TF-IDF features",
        "requested_feature_count": len(requested), "total_occurrence_count": total,
        "returned_occurrence_count": len(returned), "omitted_occurrence_count": total - len(returned),
        "occurrences": returned,
    }


def explain_prediction(subject: str, body: str, top_k: int = DEFAULT_TOP_K,
                       explanation_artifacts: dict[str, Any] | None = None,
                       expected_result: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(subject, str) or not isinstance(body, str):
        raise TypeError("subject and body must be strings")
    if not subject.strip() and not body.strip():
        raise ValueError("Provide a non-empty subject or body")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    loaded = explanation_artifacts or load_final_artifact_components()
    vectorizer, model, calibrator = _validate_components(loaded)
    canonical_text = combine_subject_body(normalize_text_field(subject), normalize_text_field(body))
    matrix = vectorizer.transform([canonical_text])
    actual_raw = float(model.decision_function(matrix)[0])
    row = matrix.getrow(0)
    names, coefficients = vectorizer.get_feature_names_out(), model.coef_[0]
    contributions: list[dict[str, Any]] = []
    for feature_index, tfidf_value in sorted(zip(row.indices.tolist(), row.data.tolist(), strict=True)):
        coefficient = float(coefficients[feature_index])
        feature = str(names[feature_index])
        contributions.append({
            "feature_index": int(feature_index), "feature": feature,
            "ngram_order": len(feature.split(" ")), "tfidf_value": float(tfidf_value),
            "coefficient": coefficient, "signed_contribution": float(float(tfidf_value) * coefficient),
        })
    intercept = float(model.intercept_[0])
    contribution_sum = float(math.fsum(item["signed_contribution"] for item in contributions))
    reconstructed = float(intercept + contribution_sum)
    difference = float(reconstructed - actual_raw)
    faithful = math.isclose(reconstructed, actual_raw, rel_tol=RECONSTRUCTION_REL_TOL,
                            abs_tol=RECONSTRUCTION_ABS_TOL)
    if not faithful:
        raise ValueError(f"LinearSVC raw score reconstruction failed: difference={difference!r}")
    probability = float(calibrator.predict_proba(np.asarray([[actual_raw]], dtype=np.float64))[0, 1])
    label = "phishing" if probability >= DECISION_THRESHOLD else "legitimate"
    if expected_result is not None:
        email = expected_result.get("email_classification") or {}
        if not math.isclose(float(email.get("text_raw_decision_score")), actual_raw,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Explanation raw score differs from Step-25C classification")
        if not math.isclose(float(email.get("text_probability_phishing")), probability,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Explanation probability differs from Step-25C classification")
        if email.get("predicted_label") != label:
            raise ValueError("Explanation label differs from Step-25C classification")
    positive = sorted((item for item in contributions if item["signed_contribution"] > 0.0),
                      key=lambda item: (-item["signed_contribution"], item["feature_index"]))[:top_k]
    negative = sorted((item for item in contributions if item["signed_contribution"] < 0.0),
                      key=lambda item: (item["signed_contribution"], item["feature_index"]))[:top_k]
    return {
        "schema_version": 1,
        "method": "active_feature_linear_contribution=tfidf_value*LinearSVC_coefficient",
        "representation": "Step-14 normalized subject_body view used by Step-25C",
        "class_orientation": {"model_classes": [str(value) for value in model.classes_],
                              "native_negative_class": "legitimate", "native_positive_class": "phishing",
                              "positive_contribution_supports": "phishing",
                              "negative_contribution_supports": "legitimate"},
        "raw_score": actual_raw, "intercept": intercept,
        "feature_contribution_sum": contribution_sum, "reconstructed_raw_score": reconstructed,
        "reconstruction_difference": difference,
        "reconstruction_tolerance": {"absolute": RECONSTRUCTION_ABS_TOL,
                                     "relative": RECONSTRUCTION_REL_TOL, "passed": faithful},
        "calibrated_text_probability_phishing": probability, "decision_threshold": DECISION_THRESHOLD,
        "final_email_label": label, "present_feature_count": len(contributions),
        "top_k_per_direction": top_k, "top_positive_features": positive,
        "top_negative_features": negative, "all_feature_contributions": contributions,
        "position_visualization": _feature_occurrences(canonical_text, vectorizer, positive + negative),
        "interpretation": {"feature_contributions_explain": "LinearSVC raw decision score",
                           "calibration_is_additive_token_attribution": False,
                           "positions_are_additive_attributions": False, "causal_or_semantic_proof": False,
                           "user_comprehension_validated": False},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Explain the frozen Step-25C text decision")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)
    print(json.dumps(explain_prediction(args.subject, args.body, args.top_k), indent=2,
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
