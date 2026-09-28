from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes
from src.execution_contract import current_contract
from src.utils import REPOSITORY_ROOT

DEFAULT_OUT = REPOSITORY_ROOT / "outputs/evaluation/explanations"
CHECKSUM_NAMES = ["explanation_validation.json", "explanation_examples.json", "report.json", "manifest.json"]
EXPECTED_PROTECTED = {
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/url/url_vectorizer.joblib": "182bfb0af30f56cd68657dbcdbd03571810882aeb75bcff5aad25af782478af3",
    "outputs/models/url/url_model.joblib": "f11e2a715ba993a1ac99c867964f79b289a5a982d92ae3cd8601585032c16be7",
    "outputs/models/url/url_calibrator.joblib": "110e48c05ec2c36ab7be35723c4916ca618062610c7f1d9c01ae84eac3040021",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
    "outputs/evaluation/phishingpot/checksums.sha256": "db21a28b32b3c7f5cc82b6ce9e9d8af3da789d228e977a38a94905e444ff1db6",
    "outputs/evaluation/dideco_intent/checksums.sha256": "dec06421882e73046364289ee2f425678c8b6e98b9fa9d44ef7bb49f80f0287b",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def normalize(value: str) -> str:
    value = re.sub(r"\r\n?|\n", "\n", value)
    value = "\n".join(re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", value).strip().lower()


def combine(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    return f"{subject}\n\n{body}" if subject and body else subject or body


def _load() -> tuple[TfidfVectorizer, LinearSVC, LogisticRegression]:
    vectorizer = joblib.load(REPOSITORY_ROOT / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(REPOSITORY_ROOT / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(REPOSITORY_ROOT / "outputs/models/text/calibration/text_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
          type(calibrator) is LogisticRegression, "Frozen component type mismatch")
    check(list(model.classes_) == ["legitimate", "phishing"], "LinearSVC class orientation mismatch")
    check(list(calibrator.classes_) == [0, 1] and float(calibrator.coef_[0, 0]) > 0,
          "Calibrator orientation mismatch")
    return vectorizer, model, calibrator


def _validate_example(example: dict[str, Any], components: tuple[Any, Any, Any]) -> float:
    vectorizer, model, calibrator = components
    expected_input_hash = hashlib.sha256((example["subject"] + "\x1f" + example["body"]).encode()).hexdigest()
    check(example["input_sha256"] == expected_input_hash, "Synthetic input hash mismatch")
    result = example["result"]
    explanation = result["text_explanation"]
    canonical = combine(normalize(example["subject"]), normalize(example["body"]))
    matrix = vectorizer.transform([canonical]).getrow(0)
    indices = matrix.indices.tolist()
    values = matrix.data.tolist()
    names = vectorizer.get_feature_names_out()
    expected_rows = []
    for index, value in sorted(zip(indices, values, strict=True)):
        coefficient = float(model.coef_[0, index])
        expected_rows.append({"feature_index": int(index), "feature": str(names[index]),
                              "ngram_order": len(str(names[index]).split(" ")),
                              "tfidf_value": float(value), "coefficient": coefficient,
                              "signed_contribution": float(value * coefficient)})
    check(explanation["all_feature_contributions"] == expected_rows, "Feature contributions mismatch")
    raw = float(model.decision_function(matrix)[0])
    total = float(math.fsum(row["signed_contribution"] for row in expected_rows))
    reconstructed = float(model.intercept_[0] + total)
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    label = "phishing" if probability >= 0.5 else "legitimate"
    check(math.isclose(raw, reconstructed, rel_tol=1e-12, abs_tol=1e-10), "Raw score not reconstructed")
    check(explanation["raw_score"] == raw and explanation["intercept"] == float(model.intercept_[0]) and
          explanation["feature_contribution_sum"] == total and
          explanation["reconstructed_raw_score"] == reconstructed, "Persisted score fields mismatch")
    check(explanation["calibrated_text_probability_phishing"] == probability and
          explanation["final_email_label"] == label, "Prediction consistency mismatch")
    email = result["email_classification"]
    check(email["text_raw_decision_score"] == raw and email["text_probability_phishing"] == probability and
          email["predicted_label"] == label, "Step-25C result changed")
    positives = sorted((row for row in expected_rows if row["signed_contribution"] > 0),
                       key=lambda row: (-row["signed_contribution"], row["feature_index"]))[:10]
    negatives = sorted((row for row in expected_rows if row["signed_contribution"] < 0),
                       key=lambda row: (row["signed_contribution"], row["feature_index"]))[:10]
    check(explanation["top_positive_features"] == positives and
          explanation["top_negative_features"] == negatives, "Deterministic ranking mismatch")
    active = set(indices)
    pattern = re.compile(vectorizer.token_pattern)
    tokens = [(match.group(0), match.start(), match.end()) for match in pattern.finditer(canonical)]
    for occurrence in explanation["position_visualization"]["occurrences"]:
        check("signed_contribution" not in occurrence, "Position occurrence contains contribution")
        check(occurrence["feature_index"] in active, "Position occurrence is not an active feature")
        start, end = occurrence["start"], occurrence["end"]
        check(0 <= start < end <= len(canonical), "Position span out of range")
        order = occurrence["ngram_order"]
        matching = [window for offset in range(len(tokens) - order + 1)
                    if (window := tokens[offset:offset + order])[0][1] == start and
                    window[-1][2] == end and " ".join(item[0] for item in window) == occurrence["feature"]]
        check(bool(matching), "Position span does not match analyzer tokens")
    return abs(float(explanation["reconstruction_difference"]))


def validate(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    protected = resolve_current_protected_hashes(REPOSITORY_ROOT, EXPECTED_PROTECTED)
    for relative, expected in protected.items():
        check(sha256(REPOSITORY_ROOT / relative) == expected, f"Protected input changed: {relative}")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["selection_provenance"] == current_selection_provenance(REPOSITORY_ROOT),
          "Selection provenance chain mismatch")
    validation = json.loads((out / "explanation_validation.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    examples = json.loads((out / "explanation_examples.json").read_text(encoding="utf-8"))
    check(manifest["input_sha256"] == protected, "Protected inventory mismatch")
    check(manifest["schema_version"] == 2 and manifest["method_specification"]["version"] == 2,
          "Explanation-evaluation schema version mismatch")
    expected_implementation = {
        "src/explain.py", "src/unified_artifact.py", "src/final_artifact.py",
        "src/explanation_evaluation.py", "scripts/validate_explanation_evaluation.py",
    }
    check(set(manifest["implementation_sha256"]) == expected_implementation,
          "Implementation inventory mismatch")
    for relative, expected in manifest["implementation_sha256"].items():
        check(sha256(REPOSITORY_ROOT / relative) == expected, f"Implementation hash mismatch: {relative}")
    lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    expected_lines = [f"{sha256(out / name)}  {name}" for name in CHECKSUM_NAMES]
    check(lines == expected_lines, "Step-31 checksum inventory mismatch")
    check(manifest["output_sha256"] == {name: sha256(out / name) for name in CHECKSUM_NAMES[:-1]},
          "Manifest output hashes mismatch")
    components = _load()
    differences = [_validate_example(example, components) for example in examples]
    contract = current_contract(REPOSITORY_ROOT)
    expected_cases = (contract.payload.get("expected", {}).get("explanation_case_count", 10)
                      if contract.is_fixture else 8)
    check(len(examples) == expected_cases and {example["result"]["email_classification"]["predicted_label"]
                                 for example in examples} == {"legitimate", "phishing"},
          "Synthetic coverage mismatch")
    check(validation["maximum_absolute_reconstruction_difference"] == max(differences),
          "Maximum reconstruction difference mismatch")
    check(report["validation"] == validation and report["boundaries"] == {
        "calibrator_fitted": False, "external_evaluation_performed": False,
        "holdout_accessed": False, "model_fitted": False, "step32_started": False,
        "threshold_changed": False, "url_explanation_added": False,
    }, "Report boundary mismatch")
    return {"status": "validated", "evaluation_run_id": manifest["evaluation_run_id"],
            "case_count": len(examples), "maximum_absolute_reconstruction_difference": max(differences)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.out), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
