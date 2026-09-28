from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocess import iter_extracted_urls, normalize_text_field
from src.url_model_bundle import load_bundle as load_url_bundle
from src.url_model_bundle import normalize_url_model_input
from src.url_model_bundle import score_normalized_urls
from src.utils import REPOSITORY_ROOT

OUT = REPOSITORY_ROOT / "outputs/models/final_artifact"
IMPLEMENTATION_PATHS = {"src/final_artifact.py", "src/url_model_bundle.py", "src/unified_artifact.py", "src/explain.py"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _combine(subject: str, body: str) -> str:
    subject, body = subject.strip(), body.strip()
    return f"{subject}\n\n{body}" if subject and body else subject or body


def _independent_prediction(root: Path, subject: str, body: str) -> dict[str, Any]:
    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    canonical = _combine(normalize_text_field(subject), normalize_text_field(body))
    raw = float(model.decision_function(vectorizer.transform([canonical]))[0])
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    urls = []
    for value in (subject, body):
        for _start, extracted in iter_extracted_urls(value):
            try:
                urls.append(normalize_url_model_input(extracted))
            except ValueError:
                pass
    unique = list(dict.fromkeys(urls))
    scores = {}
    if unique:
        result = score_normalized_urls(unique, load_url_bundle(root / "outputs/models/url"))
        scores = dict(zip(unique, map(float, result["probability_phishing"]), strict=True))
    return {
        "raw": raw,
        "probability": probability,
        "label": "phishing" if probability >= 0.5 else "legitimate",
        "scores": scores,
    }


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    check(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository root unsupported")
    from scripts.validate_text_calibration import validate as validate_text_calibration
    from scripts.validate_url_model import validate as validate_url_model
    from src.build_final_artifact import _semantic_identity, build_payloads, semantic_predecessors
    from src.provenance import CONTRACT_VERSION

    text_validation = validate_text_calibration(root)
    url_validation = validate_url_model(root)
    architecture = json.loads((out / "architecture.json").read_text(encoding="utf-8"))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    text_manifest = json.loads((root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8"))
    url_manifest = json.loads((root / "outputs/models/url/manifest.json").read_text(encoding="utf-8"))
    expected_selection = {
        "text": text_manifest["selection_provenance"],
        "url": url_manifest["selection_provenance"],
    }
    check(manifest["selection_provenance"] == report["selection_provenance"] == expected_selection,
          "Selection provenance chain mismatch")
    check(text_validation["selection_semantic_sha256"] == expected_selection["text"]["semantic_sha256"] and
          url_validation["selection_semantic_sha256"] == expected_selection["url"]["semantic_sha256"],
          "Validated upstream selection fingerprints differ")
    check(architecture["decision_threshold"] == 0.5, "Decision threshold changed")
    check(architecture["fusion_applied"] is False, "Fusion must be disabled")
    check(architecture["url_evidence"]["affects_email_classification"] is False,
          "URL evidence affects classification")
    check("historical_step25_fusion" not in architecture, "Historical fusion leaked into active architecture")

    required_false = {
        "model_fitted", "calibrator_fitted", "threshold_optimized", "fusion_applied",
        "holdout_opened", "holdout_evaluated", "external_data_used", "rq2_or_rq3_started",
        "step26_started",
    }
    check(set(manifest["flags"]) == required_false and all(manifest["flags"][name] is False for name in required_false),
          "Invalid methodological flags")
    check(report["flags"] == manifest["flags"], "Report/manifest flag mismatch")
    expected_architecture, expected_protected, expected_implementation = build_payloads(root)
    expected_run_id, expected_semantic = _semantic_identity(
        semantic_predecessors(root), expected_implementation, expected_architecture
    )
    check(manifest["predecessor_semantic_sha256"] == report["predecessor_semantic_sha256"] ==
          semantic_predecessors(root), "Predecessor semantic chain mismatch")
    check(architecture == expected_architecture, "Architecture payload mismatch")
    check(manifest["contract_version"] == report["contract_version"] ==
          f"{CONTRACT_VERSION}:final-artifact-v2", "Provenance contract mismatch")
    check(manifest["artifact_version"] == 2 and architecture["artifact_version"] == 2,
          "Final artifact schema version mismatch")
    check(__import__("re").fullmatch(r"step25c-[0-9a-f]{32}", expected_run_id) is not None,
          "Non-canonical Step-25C run ID")
    check(manifest["architecture_run_id"] == report["architecture_run_id"] == expected_run_id and
          manifest["semantic_sha256"] == report["semantic_sha256"] == expected_semantic,
          "Final artifact semantic identity mismatch")

    for relative, expected in manifest["protected_inputs"].items():
        check(digest(root / relative) == expected, f"Protected input hash mismatch: {relative}")
    check(set(manifest["implementation_sha256"]) == IMPLEMENTATION_PATHS,
          "Implementation inventory mismatch")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")
    historical_prefixes = ("outputs/models/" + "fusion/", "outputs/analysis/" + "fusion_failures/",
                           "outputs/evaluation/" + "historical/")
    check(not any(path.startswith(historical_prefixes)
                  for path in manifest["protected_inputs"]),
          "Historical output leaked into active provenance")

    checksum_lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checksums = {}
    for line in checksum_lines:
        value, name = line.split("  ", 1)
        check(name != "checksums.sha256" and name not in checksums, "Recursive or duplicate checksum")
        checksums[name] = value
    check(set(checksums) == {"architecture.json", "report.json", "manifest.json"}, "Incomplete checksum inventory")
    for name, expected in checksums.items():
        check(digest(out / name) == expected, f"Output checksum mismatch: {name}")
    check(manifest["output_sha256"] == {
        "architecture.json": checksums["architecture.json"], "report.json": checksums["report.json"]},
        "Manifest output hash mismatch")

    from src.final_artifact import load_final_artifact_components, predict_final_artifact

    components = load_final_artifact_components(out)
    subject = "Account Alert"
    body = "Review HTTPS://Example.COM/Case?Token=AbC and https://second.example/Path"
    independent = _independent_prediction(root, subject, body)
    actual = predict_final_artifact(subject, body, components)
    email = actual["email_classification"]
    evidence = actual["url_model_evidence"]
    check(math.isclose(email["text_raw_decision_score"], independent["raw"], rel_tol=1e-12, abs_tol=1e-12),
          "Text raw score mismatch")
    check(math.isclose(email["text_probability_phishing"], independent["probability"], rel_tol=1e-12, abs_tol=1e-12),
          "Text probability mismatch")
    check(email["predicted_label"] == independent["label"], "Text-only label mismatch")
    check(evidence["affects_email_classification"] is False, "URL evidence override enabled")
    check(evidence["scored_unique_url_count"] == len(independent["scores"]), "URL coverage mismatch")
    for occurrence in evidence["occurrences"]:
        if occurrence["normalization_status"] == "valid":
            expected = independent["scores"][occurrence["normalized_url"]]
            check(math.isclose(occurrence["url_model_score_phishing"], expected, rel_tol=1e-12, abs_tol=1e-12),
                  "URL model score mismatch")
        check("predicted_label" not in occurrence and "url_prediction" not in occurrence,
              "URL hard label leaked into evidence")
    forbidden = {"final_risk_score", "policy_name", "policy_basis", "url_prediction"}
    check(not forbidden.intersection(actual), "Historical fusion field leaked into public output")
    return {
        "status": "validated",
        "architecture_run_id": manifest["architecture_run_id"],
        "scored_unique_urls": evidence["scored_unique_url_count"],
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
