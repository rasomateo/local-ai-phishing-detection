from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.load_data import NORMALIZED_COLUMNS
from src.preprocess import DUPLICATE_RULE, normalize_text_field
from src.utils import REPOSITORY_ROOT
from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

RUN = ""
SOURCE = "phishingpot"
COMMIT = "49f63777126b0bdb9eb1f6e770a5c3f9df2b0306"
THRESHOLD = 0.5
CONFIDENCE = 0.95
Z = 1.959963984540054
EMPTY_ID = "phishingpot:de3cc5649e8e5951078d42a204cf14e345b08e53ac4e11f5aa8c91388282bca9"
PATH_DIGEST = "d940fe055abda78f20fdd6117ef1fd5717b137521769dd93ae0b88a2eb4cc754"
CONTENT_DIGEST = "d0e132b2e6239bb83c77f34cef59cf65c396ae225d284dfeb700881b08a13684"
EXPECTED_RAW = 8_614
EXPECTED_DUPLICATES = 648
EXPECTED_CANONICAL = 7_966
EXPECTED_EMPTY_RAW = 6
EXPECTED_EMPTY_EXCLUSIONS = 1
EXPECTED_EVALUATED = 7_965
POT_ROOT = REPOSITORY_ROOT / "data/raw/PhishingPot/phishing_pot"
OUT = REPOSITORY_ROOT / "outputs/evaluation/phishingpot"

EXPECTED_PROTECTED = {
    "data/raw/PhishingPot/phishing_pot/README.md": "fea48a02fe962a72a72f62c825da61f45bf07ad39708840ca8a9b20604f471ce",
    "outputs/dataset_acquisition_manifest.json": "38937b96b86d8e938f4ded324fa885ab41b41aeb0f8369f907fb34b19bbb147d",
    "outputs/dataset_roles/phishingpot_external_evaluation.csv": "39762e5d6af8c592ed93313695ec4113bcb5e8097c32eb2aa61df15fedb0d393",
    "outputs/preprocessing_report.json": "cfbf68fdc5be8b8a1be28ba211278b277821fda92067cedd44d317b2e2c8b425",
    "outputs/preprocessed_dataset.csv": "cb8ab274790414f518f0b7bb4122ce8655194044bf23798442185c95226963b6",
    "outputs/splits/manifest.json": "471dfc7558b80ead96beab66e60054f28aacdfd4c9a9ba96aeac39497de9d23d",
    f"outputs/splits/runs/{RUN}/split_assignments.csv": "497acde2ed8ec6fd0a1d3d6f8c004398a0e2c3063d6771734b2d8eae41923b30",
    "outputs/leakage/reserved_sources/summary.json": "6b96cb2d71f8073c1743a23435701c57921b3d62744db4545d7a37631d155e67",
    "outputs/leakage/reserved_sources/exact_overlaps.csv": "9d08b43e504c871da049c516fe9430a8a96de2f66779aade95dd012053455379",
    "outputs/models/text/classifier/manifest.json": "7f7da1edb895bcc825b9f64b5f51c0bb4c125016498e5210116558346a4a2698",
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/manifest.json": "b20970982cbcb1279302037dbda16537915839f26567a7e3e74399c4b8e78752",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/final_artifact/architecture.json": "f730617eab8fd9295a740fd8f2bb280f66687368583d1c6fb3793d378bf68924",
    "outputs/models/final_artifact/manifest.json": "df012321b671c8d3be0edfddaa4ca7fa836ada6226c717ba15eacc58fb72b21f",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
}
PREDICTION_COLUMNS = ["evaluation_run_id", "record_id", "source", "source_role", "collection_period", "label",
                      "text_raw_score", "calibrated_probability_phishing", "prediction", "correct"]
METRIC_KEYS = ["n", "tp", "fn", "recall", "miss_rate", "recall_ci_95_lower",
               "recall_ci_95_upper", "ci_method", "confidence_level", "z"]
CHECKSUM_NAMES = ["external_predictions.csv", "external_metrics.json", "report.json", "manifest.json"]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == fields, f"Unexpected schema: {path}")
        return list(reader)


def _decode_header(value: str | None) -> str:
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
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _body(message: Message) -> str:
    if not message.is_multipart():
        return _decode_payload(message).strip()
    plain: list[str] = []
    html: list[str] = []
    for part in message.walk():
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        payload = _decode_payload(part)
        if payload and part.get_content_type() == "text/plain":
            plain.append(payload)
        elif payload and part.get_content_type() == "text/html":
            html.append(payload)
    return "\n\n".join(plain or html).strip()


def stable_id(source_file: str, source_record: str) -> str:
    identity = "\x1f".join((SOURCE, "git_snapshot", source_file, source_record))
    return f"{SOURCE}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def independent_population(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    files = sorted(path for path in (POT_ROOT / "email").rglob("*.eml") if path.is_file())
    tracked = [path.relative_to(POT_ROOT).as_posix() for path in files]
    check(len(tracked) == EXPECTED_RAW and hashlib.sha256("\n".join(tracked).encode()).hexdigest() == PATH_DIGEST,
          "PhishingPot path inventory mismatch")
    source_file = f"data/raw/PhishingPot/phishing_pot/@{COMMIT}"
    parsed: list[dict[str, str]] = []
    seen_paths: list[str] = []
    snapshot = hashlib.sha256()
    for path in files:
            relative = path.relative_to(POT_ROOT).as_posix()
            seen_paths.append(relative)
            payload = path.read_bytes()
            snapshot.update(relative.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
            message = BytesParser(policy=policy.default).parsebytes(payload)
            subject = normalize_text_field(_decode_header(message.get("Subject")))
            body = normalize_text_field(_body(message))
            record = dict.fromkeys(NORMALIZED_COLUMNS, "")
            record.update({"record_id": stable_id(source_file, relative), "dataset": SOURCE,
                           "source_subset": "git_snapshot", "source_file": source_file,
                           "source_record": relative, "source_role": "external_evaluation",
                           "subject": subject, "body": body, "label": "phishing", "raw_label": "phishing",
                           "label_provenance": "source_corpus_membership:phishing_pot",
                           "inclusion_reason": "reserved evaluation only; excluded from development and selection"})
            parsed.append(record)
    check(sorted(seen_paths) == tracked and snapshot.hexdigest() == CONTENT_DIGEST,
          "Parsed PhishingPot snapshot mismatch")
    canonical = rows(root / "outputs/dataset_roles/phishingpot_external_evaluation.csv", NORMALIZED_COLUMNS)
    check(parsed == canonical, "Independent raw parsing differs from canonical role artifact")
    check(len(parsed) == len({row["record_id"] for row in parsed}) == EXPECTED_RAW, "Raw ID coverage mismatch")
    empty_raw = [row for row in parsed if not row["subject"] and not row["body"]]
    check(len(empty_raw) == EXPECTED_EMPTY_RAW, "Raw empty count mismatch")
    retained: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in parsed:
        pair = (row["subject"], row["body"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            retained.append(row)
    empty_retained = [row for row in retained if not row["subject"] and not row["body"]]
    check(len(retained) == EXPECTED_CANONICAL and len(parsed) - len(retained) == EXPECTED_DUPLICATES,
          "Independent exact-dedup reconciliation mismatch")
    check([row["record_id"] for row in empty_retained] == ([] if EXPECTED_EMPTY_EXCLUSIONS == 0 else [EMPTY_ID]),
          "Wrong empty representative")
    evaluated = [row for row in retained if row["subject"] or row["body"]]
    check(len(evaluated) == EXPECTED_EVALUATED, "Independent evaluation population mismatch")
    return evaluated, {"raw_records": EXPECTED_RAW, "raw_empty_records": EXPECTED_EMPTY_RAW,
                       "raw_empty_record_ids": [row["record_id"] for row in empty_raw],
                       "exact_duplicate_exclusions": EXPECTED_DUPLICATES, "canonical_retained": EXPECTED_CANONICAL,
                       "evaluation_only_empty_exclusions": EXPECTED_EMPTY_EXCLUSIONS,
                       "evaluation_only_excluded_record_id": (EMPTY_ID if EXPECTED_EMPTY_EXCLUSIONS else None),
                       "evaluated_records": EXPECTED_EVALUATED, "parse_failures": 0,
                       "duplicate_rule": DUPLICATE_RULE,
                       "evaluation_input_validity_rule": "at least one Step-14-normalized field subject or body is nonempty"}


def combine(subject: str, body: str) -> str:
    left, right = subject.strip(), body.strip()
    return f"{left}\n\n{right}" if left and right else left or right


def wilson(successes: int, total: int) -> tuple[float, float]:
    check(total > 0 and 0 <= successes <= total, "Invalid Wilson inputs")
    p = successes / total
    denominator = 1.0 + Z * Z / total
    center = (p + Z * Z / (2.0 * total)) / denominator
    margin = Z * math.sqrt(p * (1.0 - p) / total + Z * Z / (4.0 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - margin),
            1.0 if successes == total else min(1.0, center + margin))


def float_matches(stored: float, reproduced: float) -> tuple[bool, str]:
    if struct.pack("!d", float(stored)) == struct.pack("!d", float(reproduced)):
        return True, "binary64_exact"
    return math.isclose(stored, reproduced, rel_tol=1e-12, abs_tol=1e-15), "documented_tolerance_fallback"


def method_specification(protected: dict[str, str] = EXPECTED_PROTECTED) -> dict[str, Any]:
    return {"version": 1, "protected_sha256": protected,
            "source": SOURCE, "source_commit": COMMIT, "raw_records": EXPECTED_RAW,
            "exact_duplicate_rule": DUPLICATE_RULE, "exact_duplicate_exclusions": EXPECTED_DUPLICATES,
            "canonical_retained": EXPECTED_CANONICAL,
            "evaluation_input_validity_rule": "subject or body nonempty after Step-14 normalization",
            "evaluation_only_excluded_record_id": (EMPTY_ID if EXPECTED_EMPTY_EXCLUSIONS else None),
            "evaluated_records": EXPECTED_EVALUATED,
            "prediction_schema": PREDICTION_COLUMNS, "metric_schema": METRIC_KEYS,
            "decision_threshold": THRESHOLD, "decision_rule": "phishing iff p_text >= 0.5",
            "confidence_level": CONFIDENCE, "wilson_z": Z, "ordering": "record_id ascending",
            "float_serialization": ".17g"}


def expected_run_id(protected: dict[str, str] = EXPECTED_PROTECTED,
                    root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    return "step29:" + hashlib.sha256(encoded).hexdigest()[:32]


def forbidden_metric_keys(value: Any) -> set[str]:
    forbidden = {"accuracy", "precision", "f1", "tn", "fp", "false_positive_rate", "true_negatives", "false_positives"}
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(forbidden.intersection(value))
        for nested in value.values():
            found.update(forbidden_metric_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(forbidden_metric_keys(nested))
    return found


def validate_isolation(root: Path, evaluated: list[dict[str, str]]) -> None:
    ids = {row["record_id"] for row in evaluated}
    pairs = {(row["subject"], row["body"]) for row in evaluated}
    assignments = rows(root / f"outputs/splits/runs/{RUN}/split_assignments.csv",
                       ["record_id", "label", "partition"])
    check(not ids.intersection(row["record_id"] for row in assignments), "Internal split ID overlap")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (root / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal schema")
        for row in reader:
            check(row["source_role"] == "text_development" and row["dataset"] != SOURCE,
                  "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    check(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271) and not ids.intersection(internal_ids) and
          not pairs.intersection(internal_pairs), "Internal isolation mismatch")
    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    check(leakage["exact_overlaps_by_scope"]["internal|phishingpot"] == 0 and
          leakage["source_role_isolation_passed"] is True, "Step-16 isolation mismatch")


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    global RUN, COMMIT, PATH_DIGEST, CONTENT_DIGEST
    global EXPECTED_RAW, EXPECTED_DUPLICATES, EXPECTED_CANONICAL
    global EXPECTED_EMPTY_RAW, EXPECTED_EMPTY_EXCLUSIONS, EXPECTED_EVALUATED
    contract = current_contract(root)
    COMMIT = contract.expected("phishingpot_commit", COMMIT)
    PATH_DIGEST = contract.expected("phishingpot_path_digest", PATH_DIGEST)
    CONTENT_DIGEST = contract.expected("phishingpot_content_digest", CONTENT_DIGEST)
    EXPECTED_RAW = contract.expected("phishingpot_raw_count", EXPECTED_RAW)
    EXPECTED_DUPLICATES = contract.expected("phishingpot_duplicate_exclusions", EXPECTED_DUPLICATES)
    EXPECTED_CANONICAL = contract.expected("phishingpot_canonical_count", EXPECTED_CANONICAL)
    EXPECTED_EMPTY_RAW = contract.expected("phishingpot_empty_raw", EXPECTED_EMPTY_RAW)
    EXPECTED_EMPTY_EXCLUSIONS = contract.expected("phishingpot_empty_exclusions", EXPECTED_EMPTY_EXCLUSIONS)
    EXPECTED_EVALUATED = contract.expected("phishingpot_evaluated_count", EXPECTED_EVALUATED)
    RUN = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    check(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository root unsupported")
    protected = resolve_current_protected_hashes(root, EXPECTED_PROTECTED)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["selection_provenance"] == current_selection_provenance(root),
          "Selection provenance chain mismatch")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "external_metrics.json").read_text(encoding="utf-8"))
    check(manifest["artifact_type"] == "step29_phishingpot_external_evaluation", "Artifact type mismatch")
    check(manifest["protected_sha256"] == protected, "Protected inventory mismatch")
    check(manifest["schemas"] == {"external_predictions.csv": PREDICTION_COLUMNS,
                                  "external_metrics.json": METRIC_KEYS}, "Schema manifest mismatch")
    check(set(metrics) == set(METRIC_KEYS) and len(metrics) == len(METRIC_KEYS), "Metric schema mismatch")
    check(not forbidden_metric_keys(metrics) and not forbidden_metric_keys(report), "Forbidden metric found")
    expected_run = expected_run_id(protected, root)
    check(manifest["method_specification"] == method_specification(protected),
          "Method specification mismatch")
    check(manifest["evaluation_run_id"] == report["evaluation_run_id"] == expected_run, "Run ID mismatch")
    expected_implementation = {
        "src/phishingpot_evaluation.py", "src/load_data.py", "src/preprocess.py",
        "src/prepare_reserved_evaluation.py", "src/final_artifact.py",
        "scripts/validate_phishingpot_evaluation.py",
    }
    check(set(manifest["implementation_sha256"]) == expected_implementation,
          "Unexpected implementation inventory")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")
    for name in CHECKSUM_NAMES[:3]:
        check(manifest["output_sha256"][name] == digest(out / name), f"Output hash mismatch: {name}")
    checksum_lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    check(checksum_lines == [f"{digest(out / name)}  {name}" for name in CHECKSUM_NAMES],
          "Checksum inventory mismatch or recursive self-hash")

    evaluated, population = independent_population(root)
    check(report["population"] == population, "Reported population reconciliation mismatch")
    validate_isolation(root, evaluated)
    stored = rows(out / "external_predictions.csv", PREDICTION_COLUMNS)
    expected_ids = sorted(row["record_id"] for row in evaluated)
    check(len(stored) == len({row["record_id"] for row in stored}) == EXPECTED_EVALUATED and
          [row["record_id"] for row in stored] == expected_ids, "Prediction coverage mismatch")
    check({row["evaluation_run_id"] for row in stored} == {expected_run}, "Prediction run mismatch")

    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
          type(calibrator) is LogisticRegression, "Unexpected component type")
    check(list(model.classes_) == ["legitimate", "phishing"] and list(calibrator.classes_) == [0, 1] and
          float(calibrator.coef_[0, 0]) > 0, "Component orientation mismatch")
    by_id = {row["record_id"]: row for row in evaluated}
    modes: set[str] = set()
    predictions: list[str] = []
    for actual in stored:
        source = by_id[actual["record_id"]]
        check((actual["source"], actual["source_role"], actual["collection_period"], actual["label"]) ==
              (SOURCE, "external_evaluation", "", "phishing"), "Prediction provenance mismatch")
        raw = float(model.decision_function(vectorizer.transform([combine(source["subject"], source["body"])]))[0])
        probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
        prediction = "phishing" if probability >= THRESHOLD else "legitimate"
        raw_ok, raw_mode = float_matches(float(actual["text_raw_score"]), raw)
        prob_ok, prob_mode = float_matches(float(actual["calibrated_probability_phishing"]), probability)
        check(raw_ok and prob_ok, f"Score mismatch: {actual['record_id']}")
        modes.update((raw_mode, prob_mode))
        check(actual["prediction"] == prediction and actual["correct"] in {"true", "false"} and
              (actual["correct"] == "true") == (prediction == "phishing"), "Prediction mismatch")
        predictions.append(prediction)
    tp, fn = predictions.count("phishing"), predictions.count("legitimate")
    lower, upper = wilson(tp, len(predictions))
    expected_metrics = {"n": len(predictions), "tp": tp, "fn": fn, "recall": tp / len(predictions),
                        "miss_rate": fn / len(predictions), "recall_ci_95_lower": lower,
                        "recall_ci_95_upper": upper, "ci_method": "wilson_score_two_sided",
                        "confidence_level": CONFIDENCE, "z": Z}
    check(metrics == report["metrics"] == expected_metrics and tp + fn == EXPECTED_EVALUATED,
          "Positive-only metric reproduction mismatch")
    check(report["inference"]["url_extraction_performed"] is False and
          report["inference"]["url_model_loaded"] is False and
          report["inference"]["url_model_inference_performed"] is False and
          report["inference"]["fusion_applied"] is False and report["inference"]["fit_performed"] is False,
          "Forbidden inference path enabled")
    comparison = "documented_tolerance_fallback" if "documented_tolerance_fallback" in modes else "binary64_exact"
    return {"status": "validated", "evaluation_run_id": expected_run, "records": len(stored),
            "float_comparison": comparison, "tp": tp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(validate(out=args.out), indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
