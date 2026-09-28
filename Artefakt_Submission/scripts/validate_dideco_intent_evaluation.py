from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.load_data import NORMALIZED_COLUMNS
from src.prepare_reserved_evaluation import ANNOTATION_COLUMNS
from src.utils import REPOSITORY_ROOT
from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

RUN = ""
SOURCE = "dideco"
THRESHOLD = 0.5
CONFIDENCE = 0.95
Z = 1.959963984540054
AGGREGATION = (
    "step30_protocol_majority;25_options_assumed_available_all_rounds;"
    "selected=positive;not_selected=negative;repeat_conflict=abstention;"
    "strict_majority;tie_or_no_votes=unresolved"
)
OUT = REPOSITORY_ROOT / "outputs/evaluation/dideco_intent"
RAW = REPOSITORY_ROOT / "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip"
EXPECTED_RAW_ANNOTATIONS = 639
EXPECTED_NONEMPTY_ANNOTATIONS = 609
EXPECTED_EMPTY_ANNOTATIONS = 30
EXPECTED_MESSAGES = 220
EXPECTED_EXPLICIT = 8
EXPECTED_IMPLICIT = 17
EXPECTED_INTENTS = 25
EXPECTED_ASSIGNMENTS = 5_500
EXPECTED_REPEATED = 56
EXPECTED_DIFFERING = 55
EXPECTED_CONFLICT_MESSAGES = 38
EXPECTED_CONFLICT_CELLS = 239
EXPECTED_POSITIVE = 1_074
EXPECTED_NEGATIVE = 4_180
EXPECTED_UNRESOLVED = 246

EXPECTED_PROTECTED = {
    "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip": "472eae1754c456d41d182396a043969143a04146467fbfbb200a67b5f3028b12",
    "outputs/dataset_roles/dideco_intent_evaluation.csv": "29ce1a8715181892fc30faa95720fb68d70b20d675e612d2007758cb9c220acc",
    "outputs/dataset_roles/dideco_annotations.csv": "239a93c35f4af5e5d5479e4535ead44c6ff8ca532c5b14a159ae21e5ce76ecf6",
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
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
    "outputs/evaluation/phishingpot/checksums.sha256": "db21a28b32b3c7f5cc82b6ce9e9d8af3da789d228e977a38a94905e444ff1db6",
}

PREDICTION_COLUMNS = [
    "evaluation_run_id", "record_id", "source", "source_role", "source_file", "source_record",
    "annotation_row_count", "annotator_count", "evaluation_label", "label_provenance",
    "text_raw_score", "calibrated_probability_phishing", "prediction", "outcome",
]
ASSIGNMENT_COLUMNS = [
    "evaluation_run_id", "record_id", "intent_id", "intent_name", "annotation_type",
    "aggregation_provenance", "raw_annotation_count", "annotator_count", "valid_vote_count",
    "positive_vote_count", "negative_vote_count", "conflicting_repeat_abstention_count",
    "resolved_label", "unresolved_reason",
]
METRIC_COLUMNS = [
    "intent_id", "intent_name", "annotation_type", "n", "tp", "fn", "recall", "miss_rate",
    "recall_ci_95_lower", "recall_ci_95_upper", "ci_method", "confidence_level", "z",
    "support_category",
]
OVERALL_KEYS = [
    "n", "tp", "fn", "recall", "miss_rate", "recall_ci_95_lower", "recall_ci_95_upper",
    "ci_method", "confidence_level", "z",
]
CHECKSUM_NAMES = [
    "predictions.csv", "intent_assignments.csv", "intent_metrics.csv",
    "overall_metrics.json", "report.json", "manifest.json",
]


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


def normalize(value: Any) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    text = re.sub(r"\r\n?|\n", "\n", text)
    text = "\n".join(re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip().lower()


def stable_id(dataset: str, subset: str, source_file: str, source_record: str) -> str:
    identity = "\x1f".join((dataset, subset, source_file.replace("\\", "/"), source_record))
    return f"{dataset}:{hashlib.sha256(identity.encode()).hexdigest()}"


def independent_source() -> tuple[list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    source_file = "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip"
    messages: dict[str, dict[str, str]] = {}
    annotations: list[dict[str, Any]] = []
    mapping: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(RAW) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
        check(len(members) == 30, "JSONL member count mismatch")
        for member in members:
            round_match = re.search(r"round_(\d+)", member)
            annotator_match = re.search(r"annotator(\d+)", member)
            check(bool(round_match and annotator_match), "Member provenance mismatch")
            round_id = round_match.group(1)  # type: ignore[union-attr]
            annotator = annotator_match.group(1)  # type: ignore[union-attr]
            for line_number, line in enumerate(archive.read(member).decode("utf-8").splitlines(), 1):
                raw = json.loads(line)
                displayed = raw.get("displayed_text", "")
                check(isinstance(displayed, str), "Non-string displayed_text")
                message_record_id = ""
                if displayed.strip():
                    text_hash = hashlib.sha256(displayed.encode()).hexdigest()
                    source_record = f"displayed_text_sha256={text_hash}"
                    message_record_id = stable_id(SOURCE, "message", source_file, source_record)
                    messages.setdefault(message_record_id, {
                        "record_id": message_record_id, "subject": "", "body": normalize(displayed),
                        "source_file": source_file, "source_record": source_record,
                    })
                payload = raw.get("label_annotations", {}).get("intent_selection", {})
                check(isinstance(payload, dict), "Intent payload mismatch")
                selected: set[tuple[str, str]] = set()
                for name, value in payload.items():
                    value = str(value)
                    kind = "explicit" if value.isdigit() else "implicit"
                    check(value.isdigit() or value == f"intent_selection:::{name}", "Unknown intent encoding")
                    key = (kind, str(name))
                    check(key not in mapping or mapping[key] == value, "Intent ID mismatch")
                    mapping[key] = value
                    selected.add(key)
                annotations.append({"record_id": message_record_id, "annotator": annotator,
                                    "round": round_id, "selected": selected,
                                    "empty": not bool(displayed.strip())})
    explicit = [{"annotation_type": kind, "intent_name": name, "intent_id": value}
                for (kind, name), value in mapping.items() if kind == "explicit"]
    implicit = [{"annotation_type": kind, "intent_name": name, "intent_id": value}
                for (kind, name), value in mapping.items() if kind == "implicit"]
    explicit.sort(key=lambda item: int(item["intent_id"]))
    implicit.sort(key=lambda item: item["intent_name"])
    intents = explicit + implicit
    result = sorted(messages.values(), key=lambda item: item["record_id"])
    check(len(result) == EXPECTED_MESSAGES and len(annotations) == EXPECTED_RAW_ANNOTATIONS and
          sum(not row["empty"] for row in annotations) == EXPECTED_NONEMPTY_ANNOTATIONS and
          len(explicit) == EXPECTED_EXPLICIT and len(implicit) == EXPECTED_IMPLICIT,
          "Independent source counts mismatch")
    canonical = rows(REPOSITORY_ROOT / "outputs/dataset_roles/dideco_intent_evaluation.csv", NORMALIZED_COLUMNS)
    by_id = {row["record_id"]: row for row in canonical}
    check(set(by_id) == set(messages), "Canonical message ID mismatch")
    for row in result:
        actual = by_id[row["record_id"]]
        check((actual["subject"], actual["body"], actual["source_file"], actual["source_record"],
               actual["source_role"], actual["label"], actual["label_provenance"]) ==
              ("", row["body"], source_file, row["source_record"], "rq2_evaluation", "phishing",
               "source_corpus_membership:dideco;not_message_gold_label"), "Canonical row mismatch")
    return result, annotations, intents


def independent_aggregation(
    messages: list[dict[str, str]], annotations: list[dict[str, Any]], intents: list[dict[str, str]], run_id: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    attached = [row for row in annotations if row["record_id"]]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_message: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attached:
        grouped[(row["record_id"], row["annotator"])].append(row)
        by_message[row["record_id"]].append(row)
    repeated = sum(len(value) > 1 for value in grouped.values())
    differing = sum(len(value) > 1 and len({tuple(sorted(row["selected"])) for row in value}) > 1
                    for value in grouped.values())
    conflict_messages: set[str] = set()
    conflict_cells = 0
    for (record_id, _), value in grouped.items():
        for intent in intents:
            key = (intent["annotation_type"], intent["intent_name"])
            if len({key in row["selected"] for row in value}) > 1:
                conflict_cells += 1
                conflict_messages.add(record_id)
    output: list[dict[str, str]] = []
    abstentions_by_intent: Counter[tuple[str, str, str]] = Counter()
    for message in messages:
        record_id = message["record_id"]
        annotators = sorted({row["annotator"] for row in by_message[record_id]})
        for intent in intents:
            key = (intent["annotation_type"], intent["intent_name"])
            positive = negative = abstain = 0
            for annotator in annotators:
                values = {key in row["selected"] for row in grouped[(record_id, annotator)]}
                if len(values) > 1:
                    abstain += 1
                    abstentions_by_intent[(intent["intent_id"], intent["intent_name"],
                                            intent["annotation_type"])] += 1
                elif values == {True}:
                    positive += 1
                else:
                    negative += 1
            valid = positive + negative
            if valid == 0:
                resolved, reason = "unresolved", "no_valid_votes"
            elif positive == negative:
                resolved, reason = "unresolved", "tie"
            elif positive > negative:
                resolved, reason = "positive", ""
            else:
                resolved, reason = "negative", ""
            output.append({
                "evaluation_run_id": run_id, "record_id": record_id, **intent,
                "aggregation_provenance": AGGREGATION,
                "raw_annotation_count": str(len(by_message[record_id])),
                "annotator_count": str(len(annotators)), "valid_vote_count": str(valid),
                "positive_vote_count": str(positive), "negative_vote_count": str(negative),
                "conflicting_repeat_abstention_count": str(abstain), "resolved_label": resolved,
                "unresolved_reason": reason,
            })
    counts = Counter(row["resolved_label"] for row in output)
    check(len(output) == EXPECTED_ASSIGNMENTS and counts == Counter(
          positive=EXPECTED_POSITIVE, negative=EXPECTED_NEGATIVE, unresolved=EXPECTED_UNRESOLVED),
          "Independent assignment counts mismatch")
    check((repeated, differing, len(conflict_messages), conflict_cells) ==
          (EXPECTED_REPEATED, EXPECTED_DIFFERING, EXPECTED_CONFLICT_MESSAGES, EXPECTED_CONFLICT_CELLS),
          "Independent repeat-conflict counts mismatch")
    summary = {
        "raw_annotation_rows": EXPECTED_RAW_ANNOTATIONS,
        "attached_nonempty_annotation_rows": EXPECTED_NONEMPTY_ANNOTATIONS,
        "unattached_empty_displayed_text_rows": EXPECTED_EMPTY_ANNOTATIONS,
        "unique_messages": EXPECTED_MESSAGES, "raw_labels": EXPECTED_INTENTS,
        "explicit_labels": EXPECTED_EXPLICIT, "implicit_labels": EXPECTED_IMPLICIT,
        "assignment_cells": EXPECTED_ASSIGNMENTS,
        "repeated_message_annotator_pairs": repeated,
        "repeated_pairs_with_different_intent_sets": differing,
        "messages_with_repeat_conflicts": len(conflict_messages),
        "intent_level_repeat_conflict_abstentions": conflict_cells,
        "resolved_label_counts": dict(sorted(counts.items())),
        "abstentions_by_intent": [
            {"intent_id": key[0], "intent_name": key[1], "annotation_type": key[2], "count": count}
            for key, count in sorted(abstentions_by_intent.items())
        ],
    }
    return output, summary


def combine(subject: str, body: str) -> str:
    left, right = subject.strip(), body.strip()
    return f"{left}\n\n{right}" if left and right else left or right


def wilson(successes: int, total: int) -> tuple[float, float] | None:
    if total == 0:
        return None
    check(0 <= successes <= total, "Invalid Wilson inputs")
    p = successes / total
    denominator = 1.0 + Z * Z / total
    center = (p + Z * Z / (2.0 * total)) / denominator
    margin = Z * math.sqrt(p * (1.0 - p) / total + Z * Z / (4.0 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - margin),
            1.0 if successes == total else min(1.0, center + margin))


def support(total: int) -> str:
    return "primary" if total >= 20 else "exploratory" if total >= 10 else "insufficient" if total else "not_determinable"


def float_matches(stored: float, reproduced: float) -> tuple[bool, str]:
    if struct.pack("!d", stored) == struct.pack("!d", reproduced):
        return True, "binary64_exact"
    return math.isclose(stored, reproduced, rel_tol=1e-12, abs_tol=1e-15), "documented_tolerance_fallback"


def method_specification(protected: dict[str, str] = EXPECTED_PROTECTED) -> dict[str, Any]:
    return {
        "version": 1, "protected_sha256": protected,
        "source": SOURCE, "raw_annotation_rows": EXPECTED_RAW_ANNOTATIONS,
        "evaluated_messages": EXPECTED_MESSAGES,
        "raw_labels": EXPECTED_INTENTS, "assignment_cells": EXPECTED_ASSIGNMENTS,
        "message_identity": "raw displayed_text SHA-256 bound into canonical dideco record_id",
        "annotation_entry_semantics": "completed multi-label ballot by authorized protocol assumption",
        "option_availability": "all 25 raw labels assumed available in all five rounds",
        "negative_vote_rule": "available completed ballot and not selected",
        "repeat_rule": "one vote if repeated values agree; otherwise intent-level abstention",
        "resolution_rule": "strict majority; tie or zero valid votes unresolved",
        "aggregation_provenance": AGGREGATION,
        "prediction_schema": PREDICTION_COLUMNS, "assignment_schema": ASSIGNMENT_COLUMNS,
        "intent_metric_schema": METRIC_COLUMNS, "overall_metric_schema": OVERALL_KEYS,
        "decision_threshold": THRESHOLD, "decision_rule": "phishing iff p_text >= 0.5",
        "text_view": "Step-14 normalized subject='' and body=displayed_text",
        "confidence_level": CONFIDENCE, "wilson_z": Z,
        "support_categories": {"primary": "n>=20", "exploratory": "10<=n<=19",
                               "insufficient": "1<=n<10", "not_determinable": "n=0"},
        "ordering": "record_id ascending; taxonomy explicit numeric then implicit name",
        "float_serialization": ".17g",
    }


def expected_run_id(protected: dict[str, str] = EXPECTED_PROTECTED,
                    root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    return "step30:" + hashlib.sha256(encoded).hexdigest()[:32]


def validate_isolation(messages: list[dict[str, str]]) -> None:
    ids = {row["record_id"] for row in messages}
    pairs = {("", row["body"]) for row in messages}
    split = rows(REPOSITORY_ROOT / f"outputs/splits/runs/{RUN}/split_assignments.csv",
                 ["record_id", "label", "partition"])
    check(not ids.intersection(row["record_id"] for row in split), "Split ID overlap")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with (REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        check(reader.fieldnames == NORMALIZED_COLUMNS, "Internal schema mismatch")
        for row in reader:
            check(row["source_role"] == "text_development" and row["dataset"] != SOURCE,
                  "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    check(len(internal_ids) == current_contract(REPOSITORY_ROOT).expected("text_total_count", 251_271)
          and not ids.intersection(internal_ids) and
          not pairs.intersection(internal_pairs), "DIDECO isolation mismatch")
    leakage = json.loads((REPOSITORY_ROOT / "outputs/leakage/reserved_sources/summary.json").read_text())
    check(leakage["exact_overlaps_by_scope"]["internal|dideco"] == 0 and
          leakage["source_role_isolation_passed"] is True, "Step-16 isolation mismatch")


def forbidden_metric_keys(value: Any) -> set[str]:
    forbidden = {"accuracy", "precision", "f1", "tn", "fp", "false_positive_rate"}
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(forbidden.intersection(value))
        for nested in value.values():
            found.update(forbidden_metric_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(forbidden_metric_keys(nested))
    return found


def validate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    global RUN, EXPECTED_RAW_ANNOTATIONS, EXPECTED_NONEMPTY_ANNOTATIONS, EXPECTED_EMPTY_ANNOTATIONS
    global EXPECTED_MESSAGES, EXPECTED_EXPLICIT, EXPECTED_IMPLICIT, EXPECTED_INTENTS, EXPECTED_ASSIGNMENTS
    global EXPECTED_REPEATED, EXPECTED_DIFFERING, EXPECTED_CONFLICT_MESSAGES, EXPECTED_CONFLICT_CELLS
    global EXPECTED_POSITIVE, EXPECTED_NEGATIVE, EXPECTED_UNRESOLVED
    contract = current_contract(root)
    EXPECTED_RAW_ANNOTATIONS = contract.expected("dideco_raw_annotations", EXPECTED_RAW_ANNOTATIONS)
    EXPECTED_NONEMPTY_ANNOTATIONS = contract.expected("dideco_nonempty_annotations", EXPECTED_NONEMPTY_ANNOTATIONS)
    EXPECTED_EMPTY_ANNOTATIONS = contract.expected("dideco_empty_annotations", EXPECTED_EMPTY_ANNOTATIONS)
    EXPECTED_MESSAGES = contract.expected("dideco_messages", EXPECTED_MESSAGES)
    EXPECTED_EXPLICIT = contract.expected("dideco_explicit_intents", EXPECTED_EXPLICIT)
    EXPECTED_IMPLICIT = contract.expected("dideco_implicit_intents", EXPECTED_IMPLICIT)
    EXPECTED_INTENTS = contract.expected("dideco_intents", EXPECTED_INTENTS)
    EXPECTED_ASSIGNMENTS = contract.expected("dideco_assignments", EXPECTED_ASSIGNMENTS)
    EXPECTED_REPEATED = contract.expected("dideco_repeated_pairs", EXPECTED_REPEATED)
    EXPECTED_DIFFERING = contract.expected("dideco_differing_pairs", EXPECTED_DIFFERING)
    EXPECTED_CONFLICT_MESSAGES = contract.expected("dideco_conflict_messages", EXPECTED_CONFLICT_MESSAGES)
    EXPECTED_CONFLICT_CELLS = contract.expected("dideco_conflict_abstentions", EXPECTED_CONFLICT_CELLS)
    EXPECTED_POSITIVE = contract.expected("dideco_positive_assignments", EXPECTED_POSITIVE)
    EXPECTED_NEGATIVE = contract.expected("dideco_negative_assignments", EXPECTED_NEGATIVE)
    EXPECTED_UNRESOLVED = contract.expected("dideco_unresolved_assignments", EXPECTED_UNRESOLVED)
    RUN = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    check(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate root unsupported")
    protected = resolve_current_protected_hashes(root, EXPECTED_PROTECTED)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    check(manifest["selection_provenance"] == current_selection_provenance(root),
          "Selection provenance chain mismatch")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    overall = json.loads((out / "overall_metrics.json").read_text(encoding="utf-8"))
    predictions = rows(out / "predictions.csv", PREDICTION_COLUMNS)
    assignments = rows(out / "intent_assignments.csv", ASSIGNMENT_COLUMNS)
    metrics = rows(out / "intent_metrics.csv", METRIC_COLUMNS)
    check(manifest["artifact_type"] == "step30_dideco_intent_rq2_evaluation",
          "Manifest provenance mismatch")
    check(manifest["input_sha256"] == protected, "Input inventory mismatch")
    check(manifest["schemas"] == {"predictions.csv": PREDICTION_COLUMNS,
                                  "intent_assignments.csv": ASSIGNMENT_COLUMNS,
                                  "intent_metrics.csv": METRIC_COLUMNS,
                                  "overall_metrics.json": OVERALL_KEYS}, "Manifest schema mismatch")
    expected_run = expected_run_id(protected, root)
    check(manifest["method_specification"] == method_specification(protected),
          "Method specification mismatch")
    check(manifest["evaluation_run_id"] == report["evaluation_run_id"] == expected_run,
          "Run identity mismatch")
    expected_implementation = {
        "src/dideco_intent_evaluation.py", "src/load_data.py", "src/preprocess.py",
        "src/prepare_reserved_evaluation.py", "src/final_artifact.py",
        "scripts/validate_dideco_intent_evaluation.py",
    }
    check(set(manifest["implementation_sha256"]) == expected_implementation,
          "Unexpected implementation inventory")
    for relative, expected in manifest["implementation_sha256"].items():
        check(digest(root / relative) == expected, f"Implementation hash mismatch: {relative}")
    for name in CHECKSUM_NAMES[:-1]:
        check(manifest["output_sha256"][name] == digest(out / name), f"Output hash mismatch: {name}")
    check((out / "checksums.sha256").read_text().splitlines() ==
          [f"{digest(out / name)}  {name}" for name in CHECKSUM_NAMES], "Checksum inventory mismatch")

    messages, raw_annotations, intents = independent_source()
    validate_isolation(messages)
    reproduced_assignments, aggregation_summary = independent_aggregation(
        messages, raw_annotations, intents, expected_run)
    check(assignments == reproduced_assignments, "Stored intent assignments differ")
    check(report["aggregation"] | {} and
          {key: report["aggregation"][key] for key in aggregation_summary} == aggregation_summary,
          "Aggregation report mismatch")
    check(len(predictions) == len({row["record_id"] for row in predictions}) == EXPECTED_MESSAGES and
          [row["record_id"] for row in predictions] == [row["record_id"] for row in messages],
          "Prediction coverage mismatch")
    check({row["evaluation_run_id"] for row in predictions} == {expected_run}, "Prediction run mismatch")

    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    check(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
          type(calibrator) is LogisticRegression, "Frozen component type mismatch")
    by_id = {row["record_id"]: row for row in messages}
    predicted: dict[str, str] = {}
    modes: set[str] = set()
    for actual in predictions:
        source = by_id[actual["record_id"]]
        raw = float(model.decision_function(vectorizer.transform([combine("", source["body"])]))[0])
        probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
        prediction = "phishing" if probability >= THRESHOLD else "legitimate"
        raw_ok, raw_mode = float_matches(float(actual["text_raw_score"]), raw)
        prob_ok, prob_mode = float_matches(float(actual["calibrated_probability_phishing"]), probability)
        check(raw_ok and prob_ok and actual["prediction"] == prediction and
              actual["outcome"] == ("detected" if prediction == "phishing" else "missed"),
              f"Prediction mismatch: {actual['record_id']}")
        check((actual["source"], actual["source_role"], actual["evaluation_label"], actual["label_provenance"]) ==
              (SOURCE, "rq2_evaluation", "phishing", "source_corpus_membership:dideco;not_message_gold_label"),
              "Prediction provenance mismatch")
        modes.update((raw_mode, prob_mode))
        predicted[actual["record_id"]] = prediction
    tp = sum(value == "phishing" for value in predicted.values())
    fn = EXPECTED_MESSAGES - tp
    interval = wilson(tp, EXPECTED_MESSAGES)
    check(interval is not None, "Overall interval undefined")
    expected_overall = {"n": EXPECTED_MESSAGES, "tp": tp, "fn": fn,
                        "recall": tp / EXPECTED_MESSAGES,
                        "miss_rate": fn / EXPECTED_MESSAGES, "recall_ci_95_lower": interval[0],
                        "recall_ci_95_upper": interval[1], "ci_method": "wilson_score_two_sided",
                        "confidence_level": CONFIDENCE, "z": Z}
    check(overall == report["overall_metrics"] == expected_overall, "Overall metrics mismatch")

    expected_metric_rows: list[dict[str, str]] = []
    for intent in intents:
        ids = [row["record_id"] for row in reproduced_assignments
               if row["intent_id"] == intent["intent_id"] and
               row["annotation_type"] == intent["annotation_type"] and row["resolved_label"] == "positive"]
        n = len(ids); intent_tp = sum(predicted[value] == "phishing" for value in ids); intent_fn = n - intent_tp
        bounds = wilson(intent_tp, n)
        expected_metric_rows.append({**intent, "n": str(n), "tp": str(intent_tp), "fn": str(intent_fn),
            "recall": "" if n == 0 else format(intent_tp / n, ".17g"),
            "miss_rate": "" if n == 0 else format(intent_fn / n, ".17g"),
            "recall_ci_95_lower": "" if bounds is None else format(bounds[0], ".17g"),
            "recall_ci_95_upper": "" if bounds is None else format(bounds[1], ".17g"),
            "ci_method": "" if n == 0 else "wilson_score_two_sided",
            "confidence_level": "" if n == 0 else format(CONFIDENCE, ".17g"),
            "z": "" if n == 0 else format(Z, ".17g"), "support_category": support(n)})
    check(metrics == expected_metric_rows and len(metrics) == EXPECTED_INTENTS, "Intent metrics mismatch")
    check(not forbidden_metric_keys(overall) and not forbidden_metric_keys(report), "Forbidden metric present")
    check(report["inference"]["url_model_loaded"] is False and report["inference"]["fusion_applied"] is False and
          report["inference"]["fit_performed"] is False and
          report["boundaries"]["holdout_metrics_computed"] is False, "Boundary violation")
    comparison = "documented_tolerance_fallback" if "documented_tolerance_fallback" in modes else "binary64_exact"
    return {"status": "validated", "evaluation_run_id": expected_run, "records": EXPECTED_MESSAGES,
            "assignment_cells": EXPECTED_ASSIGNMENTS, "float_comparison": comparison, "tp": tp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(validate(out=args.out), indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
