from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.load_data import NORMALIZED_COLUMNS
from src.prepare_reserved_evaluation import ANNOTATION_COLUMNS, DIDECO_PATH, load_dideco
from src.preprocess import normalize_text_field
from src.final_artifact import _combine_subject_body
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

SPLIT_RUN_ID = ""
SOURCE = "dideco"
SOURCE_URL = "https://aclanthology.org/2026.lrec-1.542/"
SOURCE_REPOSITORY = "https://github.com/Sneusneu/Dideco/"
EXPECTED_RAW_ANNOTATIONS = 639
EXPECTED_NONEMPTY_ANNOTATIONS = 609
EXPECTED_EMPTY_ANNOTATIONS = 30
EXPECTED_MESSAGES = 220
EXPECTED_EXPLICIT_INTENTS = 8
EXPECTED_IMPLICIT_INTENTS = 17
EXPECTED_INTENTS = 25
EXPECTED_ASSIGNMENTS = 5_500
EXPECTED_REPEATED_ANNOTATOR_MESSAGE_PAIRS = 56
EXPECTED_DIFFERING_REPEAT_PAIRS = 55
EXPECTED_REPEAT_CONFLICT_MESSAGES = 38
EXPECTED_CONFLICT_ABSTENTIONS = 239
EXPECTED_POSITIVE_ASSIGNMENTS = 1_074
EXPECTED_NEGATIVE_ASSIGNMENTS = 4_180
EXPECTED_UNRESOLVED_ASSIGNMENTS = 246
DECISION_THRESHOLD = 0.5
CONFIDENCE_LEVEL = 0.95
WILSON_Z = 1.959963984540054
AGGREGATION_PROVENANCE = (
    "step30_protocol_majority;25_options_assumed_available_all_rounds;"
    "selected=positive;not_selected=negative;repeat_conflict=abstention;"
    "strict_majority;tie_or_no_votes=unresolved"
)

CANONICAL = REPOSITORY_ROOT / "outputs/dataset_roles/dideco_intent_evaluation.csv"
ANNOTATIONS = REPOSITORY_ROOT / "outputs/dataset_roles/dideco_annotations.csv"
INTERNAL = REPOSITORY_ROOT / "outputs/preprocessed_dataset.csv"
OUT = REPOSITORY_ROOT / "outputs/evaluation/dideco_intent"

EXPECTED_SHA256 = {
    "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip": "472eae1754c456d41d182396a043969143a04146467fbfbb200a67b5f3028b12",
    "outputs/dataset_roles/dideco_intent_evaluation.csv": "29ce1a8715181892fc30faa95720fb68d70b20d675e612d2007758cb9c220acc",
    "outputs/dataset_roles/dideco_annotations.csv": "239a93c35f4af5e5d5479e4535ead44c6ff8ca532c5b14a159ae21e5ce76ecf6",
    "outputs/preprocessed_dataset.csv": "cb8ab274790414f518f0b7bb4122ce8655194044bf23798442185c95226963b6",
    "outputs/splits/manifest.json": "471dfc7558b80ead96beab66e60054f28aacdfd4c9a9ba96aeac39497de9d23d",
    f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv": "497acde2ed8ec6fd0a1d3d6f8c004398a0e2c3063d6771734b2d8eae41923b30",
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
INTENT_METRIC_COLUMNS = [
    "intent_id", "intent_name", "annotation_type", "n", "tp", "fn", "recall", "miss_rate",
    "recall_ci_95_lower", "recall_ci_95_upper", "ci_method", "confidence_level", "z",
    "support_category",
]
OVERALL_METRIC_KEYS = [
    "n", "tp", "fn", "recall", "miss_rate", "recall_ci_95_lower", "recall_ci_95_upper",
    "ci_method", "confidence_level", "z",
]
CHECKSUM_NAMES = [
    "predictions.csv", "intent_assignments.csv", "intent_metrics.csv",
    "overall_metrics.json", "report.json", "manifest.json",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == columns, f"Unexpected schema: {path}")
        return list(reader)


def validate_source(
    raw_messages: list[dict[str, str]], raw_annotations: list[dict[str, str]], metadata: dict[str, Any],
    canonical_messages: list[dict[str, str]], canonical_annotations: list[dict[str, str]],
) -> None:
    require(raw_messages == canonical_messages, "Reparsed DIDECO messages differ from canonical artifact")
    require(raw_annotations == canonical_annotations, "Reparsed DIDECO annotations differ from canonical artifact")
    require(len(raw_messages) == len({row["record_id"] for row in raw_messages}) == EXPECTED_MESSAGES,
            "DIDECO message identity mismatch")
    require(len(raw_annotations) == EXPECTED_RAW_ANNOTATIONS, "DIDECO annotation count mismatch")
    require(sum(row["displayed_text_empty"] == "false" for row in raw_annotations) == EXPECTED_NONEMPTY_ANNOTATIONS,
            "DIDECO nonempty annotation count mismatch")
    require(sum(row["displayed_text_empty"] == "true" for row in raw_annotations) == EXPECTED_EMPTY_ANNOTATIONS,
            "DIDECO empty annotation count mismatch")
    require(all(not row["message_record_id"] for row in raw_annotations if row["displayed_text_empty"] == "true"),
            "Empty displayed_text was assigned to a message")
    contract = current_contract(REPOSITORY_ROOT)
    source_hash = (contract.payload.get("raw_sha256", {}).get(
        "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip")
        if contract.is_fixture else EXPECTED_SHA256["data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip"])
    require(metadata == {
        "raw_annotation_rows": EXPECTED_RAW_ANNOTATIONS, "retained_messages": EXPECTED_MESSAGES,
        "empty_displayed_text_annotation_rows": EXPECTED_EMPTY_ANNOTATIONS, "rounds": ["1", "2", "3", "4", "5"],
        "annotators": ["1", "2", "3", "4", "5", "6"], "jsonl_files": 30,
        "source_sha256": source_hash,
    }, "DIDECO source metadata mismatch")
    for row in raw_messages:
        require(row["dataset"] == SOURCE and row["source_subset"] == "message" and
                row["source_role"] == "rq2_evaluation" and not row["subject"] and row["body"],
                "Unexpected DIDECO canonical row")
        require(row["label"] == row["raw_label"] == "phishing" and
                row["label_provenance"] == "source_corpus_membership:dideco;not_message_gold_label",
                "Unexpected DIDECO label provenance")
        require(normalize_text_field(row["body"]) == row["body"] and
                row["record_id"].startswith("dideco:"), "Invalid DIDECO text or record ID")


def taxonomy(annotations: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    mapping: dict[tuple[str, str], str] = {}
    for row in annotations:
        for annotation_type, field in (("explicit", "explicit_intents_json"),
                                       ("implicit", "implicit_intents_json")):
            values = json.loads(row[field])
            require(isinstance(values, dict), "Invalid DIDECO intent payload")
            for name, intent_id in values.items():
                key = (annotation_type, str(name))
                value = str(intent_id)
                require(key not in mapping or mapping[key] == value, "Inconsistent DIDECO intent ID mapping")
                mapping[key] = value
    explicit = [{"annotation_type": kind, "intent_name": name, "intent_id": value}
                for (kind, name), value in mapping.items() if kind == "explicit"]
    implicit = [{"annotation_type": kind, "intent_name": name, "intent_id": value}
                for (kind, name), value in mapping.items() if kind == "implicit"]
    explicit.sort(key=lambda row: int(row["intent_id"]))
    implicit.sort(key=lambda row: row["intent_name"])
    result = explicit + implicit
    require(len(explicit) == EXPECTED_EXPLICIT_INTENTS and len(implicit) == EXPECTED_IMPLICIT_INTENTS and
            len(result) == EXPECTED_INTENTS, "DIDECO raw-label taxonomy mismatch")
    require(all(row["intent_id"].isdigit() for row in explicit), "Invalid explicit intent ID")
    require(all(row["intent_id"] == f"intent_selection:::{row['intent_name']}" for row in implicit),
            "Invalid implicit intent ID")
    return result


def aggregate_annotations(
    messages: list[dict[str, str]], annotations: list[dict[str, str]], intents: list[dict[str, str]],
    run_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attached = [row for row in annotations if row["message_record_id"]]
    by_message_annotator: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_message: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in attached:
        by_message_annotator[(row["message_record_id"], row["annotator"])].append(row)
        by_message[row["message_record_id"]].append(row)
    require(set(by_message) == {row["record_id"] for row in messages},
            "Annotation message coverage mismatch")

    def selected(row: dict[str, str], intent: dict[str, str]) -> bool:
        field = "explicit_intents_json" if intent["annotation_type"] == "explicit" else "implicit_intents_json"
        values = json.loads(row[field])
        return values.get(intent["intent_name"]) == intent["intent_id"]

    repeated = differing = conflict_cells = 0
    conflict_messages: set[str] = set()
    for (record_id, _annotator), rows in by_message_annotator.items():
        if len(rows) > 1:
            repeated += 1
            signatures = {row["raw_intent_json"] for row in rows}
            if len(signatures) > 1:
                differing += 1
                conflict_messages.add(record_id)
        for intent in intents:
            votes = {selected(row, intent) for row in rows}
            if len(votes) > 1:
                conflict_cells += 1
                conflict_messages.add(record_id)

    assignments: list[dict[str, Any]] = []
    abstentions_by_intent: Counter[tuple[str, str, str]] = Counter()
    for message in sorted(messages, key=lambda row: row["record_id"]):
        record_id = message["record_id"]
        annotation_rows = by_message[record_id]
        annotators = sorted({row["annotator"] for row in annotation_rows})
        for intent in intents:
            positive = negative = abstentions = 0
            for annotator in annotators:
                repeated_rows = by_message_annotator[(record_id, annotator)]
                values = {selected(row, intent) for row in repeated_rows}
                if len(values) != 1:
                    abstentions += 1
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
            assignments.append({
                "evaluation_run_id": run_id, "record_id": record_id, **intent,
                "aggregation_provenance": AGGREGATION_PROVENANCE,
                "raw_annotation_count": len(annotation_rows), "annotator_count": len(annotators),
                "valid_vote_count": valid, "positive_vote_count": positive,
                "negative_vote_count": negative, "conflicting_repeat_abstention_count": abstentions,
                "resolved_label": resolved, "unresolved_reason": reason,
            })

    counts = Counter(row["resolved_label"] for row in assignments)
    require(len(assignments) == EXPECTED_ASSIGNMENTS and counts == Counter({
        "positive": EXPECTED_POSITIVE_ASSIGNMENTS, "negative": EXPECTED_NEGATIVE_ASSIGNMENTS,
        "unresolved": EXPECTED_UNRESOLVED_ASSIGNMENTS,
    }), "Aggregated DIDECO assignment counts changed")
    require(repeated == EXPECTED_REPEATED_ANNOTATOR_MESSAGE_PAIRS and
            differing == EXPECTED_DIFFERING_REPEAT_PAIRS and
            len(conflict_messages) == EXPECTED_REPEAT_CONFLICT_MESSAGES and
            conflict_cells == EXPECTED_CONFLICT_ABSTENTIONS,
            "DIDECO repeat-conflict reconciliation mismatch")
    return assignments, {
        "raw_annotation_rows": len(annotations), "attached_nonempty_annotation_rows": len(attached),
        "unattached_empty_displayed_text_rows": len(annotations) - len(attached),
        "unique_messages": len(messages), "raw_labels": len(intents),
        "explicit_labels": sum(row["annotation_type"] == "explicit" for row in intents),
        "implicit_labels": sum(row["annotation_type"] == "implicit" for row in intents),
        "assignment_cells": len(assignments), "repeated_message_annotator_pairs": repeated,
        "repeated_pairs_with_different_intent_sets": differing,
        "messages_with_repeat_conflicts": len(conflict_messages),
        "intent_level_repeat_conflict_abstentions": conflict_cells,
        "resolved_label_counts": dict(sorted(counts.items())),
        "abstentions_by_intent": [
            {"intent_id": key[0], "intent_name": key[1], "annotation_type": key[2], "count": count}
            for key, count in sorted(abstentions_by_intent.items())
        ],
    }


def validate_isolation(rows: list[dict[str, str]], root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    rq2_ids = {row["record_id"] for row in rows}
    rq2_pairs = {(row["subject"], row["body"]) for row in rows}
    assignments = read_csv(root / f"outputs/splits/runs/{SPLIT_RUN_ID}/split_assignments.csv",
                           ["record_id", "label", "partition"])
    require(not rq2_ids.intersection(row["record_id"] for row in assignments),
            "DIDECO record_id appears in internal split")
    internal_ids: set[str] = set()
    internal_pairs: set[tuple[str, str]] = set()
    with INTERNAL.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == NORMALIZED_COLUMNS, "Unexpected internal schema")
        for row in reader:
            require(row["source_role"] == "text_development" and row["dataset"] != SOURCE,
                    "Reserved source in internal corpus")
            internal_ids.add(row["record_id"])
            internal_pairs.add((row["subject"], row["body"]))
    require(len(internal_ids) == current_contract(root).expected("text_total_count", 251_271), "Internal corpus coverage mismatch")
    require(not rq2_ids.intersection(internal_ids), "DIDECO ID overlaps internal corpus")
    require(not rq2_pairs.intersection(internal_pairs), "DIDECO exact text overlaps internal corpus")
    leakage = json.loads((root / "outputs/leakage/reserved_sources/summary.json").read_text(encoding="utf-8"))
    require(leakage["exact_overlaps_by_scope"]["internal|dideco"] == 0 and
            leakage["source_role_isolation_passed"] is True and
            leakage["reserved_source_counts"]["dideco"]["retained_messages"] == EXPECTED_MESSAGES,
            "Step-16 DIDECO isolation changed")
    return {"record_id_overlap_count": 0, "exact_normalized_subject_body_overlap_count": 0,
            "step16_internal_dideco_overlap_count": 0, "status": "passed"}


def validate_protected_hashes(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    actual = resolve_current_protected_hashes(root, EXPECTED_SHA256)
    architecture = json.loads((root / "outputs/models/final_artifact/architecture.json").read_text(encoding="utf-8"))
    require(architecture["decision_threshold"] == DECISION_THRESHOLD and
            architecture["fusion_applied"] is False and architecture["email_decision_source"] ==
            "frozen_step19_text_model_plus_step20_inference_calibrator_only",
            "Step-25C text decision contract changed")
    step19 = json.loads((root / "outputs/models/text/classifier/manifest.json").read_text(encoding="utf-8"))
    step20 = json.loads((root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8"))
    require(step19["winner_config_id"] == step20["winner_config_id"], "Text configuration changed")
    if current_contract(root).is_production:
        require(step19["winner_config_id"] ==
                "subject_body|TFIDF_B|LinearSVC|C=2;class_weight=balanced", "Text configuration changed")
    require(step19["flags"]["rq2_data_used"] is False and step20["flags"]["rq2_data_used"] is False,
            "RQ2 data was used during model development")
    return actual


def load_text_components(root: Path = REPOSITORY_ROOT) -> tuple[Any, Any, Any]:
    vectorizer = joblib.load(root / "outputs/models/text/classifier/text_vectorizer.joblib")
    model = joblib.load(root / "outputs/models/text/classifier/text_model.joblib")
    calibrator = joblib.load(root / "outputs/models/text/calibration/text_calibrator.joblib")
    require(type(vectorizer) is TfidfVectorizer and type(model) is LinearSVC and
            type(calibrator) is LogisticRegression, "Unexpected frozen component type")
    require(list(model.classes_) == ["legitimate", "phishing"] and list(calibrator.classes_) == [0, 1] and
            calibrator.coef_.shape == (1, 1) and float(calibrator.coef_[0, 0]) > 0,
            "Frozen component orientation changed")
    return vectorizer, model, calibrator


def score_text(body: str, components: tuple[Any, Any, Any]) -> tuple[float, float, str]:
    subject, body = "", normalize_text_field(body)
    require(bool(body), "Fully empty DIDECO message is not evaluable")
    vectorizer, model, calibrator = components
    raw = float(model.decision_function(vectorizer.transform([_combine_subject_body(subject, body)]))[0])
    probability = float(calibrator.predict_proba(np.asarray([[raw]], dtype=np.float64))[0, 1])
    require(math.isfinite(raw) and math.isfinite(probability) and 0.0 <= probability <= 1.0,
            "Invalid frozen text score")
    return raw, probability, "phishing" if probability >= DECISION_THRESHOLD else "legitimate"


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    require(total >= 0 and 0 <= successes <= total, "Invalid Wilson inputs")
    if total == 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - margin),
            1.0 if successes == total else min(1.0, center + margin))


def metric_values(predictions: Iterable[str]) -> dict[str, Any]:
    values = list(predictions)
    tp, fn = values.count("phishing"), values.count("legitimate")
    require(values and tp + fn == len(values), "Invalid positive-only predictions")
    interval = wilson_interval(tp, len(values))
    require(interval is not None, "Overall Wilson interval is undefined")
    return {"n": len(values), "tp": tp, "fn": fn, "recall": tp / len(values),
            "miss_rate": fn / len(values), "recall_ci_95_lower": interval[0],
            "recall_ci_95_upper": interval[1], "ci_method": "wilson_score_two_sided",
            "confidence_level": CONFIDENCE_LEVEL, "z": WILSON_Z}


def support_category(total: int) -> str:
    require(total >= 0, "Negative support")
    if total >= 20:
        return "primary"
    if total >= 10:
        return "exploratory"
    if total >= 1:
        return "insufficient"
    return "not_determinable"


def intent_metrics(
    assignments: list[dict[str, Any]], predictions_by_id: dict[str, str], intents: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intent in intents:
        positive_ids = [row["record_id"] for row in assignments
                        if row["intent_id"] == intent["intent_id"] and
                        row["annotation_type"] == intent["annotation_type"] and
                        row["resolved_label"] == "positive"]
        n = len(positive_ids)
        tp = sum(predictions_by_id[record_id] == "phishing" for record_id in positive_ids)
        fn = n - tp
        interval = wilson_interval(tp, n)
        rows.append({**intent, "n": n, "tp": tp, "fn": fn,
                     "recall": "" if n == 0 else tp / n,
                     "miss_rate": "" if n == 0 else fn / n,
                     "recall_ci_95_lower": "" if interval is None else interval[0],
                     "recall_ci_95_upper": "" if interval is None else interval[1],
                     "ci_method": "" if n == 0 else "wilson_score_two_sided",
                     "confidence_level": "" if n == 0 else CONFIDENCE_LEVEL,
                     "z": "" if n == 0 else WILSON_Z, "support_category": support_category(n)})
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    return str(value)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row[field]) for field in fields})


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def validate_execution_context() -> None:
    require(not OUT.exists() or not any(OUT.iterdir()), "Step-30 output directory is not empty")


def method_specification(protected: dict[str, str]) -> dict[str, Any]:
    return {
        "version": 1, "protected_sha256": protected,
        "source": SOURCE, "raw_annotation_rows": EXPECTED_RAW_ANNOTATIONS,
        "evaluated_messages": EXPECTED_MESSAGES, "raw_labels": EXPECTED_INTENTS,
        "assignment_cells": EXPECTED_ASSIGNMENTS,
        "message_identity": "raw displayed_text SHA-256 bound into canonical dideco record_id",
        "annotation_entry_semantics": "completed multi-label ballot by authorized protocol assumption",
        "option_availability": "all 25 raw labels assumed available in all five rounds",
        "negative_vote_rule": "available completed ballot and not selected",
        "repeat_rule": "one vote if repeated values agree; otherwise intent-level abstention",
        "resolution_rule": "strict majority; tie or zero valid votes unresolved",
        "aggregation_provenance": AGGREGATION_PROVENANCE,
        "prediction_schema": PREDICTION_COLUMNS, "assignment_schema": ASSIGNMENT_COLUMNS,
        "intent_metric_schema": INTENT_METRIC_COLUMNS, "overall_metric_schema": OVERALL_METRIC_KEYS,
        "decision_threshold": DECISION_THRESHOLD, "decision_rule": "phishing iff p_text >= 0.5",
        "text_view": "Step-14 normalized subject='' and body=displayed_text",
        "confidence_level": CONFIDENCE_LEVEL, "wilson_z": WILSON_Z,
        "support_categories": {"primary": "n>=20", "exploratory": "10<=n<=19",
                               "insufficient": "1<=n<10", "not_determinable": "n=0"},
        "ordering": "record_id ascending; taxonomy explicit numeric then implicit name",
        "float_serialization": ".17g",
    }


def make_run_id(protected: dict[str, str], root: Path = REPOSITORY_ROOT) -> str:
    specification = method_specification(semantic_predecessor_identity(root, protected))
    encoded = json.dumps(specification, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "step30:" + hashlib.sha256(encoded).hexdigest()[:32]


def run(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    global SPLIT_RUN_ID, EXPECTED_RAW_ANNOTATIONS, EXPECTED_NONEMPTY_ANNOTATIONS
    global EXPECTED_EMPTY_ANNOTATIONS, EXPECTED_MESSAGES, EXPECTED_EXPLICIT_INTENTS
    global EXPECTED_IMPLICIT_INTENTS, EXPECTED_INTENTS, EXPECTED_ASSIGNMENTS
    global EXPECTED_REPEATED_ANNOTATOR_MESSAGE_PAIRS, EXPECTED_DIFFERING_REPEAT_PAIRS
    global EXPECTED_REPEAT_CONFLICT_MESSAGES, EXPECTED_CONFLICT_ABSTENTIONS
    global EXPECTED_POSITIVE_ASSIGNMENTS, EXPECTED_NEGATIVE_ASSIGNMENTS, EXPECTED_UNRESOLVED_ASSIGNMENTS
    contract = current_contract(root)
    EXPECTED_RAW_ANNOTATIONS = contract.expected("dideco_raw_annotations", EXPECTED_RAW_ANNOTATIONS)
    EXPECTED_NONEMPTY_ANNOTATIONS = contract.expected("dideco_nonempty_annotations", EXPECTED_NONEMPTY_ANNOTATIONS)
    EXPECTED_EMPTY_ANNOTATIONS = contract.expected("dideco_empty_annotations", EXPECTED_EMPTY_ANNOTATIONS)
    EXPECTED_MESSAGES = contract.expected("dideco_messages", EXPECTED_MESSAGES)
    EXPECTED_EXPLICIT_INTENTS = contract.expected("dideco_explicit_intents", EXPECTED_EXPLICIT_INTENTS)
    EXPECTED_IMPLICIT_INTENTS = contract.expected("dideco_implicit_intents", EXPECTED_IMPLICIT_INTENTS)
    EXPECTED_INTENTS = contract.expected("dideco_intents", EXPECTED_INTENTS)
    EXPECTED_ASSIGNMENTS = contract.expected("dideco_assignments", EXPECTED_ASSIGNMENTS)
    EXPECTED_REPEATED_ANNOTATOR_MESSAGE_PAIRS = contract.expected("dideco_repeated_pairs", EXPECTED_REPEATED_ANNOTATOR_MESSAGE_PAIRS)
    EXPECTED_DIFFERING_REPEAT_PAIRS = contract.expected("dideco_differing_pairs", EXPECTED_DIFFERING_REPEAT_PAIRS)
    EXPECTED_REPEAT_CONFLICT_MESSAGES = contract.expected("dideco_conflict_messages", EXPECTED_REPEAT_CONFLICT_MESSAGES)
    EXPECTED_CONFLICT_ABSTENTIONS = contract.expected("dideco_conflict_abstentions", EXPECTED_CONFLICT_ABSTENTIONS)
    EXPECTED_POSITIVE_ASSIGNMENTS = contract.expected("dideco_positive_assignments", EXPECTED_POSITIVE_ASSIGNMENTS)
    EXPECTED_NEGATIVE_ASSIGNMENTS = contract.expected("dideco_negative_assignments", EXPECTED_NEGATIVE_ASSIGNMENTS)
    EXPECTED_UNRESOLVED_ASSIGNMENTS = contract.expected("dideco_unresolved_assignments", EXPECTED_UNRESOLVED_ASSIGNMENTS)
    SPLIT_RUN_ID = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))["run_id"]
    require(root.resolve() == REPOSITORY_ROOT.resolve(), "Alternate repository root unsupported")
    validate_execution_context()
    protected = validate_protected_hashes(root)
    raw_messages, raw_annotations, source_metadata = load_dideco(DIDECO_PATH)
    canonical_messages = read_csv(CANONICAL, NORMALIZED_COLUMNS)
    canonical_annotations = read_csv(ANNOTATIONS, ANNOTATION_COLUMNS)
    validate_source(raw_messages, raw_annotations, source_metadata, canonical_messages, canonical_annotations)
    intents = taxonomy(raw_annotations)
    provisional_assignments, aggregation = aggregate_annotations(raw_messages, raw_annotations, intents)
    leakage = validate_isolation(raw_messages, root)
    components = load_text_components(root)
    run_id = make_run_id(protected, root)
    message_annotations = Counter(row["message_record_id"] for row in raw_annotations if row["message_record_id"])
    message_annotators: dict[str, set[str]] = defaultdict(set)
    for row in raw_annotations:
        if row["message_record_id"]:
            message_annotators[row["message_record_id"]].add(row["annotator"])

    predictions: list[dict[str, Any]] = []
    for row in sorted(raw_messages, key=lambda item: item["record_id"]):
        raw, probability, prediction = score_text(row["body"], components)
        predictions.append({
            "evaluation_run_id": run_id, "record_id": row["record_id"], "source": SOURCE,
            "source_role": "rq2_evaluation", "source_file": row["source_file"],
            "source_record": row["source_record"], "annotation_row_count": message_annotations[row["record_id"]],
            "annotator_count": len(message_annotators[row["record_id"]]), "evaluation_label": "phishing",
            "label_provenance": "source_corpus_membership:dideco;not_message_gold_label",
            "text_raw_score": raw, "calibrated_probability_phishing": probability,
            "prediction": prediction, "outcome": "detected" if prediction == "phishing" else "missed",
        })
    assignments = [{**row, "evaluation_run_id": run_id} for row in provisional_assignments]
    predictions_by_id = {row["record_id"]: row["prediction"] for row in predictions}
    overall = metric_values(predictions_by_id.values())
    per_intent = intent_metrics(assignments, predictions_by_id, intents)

    report = {
        "status": "completed", "evaluation_run_id": run_id,
        "source_provenance": {
            "paper": SOURCE_URL, "official_repository": SOURCE_REPOSITORY,
            "local_path": "data/raw/DIDECO/2026.lrec-1.542.OptionalSupplementaryMaterial.zip",
            "source_role": "rq2_evaluation", "evaluation_label": "phishing",
            "label_provenance": "source_corpus_membership:dideco;not_message_gold_label",
        },
        "message_reconciliation": {
            "raw_annotation_rows": EXPECTED_RAW_ANNOTATIONS,
            "nonempty_displayed_text_annotation_rows": EXPECTED_NONEMPTY_ANNOTATIONS,
            "empty_displayed_text_annotation_rows": EXPECTED_EMPTY_ANNOTATIONS,
            "unique_evaluated_messages": EXPECTED_MESSAGES,
            "identity_rule": "exact raw displayed_text SHA-256; id/instance_id not used as message identity",
        },
        "taxonomy": {
            "paper_claimed_category_count": 20, "raw_label_count": EXPECTED_INTENTS,
            "explicit_raw_labels": EXPECTED_EXPLICIT_INTENTS, "implicit_raw_labels": EXPECTED_IMPLICIT_INTENTS,
            "raw_labels_preserved_without_mapping_or_collapse": True, "labels": intents,
        },
        "aggregation": {
            **aggregation,
            "labels_are": "Step-30 protocol-majority labels; not official DIDECO gold or author-confirmed consensus",
            "completed_ballot_assumption": True,
            "all_25_options_available_all_rounds_assumption": True,
            "potato_completion_or_ui_configuration_technically_available": False,
            "assumption_basis": "authorized interpretation of the paper's standardized multi-label protocol",
        },
        "leakage": {**leakage,
                    "near_duplicate_scope": "not systematically assessed; Step-16C exact-only decision retained"},
        "inference": {
            "text_view": "Step-14 normalized subject='' and body=displayed_text",
            "missing_real_subject_is_limitation": True, "decision_threshold": DECISION_THRESHOLD,
            "decision_rule": "phishing iff p_text >= 0.5", "url_extraction_performed": False,
            "url_model_loaded": False, "fusion_applied": False, "explanation_component_used": False,
            "fit_performed": False,
        },
        "overall_metrics": overall,
        "intent_metrics": per_intent,
        "support_interpretation": {
            "primary": "n>=20", "exploratory": "10-19", "insufficient": "1-9",
            "not_determinable": "n=0",
        },
        "limitations": [
            "DIDECO is positive-only; accuracy, precision, F1, TN, and FP are not identifiable.",
            "The source consists of English LLM-generated spear-phishing messages from one synthetic corpus.",
            "No real Subject field is available; inference uses an empty Subject.",
            "The paper states 20 categories while the released ZIP encodes 25 raw labels.",
            "All 25 options being available in every round and each exported row being complete are authorized protocol-based assumptions, not technically verified POTATO UI facts.",
            "Intent groups overlap, so supports sum beyond the message count and differences are not independent.",
            "Unresolved message-intent pairs are neither positive nor negative and are excluded from intent support.",
            "Recall differences are descriptive and do not establish causality or statistical significance.",
            "Non-exact near-duplicate or template-level overlap was not systematically assessed.",
        ],
        "boundaries": {
            "training_performed": False, "calibration_performed": False, "threshold_optimized": False,
            "url_model_used": False, "fusion_used": False, "holdout_metrics_computed": False,
            "previous_evaluations_rerun": False, "thesis_updated": False, "step31_started": False,
        },
    }

    implementation = {relative: sha256(root / relative) for relative in [
        "src/dideco_intent_evaluation.py", "scripts/validate_dideco_intent_evaluation.py",
        "src/load_data.py", "src/preprocess.py", "src/prepare_reserved_evaluation.py",
        "src/final_artifact.py",
    ]}
    manifest = {
        "selection_provenance": current_selection_provenance(root),
        "artifact_type": "step30_dideco_intent_rq2_evaluation", "schema_version": 1,
        "status": "completed", "evaluation_run_id": run_id,
        "method_specification": method_specification(protected),
        "input_sha256": protected, "model_sha256": {
            key: value for key, value in protected.items() if key.endswith(".joblib")
        }, "implementation_sha256": implementation,
        "schemas": {
            "predictions.csv": PREDICTION_COLUMNS,
            "intent_assignments.csv": ASSIGNMENT_COLUMNS,
            "intent_metrics.csv": INTENT_METRIC_COLUMNS,
            "overall_metrics.json": OVERALL_METRIC_KEYS,
        },
        "runtime": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "joblib": importlib.metadata.version("joblib")},
        "output_sha256": {},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dideco-intent-evaluation-staging-", dir=OUT.parent))
    try:
        write_csv(staging / "predictions.csv", PREDICTION_COLUMNS, predictions)
        write_csv(staging / "intent_assignments.csv", ASSIGNMENT_COLUMNS, assignments)
        write_csv(staging / "intent_metrics.csv", INTENT_METRIC_COLUMNS, per_intent)
        write_json(staging / "overall_metrics.json", overall)
        write_json(staging / "report.json", report)
        manifest["output_sha256"] = {name: sha256(staging / name) for name in CHECKSUM_NAMES[:-1]}
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in CHECKSUM_NAMES),
            encoding="utf-8", newline="\n")
        subprocess.run([sys.executable, "-m", "scripts.validate_dideco_intent_evaluation",
                        "--out", str(staging)], cwd=root, check=True)
        require(not OUT.exists(), "Step-30 output path appeared during staging")
        os.replace(staging, OUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "completed", "evaluation_run_id": run_id, **overall}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-rq2-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.execute_rq2_evaluation:
        parser.error("Step 30 is inert without --execute-rq2-evaluation")
    print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
