from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ActiveArtifact:
    artifact_id: str
    producer: str
    output_paths: tuple[str, ...]
    consumers: tuple[str, ...]
    validator: str
    predecessors: tuple[str, ...]
    provenance: str


ROOT_INPUTS = {"raw_datasets", "approved_source"}
ARTIFACTS = (
    ActiveArtifact("acquisition", "scripts.verify_acquired_datasets", (
        "outputs/dataset_acquisition_manifest.json", "outputs/dataset_acquisition_report.md"),
        ("src.load_data",), "scripts.validate_dataset_integration", ("raw_datasets", "approved_source"),
        "immutable raw hashes and acquisition contract"),
    ActiveArtifact("integrated_datasets", "src.load_data", (
        "outputs/merged_raw_dataset.csv",),
        ("src.preprocess", "src.prepare_reserved_evaluation"), "scripts.validate_dataset_integration",
        ("acquisition",), "acquisition manifest, source counts and raw hashes"),
    ActiveArtifact("preprocessed_email_dataset", "src.preprocess", (
        "outputs/preprocessed_dataset.csv", "outputs/extracted_urls.csv", "outputs/preprocessing_report.json",
        "outputs/dataset_roles/nazario_2025_temporal_evaluation.csv",
        "outputs/dataset_roles/phishload_url_only.csv", "outputs/dataset_roles/trec_excluded.csv"),
        ("src.split_data",), "scripts.validate_data_splits", ("integrated_datasets",),
        "integrated dataset hash and deterministic normalization contract"),
    ActiveArtifact("email_splits", "src.split_data", (
        "outputs/splits/manifest.json", "outputs/splits/runs/*/*.csv", "outputs/splits/runs/*/manifest.json"),
        ("src.reserved_source_audit", "src.prepare_text_inputs"), "scripts.validate_data_splits",
        ("preprocessed_email_dataset",), "content-addressed split contract, seeds, counts and input hashes"),
    ActiveArtifact("reserved_source_audit", "src.reserved_source_audit", (
        "outputs/leakage/reserved_sources/*.csv", "outputs/leakage/reserved_sources/summary.json",
        "outputs/dataset_roles/ephishllm_english_external_evaluation.csv",
        "outputs/dataset_roles/phishingpot_external_evaluation.csv",
        "outputs/dataset_roles/dideco_intent_evaluation.csv", "outputs/dataset_roles/dideco_annotations.csv"),
        ("src.prepare_text_inputs",), "scripts.validate_leakage_audit", ("email_splits",),
        "split fingerprint, reserved-source hashes and overlap exclusions"),
    ActiveArtifact("text_inputs", "src.prepare_text_inputs", ("outputs/features/text/*.csv", "outputs/features/text/report.json"),
        ("src.text_model_selection", "src.train_text_model", "src.internal_holdout_evaluation"),
        "scripts.validate_text_inputs", ("reserved_source_audit",),
        "current split fingerprint, artifact hashes and deterministic counts"),
    ActiveArtifact("text_selection", "src.text_model_selection", ("outputs/benchmarks/text/*",),
        ("src.train_text_model",), "scripts.validate_text_model_selection", ("text_inputs",),
        "current text inputs, folds, candidates, views, parameters, seeds and OOF results"),
    ActiveArtifact("text_model", "src.train_text_model", ("outputs/models/text/classifier/*",),
        ("src.calibrate_text_model",), "scripts.validate_text_model", ("text_selection",),
        "validated current selection winner and training inputs"),
    ActiveArtifact("text_calibration", "src.calibrate_text_model", ("outputs/models/text/calibration/*",),
        ("src.build_final_artifact",), "scripts.validate_text_calibration", ("text_model",),
        "current text model, OOF scores and calibration contract"),
    ActiveArtifact("phishload_audit", "src.phishload_dataset audit", ("outputs/datasets/url/audit/*",),
        ("src.phishload_dataset prepare",), "scripts.validate_url_dataset_audit", ("integrated_datasets",),
        "raw Phishload identity and semantic audit fingerprint"),
    ActiveArtifact("url_dataset", "src.phishload_dataset prepare", ("outputs/datasets/url/canonical/*",),
        ("src.url_model_selection", "src.train_url_model", "src.internal_holdout_evaluation"),
        "scripts.validate_url_dataset", ("phishload_audit",),
        "validated audit fingerprint, canonicalization, splits, folds, seeds and counts"),
    ActiveArtifact("url_selection", "src.url_model_selection", ("outputs/benchmarks/url/*",),
        ("src.train_url_model",), "scripts.validate_url_model_selection", ("url_dataset",),
        "current URL dataset, folds, representations, candidates, parameters, seeds and OOF results"),
    ActiveArtifact("url_model", "src.train_url_model", ("outputs/models/url/*",),
        ("src.build_final_artifact",), "scripts.validate_url_model", ("url_selection",),
        "validated current URL selection winner and calibration inputs"),
    ActiveArtifact("final_artifact", "src.build_final_artifact", ("outputs/models/final_artifact/*",),
        ("src.internal_holdout_evaluation", "src.temporal_evaluation", "src.ephishllm_evaluation",
         "src.phishingpot_evaluation", "src.dideco_intent_evaluation", "src.explanation_evaluation",
         "src.technical_evaluation", "src.unified_artifact"),
        "scripts.validate_final_artifact", ("text_calibration", "url_model"),
        "current text and URL predecessor fingerprints and concrete artifact hashes"),
    ActiveArtifact("internal_holdout", "src.internal_holdout_evaluation", ("outputs/evaluation/internal_holdout/*",),
        ("src.explanation_evaluation", "src.technical_evaluation"), "scripts.validate_internal_holdout",
        ("final_artifact", "text_inputs", "url_dataset"), "current final artifact and held-out split provenance"),
    ActiveArtifact("temporal_evaluation", "src.temporal_evaluation", ("outputs/evaluation/temporal/*",),
        ("src.explanation_evaluation", "src.technical_evaluation"), "scripts.validate_temporal_evaluation",
        ("final_artifact", "integrated_datasets", "internal_holdout"),
        "current artifact, holdout completion and Nazario-2025 population hashes"),
    ActiveArtifact("ephishllm_evaluation", "src.ephishllm_evaluation", ("outputs/evaluation/ephishllm/*",),
        ("src.explanation_evaluation", "src.technical_evaluation"), "scripts.validate_ephishllm_evaluation",
        ("final_artifact", "integrated_datasets", "internal_holdout", "temporal_evaluation"),
        "current artifact, preceding evaluation chain and E-PhishLLM population hashes"),
    ActiveArtifact("phishingpot_evaluation", "src.phishingpot_evaluation", ("outputs/evaluation/phishingpot/*",),
        ("src.explanation_evaluation", "src.technical_evaluation"), "scripts.validate_phishingpot_evaluation",
        ("final_artifact", "integrated_datasets", "internal_holdout", "temporal_evaluation",
         "ephishllm_evaluation"), "current artifact, preceding evaluation chain and PhishingPot population hashes"),
    ActiveArtifact("dideco_evaluation", "src.dideco_intent_evaluation", ("outputs/evaluation/dideco_intent/*",),
        ("src.explanation_evaluation", "src.technical_evaluation"), "scripts.validate_dideco_intent_evaluation",
        ("final_artifact", "integrated_datasets", "internal_holdout", "temporal_evaluation",
         "ephishllm_evaluation", "phishingpot_evaluation"),
        "current artifact, preceding evaluation chain and DIDECO annotation/population hashes"),
    ActiveArtifact("explanation_evaluation", "src.explanation_evaluation", ("outputs/evaluation/explanations/*",),
        ("src.technical_evaluation",), "scripts.validate_explanation_evaluation",
        ("final_artifact", "internal_holdout", "temporal_evaluation", "ephishllm_evaluation",
         "phishingpot_evaluation", "dideco_evaluation"), "current artifact and current evaluation hashes"),
    ActiveArtifact("technical_evaluation", "src.technical_evaluation", ("outputs/evaluation/technical/*",),
        ("src.unified_artifact",), "scripts.validate_technical_evaluation",
        ("final_artifact", "internal_holdout", "temporal_evaluation", "ephishllm_evaluation",
         "phishingpot_evaluation", "dideco_evaluation", "explanation_evaluation"),
        "current artifact, current evaluation hashes and runtime contract"),
)

ACTIVE_SOURCE_FILES = (
    "scripts/verify_acquired_datasets.py", "src/load_data.py", "scripts/validate_dataset_integration.py",
    "src/preprocess.py", "src/split_data.py", "scripts/validate_data_splits.py",
    "src/reserved_source_audit.py", "scripts/validate_leakage_audit.py",
    "src/prepare_text_inputs.py", "scripts/validate_text_inputs.py",
    "src/text_model_selection.py", "scripts/validate_text_model_selection.py",
    "src/train_text_model.py", "scripts/validate_text_model.py",
    "src/calibrate_text_model.py", "scripts/validate_text_calibration.py",
    "src/phishload_dataset.py", "scripts/validate_url_dataset_audit.py", "scripts/validate_url_dataset.py",
    "src/url_model_selection.py", "scripts/validate_url_model_selection.py",
    "src/train_url_model.py", "scripts/validate_url_model.py",
    "src/build_final_artifact.py", "scripts/validate_final_artifact.py",
    "src/internal_holdout_evaluation.py", "scripts/validate_internal_holdout.py",
    "src/technical_evaluation.py", "scripts/validate_technical_evaluation.py",
    "src/explanation_evaluation.py", "scripts/validate_explanation_evaluation.py",
    "src/temporal_evaluation.py", "scripts/validate_temporal_evaluation.py",
    "src/ephishllm_evaluation.py", "scripts/validate_ephishllm_evaluation.py",
    "src/phishingpot_evaluation.py", "scripts/validate_phishingpot_evaluation.py",
    "src/dideco_intent_evaluation.py", "scripts/validate_dideco_intent_evaluation.py",
)


def validate_topology(root: Path) -> list[ActiveArtifact]:
    ids = {item.artifact_id for item in ARTIFACTS}
    if len(ids) != len(ARTIFACTS):
        raise ValueError("Duplicate active artifact ID")
    produced_paths: set[str] = set()
    seen = set(ROOT_INPUTS)
    for item in ARTIFACTS:
        if not item.output_paths or not item.validator:
            raise ValueError(f"Incomplete producer/validator contract: {item.artifact_id}")
        missing = set(item.predecessors) - seen
        if missing:
            raise ValueError(f"Consumer precedes producer: {item.artifact_id}: {sorted(missing)}")
        for path in item.output_paths:
            if path in produced_paths:
                raise ValueError(f"Output has multiple producers: {path}")
            produced_paths.add(path)
        seen.add(item.artifact_id)
    forbidden_parts = ("outputs/models/" + "fusion/", "outputs/analysis/" + "fusion_failures/",
                       "outputs/evaluation/" + "historical/")
    for item in ARTIFACTS:
        if any(path.startswith(forbidden_parts) for path in item.output_paths):
            raise ValueError(f"Historical output is active: {item.artifact_id}")
    forbidden_tokens = ("outputs/models/" + "fusion", "outputs/analysis/" + "fusion_failures",
                        "outputs/evaluation/" + "historical", "configuration_c", "paired_comparison",
                        "historical_step25c_validation", "historical_manifests_reinterpreted")
    for relative in ACTIVE_SOURCE_FILES:
        text = (root / relative).read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            if token in text:
                raise ValueError(f"Active source retains removed contract {token!r}: {relative}")
    return list(ARTIFACTS)


def as_jsonable(items: Iterable[ActiveArtifact] = ARTIFACTS) -> list[dict[str, object]]:
    return [asdict(item) for item in items]


def as_markdown(items: Iterable[ActiveArtifact] = ARTIFACTS) -> str:
    lines = ["# Active Producer-Consumer Topology", "",
             "| Artifact | Producer | Output path(s) | Immediate consumers | Validator | Upstream provenance |",
             "|---|---|---|---|---|---|"]
    for item in items:
        lines.append("| " + " | ".join((item.artifact_id, item.producer, "<br>".join(item.output_paths),
                                         "<br>".join(item.consumers), item.validator,
                                         ", ".join(item.predecessors) + ": " + item.provenance)) + " |")
    return "\n".join(lines) + "\n"
