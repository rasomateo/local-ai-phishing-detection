from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.final_artifact import (
    ARCHITECTURE_NAME,
    DECISION_THRESHOLD,
    SCHEMA_VERSION,
    URL_INTERPRETATION_WARNING,
    URL_MODEL_SCOPE,
    URL_VALIDATION_SCOPE,
)
from src.utils import REPOSITORY_ROOT
from src.provenance import CONTRACT_VERSION, content_run_id

OUT = REPOSITORY_ROOT / "outputs/models/final_artifact"
PROTECTED_PATHS = [
    "outputs/models/text/classifier/text_vectorizer.joblib",
    "outputs/models/text/classifier/text_model.joblib",
    "outputs/models/text/classifier/manifest.json",
    "outputs/models/text/classifier/report.json",
    "outputs/models/text/calibration/text_calibrator.joblib",
    "outputs/models/text/calibration/manifest.json",
    "outputs/models/text/calibration/report.json",
    "outputs/models/url/url_vectorizer.joblib",
    "outputs/models/url/url_model.joblib",
    "outputs/models/url/url_calibrator.joblib",
    "outputs/models/url/manifest.json",
    "outputs/models/url/report.json",
]
IMPLEMENTATION_PATHS = [
    "src/final_artifact.py",
    "src/url_model_bundle.py",
    "src/unified_artifact.py",
    "src/explain.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _semantic_identity(
    predecessor_semantics: dict[str, str], implementation: dict[str, str], architecture: dict[str, Any]
) -> tuple[str, str]:
    return content_run_id("step25c", {
        "contract_version": f"{CONTRACT_VERSION}:final-artifact-v2",
        "architecture": architecture,
        "implementation_sha256": implementation,
        "predecessor_semantic_sha256": predecessor_semantics,
    })


def semantic_predecessors(root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    text = json.loads(
        (root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8")
    )
    url = json.loads((root / "outputs/models/url/manifest.json").read_text(encoding="utf-8"))
    return {
        "text_calibration": text["semantic_sha256"],
        "url_model": url["semantic_sha256"],
    }


def build_payloads(root: Path = REPOSITORY_ROOT) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    protected = {name: sha256(root / name) for name in PROTECTED_PATHS}
    implementation = {name: sha256(root / name) for name in IMPLEMENTATION_PATHS}
    architecture = {
        "architecture_name": ARCHITECTURE_NAME,
        "artifact_version": 2,
        "schema_version": SCHEMA_VERSION,
        "step": "25C",
        "email_decision_source": "frozen_step19_text_model_plus_step20_inference_calibrator_only",
        "decision_threshold": DECISION_THRESHOLD,
        "decision_rule": "phishing iff text_probability_phishing >= 0.5; otherwise legitimate",
        "fusion_applied": False,
        "url_evidence": {
            "model_source": "frozen_step23_url_bundle",
            "all_valid_urls_scored": True,
            "affects_email_classification": False,
            "model_scope": URL_MODEL_SCOPE,
            "validation_scope": URL_VALIDATION_SCOPE,
            "interpretation_warning": URL_INTERPRETATION_WARNING,
        },
    }
    return architecture, protected, implementation


def generate(root: Path = REPOSITORY_ROOT, out: Path = OUT) -> dict[str, Any]:
    if root.resolve() != REPOSITORY_ROOT.resolve():
        raise ValueError("Alternate repository root unsupported")
    from scripts.validate_text_calibration import validate as validate_text_calibration
    from scripts.validate_url_model import validate as validate_url_model

    text_validation = validate_text_calibration(root)
    url_validation = validate_url_model(root)
    architecture, protected, implementation = build_payloads(root)
    text_manifest = json.loads((root / "outputs/models/text/calibration/manifest.json").read_text(encoding="utf-8"))
    url_manifest = json.loads((root / "outputs/models/url/manifest.json").read_text(encoding="utf-8"))
    selection_provenance = {
        "text": text_manifest["selection_provenance"],
        "url": url_manifest["selection_provenance"],
    }
    if text_validation["selection_semantic_sha256"] != selection_provenance["text"]["semantic_sha256"]:
        raise ValueError("Text selection provenance differs from validated calibration")
    if url_validation["selection_semantic_sha256"] != selection_provenance["url"]["semantic_sha256"]:
        raise ValueError("URL selection provenance differs from validated model")
    predecessor_semantics = semantic_predecessors(root)
    run_id, semantic_sha = _semantic_identity(predecessor_semantics, implementation, architecture)
    flags = {
        "model_fitted": False,
        "calibrator_fitted": False,
        "threshold_optimized": False,
        "fusion_applied": False,
        "holdout_opened": False,
        "holdout_evaluated": False,
        "external_data_used": False,
        "rq2_or_rq3_started": False,
        "step26_started": False,
    }
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="step25c-staging-", dir=parent))
    backup = out.with_name(f"{out.name}.previous")
    try:
        write_json(staging / "architecture.json", architecture)
        report = {
            "step": "25C",
            "architecture_run_id": run_id,
            "contract_version": f"{CONTRACT_VERSION}:final-artifact-v2",
            "semantic_sha256": semantic_sha,
            "scope": "Final offline text-led email classification with auxiliary URL model evidence",
            "selection_provenance": selection_provenance,
            "predecessor_semantic_sha256": predecessor_semantics,
            "decision_threshold": DECISION_THRESHOLD,
            "public_entrypoint": "src.unified_artifact.predict_unified",
            "final_entrypoint": "src.final_artifact.predict_final_artifact",
            "flags": flags,
            "protected_inputs": protected,
            "implementation_sha256": implementation,
            "output_sha256": {"architecture.json": sha256(staging / "architecture.json")},
        }
        write_json(staging / "report.json", report)
        manifest = {
            "artifact_type": "step25c_final_text_led_architecture",
            "artifact_version": 2,
            "architecture_run_id": run_id,
            "contract_version": f"{CONTRACT_VERSION}:final-artifact-v2",
            "semantic_sha256": semantic_sha,
            "selection_provenance": selection_provenance,
            "predecessor_semantic_sha256": predecessor_semantics,
            "protected_inputs": protected,
            "implementation_sha256": implementation,
            "flags": flags,
            "output_sha256": {
                "architecture.json": sha256(staging / "architecture.json"),
                "report.json": sha256(staging / "report.json"),
            },
            "self_hash_policy": "No JSON contains its own hash; checksums.sha256 excludes itself",
        }
        write_json(staging / "manifest.json", manifest)
        checksum_names = ["architecture.json", "report.json", "manifest.json"]
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in checksum_names),
            encoding="utf-8",
            newline="\n",
        )

        from scripts.validate_final_artifact import validate

        validate(root=root, out=staging)
        if backup.exists():
            shutil.rmtree(backup)
        if out.exists():
            os.replace(out, backup)
        os.replace(staging, out)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, sort_keys=True))
