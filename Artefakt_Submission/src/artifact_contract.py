from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.provenance import canonical_json


DYNAMIC_PIPELINE_PREFIXES = (
    "outputs/dataset_",
    "outputs/merged_raw_dataset.csv",
    "outputs/preprocessed_dataset.csv",
    "outputs/preprocessing_report.json",
    "outputs/extracted_urls.csv",
    "outputs/splits/",
    "outputs/leakage/",
    "outputs/features/text/",
    "outputs/benchmarks/text/",
    "outputs/benchmarks/url/",
    "outputs/datasets/url/",
    "outputs/models/text/",
    "outputs/models/url/",
    "outputs/models/final_artifact/",
    "outputs/evaluation/",
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_current_protected_hashes(
    root: Path,
    expected: dict[str, str],
    *,
    dynamic_prefixes: tuple[str, ...] = DYNAMIC_PIPELINE_PREFIXES,
) -> dict[str, str]:
    from src.execution_contract import current_contract
    contract = current_contract(root)
    current: dict[str, str] = {}
    split_manifest = None
    for original_relative, pinned in expected.items():
        relative = original_relative
        if relative.startswith("outputs/splits/runs/"):
            if split_manifest is None:
                split_manifest = json.loads((root / "outputs/splits/manifest.json").read_text(encoding="utf-8"))
            suffix = relative.split("/", 4)[-1]
            relative = f"{split_manifest['run_directory']}/{suffix}"
        actual = sha256(root / relative)
        dynamic = relative.startswith(dynamic_prefixes)
        immutable_expected = pinned
        if contract.is_fixture and relative.startswith("data/raw/"):
            fixture_hashes = contract.payload.get("raw_sha256", {})
            if relative not in fixture_hashes:
                raise ValueError(f"Fixture contract does not authorize raw input: {relative}")
            immutable_expected = fixture_hashes[relative]
        if not dynamic and actual != immutable_expected:
            raise ValueError(f"Protected immutable input changed: {relative}")
        current[relative] = actual
    return current


def current_selection_provenance(root: Path) -> dict[str, dict[str, str]]:
    final_dir = root / "outputs/models/final_artifact"
    final_manifest_path = final_dir / "manifest.json"
    checksums = {}
    for line in (final_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in checksums:
            raise ValueError("Duplicate final-artifact checksum entry")
        checksums[name] = digest
    if checksums.get("manifest.json") != sha256(final_manifest_path):
        raise ValueError("Final-artifact manifest checksum mismatch")
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    provenance = final_manifest.get("selection_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"text", "url"}:
        raise ValueError("Final artifact lacks the current text/URL selection provenance")
    for selection_type, item in provenance.items():
        if not isinstance(item, dict) or not isinstance(item.get("semantic_sha256"), str):
            raise ValueError(f"Invalid {selection_type} selection provenance")
        selection_manifest_path = root / item.get("manifest_path", "")
        if item.get("manifest_sha256") != sha256(selection_manifest_path):
            raise ValueError(f"Current {selection_type} selection manifest checksum mismatch")
        selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        if selection_manifest.get("selection_type") != selection_type or \
                selection_manifest.get("semantic_sha256") != item["semantic_sha256"]:
            raise ValueError(f"Current {selection_type} semantic provenance mismatch")
    return provenance


def semantic_predecessor_identity(root: Path, protected: dict[str, str]) -> dict[str, str]:
    identity: dict[str, str] = {}
    model_manifests = {
        "outputs/models/text/classifier/": "outputs/models/text/classifier/manifest.json",
        "outputs/models/text/calibration/": "outputs/models/text/calibration/manifest.json",
        "outputs/models/url/": "outputs/models/url/manifest.json",
        "outputs/models/final_artifact/": "outputs/models/final_artifact/manifest.json",
    }
    for prefix, manifest_relative in model_manifests.items():
        if any(path.startswith(prefix) for path in protected):
            manifest = json.loads((root / manifest_relative).read_text(encoding="utf-8"))
            semantic = manifest.get("semantic_sha256")
            if not isinstance(semantic, str):
                raise ValueError(f"Predecessor manifest lacks semantic fingerprint: {manifest_relative}")
            identity[f"{prefix.rstrip('/')}#semantic_sha256"] = semantic
    for relative, digest in sorted(protected.items()):
        if any(relative.startswith(prefix) for prefix in model_manifests):
            continue
        path = root / relative
        if relative == "outputs/dataset_acquisition_manifest.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            operational_keys = {
                "generated_at_utc",
                "filesystem_last_write_time_utc",
                "acquisition_date",
                "acquisition_date_note",
            }

            def semantic_projection(value: Any) -> Any:
                if isinstance(value, dict):
                    return {
                        key: semantic_projection(item)
                        for key, item in value.items()
                        if key not in operational_keys
                    }
                if isinstance(value, list):
                    return [semantic_projection(item) for item in value]
                return value

            projection = semantic_projection(payload)
            identity[f"{relative}#semantic_projection_sha256"] = hashlib.sha256(
                canonical_json(projection)
            ).hexdigest()
            continue
        if relative.endswith("/checksums.sha256") and "/evaluation/" in relative:
            manifest_path = path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = manifest.get("evaluation_run_id")
            if not isinstance(run_id, str):
                raise ValueError(f"Evaluation predecessor lacks run ID: {manifest_path}")
            identity[f"{path.parent.relative_to(root).as_posix()}#evaluation_run_id"] = run_id
            continue
        if relative.endswith(("/manifest.json", "/report.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            semantic = payload.get("semantic_sha256")
            if isinstance(semantic, str):
                identity[f"{relative}#semantic_sha256"] = semantic
                continue
        identity[relative] = digest
    return identity
