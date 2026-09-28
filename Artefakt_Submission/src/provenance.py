from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "artifact-provenance-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    forbidden = {
        "semantic_sha256", "manifest_sha256", "output_sha256", "outputs",
        "run_id", "timestamp", "timestamp_utc", "duration_seconds",
        "processing_seconds", "git_status", "cpu_count", "jobs",
    }

    def check(value: Any, trail: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key)
                if name in forbidden:
                    raise ValueError(
                        f"Semantic payload contains operational or self-derived field: {'.'.join((*trail, name))}"
                    )
                check(item, (*trail, name))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check(item, (*trail, str(index)))

    check(payload)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def content_run_id(prefix: str, payload: Mapping[str, Any]) -> tuple[str, str]:
    fingerprint = semantic_fingerprint(payload)
    return f"{prefix}-{fingerprint[:32]}", fingerprint


def artifact_inventory(base: Path, names: list[str], relative_prefix: str) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": f"{relative_prefix.rstrip('/')}/{name}",
            "sha256": sha256_file(base / name),
        }
        for name in names
    }


def validate_artifact_inventory(root: Path, artifacts: Mapping[str, Mapping[str, str]]) -> None:
    for name, spec in artifacts.items():
        if set(spec) != {"path", "sha256"}:
            raise ValueError(f"Invalid artifact specification: {name}")
        path = root / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")


def predecessor_reference(root: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = manifest.get("semantic_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"Missing semantic fingerprint: {manifest_path}")
    return {
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "semantic_sha256": fingerprint,
    }


def validate_predecessor(root: Path, reference: Mapping[str, str]) -> dict[str, Any]:
    if set(reference) != {"manifest_path", "manifest_sha256", "semantic_sha256"}:
        raise ValueError("Invalid predecessor reference")
    path = root / reference["manifest_path"]
    if sha256_file(path) != reference["manifest_sha256"]:
        raise ValueError("Predecessor manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("semantic_sha256") != reference["semantic_sha256"]:
        raise ValueError("Predecessor semantic fingerprint mismatch")
    artifacts = manifest.get("artifacts") or manifest.get("outputs")
    if isinstance(artifacts, dict):
        validate_artifact_inventory(root, artifacts)
    return manifest
