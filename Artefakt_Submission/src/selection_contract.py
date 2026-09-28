from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CONTRACT_VERSION = "model-selection-semantic-v1"
FINGERPRINT_ALGORITHM = "sha256-canonical-json"
VOLATILE_FOLD_FIELDS = {"fit_seconds", "predict_seconds"}
OPERATIONAL_PARAMETER_FIELDS = {"n_jobs"}


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items())
            if key not in OPERATIONAL_PARAMETER_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, type) and value.__module__.startswith("numpy"):
        return value.__name__
    if hasattr(value, "item"):
        return _semantic_value(value.item())
    return value


def _rows(rows: Iterable[dict[str, Any]], *, key: tuple[str, ...], excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    normalized = [
        _semantic_value({name: value for name, value in row.items() if name not in excluded})
        for row in rows
    ]
    try:
        normalized.sort(key=lambda row: tuple(str(row[name]) for name in key))
    except KeyError as exc:
        raise ValueError(f"Semantic selection row lacks stable key field: {exc.args[0]}") from exc
    stable_keys = [tuple(str(row[name]) for name in key) for row in normalized]
    if len(stable_keys) != len(set(stable_keys)):
        raise ValueError(f"Duplicate semantic selection key: {key}")
    return normalized


def build_semantic_payload(
    *,
    selection_type: str,
    benchmark_run_id: str,
    input_identity: dict[str, Any],
    method_identity: dict[str, Any],
    fold_rows: Iterable[dict[str, Any]],
    summaries: Iterable[dict[str, Any]],
    winner: dict[str, Any],
    winner_oof: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    if selection_type not in {"text", "url"}:
        raise ValueError("Unsupported selection contract type")
    oof_key = ("record_id",) if selection_type == "text" else ("canonical_url_id",)
    return {
        "contract_version": CONTRACT_VERSION,
        "selection_type": selection_type,
        "benchmark_run_id": benchmark_run_id,
        "input_identity": _semantic_value(input_identity),
        "method_identity": _semantic_value(method_identity),
        "fold_results": _rows(
            fold_rows,
            key=("config_id", "fold"),
            excluded=VOLATILE_FOLD_FIELDS,
        ),
        "config_summary": _rows(summaries, key=("config_id",)),
        "winner": _semantic_value(winner),
        "winner_oof_predictions": _rows(winner_oof, key=oof_key),
    }


def semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _semantic_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_selection_manifest(
    *,
    payload: dict[str, Any],
    input_paths: dict[str, str],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "selection_type": payload["selection_type"],
        "benchmark_run_id": payload["benchmark_run_id"],
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "semantic_sha256": semantic_sha256(payload),
        "input_identity": payload["input_identity"],
        "input_paths": dict(sorted(input_paths.items())),
        "output_paths": dict(sorted(output_paths.items())),
        "output_schema_versions": {name: 1 for name in sorted(output_paths)},
    }


def validate_selection_manifest(manifest: dict[str, Any], payload: dict[str, Any]) -> str:
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Unexpected selection contract version")
    if manifest.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        raise ValueError("Unexpected selection fingerprint algorithm")
    if manifest.get("selection_type") != payload.get("selection_type"):
        raise ValueError("Selection contract type mismatch")
    if manifest.get("benchmark_run_id") != payload.get("benchmark_run_id"):
        raise ValueError("Selection benchmark run mismatch")
    if manifest.get("input_identity") != payload.get("input_identity"):
        raise ValueError("Selection input identity mismatch")
    output_paths = manifest.get("output_paths")
    if not isinstance(output_paths, dict) or set(output_paths) != {
        "fold_results", "config_summary", "winner", "winner_oof"
    }:
        raise ValueError("Selection output path inventory mismatch")
    if manifest.get("output_schema_versions") != {name: 1 for name in sorted(output_paths)}:
        raise ValueError("Selection output schema version mismatch")
    actual = semantic_sha256(payload)
    if manifest.get("semantic_sha256") != actual:
        raise ValueError("Selection semantic fingerprint mismatch")
    return actual


def read_selection_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Selection manifest must be a JSON object")
    return manifest
