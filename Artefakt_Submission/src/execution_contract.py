from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "execution-scope-v1"
ENV_CONTRACT = "THESIS_EXECUTION_CONTRACT"
SCOPES = {"production", "integration_fixture"}
FIXTURE_SENTINEL = ".integration-fixture-root"

PRODUCTION_METHOD = {
    "text_candidate_count": 120,
    "text_fold_fit_count": 600,
    "url_candidate_count": 60,
    "url_fold_fit_count": 300,
    "fold_ids": [0, 1, 2, 3, 4],
    "text_views": ["subject", "body", "subject_body"],
    "tfidf_configs": ["TFIDF_A", "TFIDF_B"],
    "inner_max_num_threads": 1,
    "random_forest_n_jobs": 1,
}


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def contract_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("contract_hash", None)
    return hashlib.sha256(_canonical(material)).hexdigest()


@dataclass(frozen=True)
class ExecutionContract:
    path: Path
    root: Path
    scope: str
    contract_hash: str
    payload: dict[str, Any]

    @property
    def is_fixture(self) -> bool:
        return self.scope == "integration_fixture"

    @property
    def is_production(self) -> bool:
        return self.scope == "production"

    def expected(self, key: str, production_value: Any) -> Any:
        if self.scope == "production":
            actual = self.payload.get("expected", {}).get(key, production_value)
            if actual != production_value:
                raise ValueError(f"Production contract changes protected value {key!r}: {actual!r} != {production_value!r}")
            return production_value
        expected = self.payload.get("expected", {})
        if key not in expected:
            raise ValueError(f"Fixture contract lacks expected value {key!r}")
        return expected[key]

    def stamp(self) -> dict[str, Any]:
        return {
            "execution_scope": self.scope,
            "execution_contract_version": CONTRACT_VERSION,
            "execution_contract_sha256": self.contract_hash,
        }

    def validate_stamp(self, value: Mapping[str, Any], *, source: str) -> None:
        expected = self.stamp()
        actual = {key: value.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"Execution-scope chain mismatch in {source}: {actual!r} != {expected!r}")

    def stamp_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        overlap = set(result) & set(self.stamp())
        if overlap:
            raise ValueError(f"Payload already contains execution contract fields: {sorted(overlap)}")
        result.update(self.stamp())
        return result


def _validate_method(payload: Mapping[str, Any], scope: str) -> None:
    method = payload.get("method")
    if not isinstance(method, dict):
        raise ValueError("Execution contract must contain a method object")
    if scope == "production" and method != PRODUCTION_METHOD:
        raise ValueError("Production method contract differs from the protected full scientific method")
    for key in ("text_candidate_count", "text_fold_fit_count", "url_candidate_count", "url_fold_fit_count"):
        if method.get(key) != PRODUCTION_METHOD[key]:
            raise ValueError(f"Execution contract reduces or changes protected search-space value {key!r}")
    if method.get("fold_ids") != PRODUCTION_METHOD["fold_ids"]:
        raise ValueError("Execution contract must retain the complete five-fold contract")
    if method.get("text_views") != PRODUCTION_METHOD["text_views"]:
        raise ValueError("Execution contract must retain every text view")
    if method.get("tfidf_configs") != PRODUCTION_METHOD["tfidf_configs"]:
        raise ValueError("Execution contract must retain TFIDF_A and TFIDF_B")


def load_execution_contract(*, root: Path, environ: Mapping[str, str] | None = None) -> ExecutionContract:
    env = os.environ if environ is None else environ
    raw_path = env.get(ENV_CONTRACT, "").strip()
    if not raw_path:
        raise ValueError(f"{ENV_CONTRACT} must explicitly name an execution contract")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"Execution contract does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Unknown execution contract version")
    scope = payload.get("execution_scope")
    if scope not in SCOPES:
        raise ValueError(f"Unknown execution scope: {scope!r}")
    expected_hash = contract_hash(payload)
    if payload.get("contract_hash") != expected_hash:
        raise ValueError("Execution contract hash mismatch")
    resolved_root = root.resolve()
    if Path(payload.get("repository_root", "")).resolve() != resolved_root:
        raise ValueError("Execution contract repository root mismatch")
    _validate_method(payload, scope)
    sentinel = resolved_root / FIXTURE_SENTINEL
    if scope == "integration_fixture":
        if not sentinel.is_file():
            raise ValueError("Fixture scope requires a fixture-root sentinel")
        if payload.get("fixture_sentinel_sha256") != hashlib.sha256(sentinel.read_bytes()).hexdigest():
            raise ValueError("Fixture sentinel hash mismatch")
        if "fixture" not in resolved_root.as_posix().lower():
            raise ValueError("Fixture execution is only allowed beneath an explicit temporary fixture root")
    else:
        if sentinel.exists():
            raise ValueError("Production scope rejects a fixture-root sentinel")
        encoded = _canonical(payload).lower()
        if b"integration_fixture" in encoded or b"fixture_raw" in encoded:
            raise ValueError("Production contract contains fixture markers")
    return ExecutionContract(path, resolved_root, scope, expected_hash, payload)


def write_contract(path: Path, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value["contract_hash"] = contract_hash(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return value["contract_hash"]


def current_contract(root: Path) -> ExecutionContract:
    return load_execution_contract(root=root)


def expected_value(root: Path, key: str, production_value: Any) -> Any:
    return current_contract(root).expected(key, production_value)
