from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.active_pipeline_contract import ARTIFACTS, as_jsonable, validate_topology
from src.execution_contract import ENV_CONTRACT, ExecutionContract, load_execution_contract
from src.utils import REPOSITORY_ROOT


HISTORICAL_DIRS = (
    "outputs/models/" + "fusion",
    "outputs/analysis/" + "fusion_failures",
    "outputs/evaluation/" + "historical",
)


def commands(text_jobs: int, url_jobs: int) -> list[tuple[str, str, list[str]]]:
    py = sys.executable
    return [
        ("acquisition", "producer", [py, "-B", "-m", "scripts.verify_acquired_datasets"]),
        ("integrated_datasets", "producer", [py, "-B", "-m", "src.load_data"]),
        ("preprocessed_email_dataset", "producer", [py, "-B", "-m", "src.preprocess"]),
        ("integrated_datasets", "validator", [py, "-B", "-m", "scripts.validate_dataset_integration"]),
        ("email_splits", "producer", [py, "-B", "-m", "src.split_data"]),
        ("email_splits", "validator", [py, "-B", "-m", "scripts.validate_data_splits"]),
        ("reserved_source_audit", "producer", [py, "-B", "-m", "src.reserved_source_audit"]),
        ("reserved_source_audit", "validator", [py, "-B", "-m", "scripts.validate_leakage_audit"]),
        ("text_inputs", "producer", [py, "-B", "-m", "src.prepare_text_inputs"]),
        ("text_inputs", "validator", [py, "-B", "-m", "scripts.validate_text_inputs"]),
        ("text_selection", "producer", [py, "-B", "-m", "src.text_model_selection", "--jobs", str(text_jobs)]),
        ("text_selection", "validator", [py, "-B", "-m", "scripts.validate_text_model_selection"]),
        ("text_model", "producer", [py, "-B", "-m", "src.train_text_model"]),
        ("text_model", "validator", [py, "-B", "-m", "scripts.validate_text_model"]),
        ("text_calibration", "producer", [py, "-B", "-m", "src.calibrate_text_model"]),
        ("text_calibration", "validator", [py, "-B", "-m", "scripts.validate_text_calibration"]),
        ("phishload_audit", "producer", [py, "-B", "-m", "src.phishload_dataset", "audit"]),
        ("phishload_audit", "validator", [py, "-B", "-m", "scripts.validate_url_dataset_audit"]),
        ("url_dataset", "producer", [py, "-B", "-m", "src.phishload_dataset", "prepare"]),
        ("url_dataset", "validator", [py, "-B", "-m", "scripts.validate_url_dataset"]),
        ("url_selection", "producer", [py, "-B", "-m", "src.url_model_selection", "--jobs", str(url_jobs)]),
        ("url_selection", "validator", [py, "-B", "-m", "scripts.validate_url_model_selection"]),
        ("url_model", "producer", [py, "-B", "-m", "src.train_url_model"]),
        ("url_model", "validator", [py, "-B", "-m", "scripts.validate_url_model"]),
        ("final_artifact", "producer", [py, "-B", "-m", "src.build_final_artifact"]),
        ("final_artifact", "validator", [py, "-B", "-m", "scripts.validate_final_artifact"]),
        ("internal_holdout", "producer", [py, "-B", "-m", "src.internal_holdout_evaluation", "--execute-final-holdout"]),
        ("internal_holdout", "validator", [py, "-B", "-m", "scripts.validate_internal_holdout"]),
        ("temporal_evaluation", "producer", [py, "-B", "-m", "src.temporal_evaluation", "--execute-temporal-evaluation"]),
        ("temporal_evaluation", "validator", [py, "-B", "-m", "scripts.validate_temporal_evaluation"]),
        ("ephishllm_evaluation", "producer", [py, "-B", "-m", "src.ephishllm_evaluation", "--execute-external-evaluation"]),
        ("ephishllm_evaluation", "validator", [py, "-B", "-m", "scripts.validate_ephishllm_evaluation"]),
        ("phishingpot_evaluation", "producer", [py, "-B", "-m", "src.phishingpot_evaluation", "--execute-external-evaluation"]),
        ("phishingpot_evaluation", "validator", [py, "-B", "-m", "scripts.validate_phishingpot_evaluation"]),
        ("dideco_evaluation", "producer", [py, "-B", "-m", "src.dideco_intent_evaluation", "--execute-rq2-evaluation"]),
        ("dideco_evaluation", "validator", [py, "-B", "-m", "scripts.validate_dideco_intent_evaluation"]),
        ("explanation_evaluation", "producer", [py, "-B", "-m", "src.explanation_evaluation", "--execute-rq3-evaluation"]),
        ("explanation_evaluation", "validator", [py, "-B", "-m", "scripts.validate_explanation_evaluation"]),
        ("technical_evaluation", "producer", [py, "-B", "-m", "src.technical_evaluation", "--execute-technical-evaluation"]),
        ("technical_evaluation", "validator", [py, "-B", "-m", "scripts.validate_technical_evaluation"]),
        ("public_api", "smoke", [py, "-B", "-m", "src.unified_artifact", "--subject", "Project update",
                                  "--body", "The meeting remains scheduled for tomorrow."]),
        ("public_api", "smoke", [py, "-B", "-m", "src.unified_artifact", "--subject", "Account alert",
                                  "--body", "Review https://example.com/login immediately."]),
        ("public_api", "smoke", [py, "-B", "-m", "src.unified_artifact", "--subject", "Reference",
                                  "--body", "Review example.com for details."]),
    ]


def output_files() -> set[str]:
    output = REPOSITORY_ROOT / "outputs"
    if not output.exists():
        return set()
    return {path.relative_to(REPOSITORY_ROOT).as_posix() for path in output.rglob("*") if path.is_file()}


def registered(path: str) -> bool:
    if path.startswith("outputs/.execution_scope/") and path.endswith(".json"):
        return Path(path).stem in {item.artifact_id for item in ARTIFACTS}
    return any(fnmatch.fnmatchcase(path, pattern) for artifact in ARTIFACTS for pattern in artifact.output_paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(artifact_id: str) -> Path:
    return REPOSITORY_ROOT / "outputs/.execution_scope" / f"{artifact_id}.json"


def validate_predecessor_sidecars(artifact_id: str, contract: ExecutionContract) -> None:
    item = next(entry for entry in ARTIFACTS if entry.artifact_id == artifact_id)
    by_id = {entry.artifact_id: entry for entry in ARTIFACTS}
    for predecessor in item.predecessors:
        if predecessor not in by_id:
            continue
        path = sidecar_path(predecessor)
        if not path.is_file():
            raise ValueError(f"Missing execution-scope sidecar for predecessor {predecessor!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract.validate_stamp(payload, source=str(path))
        inventory = payload.get("artifact_sha256")
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError(f"Empty artifact inventory in {path}")
        for relative, expected in inventory.items():
            target = REPOSITORY_ROOT / relative
            if not target.is_file() or sha256(target) != expected:
                raise ValueError(f"Predecessor artifact hash mismatch: {relative}")


def validate_own_sidecar(artifact_id: str, contract: ExecutionContract) -> None:
    path = sidecar_path(artifact_id)
    if not path.is_file():
        raise ValueError(f"Missing execution-scope sidecar for artifact {artifact_id!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    contract.validate_stamp(payload, source=str(path))
    for relative, expected in payload.get("artifact_sha256", {}).items():
        target = REPOSITORY_ROOT / relative
        if not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Artifact hash mismatch before validation: {relative}")


def write_sidecar(artifact_id: str, contract: ExecutionContract, created_or_changed: set[str]) -> None:
    inventory = {
        relative: sha256(REPOSITORY_ROOT / relative)
        for relative in sorted(created_or_changed)
        if not relative.startswith("outputs/.execution_scope/") and (REPOSITORY_ROOT / relative).is_file()
    }
    if not inventory:
        raise ValueError(f"Producer {artifact_id!r} did not create or change a registered output")
    path = sidecar_path(artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.stamp_payload({
        "artifact_id": artifact_id,
        "producer": next(item.producer for item in ARTIFACTS if item.artifact_id == artifact_id),
        "predecessors": list(next(item.predecessors for item in ARTIFACTS if item.artifact_id == artifact_id)),
        "artifact_sha256": inventory,
    })
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    contract: ExecutionContract,
    expected_scope: str,
    ledger_path: Path,
    start_state: str,
    text_jobs: int,
    url_jobs: int,
) -> None:
    if contract.scope != expected_scope:
        raise ValueError(f"Pipeline requires {expected_scope} scope, received {contract.scope}")
    if text_jobs < 1 or url_jobs < 1:
        raise ValueError("Pipeline job counts must be at least 1")
    if start_state not in {"absent", "empty"}:
        raise ValueError(f"Unsupported pipeline start state: {start_state}")
    outputs = REPOSITORY_ROOT / "outputs"
    if start_state == "absent" and outputs.exists():
        raise ValueError("Absent-start pipeline requires a completely non-existent outputs directory")
    if start_state == "empty" and (not outputs.is_dir() or any(outputs.iterdir())):
        raise ValueError("Empty-start pipeline requires an existing, empty outputs directory")
    items = validate_topology(REPOSITORY_ROOT)
    ledger: dict[str, Any] = {"started_utc": datetime.now(timezone.utc).isoformat(),
                              "start_state": start_state, "execution_scope": contract.scope,
                              "execution_contract_sha256": contract.contract_hash,
                              "topology": as_jsonable(items), "commands": []}
    before_hashes: dict[str, str] = {}
    child_env = dict(os.environ)
    child_env[ENV_CONTRACT] = str(contract.path)
    for number, (artifact, role, command) in enumerate(commands(text_jobs, url_jobs), 1):
        if artifact != "public_api":
            validate_predecessor_sidecars(artifact, contract)
            if role == "validator":
                validate_own_sidecar(artifact, contract)
        else:
            validate_predecessor_sidecars("technical_evaluation", contract)
        started = datetime.now(timezone.utc)
        process = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, env=child_env)
        ended = datetime.now(timezone.utc)
        after = output_files()
        after_hashes = {relative: sha256(REPOSITORY_ROOT / relative) for relative in after}
        changed = {relative for relative, digest in after_hashes.items() if before_hashes.get(relative) != digest}
        created = sorted(set(after_hashes) - set(before_hashes))
        if process.returncode == 0 and role == "producer":
            write_sidecar(artifact, contract, changed)
            after = output_files()
            after_hashes = {relative: sha256(REPOSITORY_ROOT / relative) for relative in after}
        entry = {"number": number, "artifact": artifact, "role": role, "command": command,
                 "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
                 "duration_seconds": (ended - started).total_seconds(), "exit_code": process.returncode,
                 "stdout": process.stdout, "stderr": process.stderr, "created_outputs": created}
        ledger["commands"].append(entry)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(process.stdout, end="", flush=True)
        print(process.stderr, end="", file=sys.stderr, flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
        before_hashes = after_hashes
    for relative in HISTORICAL_DIRS:
        if (REPOSITORY_ROOT / relative).exists():
            raise ValueError(f"Historical output was created: {relative}")
    unregistered = sorted(path for path in output_files() if not registered(path))
    if unregistered:
        raise ValueError(f"Outputs without registered active producer: {unregistered}")
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["status"] = "passed"
    ledger["generated_output_count"] = len(output_files())
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "commands": len(ledger["commands"]),
                      "generated_output_count": ledger["generated_output_count"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the active pipeline from a genuinely absent outputs directory")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--start-state", choices=("absent", "empty"), required=True)
    parser.add_argument("--text-jobs", type=int, default=1)
    parser.add_argument("--url-jobs", type=int, default=1)
    args = parser.parse_args()
    contract = load_execution_contract(root=REPOSITORY_ROOT,
                                       environ={**os.environ, ENV_CONTRACT: str(args.contract.resolve())})
    run_pipeline(
        contract=contract,
        expected_scope="integration_fixture",
        ledger_path=args.ledger,
        start_state=args.start_state,
        text_jobs=args.text_jobs,
        url_jobs=args.url_jobs,
    )


if __name__ == "__main__":
    main()
