from __future__ import annotations

import argparse
import copy
from pathlib import Path

from scripts.run_from_scratch_gate import run_pipeline
from src.execution_contract import (
    CONTRACT_VERSION,
    ENV_CONTRACT,
    FIXTURE_SENTINEL,
    PRODUCTION_METHOD,
    load_execution_contract,
    write_contract,
)
from src.utils import REPOSITORY_ROOT


PRODUCTION_RUN_DIR = REPOSITORY_ROOT / "production_run"
CONTRACT_PATH = PRODUCTION_RUN_DIR / "execution_contract.json"
LEDGER_PATH = PRODUCTION_RUN_DIR / "ledger.json"


def positive_jobs(value: str) -> int:
    jobs = int(value)
    if jobs < 1:
        raise argparse.ArgumentTypeError("job count must be at least 1")
    return jobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the complete production pipeline from absent outputs")
    parser.add_argument("--text-jobs", type=positive_jobs, default=1)
    parser.add_argument("--url-jobs", type=positive_jobs, default=1)
    return parser.parse_args(argv)


def production_contract_payload(root: Path = REPOSITORY_ROOT) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "execution_scope": "production",
        "repository_root": str(root.resolve()),
        "method": copy.deepcopy(PRODUCTION_METHOD),
        "expected": {},
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = REPOSITORY_ROOT / "outputs"
    if outputs.exists():
        raise ValueError("Production requires a completely non-existent outputs directory")
    if (REPOSITORY_ROOT / FIXTURE_SENTINEL).exists():
        raise ValueError("Production rejects a fixture-root sentinel")

    PRODUCTION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_contract(CONTRACT_PATH, production_contract_payload())
    contract = load_execution_contract(
        root=REPOSITORY_ROOT,
        environ={ENV_CONTRACT: str(CONTRACT_PATH.resolve())},
    )
    run_pipeline(
        contract=contract,
        expected_scope="production",
        ledger_path=LEDGER_PATH,
        start_state="absent",
        text_jobs=args.text_jobs,
        url_jobs=args.url_jobs,
    )


if __name__ == "__main__":
    main()
