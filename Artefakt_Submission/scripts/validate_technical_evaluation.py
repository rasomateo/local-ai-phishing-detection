from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.technical_evaluation import (
    FIXTURES, IMPLEMENTATION_FILES, OUTPUT_NAMES, PROTECTED_SHA256, evaluation_fixtures,
)
from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes
from src.utils import REPOSITORY_ROOT

DEFAULT_OUT = REPOSITORY_ROOT / "outputs/evaluation/technical"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered), "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0], "max_ms": ordered[-1], "p95_method": "nearest_rank",
    }


def validate(out: Path = DEFAULT_OUT) -> dict[str, Any]:
    fixtures = evaluation_fixtures()
    require(out.is_dir(), f"Missing Step-34 output directory: {out}")
    for name in OUTPUT_NAMES + ["checksums.sha256"]:
        require((out / name).is_file(), f"Missing Step-34 output: {name}")
    checksum_lines = (out / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    require(checksum_lines == [f"{digest(out / name)}  {name}" for name in OUTPUT_NAMES],
            "Step-34 checksum inventory mismatch")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["selection_provenance"] == current_selection_provenance(REPOSITORY_ROOT),
            "Selection provenance chain mismatch")
    require(manifest["artifact_type"] == "step34_cli_only_technical_evaluation", "Artifact type")
    require(manifest["schema_version"] == 2 and manifest["method_specification"]["version"] == 2,
            "Technical-evaluation schema version")
    require(manifest["method_specification"]["actual_cli_command"] ==
            "python -B -m src.unified_artifact", "CLI command identity")
    require(manifest["method_specification"]["fixture_ids"] ==
            [item["case_id"] for item in fixtures], "Fixture order")
    protected = resolve_current_protected_hashes(REPOSITORY_ROOT, PROTECTED_SHA256)
    for name, expected in protected.items():
        require(digest(REPOSITORY_ROOT / name) == expected, f"Protected input changed: {name}")
        require(manifest["protected_input_sha256"][name] == expected, f"Manifest protected hash: {name}")
    require(manifest["protected_input_sha256"] == protected, "Protected inventory mismatch")
    require(set(manifest["implementation_sha256"]) == set(IMPLEMENTATION_FILES), "Implementation inventory")
    for name, expected in manifest["implementation_sha256"].items():
        require(digest(REPOSITORY_ROOT / name) == expected, f"Implementation changed: {name}")
    require(manifest["output_sha256"] ==
            {name: digest(out / name) for name in OUTPUT_NAMES[:-1]}, "Output hash inventory")

    technical = json.loads((out / "technical_metrics.json").read_text(encoding="utf-8"))
    groups: dict[str, list[float]] = {}
    with (out / "inference_benchmark.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["phase", "scenario_id", "repetition", "elapsed_ms"], "CSV schema")
        for row in reader:
            value = float(row["elapsed_ms"])
            require(math.isfinite(value) and value >= 0.0, "Invalid timing")
            groups.setdefault(row["scenario_id"], []).append(value)
    require({name: stats(values) for name, values in sorted(groups.items())} ==
            technical["timing_aggregates"], "Timing aggregates")
    require(len(groups["fresh_cli_end_to_end"]) == 30, "Fresh CLI repetition count")
    require(all(len(values) == 500 for name, values in groups.items()
                if name.startswith(("core_", "public_", "cli_parse"))), "Inference repetitions")

    sizes = technical["bundle_sizes"]
    for section in ("model_components", "decision_contract_metadata", "cli_runtime_source_files"):
        for row in sizes[section]:
            path = REPOSITORY_ROOT / row["path"]
            require(row["bytes"] == path.stat().st_size, f"Size mismatch: {row['path']}")
            require(row["sha256"] == digest(path), f"Size hash mismatch: {row['path']}")
    model_bytes = sum(row["bytes"] for row in sizes["model_components"])
    productive = model_bytes + sum(row["bytes"] for row in sizes["decision_contract_metadata"])
    delivery = productive + sum(row["bytes"] for row in sizes["cli_runtime_source_files"])
    require(sizes["productive_model_bundle_total"]["bytes"] == productive, "Productive bundle total")
    require(sizes["complete_cli_delivery_total"]["bytes"] == delivery, "CLI delivery total")

    end_to_end = json.loads((out / "end_to_end_validation.json").read_text(encoding="utf-8"))
    require([row["case_id"] for row in end_to_end["cases"]] ==
            [row["case_id"] for row in fixtures], "Persisted fixture order")
    require(all(row["status"] == "passed" for row in end_to_end["cases"]), "Fixture failure")
    fresh = end_to_end["fresh_process_cli"]
    require(fresh["command"] == "python -B -m src.unified_artifact", "Fresh CLI command")
    require(fresh["returncode"] == 0 and fresh["stdout_json_only"], "Fresh CLI execution")
    require(all(fresh["checks"].values()), "Fresh CLI checks")
    modules = set(fresh["imported_project_modules"])
    require("src.final_artifact" in modules and "src.explain" in modules, "Active module imports")
    require("src.fusion_artifact" not in modules, "Historical fusion imported")
    require("src.local_unified_interface" not in modules, "Web interface imported")
    require(all(end_to_end["url_score_isolation"]["checks"].values()), "URL isolation")
    require(end_to_end["offline_cli_validation"]["status"] == "passed", "Offline CLI")

    resources = json.loads((out / "resource_usage.json").read_text(encoding="utf-8"))
    memory = resources["memory"]
    require(memory["trial_count"] == 10 and len(memory["trials"]) == 10, "Memory trials")
    for trial in memory["trials"]:
        require(trial["peak_vmhwm_bytes"] >= max(
            trial[name] for name in ("baseline_rss_bytes", "after_project_import_rss_bytes",
                                     "after_model_load_rss_bytes", "after_warmup_rss_bytes",
                                     "after_full_inference_rss_bytes")), "Peak RSS")

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    require(report["step"] == 34 and "historical_step32_overwritten" not in report, "Report scope")

    process = subprocess.run(
        [sys.executable, "-B", "-m", "src.unified_artifact", "--subject", FIXTURES[3]["subject"],
         "--body", FIXTURES[3]["body"], "--explanation-top-k", "3"],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True,
    )
    require(process.returncode == 0 and not process.stderr, "Independent fresh CLI invocation")
    result = json.loads(process.stdout)
    require(result["architecture"] == {
        "name": "step25c_text_led_email_classifier_with_auxiliary_url_evidence",
        "schema_version": 2, "fusion_applied": False,
    }, "Independent active architecture")
    require(result["url_model_evidence"]["affects_email_classification"] is False, "URL auxiliary contract")
    require(result["text_explanation"]["reconstruction_tolerance"]["passed"] is True, "Explanation")
    return {"status": "validated", "evaluation_run_id": manifest["evaluation_run_id"],
            "case_count": len(end_to_end["cases"]), "benchmark_rows": sum(map(len, groups.values())),
            "memory_trials": len(memory["trials"]), "fresh_cli": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.out), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
