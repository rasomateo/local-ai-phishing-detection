from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Iterator

from src.artifact_contract import current_selection_provenance, resolve_current_protected_hashes, semantic_predecessor_identity
from src.execution_contract import current_contract
from src.utils import REPOSITORY_ROOT

OUT = REPOSITORY_ROOT / "outputs/evaluation/technical"
WARMUP_REPETITIONS = 25
INFERENCE_REPETITIONS = 500
LOAD_REPETITIONS = 30
MEMORY_REPETITIONS = 10
OUTPUT_NAMES = [
    "technical_metrics.json", "inference_benchmark.csv", "resource_usage.json",
    "end_to_end_validation.json", "report.json", "manifest.json",
]
PROTECTED_SHA256 = {
    "outputs/models/text/classifier/text_vectorizer.joblib": "813933dcf2fe63934db12dea7c90c83cf7de4b1c9c3902182d1af6da23b664d0",
    "outputs/models/text/classifier/text_model.joblib": "1cb802457451c69f6a02921c97353e0889bc3d4bbdb0b2f39aae7f5d29e204b9",
    "outputs/models/text/calibration/text_calibrator.joblib": "5129cee2c6e52310738f91ac004cde5e5f09cfca065c16de8a2fe4847de27e59",
    "outputs/models/url/url_vectorizer.joblib": "182bfb0af30f56cd68657dbcdbd03571810882aeb75bcff5aad25af782478af3",
    "outputs/models/url/url_model.joblib": "f11e2a715ba993a1ac99c867964f79b289a5a982d92ae3cd8601585032c16be7",
    "outputs/models/url/url_calibrator.joblib": "110e48c05ec2c36ab7be35723c4916ca618062610c7f1d9c01ae84eac3040021",
    "outputs/models/final_artifact/architecture.json": "f730617eab8fd9295a740fd8f2bb280f66687368583d1c6fb3793d378bf68924",
    "outputs/models/final_artifact/manifest.json": "df012321b671c8d3be0edfddaa4ca7fa836ada6226c717ba15eacc58fb72b21f",
    "outputs/models/final_artifact/checksums.sha256": "bb4797a50a82513149ab4c7b83505c164b923b0c0a866f7063bae0f9532ab05b",
    "outputs/evaluation/internal_holdout/checksums.sha256": "10ae68aed6d1eba52ab9ef006deb2298a5d4dc0578d9e15340e0e63d39f285cb",
    "outputs/evaluation/temporal/checksums.sha256": "4a2e196d162ad06bc26e2a61730d197e1961972c03d43fdbe597d9c1194bcdec",
    "outputs/evaluation/ephishllm/checksums.sha256": "29b1e6fa17cdc6f859e7723dc437375bd0c036b04e65bd79312799c6193669bc",
    "outputs/evaluation/phishingpot/checksums.sha256": "db21a28b32b3c7f5cc82b6ce9e9d8af3da789d228e977a38a94905e444ff1db6",
    "outputs/evaluation/dideco_intent/checksums.sha256": "dec06421882e73046364289ee2f425678c8b6e98b9fa9d44ef7bb49f80f0287b",
    "outputs/evaluation/explanations/checksums.sha256": "918d101ee160ad6f5d791604188529ab89fe4761d96474cce5c85be8229883e6",
}
MODEL_FILES = list(PROTECTED_SHA256)[:6]
POLICY_FILES = [
    "outputs/models/final_artifact/architecture.json",
    "outputs/models/final_artifact/manifest.json",
]
RUNTIME_SOURCE_FILES = [
    "src/unified_artifact.py", "src/final_artifact.py", "src/explain.py",
    "src/preprocess.py", "src/load_data.py", "src/utils.py",
    "src/phishload_dataset.py", "src/url_model_bundle.py",
]
IMPLEMENTATION_FILES = [
    "src/technical_evaluation.py",
    "scripts/validate_technical_evaluation.py",
]
EXPECTED_ACTIVE_MODULES = {
    "src.utils", "src.load_data", "src.preprocess",
    "src.phishload_dataset", "src.url_model_bundle",
    "src.final_artifact", "src.explain",
}
FORBIDDEN_ACTIVE_MODULES = {"src.fusion_artifact", "src.local_unified_interface"}

LONG_BODY = (
    "The team reviewed the schedule, minutes, budget, and documentation for tomorrow. " * 80
    + "Review https://example.com/alpha, https://second.com/beta, and https://third.org/gamma."
)
FIXTURES = [
    {"case_id": "subject_body_phishing", "categories": ["subject_and_body", "no_url", "phishing"],
     "subject": "Urgent account alert", "body": "Verify your account immediately", "expected_label": "phishing"},
    {"case_id": "subject_only_legitimate", "categories": ["subject_only", "no_url", "legitimate"],
     "subject": "Quarterly team meeting agenda", "body": "", "expected_label": "legitimate"},
    {"case_id": "body_only_legitimate", "categories": ["body_only", "no_url", "legitimate"],
     "subject": "", "body": "Please review the attached minutes for tomorrow", "expected_label": "legitimate"},
    {"case_id": "one_valid_url", "categories": ["subject_and_body", "one_valid_url"],
     "subject": "Project reference", "body": "Review https://example.com/case for the meeting"},
    {"case_id": "multiple_distinct_urls", "categories": ["multiple_distinct_urls"],
     "subject": "Reference links", "body": "Review https://example.com/a and http://second.com/path?x=1"},
    {"case_id": "duplicate_normalized_url", "categories": ["duplicate_url", "url_normalization"],
     "subject": "One HTTPS://Example.COM/Case?Token=AbC",
     "body": "Again https://example.com/Case?Token=AbC."},
    {"case_id": "unsupported_url_like", "categories": ["unsupported_url", "no_scorable_url"],
     "subject": "Contact details", "body": "Write mailto:test@example.com or see https://"},
    {"case_id": "long_multiple_urls", "categories": ["long_text", "multiple_distinct_urls"],
     "subject": "Monthly project update", "body": LONG_BODY},
    {"case_id": "special_html_unicode", "categories": ["special_characters", "unicode", "no_url"],
     "subject": "<script>alert(1)</script> & Grüße",
     "body": "Meeting notes: <b>review</b> — café & tea"},
]


def evaluation_fixtures() -> list[dict[str, Any]]:
    contract = current_contract(REPOSITORY_ROOT)
    if not contract.is_fixture:
        return [dict(row) for row in FIXTURES]
    probes = [{key: value for key, value in row.items() if key != "expected_label"}
              for row in FIXTURES]
    with (REPOSITORY_ROOT / "outputs/features/text/holdout_inputs.csv").open(
            encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for label in ("legitimate", "phishing"):
        row = next(item for item in rows if item["label"] == label)
        probes.append({
            "case_id": f"fixture_holdout_{label}",
            "categories": ["fixture_holdout", label],
            "subject": row["subject"],
            "body": row["body"],
            "expected_label": label,
        })
    return probes


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def input_sha256(subject: str, body: str) -> str:
    return hashlib.sha256((subject + "\x1f" + body).encode("utf-8")).hexdigest()


def validate_protected() -> dict[str, str]:
    return resolve_current_protected_hashes(REPOSITORY_ROOT, PROTECTED_SHA256)


def aggregate_milliseconds(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot aggregate an empty sample")
    return {
        "count": len(ordered), "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[max(1, math.ceil(0.95 * len(ordered))) - 1],
        "min_ms": ordered[0], "max_ms": ordered[-1], "p95_method": "nearest_rank",
    }


def file_size_record(relative: str) -> dict[str, Any]:
    path = REPOSITORY_ROOT / relative
    size = path.stat().st_size
    return {"path": relative, "bytes": size, "mib": size / (1024 ** 2), "sha256": sha256(path)}


def collect_bundle_sizes() -> dict[str, Any]:
    model = [file_size_record(name) for name in MODEL_FILES]
    policy = [file_size_record(name) for name in POLICY_FILES]
    source = [file_size_record(name) for name in RUNTIME_SOURCE_FILES]
    model_bytes = sum(row["bytes"] for row in model)
    productive_bytes = model_bytes + sum(row["bytes"] for row in policy)
    delivery_bytes = productive_bytes + sum(row["bytes"] for row in source)
    return {
        "unit_definition": "1 MiB = 1,048,576 bytes",
        "model_components": model, "decision_contract_metadata": policy,
        "cli_runtime_source_files": source,
        "model_components_total": {"bytes": model_bytes, "mib": model_bytes / (1024 ** 2)},
        "productive_model_bundle_total": {"bytes": productive_bytes, "mib": productive_bytes / (1024 ** 2)},
        "complete_cli_delivery_total": {"bytes": delivery_bytes, "mib": delivery_bytes / (1024 ** 2)},
        "excluded": ["historical outputs", "training datasets", "evaluation outputs", "Python environment"],
    }


def _proc_status_bytes(field: str) -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"Missing {field} in /proc/self/status")


def _worker_load() -> dict[str, Any]:
    started = time.perf_counter_ns()
    from src.unified_artifact import load_unified_artifact_components
    imported = time.perf_counter_ns()
    components = load_unified_artifact_components()
    loaded = time.perf_counter_ns()
    if not components:
        raise RuntimeError("Components not loaded")
    return {
        "module_import_ms": (imported - started) / 1_000_000,
        "component_load_ms": (loaded - imported) / 1_000_000,
        "module_import_and_component_load_ms": (loaded - started) / 1_000_000,
    }


def _worker_memory() -> dict[str, Any]:
    baseline = _proc_status_bytes("VmRSS")
    from src.unified_artifact import load_unified_artifact_components, predict_unified
    imported = _proc_status_bytes("VmRSS")
    components = load_unified_artifact_components()
    loaded = _proc_status_bytes("VmRSS")
    predict_unified("Project reference", FIXTURES[3]["body"], components)
    warmed = _proc_status_bytes("VmRSS")
    predict_unified("Monthly project update", LONG_BODY, components)
    inferred = _proc_status_bytes("VmRSS")
    peak = _proc_status_bytes("VmHWM")
    return {
        "baseline_rss_bytes": baseline, "after_project_import_rss_bytes": imported,
        "after_model_load_rss_bytes": loaded, "after_warmup_rss_bytes": warmed,
        "after_full_inference_rss_bytes": inferred, "peak_vmhwm_bytes": peak,
        "model_load_rss_delta_bytes": loaded - baseline,
        "warmup_rss_delta_bytes": warmed - baseline,
    }


@contextmanager
def network_guard() -> Iterator[dict[str, int]]:
    original_connect = socket.socket.connect
    original_create = socket.create_connection
    original_dns = socket.getaddrinfo
    state = {"blocked": 0}

    def blocked(*args: Any, **kwargs: Any) -> Any:
        state["blocked"] += 1
        raise OSError("External network blocked by Step-34 process-local guard")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    try:
        yield state
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create
        socket.getaddrinfo = original_dns


def _worker_offline_cli() -> dict[str, Any]:
    with network_guard() as state:
        try:
            socket.create_connection(("192.0.2.1", 443), timeout=0.01)
        except OSError:
            pass
        control = state["blocked"]
        from src.unified_artifact import main
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--subject", "Reference links", "--body", FIXTURES[4]["body"],
                  "--explanation-top-k", "3"])
        result = json.loads(output.getvalue())
        artifact_attempts = state["blocked"] - control
    return {
        "status": "passed" if control == 1 and artifact_attempts == 0 else "failed",
        "guard_scope": "process-local socket connect/create_connection/getaddrinfo interception",
        "control_blocked_attempts": control, "artifact_external_attempts": artifact_attempts,
        "scored_unique_url_count": result["url_model_evidence"]["scored_unique_url_count"],
        "explanation_reconstruction_passed": result["text_explanation"]["reconstruction_tolerance"]["passed"],
    }


def run_worker(name: str) -> dict[str, Any]:
    workers: dict[str, Callable[[], dict[str, Any]]] = {
        "load": _worker_load, "memory": _worker_memory, "offline_cli": _worker_offline_cli,
    }
    if name not in workers:
        raise ValueError(f"Unknown worker: {name}")
    return workers[name]()


def invoke_worker(name: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter_ns()
    process = subprocess.run(
        [sys.executable, "-B", "-m", "src.technical_evaluation", "--worker", name],
        cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(process.stdout), (time.perf_counter_ns() - started) / 1_000_000


def invoke_actual_cli(subject: str, body: str, top_k: int = 10, *, import_trace: bool = False) -> dict[str, Any]:
    command = [sys.executable]
    if import_trace:
        command += ["-X", "importtime"]
    command += ["-B", "-m", "src.unified_artifact", "--subject", subject, "--body", body,
                "--explanation-top-k", str(top_k)]
    started = time.perf_counter_ns()
    process = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000
    parsed = json.loads(process.stdout) if process.returncode == 0 else None
    modules = sorted({line.rsplit("|", 1)[-1].strip() for line in process.stderr.splitlines()
                      if line.startswith("import time:") and line.rsplit("|", 1)[-1].strip().startswith("src.")})
    return {
        "command": "python -B -m src.unified_artifact",
        "returncode": process.returncode, "stdout_json_only": parsed is not None,
        "stderr_empty": not process.stderr if not import_trace else None,
        "wall_ms": wall_ms, "result": parsed, "imported_project_modules": modules,
    }


def validate_fresh_cli() -> dict[str, Any]:
    probe = invoke_actual_cli(FIXTURES[3]["subject"], FIXTURES[3]["body"], 3, import_trace=True)
    result = probe["result"]
    imported = set(probe["imported_project_modules"])
    checks = {
        "successful_exit": probe["returncode"] == 0,
        "json_output": probe["stdout_json_only"],
        "expected_modules_imported": EXPECTED_ACTIVE_MODULES <= imported,
        "inactive_fusion_module_not_imported": "src.fusion_artifact" not in imported,
        "web_interface_not_imported": "src.local_unified_interface" not in imported,
        "architecture_name": result["architecture"]["name"] == "step25c_text_led_email_classifier_with_auxiliary_url_evidence",
        "schema_version": result["architecture"]["schema_version"] == 2,
        "fusion_disabled": result["architecture"]["fusion_applied"] is False,
        "fixed_threshold": result["email_classification"]["decision_threshold"] == 0.5,
        "url_auxiliary": result["url_model_evidence"]["affects_email_classification"] is False,
        "url_evidence_present": result["url_model_evidence"]["scored_unique_url_count"] == 1,
        "explanation_present": result["text_explanation"]["reconstruction_tolerance"]["passed"] is True,
    }
    empty = subprocess.run(
        [sys.executable, "-B", "-m", "src.unified_artifact", "--subject", "", "--body", ""],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True,
    )
    checks["empty_input_fails"] = empty.returncode != 0
    invalid_top_k = subprocess.run(
        [sys.executable, "-B", "-m", "src.unified_artifact", "--subject", "x",
         "--explanation-top-k", "0"], cwd=REPOSITORY_ROOT, capture_output=True, text=True,
    )
    checks["invalid_explanation_limit_fails"] = invalid_top_k.returncode != 0
    if not all(checks.values()):
        raise ValueError(f"Fresh CLI validation failed: {checks}")
    return {**probe, "result": result, "checks": checks, "status": "passed"}


def validate_url_isolation(components: dict[str, Any]) -> dict[str, Any]:
    from src import final_artifact as final
    from src.unified_artifact import predict_unified
    subject, body = FIXTURES[1]["subject"], "Review https://example.com/case"
    baseline = predict_unified(subject, body, components)
    original = final.score_normalized_urls
    forced: dict[str, dict[str, Any]] = {}
    try:
        for name, probability in (("zero", 0.0), ("one", 1.0)):
            final.score_normalized_urls = lambda urls, bundle, p=probability: {
                "decision_score": [(-999.0 if p == 0.0 else 999.0)] * len(urls),
                "probability_phishing": [p] * len(urls),
            }
            forced[name] = predict_unified(subject, body, components)
    finally:
        final.score_normalized_urls = original
    baseline_email = baseline["email_classification"]
    checks = {
        "zero_score_email_bit_identical": forced["zero"]["email_classification"] == baseline_email,
        "one_score_email_bit_identical": forced["one"]["email_classification"] == baseline_email,
        "zero_recorded": forced["zero"]["url_model_evidence"]["maximum_url_model_score_phishing"] == 0.0,
        "one_recorded": forced["one"]["url_model_evidence"]["maximum_url_model_score_phishing"] == 1.0,
        "url_declared_auxiliary": all(not forced[name]["url_model_evidence"]["affects_email_classification"]
                                      for name in forced),
    }
    if not all(checks.values()):
        raise ValueError(f"URL isolation failed: {checks}")
    return {"status": "passed", "baseline_email_classification": baseline_email, "checks": checks}


def validate_end_to_end(components: dict[str, Any]) -> dict[str, Any]:
    from src.final_artifact import predict_final_artifact
    from src.unified_artifact import predict_unified
    cases = []
    labels: set[str] = set()
    fixtures = evaluation_fixtures()
    for fixture in fixtures:
        core = predict_final_artifact(fixture["subject"], fixture["body"], components)
        public = predict_unified(fixture["subject"], fixture["body"], components)
        email, evidence, explanation = (public["email_classification"], public["url_model_evidence"],
                                        public["text_explanation"])
        labels.add(email["predicted_label"])
        expected = fixture.get("expected_label")
        checks = {
            "core_email_equal": email == core["email_classification"],
            "core_url_equal": evidence == core["url_model_evidence"],
            "threshold_reproduces_label": email["predicted_label"] ==
                ("phishing" if email["text_probability_phishing"] >= 0.5 else "legitimate"),
            "url_auxiliary": evidence["affects_email_classification"] is False,
            "explanation_label_equal": explanation["final_email_label"] == email["predicted_label"],
            "explanation_probability_equal": explanation["calibrated_text_probability_phishing"] ==
                email["text_probability_phishing"],
            "reconstruction_passed": explanation["reconstruction_tolerance"]["passed"] is True,
            "expected_label_equal": expected is None or email["predicted_label"] == expected,
            "unicode_json_roundtrip": json.loads(json.dumps(public, ensure_ascii=False)) == public,
        }
        cases.append({**fixture, "synthetic": True,
                      "input_sha256": input_sha256(fixture["subject"], fixture["body"]),
                      "observed": {"predicted_label": email["predicted_label"],
                                   "text_probability_phishing": email["text_probability_phishing"],
                                   "scored_unique_url_count": evidence["scored_unique_url_count"]},
                      "checks": checks, "status": "passed" if all(checks.values()) else "failed"})
    duplicate = next(case for case in cases if case["case_id"] == "duplicate_normalized_url")
    unsupported = next(case for case in cases if case["case_id"] == "unsupported_url_like")
    coverage = {
        "fixture_order_preserved": [case["case_id"] for case in cases] == [case["case_id"] for case in fixtures],
        "both_labels_observed": labels == {"legitimate", "phishing"},
        "duplicate_scored_once": duplicate["observed"]["scored_unique_url_count"] == 1,
        "unsupported_not_scored": unsupported["observed"]["scored_unique_url_count"] == 0,
        "all_cases_passed": all(case["status"] == "passed" for case in cases),
    }
    if not all(coverage.values()):
        raise ValueError(f"End-to-end validation failed: {coverage}")
    return {"status": "passed", "synthetic_only": True, "performance_evaluation": False,
            "coverage": coverage, "observed_labels": sorted(labels), "cases": cases}


def run_benchmarks(components: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repetition in range(1, LOAD_REPETITIONS + 1):
        child, wall_ms = invoke_worker("load")
        for scenario, value in (("fresh_python_process_wall", wall_ms),
                                ("module_import", child["module_import_ms"]),
                                ("component_load", child["component_load_ms"]),
                                ("module_import_and_component_load", child["module_import_and_component_load_ms"])):
            rows.append({"phase": "initialization", "scenario_id": scenario,
                         "repetition": repetition, "elapsed_ms": value})
    for repetition in range(1, LOAD_REPETITIONS + 1):
        probe = invoke_actual_cli(FIXTURES[3]["subject"], FIXTURES[3]["body"], 10)
        if probe["returncode"] != 0 or not probe["stdout_json_only"] or not probe["stderr_empty"]:
            raise ValueError("Fresh CLI timing probe failed")
        rows.append({"phase": "cli", "scenario_id": "fresh_cli_end_to_end",
                     "repetition": repetition, "elapsed_ms": probe["wall_ms"]})
    from src.final_artifact import predict_final_artifact
    from src.unified_artifact import main as cli_main, predict_unified
    scenarios: list[tuple[str, Callable[[], Any]]] = [
        ("core_no_url", lambda: predict_final_artifact(FIXTURES[1]["subject"], FIXTURES[1]["body"], components)),
        ("core_one_url", lambda: predict_final_artifact(FIXTURES[3]["subject"], FIXTURES[3]["body"], components)),
        ("core_multiple_urls", lambda: predict_final_artifact(FIXTURES[4]["subject"], FIXTURES[4]["body"], components)),
        ("public_with_explanation", lambda: predict_unified(FIXTURES[3]["subject"], FIXTURES[3]["body"], components)),
        ("public_long_multiple_urls_with_explanation", lambda: predict_unified(FIXTURES[7]["subject"], FIXTURES[7]["body"], components)),
        ("cli_parse_infer_json", lambda: _timed_cli_main(cli_main, FIXTURES[3])),
    ]
    for scenario, call in scenarios:
        for _ in range(WARMUP_REPETITIONS):
            call()
        for repetition in range(1, INFERENCE_REPETITIONS + 1):
            started = time.perf_counter_ns()
            call()
            rows.append({"phase": "inference", "scenario_id": scenario, "repetition": repetition,
                         "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000})
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(float(row["elapsed_ms"]))
    return rows, {name: aggregate_milliseconds(values) for name, values in sorted(grouped.items())}


def _timed_cli_main(cli_main: Callable[[list[str]], None], fixture: dict[str, Any]) -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        cli_main(["--subject", fixture["subject"], "--body", fixture["body"], "--explanation-top-k", "10"])
    json.loads(output.getvalue())


def collect_memory() -> dict[str, Any]:
    trials = []
    for repetition in range(1, MEMORY_REPETITIONS + 1):
        value, _ = invoke_worker("memory")
        trials.append({"repetition": repetition, **value})
    fields = ["baseline_rss_bytes", "after_project_import_rss_bytes", "after_model_load_rss_bytes",
              "after_warmup_rss_bytes", "after_full_inference_rss_bytes", "peak_vmhwm_bytes",
              "model_load_rss_delta_bytes", "warmup_rss_delta_bytes"]
    aggregates = {}
    for field in fields:
        values = [int(trial[field]) for trial in trials]
        aggregates[field] = {"count": len(values), "median_bytes": statistics.median(values),
                             "mean_bytes": statistics.fmean(values), "min_bytes": min(values),
                             "max_bytes": max(values), "median_mib": statistics.median(values) / (1024 ** 2),
                             "max_mib": max(values) / (1024 ** 2)}
    return {"measurement_method": "/proc/self/status VmRSS and VmHWM",
            "platform_scope": "WSL2 Linux process", "trial_count": MEMORY_REPETITIONS,
            "trials": trials, "aggregates": aggregates}


def collect_environment() -> dict[str, Any]:
    return {
        "measurement_context": "WSL2 project virtual environment on a Windows host",
        "platform": platform.platform(), "kernel": platform.release(), "logical_cpus": os.cpu_count(),
        "runtime": {"python": platform.python_version(), "python_executable": sys.executable,
                    "numpy": importlib.metadata.version("numpy"),
                    "scikit_learn": importlib.metadata.version("scikit-learn"),
                    "joblib": importlib.metadata.version("joblib"),
                    "urlextract": importlib.metadata.version("urlextract")},
        "cpu_only": True, "gpu_benchmark_performed": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "scenario_id", "repetition", "elapsed_ms"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run() -> dict[str, Any]:
    if OUT.exists():
        raise ValueError(f"Step-34 output already exists: {OUT}")
    protected = validate_protected()
    from src.unified_artifact import load_unified_artifact_components
    components = load_unified_artifact_components()
    architecture = components["architecture"]
    if architecture["architecture_name"] != "step25c_text_led_email_classifier_with_auxiliary_url_evidence":
        raise ValueError("Unexpected active architecture")
    end_to_end = validate_end_to_end(components)
    end_to_end["fresh_process_cli"] = validate_fresh_cli()
    end_to_end["url_score_isolation"] = validate_url_isolation(components)
    offline, _ = invoke_worker("offline_cli")
    if offline["status"] != "passed":
        raise ValueError("Offline CLI validation failed")
    end_to_end["offline_cli_validation"] = offline
    benchmark_rows, timing = run_benchmarks(components)
    sizes = collect_bundle_sizes()
    memory = collect_memory()
    environment = collect_environment()
    technical = {
        "timing_method": "CPU-only wall-clock measured with time.perf_counter_ns; milliseconds",
        "initialization_repetitions": LOAD_REPETITIONS,
        "warmup_repetitions_per_inference_scenario": WARMUP_REPETITIONS,
        "measured_repetitions_per_inference_scenario": INFERENCE_REPETITIONS,
        "p95_method": "nearest_rank", "timing_aggregates": timing, "bundle_sizes": sizes,
    }
    resources = {"environment": environment, "memory": memory}
    summary = [
        {"metric": "Productive model bundle", "value": sizes["productive_model_bundle_total"]["mib"], "unit": "MiB"},
        {"metric": "Complete CLI delivery", "value": sizes["complete_cli_delivery_total"]["mib"], "unit": "MiB"},
        {"metric": "Fresh CLI process median", "value": timing["fresh_cli_end_to_end"]["median_ms"], "unit": "ms"},
        {"metric": "Full long multi-URL inference p95", "value": timing["public_long_multiple_urls_with_explanation"]["p95_ms"], "unit": "ms"},
        {"metric": "Process peak RSS maximum", "value": memory["aggregates"]["peak_vmhwm_bytes"]["max_mib"], "unit": "MiB"},
        {"metric": "Offline CLI inference", "value": offline["status"], "unit": "status"},
    ]
    report = {
        "status": "completed", "step": 34,
        "evaluation_type": "cli_only_end_to_end_technical_evaluation",
        "classification_performance_claim": False,
        "summary_table": summary,
        "limitations": ["Timing and memory characterize only the recorded environment.",
                        "Synthetic fixtures do not estimate classification performance."],
    }
    implementation = {name: sha256(REPOSITORY_ROOT / name) for name in IMPLEMENTATION_FILES}
    method = {"version": 2, "fixture_ids": [row["case_id"] for row in evaluation_fixtures()],
              "load_repetitions": LOAD_REPETITIONS, "warmup_repetitions": WARMUP_REPETITIONS,
              "inference_repetitions": INFERENCE_REPETITIONS, "memory_repetitions": MEMORY_REPETITIONS,
              "actual_cli_command": "python -B -m src.unified_artifact"}
    run_id = "step34:" + hashlib.sha256(json.dumps(
        {"method": method,
         "predecessor_semantics": semantic_predecessor_identity(REPOSITORY_ROOT, protected),
         "implementation": implementation},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    manifest = {
        "selection_provenance": current_selection_provenance(REPOSITORY_ROOT),
        "artifact_type": "step34_cli_only_technical_evaluation", "schema_version": 2,
        "evaluation_run_id": run_id, "method_specification": method,
        "protected_input_sha256": protected, "implementation_sha256": implementation,
        "runtime": environment["runtime"], "output_sha256": {},
        "self_hash_policy": "manifest excludes its own hash; checksums.sha256 excludes itself",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".technical-evaluation-staging-", dir=OUT.parent))
    try:
        write_json(staging / "technical_metrics.json", technical)
        _write_csv(staging / "inference_benchmark.csv", benchmark_rows)
        write_json(staging / "resource_usage.json", resources)
        write_json(staging / "end_to_end_validation.json", end_to_end)
        write_json(staging / "report.json", report)
        manifest["output_sha256"] = {name: sha256(staging / name) for name in OUTPUT_NAMES[:-1]}
        write_json(staging / "manifest.json", manifest)
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(staging / name)}  {name}\n" for name in OUTPUT_NAMES),
            encoding="utf-8", newline="\n")
        subprocess.run([sys.executable, "-B", "-m", "scripts.validate_technical_evaluation",
                        "--out", str(staging)], cwd=REPOSITORY_ROOT, check=True)
        os.replace(staging, OUT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "completed", "evaluation_run_id": run_id, "summary_table": summary}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-technical-evaluation", action="store_true")
    parser.add_argument("--worker", choices=["load", "memory", "offline_cli"])
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(run_worker(args.worker), ensure_ascii=False, sort_keys=True), flush=True)
    elif args.execute_technical_evaluation:
        print(json.dumps(run(), indent=2, ensure_ascii=False, sort_keys=True), flush=True)
    else:
        parser.error("Step 34 is inert without --execute-technical-evaluation")


if __name__ == "__main__":
    main()
