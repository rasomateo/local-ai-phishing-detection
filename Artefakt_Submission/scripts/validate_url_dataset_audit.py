from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from src.load_data import PHISHLOAD_WEBSITE_COLUMNS, _extract_insert_values, _parse_sql_insert_rows
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract
from src.phishload_dataset import AUDIT_VERSION, protected_hashes
from src.provenance import content_run_id, validate_predecessor

csv.field_size_limit(sys.maxsize)
CANONICAL = ("url_dataset.csv", "split_assignments.csv", "cv_assignments.csv")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def independent_normalize(value: str) -> str:
    value = value.strip()
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return ""
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if parsed.scheme.lower() not in ("http", "https") or not host:
            return ""
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or any(char.isspace() for char in host) or any(not label or len(label) > 63 for label in host.split(".")):
            return ""
        port = parsed.port
        credentials = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
        netloc = credentials + ("[" + host + "]" if ":" in host else host) + (":" + str(port) if port is not None else "")
        return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, ""))
    except (ValueError, UnicodeError):
        return ""


def validate(root: Path = REPOSITORY_ROOT) -> dict:
    contract = current_contract(root)
    out = root / "outputs/datasets/url/audit"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    check(report["status"] == "blocked" and report["source"] == "phishload" and report["source_role"] == "url_only", "Status or source role")
    for key in ("canonical_url_dataset_published", "canonical_split_published", "canonical_cv_published", "model_training_performed", "step22_authorized", "calibration_performed", "threshold_tuned", "fusion_performed", "holdout_evaluated", "external_data_used", "website_html_used_as_email_text", "text_data_used_for_url_training"):
        check(report.get(key) is False, f"False flag required: {key}")
    check(all(not (out / name).exists() for name in CANONICAL), "Canonical artifact published while conflicts remain")
    validate_predecessor(root, report["predecessor"])
    expected_protected = protected_hashes(root)
    check(report["protected_sha256"] == expected_protected, "Protected inventory differs")
    for name, expected in expected_protected.items():
        check(sha(root / name) == expected, f"Protected artifact hash changed: {name}")
    archive = root / "data/raw/phishload/current.zip"
    with zipfile.ZipFile(archive) as source:
        check(set(source.namelist()) == {"phishload_120322.sql", "phishload_readme.txt"}, "Archive member inventory")
        sql_bytes = source.read("phishload_120322.sql")
        readme_bytes = source.read("phishload_readme.txt")
    expected_sources = {"data/raw/phishload/current.zip": sha(archive), "phishload_120322.sql": hashlib.sha256(sql_bytes).hexdigest(), "phishload_readme.txt": hashlib.sha256(readme_bytes).hexdigest()}
    check(report["source_files_sha256"] == expected_sources, "Source hashes differ")
    expected_code_paths = {"src/phishload_dataset.py", "src/load_data.py"}
    check(set(report["code_sha256"]) == expected_code_paths, "Audited code inventory differs")
    for name, expected in report["code_sha256"].items():
        check(sha(root / name) == expected, f"Audited code hash differs: {name}")
    readme = readme_bytes.decode("utf-8", errors="replace").lower()
    documented = "isphish: is this webpage url from a phishinglist (1) or non-fraudulent (0)" in readme and 'the most important state is "3" all websites with this state are real phishing websites' in readme
    check(report["label_semantics_verified_from_readme"] is documented, "README semantics evidence differs")
    check("globally" in report["legitimate_label_scope"] and "scan" in report["legitimate_label_scope"], "Legitimate scope not qualified")
    raw_rows = _parse_sql_insert_rows(_extract_insert_values(sql_bytes.decode("utf-8", errors="replace"), "websites"))
    source_rows = [{name: "" if value is None else str(value) for name, value in zip(PHISHLOAD_WEBSITE_COLUMNS, values)} for values in raw_rows]
    check(all(len(values) == len(PHISHLOAD_WEBSITE_COLUMNS) for values in raw_rows), "Source schema")
    check(len({row["id"] for row in source_rows}) == len(source_rows), "Source ID uniqueness")
    grouped = defaultdict(list)
    raw_grouped = defaultdict(list)
    expected_exclusions = Counter()
    candidate_counts = Counter()
    raw_isphish = Counter()
    raw_states = Counter()
    domain_sizes = Counter()
    domain_missing = 0
    domain_mismatch = 0
    for row in source_rows:
        raw_isphish[row["isPhish"] or "<null>"] += 1
        raw_states[row["state"] or "<null>"] += 1
        normalized = independent_normalize(row["url"])
        label = "legitimate" if row["isPhish"] == "0" and not row["state"] else "phishing" if row["isPhish"] == "1" and row["state"] == "3" else ""
        candidate_counts[label or "ambiguous"] += 1
        if not normalized or not label:
            expected_exclusions[(row["id"], "missing_url" if not row["url"].strip() else "invalid_url" if not normalized else "ambiguous_label")] += 1
        if normalized and label:
            grouped[normalized].append((row, label))
            raw_grouped[row["url"]].append((row, label))
        if normalized:
            domain = row["urlBasedomain"].lower().rstrip(".")
            if domain:
                domain_sizes[domain] += 1
                host = urlsplit(normalized).hostname or ""
                if not (host == domain or host.endswith("." + domain)):
                    domain_mismatch += 1
            else:
                domain_missing += 1
    expected_conflicts = {url: values for url, values in grouped.items() if len({label for _, label in values}) > 1}
    conflicts = csv_rows(out / "label_conflicts.csv")
    exclusions = csv_rows(out / "exclusions.csv")
    check(report["label_conflict_count"] == len(expected_conflicts) > 0, "Expected unresolved conflicts absent or count wrong")
    check(report["counts"]["raw_record_count"] == len(source_rows) and report["counts"]["candidate_label_counts"] == dict(candidate_counts), "Source/candidate counts")
    check(report["counts"]["raw_isPhish_counts"] == dict(raw_isphish) and report["counts"]["raw_state_counts"] == dict(raw_states), "Raw state/isPhish counts")
    check(report["counts"]["raw_url_nonempty_count"] == sum(bool(r["url"].strip()) for r in source_rows), "Raw URL count")
    check(report["counts"]["raw_unique_url_count"] == len({r["url"] for r in source_rows if r["url"].strip()}), "Raw unique URL count")
    check(report["counts"]["supervised_candidate_count_before_conflict_gate"] == sum(len(x) for x in grouped.values()), "Candidate count")
    check(report["counts"]["unique_normalized_url_count_before_conflict_gate"] == len(grouped), "Unique normalized URL count")
    def duplicate_groups(groups: dict, conflict: bool) -> int:
        return sum(len(values) > 1 and (len({label for _, label in values}) > 1) == conflict for values in groups.values())
    raw_conflicting = {url for url, values in raw_grouped.items() if len({label for _, label in values}) > 1}
    normalized_only = sum(not any(row["url"] in raw_conflicting for row, _ in values) for url, values in expected_conflicts.items())
    checks = {
        "raw_exact_duplicate_url_groups_same_label": duplicate_groups(raw_grouped, False),
        "normalized_duplicate_url_groups_same_label": duplicate_groups(grouped, False),
        "raw_exact_duplicate_url_groups_conflicting_label": duplicate_groups(raw_grouped, True),
        "normalized_only_duplicate_url_groups_conflicting_label": normalized_only,
        "raw_exact_duplicate_extra_records": sum(len(v) - 1 for v in raw_grouped.values() if len(v) > 1),
        "normalized_duplicate_extra_records": sum(len(v) - 1 for v in grouped.values() if len(v) > 1),
        "duplicate_max_multiplicity": max((len(v) for v in grouped.values()), default=0),
        "source_base_domain_count": len(domain_sizes),
        "source_base_domain_missing_count": domain_missing,
        "source_base_domain_host_mismatch_count": domain_mismatch,
    }
    for key, value in checks.items():
        check(report["counts"][key] == value, f"Count mismatch: {key}")
    check(report["counts"]["source_base_domain_sizes"] == {"singletons": sum(v == 1 for v in domain_sizes.values()), "size_2_to_9": sum(2 <= v <= 9 for v in domain_sizes.values()), "size_10_plus": sum(v >= 10 for v in domain_sizes.values()), "max": max(domain_sizes.values(), default=0)}, "Domain size distribution")
    check(report["counts"]["label_conflict_record_count"] == len(conflicts), "Conflict record count")
    check(Counter((r["website_id"], r["reason"]) for r in exclusions) == expected_exclusions, "Exclusion coverage")
    check(report["counts"]["exclusion_reason_counts"] == dict(Counter(r["reason"] for r in exclusions)), "Exclusion counts")
    check(len(conflicts) == sum(len(values) for values in expected_conflicts.values()), "Conflict source-record coverage")
    seen = set()
    for item in conflicts:
        normalized = item["normalized_url"]
        check(normalized in expected_conflicts, "Unrecognized conflict URL")
        check(item["conflict_id"] == "url-conflict:" + hashlib.sha256(normalized.encode()).hexdigest()[:20], "Conflict ID mismatch")
        check((normalized, item["website_id"]) not in seen, "Repeated conflict source ID")
        seen.add((normalized, item["website_id"]))
        matches = [(row, label) for row, label in expected_conflicts[normalized] if row["id"] == item["website_id"]]
        check(len(matches) == 1, "Conflict source record absent")
        row, label = matches[0]
        for field in ("url", "isPhish", "state", "urlBasedomain", "finalUrl", "finalUrlBasedomain", "scanned", "created", "rescan", "statusCode", "parent", "duplicatedFrom", "urlHash", "phishtank_id", "phishtank_submissiontime", "phishtank_verified", "phishtank_verifiedtime", "phishtank_isonline"):
            check(item["raw_url" if field == "url" else field] == row[field], f"Conflict field mismatch: {field}")
        check(item["candidate_label"] == label and item["source"] == "phishload" and item["source_role"] == "url_only", "Conflict label/role mismatch")
    known = {} if contract.is_fixture else {
        "http://webslogin.ordinancetrust.com/myaccount/Login.php": {"1937", "10264"},
        "http://home2.btconnect.com/mcleodscrap/tiny_mce/ogbomosho.html": {"3317", "11206"},
    }
    for raw, ids in known.items():
        records = [r for r in conflicts if r["raw_url"] == raw]
        check({r["website_id"] for r in records} == ids and {r["normalized_url"] for r in records} == {raw}, "Known raw conflict pair missing")
        check(len({r["scanned"] for r in records}) == 2, "Known conflict scan times not preserved")
    for name in ("label_conflicts.csv", "exclusions.csv"):
        check(report["outputs"][name] == {"path": "outputs/datasets/url/audit/" + name, "sha256": sha(out / name)}, f"Output hash/path mismatch: {name}")
    check("unresolved_normalized_url_label_conflicts" in report["blocking_reason"], "Missing conflict blocker")
    semantic_payload = {
        "contract_version": report["contract_version"], "audit_version": AUDIT_VERSION,
        "inputs": report["source_files_sha256"],
        "normalization": "strip; require HTTP(S); lowercase scheme/IDNA host; remove terminal host dot and fragment; preserve userinfo/port/path/query",
        "label_mapping": report["label_mapping"], "counts": report["counts"],
        "state_lookup": report["state_lookup"],
        "conflict_rule": "report unresolved normalized URL identities with both labels",
    }
    expected_run_id, expected_semantic = content_run_id("phishload-audit", semantic_payload)
    check(report["url_run_id"] == expected_run_id and report["semantic_sha256"] == expected_semantic,
          "Audit semantic identity mismatch")
    return report


if __name__ == "__main__":
    result = validate()
    print(f"validated blocked Step 21: {result['label_conflict_count']} unresolved URL conflicts")
