from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import ipaddress
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from sklearn.model_selection import StratifiedGroupKFold

from src.load_data import PHISHLOAD_WEBSITE_COLUMNS, _extract_insert_values, _parse_sql_insert_rows
from src.provenance import content_run_id, predecessor_reference, validate_artifact_inventory
from src.utils import REPOSITORY_ROOT

ARCHIVE = Path("data/raw/phishload/current.zip")
AUDIT_OUT = Path("outputs/datasets/url/audit")
SQL_NAME = "phishload_120322.sql"
README_NAME = "phishload_readme.txt"
AUDIT_VERSION = "step21-diagnostic-v1"

AUDIT_CONFLICT_COLUMNS = ["conflict_id", "normalized_url", "url_record_id", "website_id", "source", "source_role", "raw_url", "isPhish", "state", "state_name", "candidate_label", "urlBasedomain", "finalUrl", "finalUrlBasedomain", "scanned", "created", "rescan", "statusCode", "parent", "duplicatedFrom", "urlHash", "phishtank_id", "phishtank_submissiontime", "phishtank_verified", "phishtank_verifiedtime", "phishtank_isonline"]
AUDIT_EXCLUSION_COLUMNS = ["url_record_id", "website_id", "raw_url", "normalized_url", "isPhish", "state", "state_name", "candidate_label", "reason"]
AUDIT_CANONICAL_OUTPUTS = ("url_dataset.csv", "split_assignments.csv", "cv_assignments.csv")
AUDIT_FLAGS = {key: False for key in ("text_data_used_for_url_training", "website_html_used_as_email_text", "model_training_performed", "calibration_performed", "threshold_tuned", "fusion_performed", "holdout_evaluated", "external_data_used", "step22_authorized", "canonical_url_dataset_published", "canonical_split_published", "canonical_cv_published")}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes(root: Path) -> dict[str, str]:
    from scripts.validate_text_calibration import validate
    validate(root)
    base = root / "outputs/models/text/calibration"
    names = ("text_calibrator.joblib", "development_crossfit_calibrated_scores.csv",
             "calibration_diagnostics.csv", "manifest.json", "report.json")
    return {f"outputs/models/text/calibration/{name}": digest(base / name) for name in names}


def normalize_url(raw: str) -> str:
    value = raw.strip()
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("missing_or_control_character")
    try:
        parts = urlsplit(value)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc or not parts.hostname:
            raise ValueError("invalid_scheme_or_host")
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or any(c.isspace() for c in host):
            raise ValueError("invalid_host")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not all(label and len(label) <= 63 for label in host.split(".")):
                raise ValueError("invalid_host")
        port = parts.port
        userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
        normalized_host = f"[{host}]" if ":" in host else host
        netloc = userinfo + normalized_host + (f":{port}" if port is not None else "")
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, ""))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid_url") from exc


def url_record_id(website_id: str, normalized_url: str) -> str:
    identity = "\x1f".join((AUDIT_VERSION, "phishload", website_id, normalized_url))
    return "phishload-url:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def candidate_label(is_phish: str, state: str) -> str:
    if is_phish == "0" and not state:
        return "legitimate"
    if is_phish == "1" and state == "3":
        return "phishing"
    return ""


def source_rows(sql: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    states = {str(row[0]): str(row[1]) for row in _parse_sql_insert_rows(_extract_insert_values(sql, "states"))}
    rows = []
    for values in _parse_sql_insert_rows(_extract_insert_values(sql, "websites")):
        if len(values) != len(PHISHLOAD_WEBSITE_COLUMNS):
            raise ValueError("Unexpected Phishload website schema")
        rows.append({name: "" if value is None else str(value) for name, value in zip(PHISHLOAD_WEBSITE_COLUMNS, values)})
    return rows, states


def audit_records(rows: list[dict[str, str]], states: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    records = []
    exclusions = []
    ids = set()
    for row in rows:
        website_id = row["id"]
        if not website_id or website_id in ids:
            raise ValueError("Missing or duplicated website ID")
        ids.add(website_id)
        raw = row["url"]
        try:
            normalized = normalize_url(raw)
            url_error = ""
        except ValueError:
            normalized = ""
            url_error = "missing_url" if not raw.strip() else "invalid_url"
        label = candidate_label(row["isPhish"], row["state"])
        item = {**row, "website_id": website_id, "raw_url": raw, "normalized_url": normalized,
                "source": "phishload", "source_role": "url_only", "state_name": states.get(row["state"], ""),
                "candidate_label": label, "url_record_id": url_record_id(website_id, normalized)}
        records.append(item)
        if url_error or not label:
            exclusions.append({k: item.get(k, "") for k in AUDIT_EXCLUSION_COLUMNS if k != "reason"} | {"reason": url_error or "ambiguous_label"})
    by_normal = defaultdict(list)
    by_raw = defaultdict(list)
    for item in records:
        if item["normalized_url"] and item["candidate_label"]:
            by_normal[item["normalized_url"]].append(item)
            by_raw[item["raw_url"]].append(item)
    conflict_urls = {key for key, values in by_normal.items() if len({v["candidate_label"] for v in values}) > 1}
    conflicts = []
    for normalized in sorted(conflict_urls):
        conflict_id = "url-conflict:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        for item in sorted(by_normal[normalized], key=lambda r: r["website_id"]):
            conflicts.append({"conflict_id": conflict_id, **{key: item.get(key, "") for key in AUDIT_CONFLICT_COLUMNS if key != "conflict_id"}})
    def group_count(groups, same):
        return sum(len(values) > 1 and (len({x["candidate_label"] for x in values}) == 1) == same for values in groups.values())
    raw_conflict_urls = {key for key, values in by_raw.items() if len({v["candidate_label"] for v in values}) > 1}
    normalized_only_conflicts = sum(not any(raw in raw_conflict_urls for raw in {v["raw_url"] for v in by_normal[key]}) for key in conflict_urls)
    domains = Counter(item["urlBasedomain"].lower() for item in records if item["normalized_url"] and item["urlBasedomain"])
    domain_missing = 0
    domain_mismatch = 0
    for item in records:
        if not item["normalized_url"]:
            continue
        basedomain = item["urlBasedomain"].strip().lower().rstrip(".")
        if not basedomain:
            domain_missing += 1
        else:
            host = urlsplit(item["normalized_url"]).hostname or ""
            if not (host == basedomain or host.endswith("." + basedomain)):
                domain_mismatch += 1
    supervised_candidates = [r for r in records if r["normalized_url"] and r["candidate_label"]]
    diagnostics = {
        "raw_record_count": len(rows), "raw_url_nonempty_count": sum(bool(r["url"].strip()) for r in rows),
        "raw_unique_url_count": len({r["url"] for r in rows if r["url"].strip()}),
        "raw_isPhish_counts": dict(sorted(Counter(r["isPhish"] or "<null>" for r in rows).items())),
        "raw_state_counts": dict(sorted(Counter(r["state"] or "<null>" for r in rows).items())),
        "raw_state_isPhish_counts": {f"state={k[0] or '<null>'};isPhish={k[1] or '<null>'}": v for k, v in sorted(Counter((r["state"], r["isPhish"]) for r in rows).items())},
        "candidate_label_counts": dict(sorted(Counter(r["candidate_label"] or "ambiguous" for r in records).items())),
        "exclusion_reason_counts": dict(sorted(Counter(r["reason"] for r in exclusions).items())),
        "supervised_candidate_count_before_conflict_gate": len(supervised_candidates),
        "unique_normalized_url_count_before_conflict_gate": len(by_normal),
        "raw_exact_duplicate_url_groups_same_label": group_count(by_raw, True),
        "normalized_duplicate_url_groups_same_label": group_count(by_normal, True),
        "raw_exact_duplicate_url_groups_conflicting_label": len(raw_conflict_urls),
        "normalized_only_duplicate_url_groups_conflicting_label": normalized_only_conflicts,
        "raw_exact_duplicate_extra_records": sum(len(v) - 1 for v in by_raw.values() if len(v) > 1),
        "normalized_duplicate_extra_records": sum(len(v) - 1 for v in by_normal.values() if len(v) > 1),
        "duplicate_max_multiplicity": max((len(v) for v in by_normal.values()), default=0),
        "label_conflict_count": len(conflict_urls), "label_conflict_record_count": len(conflicts),
        "source_base_domain_count": len(domains), "source_base_domain_sizes": {"singletons": sum(v == 1 for v in domains.values()), "size_2_to_9": sum(2 <= v <= 9 for v in domains.values()), "size_10_plus": sum(v >= 10 for v in domains.values()), "max": max(domains.values(), default=0)},
        "source_base_domain_missing_count": domain_missing, "source_base_domain_host_mismatch_count": domain_mismatch,
    }
    return conflicts, exclusions, diagnostics


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phishload_audit_semantic_payload(
    input_hashes: dict[str, str], counts: dict, states: dict[str, str]
) -> dict:
    return {
        "contract_version": "artifact-provenance-v1:phishload-audit-v1",
        "audit_version": AUDIT_VERSION,
        "inputs": input_hashes,
        "normalization": "strip; require HTTP(S); lowercase scheme/IDNA host; remove terminal host dot and fragment; preserve userinfo/port/path/query",
        "label_mapping": {"isPhish=0;state=<null>": "legitimate", "isPhish=1;state=3": "phishing", "otherwise": "ambiguous/excluded"},
        "counts": counts,
        "state_lookup": states,
        "conflict_rule": "report unresolved normalized URL identities with both labels",
    }


def run_audit(root: Path = REPOSITORY_ROOT) -> dict:
    start = time.perf_counter()
    protected = protected_hashes(root)
    protected_seconds = time.perf_counter() - start
    archive = root / ARCHIVE
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != {SQL_NAME, README_NAME}:
            raise ValueError("Unexpected Phishload archive members")
        sql_bytes = source.read(SQL_NAME)
        readme_bytes = source.read(README_NAME)
    readme = readme_bytes.decode("utf-8", errors="replace").lower()
    semantics_verified = all(term in readme for term in ("isphish: is this webpage url from a phishinglist (1) or non-fraudulent (0)", 'the most important state is "3" all websites with this state are real phishing websites'))
    source_seconds = time.perf_counter() - start - protected_seconds
    rows, states = source_rows(sql_bytes.decode("utf-8", errors="replace"))
    conflicts, exclusions, counts = audit_records(rows, states)
    audit_seconds = time.perf_counter() - start - protected_seconds - source_seconds
    if not conflicts and semantics_verified:
        raise ValueError("Expected methodological blocker absent; canonical publication requires a separately approved step")
    out = root / AUDIT_OUT
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in AUDIT_CANONICAL_OUTPUTS):
        raise ValueError("Canonical URL artifact exists in blocked run")
    write_csv(out / "label_conflicts.csv", AUDIT_CONFLICT_COLUMNS, conflicts)
    write_csv(out / "exclusions.csv", AUDIT_EXCLUSION_COLUMNS, exclusions)
    input_hashes = {str(ARCHIVE).replace("\\", "/"): digest(archive), SQL_NAME: hashlib.sha256(sql_bytes).hexdigest(), README_NAME: hashlib.sha256(readme_bytes).hexdigest()}
    semantic_payload = phishload_audit_semantic_payload(input_hashes, counts, states)
    run_id, semantic_sha = content_run_id("phishload-audit", semantic_payload)
    reasons = []
    if conflicts:
        reasons.append("unresolved_normalized_url_label_conflicts")
    if not semantics_verified:
        reasons.append("source_label_semantics_not_verified")
    report = {
        "status": "blocked", "url_run_id": run_id, "blocking_reason": reasons,
        "contract_version": semantic_payload["contract_version"], "semantic_sha256": semantic_sha,
        "predecessor": predecessor_reference(root, root / "outputs/models/text/calibration/manifest.json"),
        "label_conflict_count": counts["label_conflict_count"], **AUDIT_FLAGS,
        "source": "phishload", "source_role": "url_only", "source_files_sha256": input_hashes,
        "source_schema": PHISHLOAD_WEBSITE_COLUMNS, "state_lookup": states,
        "label_mapping": {"isPhish=0;state=<null>": "legitimate", "isPhish=1;state=3": "phishing", "otherwise": "ambiguous/excluded"},
        "label_semantics_verified_from_readme": semantics_verified,
        "legitimate_label_scope": "source record not marked phishing/fraudulent at its scan or capture time; no globally or permanently verified legitimacy claim",
        "label_provenance": "phishload_readme.txt and websites.isPhish/state",
        "normalization": "strip surrounding whitespace; require HTTP(S); lowercase scheme and IDNA hostname; remove terminal hostname dot and fragment; preserve userinfo, port, path and query; no DNS/HTTP/network",
        "url_record_id_method": "phishload-url:SHA256(UTF-8(step21-diagnostic-v1 US phishload US website_id US normalized_url))",
        "base_domain_diagnostic": "source-native urlBasedomain; host equals it or ends with dot plus it; no online suffix lookup; no split grouping published",
        "counts": counts, "protected_sha256": protected,
        "code_sha256": {name: digest(root / name) for name in ("src/phishload_dataset.py", "src/load_data.py")},
        "outputs": {name: {"path": (AUDIT_OUT / name).as_posix(), "sha256": digest(out / name)} for name in ("label_conflicts.csv", "exclusions.csv")},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "pandas": importlib.metadata.version("pandas"), "scikit_learn": importlib.metadata.version("scikit-learn")},
        "processing_seconds": {"protected_hash_check": round(protected_seconds, 6), "source_read": round(source_seconds, 6), "parse_and_audit": round(audit_seconds, 6), "total_before_report": round(time.perf_counter() - start, 6)},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protected_hashes(root)
    return report


CANONICAL_VERSION = "step21b-canonical-url-v1"
CANONICAL_OUT = Path("outputs/datasets/url/canonical")
DATASET_COLUMNS = ["url_run_id", "canonical_url_id", "normalized_url", "representative_raw_url", "label", "leakage_group", "source", "source_role", "label_provenance", "source_record_count"]
MAPPING_COLUMNS = ["url_run_id", "canonical_url_id", "source_record_id", "website_id", "raw_url", "normalized_url", "candidate_label", "isPhish", "state", "scanned", "created", "parent", "urlBasedomain", "mapping_status"]
CANONICAL_EXCLUSION_COLUMNS = ["url_run_id", "source_record_id", "website_id", "raw_url", "normalized_url", "candidate_label", "reason"]
CANONICAL_CONFLICT_COLUMNS = ["url_run_id", "conflict_id", "normalized_url", "involved_source_record_count", "observed_labels", "resolution", "excluded_from_supervised"]
ASSIGNMENT_COLUMNS = ["url_run_id", "canonical_url_id", "label", "leakage_group", "split"]
CV_COLUMNS = ["url_run_id", "canonical_url_id", "label", "leakage_group", "cv_fold"]
STATUS_EXCLUDED = {"conflicting_ground_truth_excluded": "conflicting_ground_truth", "ambiguous_label_excluded": "ambiguous_label", "invalid_url_excluded": "invalid_url", "missing_leakage_group_excluded": "missing_leakage_group"}
CANONICAL_FLAGS = {"conflicting_identities_resolved_by_exclusion": True, "automatic_relabeling_performed": False, "majority_vote_performed": False, "time_based_relabeling_performed": False, "canonical_url_dataset_published": True, "canonical_split_published": True, "canonical_cv_published": True, "model_training_performed": False, "calibration_performed": False, "threshold_tuned": False, "fusion_performed": False, "holdout_evaluated": False, "external_data_used": False, "email_text_used": False, "website_html_used_as_email_text": False, "model_representation_selected": False, "step22_authorized": False}


def check_inputs(root: Path) -> dict[str, str]:
    expected = protected_hashes(root)
    step21 = json.loads((root / "outputs/datasets/url/audit/report.json").read_text(encoding="utf-8"))
    if step21["status"] != "blocked" or step21["label_conflict_count"] != 2 or not step21["label_semantics_verified_from_readme"]:
        raise ValueError("Step-21 source semantics or conflict precondition changed")
    validate_artifact_inventory(root, step21["outputs"])
    return {**expected, **{spec["path"]: spec["sha256"] for spec in step21["outputs"].values()},
            "outputs/datasets/url/audit/report.json": digest(root / "outputs/datasets/url/audit/report.json")}


def canonical_url_id(normalized_url: str) -> str:
    payload = "\x1f".join((CANONICAL_VERSION, "phishload", normalized_url))
    return "phishload-canonical:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def leakage_group(normalized_url: str, source_rows_for_url: list[dict[str, str]]) -> str:
    host = urlsplit(normalized_url).hostname or ""
    domains = {r["urlBasedomain"].strip().lower().rstrip(".") for r in source_rows_for_url if r["urlBasedomain"].strip()}
    if len(domains) > 1 or any(not (host == domain or host.endswith("." + domain)) for domain in domains):
        raise ValueError(f"Inconsistent source base domain: {normalized_url}")
    if domains:
        return next(iter(domains))
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return ""


def prepare(rows: list[dict[str, str]], run_id: str) -> tuple[dict[str, list[dict[str, str]]], dict]:
    prepared = []
    ids = set()
    for row in rows:
        website_id = row["id"]
        if not website_id or website_id in ids:
            raise ValueError("Source website IDs must be unique and nonempty")
        ids.add(website_id)
        try:
            normalized = normalize_url(row["url"])
        except ValueError:
            normalized = ""
        label = candidate_label(row["isPhish"], row["state"])
        prepared.append({"row": row, "normalized_url": normalized, "label": label, "status": ""})
    grouped = defaultdict(list)
    for item in prepared:
        if not item["normalized_url"]:
            item["status"] = "invalid_url_excluded"
        elif not item["label"]:
            item["status"] = "ambiguous_label_excluded"
        else:
            grouped[item["normalized_url"]].append(item)
    conflicts = {url: members for url, members in grouped.items() if len({m["label"] for m in members}) > 1}
    conflict_rows = []
    canonical = []
    for url, members in sorted(grouped.items()):
        if url in conflicts:
            for item in members:
                item["status"] = "conflicting_ground_truth_excluded"
            conflict_rows.append({"url_run_id": run_id, "conflict_id": "url-conflict:" + hashlib.sha256(url.encode()).hexdigest()[:20], "normalized_url": url, "involved_source_record_count": str(len(members)), "observed_labels": "legitimate|phishing", "resolution": "exclude_entire_identity_due_conflicting_ground_truth", "excluded_from_supervised": "true"})
            continue
        source_members = [item["row"] for item in members]
        group = leakage_group(url, source_members)
        if not group:
            for item in members:
                item["status"] = "missing_leakage_group_excluded"
            continue
        chosen = min(members, key=lambda m: m["row"]["id"])
        for item in members:
            item["status"] = "canonical_member" if item is chosen else "same_label_duplicate_member"
        canonical.append({"url_run_id": run_id, "canonical_url_id": canonical_url_id(url), "normalized_url": url, "representative_raw_url": min(m["row"]["url"] for m in members), "label": members[0]["label"], "leakage_group": group, "source": "phishload", "source_role": "url_only", "label_provenance": "phishload_readme.txt:websites.isPhish/state;scan_time_scope", "source_record_count": str(len(members))})
    by_url = {r["normalized_url"]: r for r in canonical}
    mapping = []
    exclusions = []
    for item in sorted(prepared, key=lambda m: m["row"]["id"]):
        row, status = item["row"], item["status"]
        if not status:
            raise ValueError("Unassigned source record")
        canonical_id = by_url[item["normalized_url"]]["canonical_url_id"] if status in {"canonical_member", "same_label_duplicate_member"} else ""
        common = {"url_run_id": run_id, "source_record_id": f"phishload:website:{row['id']}", "website_id": row["id"], "raw_url": row["url"], "normalized_url": item["normalized_url"], "candidate_label": item["label"]}
        mapping.append({**common, "canonical_url_id": canonical_id, "isPhish": row["isPhish"], "state": row["state"], "scanned": row["scanned"], "created": row["created"], "parent": row["parent"], "urlBasedomain": row["urlBasedomain"], "mapping_status": status})
        if status in STATUS_EXCLUDED:
            exclusions.append({**common, "reason": STATUS_EXCLUDED[status]})
    tables = {"url_dataset.csv": canonical, "source_record_mapping.csv": mapping, "exclusions.csv": exclusions, "conflict_resolution.csv": conflict_rows}
    diagnostics = {"source_records": len(rows), "valid_candidate_source_rows": sum(len(m) for m in grouped.values()), "unique_candidate_normalized_urls": len(grouped), "conflicting_identities": len(conflicts), "conflict_source_records": sum(map(len, conflicts.values())), "same_label_duplicate_identities": sum(len(m) > 1 for url, m in grouped.items() if url not in conflicts), "duplicate_source_records_collapsed": sum(len(m) - 1 for url, m in grouped.items() if url not in conflicts and m[0]["status"] != "missing_leakage_group_excluded"), "mapping_status_counts": dict(sorted(Counter(r["mapping_status"] for r in mapping).items())), "exclusion_reason_counts": dict(sorted(Counter(r["reason"] for r in exclusions).items())), "canonical_count": len(canonical), "canonical_label_counts": dict(sorted(Counter(r["label"] for r in canonical).items())), "leakage_group_count": len({r["leakage_group"] for r in canonical})}
    sizes = Counter(r["leakage_group"] for r in canonical)
    diagnostics["leakage_group_size_distribution"] = {"singletons": sum(n == 1 for n in sizes.values()), "size_2_to_9": sum(2 <= n <= 9 for n in sizes.values()), "size_10_plus": sum(n >= 10 for n in sizes.values()), "max": max(sizes.values(), default=0)}
    return tables, diagnostics


def assignments(canonical: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    canonical = sorted(canonical, key=lambda r: r["canonical_url_id"])
    index = list(range(len(canonical)))
    labels = [r["label"] for r in canonical]
    groups = [r["leakage_group"] for r in canonical]
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(index, labels, groups))
    holdout = set(outer[0][1])
    dev = [i for i in index if i not in holdout]
    split_rows = [{"url_run_id": row["url_run_id"], "canonical_url_id": row["canonical_url_id"], "label": row["label"], "leakage_group": row["leakage_group"], "split": "holdout" if i in holdout else "development"} for i, row in enumerate(canonical)]
    inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(dev, [labels[i] for i in dev], [groups[i] for i in dev]))
    cv_rows = []
    for fold, (_, validation) in enumerate(inner):
        for j in validation:
            row = canonical[dev[j]]
            cv_rows.append({"url_run_id": row["url_run_id"], "canonical_url_id": row["canonical_url_id"], "label": row["label"], "leakage_group": row["leakage_group"], "cv_fold": str(fold)})
    cv_rows.sort(key=lambda row: row["canonical_url_id"])
    if len(cv_rows) != len(dev) or len({r["canonical_url_id"] for r in cv_rows}) != len(dev):
        raise ValueError("Incomplete Development CV")
    for subset in ("development", "holdout"):
        if {r["label"] for r in split_rows if r["split"] == subset} != {"legitimate", "phishing"}:
            raise ValueError("Outer split lacks a class")
    for fold in range(5):
        if {r["label"] for r in cv_rows if r["cv_fold"] == str(fold)} != {"legitimate", "phishing"}:
            raise ValueError("Development fold lacks a class")
    if {r["leakage_group"] for r in split_rows if r["split"] == "development"} & {r["leakage_group"] for r in split_rows if r["split"] == "holdout"}:
        raise ValueError("Outer group leakage")
    by_group = defaultdict(set)
    for row in cv_rows:
        by_group[row["leakage_group"]].add(row["cv_fold"])
    if any(len(v) != 1 for v in by_group.values()):
        raise ValueError("CV group leakage")
    counts = {"split_counts": dict(sorted(Counter(r["split"] for r in split_rows).items())), "split_label_counts": {s: dict(sorted(Counter(r["label"] for r in split_rows if r["split"] == s).items())) for s in ("development", "holdout")}, "cv_fold_counts": {str(k): sum(r["cv_fold"] == str(k) for r in cv_rows) for k in range(5)}, "cv_fold_label_counts": {str(k): dict(sorted(Counter(r["label"] for r in cv_rows if r["cv_fold"] == str(k)).items())) for k in range(5)}}
    return split_rows, cv_rows, counts


def run_prepare(root: Path = REPOSITORY_ROOT) -> dict:
    start = time.perf_counter()
    protected = check_inputs(root)
    protected_seconds = time.perf_counter() - start
    with zipfile.ZipFile(root / ARCHIVE) as archive:
        sql = archive.read(SQL_NAME)
        readme = archive.read(README_NAME)
    if b'non-fraudulent (0)' not in readme or b'websites with this state are real phishing websites' not in readme:
        raise ValueError("Source label semantics changed")
    rows, _ = source_rows(sql.decode("utf-8", errors="replace"))
    source_seconds = time.perf_counter() - start - protected_seconds
    input_hashes = {ARCHIVE.as_posix(): digest(root / ARCHIVE), SQL_NAME: hashlib.sha256(sql).hexdigest(), README_NAME: hashlib.sha256(readme).hexdigest()}
    audit_report = json.loads((root / "outputs/datasets/url/audit/report.json").read_text(encoding="utf-8"))
    tables, counts = prepare(rows, "")
    prepare_seconds = time.perf_counter() - start - protected_seconds - source_seconds
    split_rows, cv_rows, split_counts = assignments(tables["url_dataset.csv"])
    split_seconds = time.perf_counter() - start - protected_seconds - source_seconds - prepare_seconds
    tables["split_assignments.csv"] = split_rows
    tables["cv_assignments.csv"] = cv_rows
    counts.update(split_counts)
    canonical_identity = {
        "version": CANONICAL_VERSION,
        "source": input_hashes,
        "audit_semantic_sha256": audit_report["semantic_sha256"],
        "normalization": "step21-normalize-url-unchanged-v1",
        "label_mapping": "isPhish=0 legitimate; isPhish=1,state=3 phishing; otherwise excluded",
        "conflict_handling": "exclude-entire-normalized-url-identity",
        "leakage_group": "source-urlBasedomain-when-host-consistent-else-literal-ip-else-exclude",
        "outer_split": {"algorithm": "StratifiedGroupKFold", "n_splits": 5,
                        "shuffle": True, "random_state": 42, "holdout_fold": 0},
        "development_cv": {"algorithm": "StratifiedGroupKFold", "n_splits": 5,
                           "shuffle": True, "random_state": 42},
        "counts": counts,
    }
    run_id, canonical_semantic_sha = content_run_id("phishload-canonical", canonical_identity)
    for rows_for_artifact in tables.values():
        for row in rows_for_artifact:
            row["url_run_id"] = run_id
    parent = root / CANONICAL_OUT.parent
    parent.mkdir(parents=True, exist_ok=True)
    destination = root / CANONICAL_OUT
    if destination.exists():
        raise ValueError("Step-21B output already exists; refusing overwrite")
    staging = Path(tempfile.mkdtemp(prefix="step21b-staging-", dir=parent))
    try:
        STEP21_INPUTS = {**{spec["path"]: spec["sha256"] for spec in audit_report["outputs"].values()},
                         "outputs/datasets/url/audit/report.json": digest(root / "outputs/datasets/url/audit/report.json")}
        columns = {"url_dataset.csv": DATASET_COLUMNS, "source_record_mapping.csv": MAPPING_COLUMNS, "exclusions.csv": CANONICAL_EXCLUSION_COLUMNS, "conflict_resolution.csv": CANONICAL_CONFLICT_COLUMNS, "split_assignments.csv": ASSIGNMENT_COLUMNS, "cv_assignments.csv": CV_COLUMNS}
        for name, fields in columns.items():
            write_csv(staging / name, fields, tables[name])
        report = {"status": "completed", "url_run_id": run_id, "source": "phishload", "source_role": "url_only", "raw_source_sha256": input_hashes, "step21_inputs_sha256": STEP21_INPUTS, "protected_sha256": protected, "source_schema": list(rows[0]), "source_semantics": "Phishload README: isPhish=0 non-fraudulent at scan/capture time only; isPhish=1 and state=3 verified phishing website", "candidate_mapping": {"isPhish=0;state=<null>": "legitimate", "isPhish=1;state=3": "phishing", "otherwise": "ambiguous"}, "conflict_resolution": "exclude entire normalized URL identity with both labels", "same_label_duplicate_rule": "one canonical URL identity; retain all original source records", "normalization": "Step-21 normalize_url unchanged; whitespace trim, HTTP(S), lowercase scheme/IDNA host, terminal host dot and fragment removed, userinfo/port/path/query retained", "canonical_id": "phishload-canonical:SHA256(UTF-8(step21b-canonical-url-v1 US phishload US normalized_url))", "representative_raw_url": "lexicographically smallest raw URL among same-label members; provenance only", "leakage_group": "source-native urlBasedomain if nonempty and host-consistent; otherwise literal IP host; otherwise exclude missing_leakage_group", "source_member_selection": "lexicographically smallest website ID", "counts": counts, "outer_split": {"algorithm": "StratifiedGroupKFold", "parameters": {"n_splits": 5, "shuffle": True, "random_state": 42}, "holdout_fold": 0}, "development_cv": {"algorithm": "StratifiedGroupKFold", "parameters": {"n_splits": 5, "shuffle": True, "random_state": 42}}, "environment": {"python": platform.python_version(), "scikit_learn": importlib.metadata.version("scikit-learn"), "domain_parser": "source-native Phishload urlBasedomain; ipaddress stdlib; no offline PSL package"}, "processing_seconds": {"protected_check": round(protected_seconds, 6), "source_read_parse": round(source_seconds, 6), "canonical_prepare": round(prepare_seconds, 6), "split_and_cv": round(split_seconds, 6), "total_before_report": round(time.perf_counter() - start, 6)}, **CANONICAL_FLAGS}
        report["contract_version"] = "artifact-provenance-v1:phishload-canonical-v1"
        report["semantic_sha256"] = canonical_semantic_sha
        report["predecessor"] = predecessor_reference(root, root / "outputs/datasets/url/audit/report.json")
        report["outputs"] = {name: {"path": (CANONICAL_OUT / name).as_posix(), "sha256": digest(staging / name)} for name in columns}
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        from scripts.validate_url_dataset import validate

        validate(root, staging)
        check_inputs(root)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit or prepare the Phishload URL dataset.")
    parser.add_argument("command", choices=("audit", "prepare"))
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = run_audit()
        print(f"Step 21 {result['status']}: {result['label_conflict_count']} unresolved normalized URL conflicts")
    else:
        result = run_prepare()
        print(f"Step 21B {result['status']}: {result['counts']['canonical_count']} canonical URL identities")


if __name__ == "__main__":
    main()
