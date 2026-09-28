from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from sklearn.model_selection import StratifiedGroupKFold

from src.load_data import PHISHLOAD_WEBSITE_COLUMNS, _extract_insert_values, _parse_sql_insert_rows
from src.utils import REPOSITORY_ROOT
from src.execution_contract import current_contract
from src.provenance import content_run_id, validate_predecessor

csv.field_size_limit(sys.maxsize)
SCHEMAS = {
    "url_dataset.csv": ["url_run_id", "canonical_url_id", "normalized_url", "representative_raw_url", "label", "leakage_group", "source", "source_role", "label_provenance", "source_record_count"],
    "source_record_mapping.csv": ["url_run_id", "canonical_url_id", "source_record_id", "website_id", "raw_url", "normalized_url", "candidate_label", "isPhish", "state", "scanned", "created", "parent", "urlBasedomain", "mapping_status"],
    "exclusions.csv": ["url_run_id", "source_record_id", "website_id", "raw_url", "normalized_url", "candidate_label", "reason"],
    "conflict_resolution.csv": ["url_run_id", "conflict_id", "normalized_url", "involved_source_record_count", "observed_labels", "resolution", "excluded_from_supervised"],
    "split_assignments.csv": ["url_run_id", "canonical_url_id", "label", "leakage_group", "split"],
    "cv_assignments.csv": ["url_run_id", "canonical_url_id", "label", "leakage_group", "cv_fold"],
}
STATUS = {"canonical_member", "same_label_duplicate_member", "conflicting_ground_truth_excluded", "ambiguous_label_excluded", "invalid_url_excluded", "missing_leakage_group_excluded"}
REASONS = {"conflicting_ground_truth_excluded": "conflicting_ground_truth", "ambiguous_label_excluded": "ambiguous_label", "invalid_url_excluded": "invalid_url", "missing_leakage_group_excluded": "missing_leakage_group"}


def check(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path, schema: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == schema, f"Unexpected schema: {path}")
        return list(reader)


def normalize(value: str) -> str:
    value = value.strip()
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return ""
    try:
        p = urlsplit(value)
        if p.scheme.lower() not in {"http", "https"} or not p.netloc or not p.hostname:
            return ""
        host = p.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or any(c.isspace() for c in host):
            return ""
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if not all(part and len(part) <= 63 for part in host.split(".")):
                return ""
        port = p.port
        user = p.netloc.rsplit("@", 1)[0] + "@" if "@" in p.netloc else ""
        return urlunsplit((p.scheme.lower(), user + (f"[{host}]" if ":" in host else host) + (f":{port}" if port is not None else ""), p.path, p.query, ""))
    except (ValueError, UnicodeError):
        return ""


def source_group(url: str, members: list[dict[str, str]]) -> str:
    host = urlsplit(url).hostname or ""
    bases = {r["urlBasedomain"].strip().lower().rstrip(".") for r in members if r["urlBasedomain"].strip()}
    check(len(bases) <= 1 and all(host == base or host.endswith("." + base) for base in bases), "Invalid source base domain")
    if bases:
        return next(iter(bases))
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        return ""


def validate(root: Path = REPOSITORY_ROOT, output_dir: Path | None = None) -> dict:
    contract = current_contract(root)
    output_dir = output_dir or root / "outputs/datasets/url/canonical"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    check(report["status"] == "completed" and report["source"] == "phishload" and report["source_role"] == "url_only", "Completion/source status")
    false_flags = ("automatic_relabeling_performed", "majority_vote_performed", "time_based_relabeling_performed", "model_training_performed", "calibration_performed", "threshold_tuned", "fusion_performed", "holdout_evaluated", "external_data_used", "email_text_used", "website_html_used_as_email_text", "model_representation_selected", "step22_authorized")
    true_flags = ("conflicting_identities_resolved_by_exclusion", "canonical_url_dataset_published", "canonical_split_published", "canonical_cv_published")
    check(all(report.get(k) is False for k in false_flags) and all(report.get(k) is True for k in true_flags), "Method flags")
    check(report["outer_split"] == {"algorithm": "StratifiedGroupKFold", "parameters": {"n_splits": 5, "shuffle": True, "random_state": 42}, "holdout_fold": 0} and report["development_cv"] == {"algorithm": "StratifiedGroupKFold", "parameters": {"n_splits": 5, "shuffle": True, "random_state": 42}}, "Split configuration")
    step21_report = json.loads((root / "outputs/datasets/url/audit/report.json").read_text(encoding="utf-8"))
    validate_predecessor(root, report["predecessor"])
    step21_inputs = {**{spec["path"]: spec["sha256"] for spec in step21_report["outputs"].values()},
                     "outputs/datasets/url/audit/report.json": sha(root / "outputs/datasets/url/audit/report.json")}
    expected_protected = {**step21_report["protected_sha256"], **step21_inputs}
    check(report["protected_sha256"] == expected_protected and report["step21_inputs_sha256"] == step21_inputs, "Protected inventory")
    for name, value in expected_protected.items():
        check(sha(root / name) == value, f"Protected file hash: {name}")
    check(step21_report["status"] == "blocked" and step21_report["label_conflict_count"] == 2 and step21_report["counts"]["label_conflict_record_count"] == 4 and step21_report["label_semantics_verified_from_readme"] is True, "Step-21 precondition")
    archive = root / "data/raw/phishload/current.zip"
    with zipfile.ZipFile(archive) as source:
        check(set(source.namelist()) == {"phishload_120322.sql", "phishload_readme.txt"}, "Raw members")
        sql, readme = source.read("phishload_120322.sql"), source.read("phishload_readme.txt")
    sources = {"data/raw/phishload/current.zip": sha(archive), "phishload_120322.sql": hashlib.sha256(sql).hexdigest(), "phishload_readme.txt": hashlib.sha256(readme).hexdigest()}
    check(report["raw_source_sha256"] == sources == step21_report["source_files_sha256"], "Raw source hash")
    raw = _parse_sql_insert_rows(_extract_insert_values(sql.decode("utf-8", errors="replace"), "websites"))
    check(all(len(values) == len(PHISHLOAD_WEBSITE_COLUMNS) for values in raw), "Raw source schema")
    source_rows = [{field: "" if value is None else str(value) for field, value in zip(PHISHLOAD_WEBSITE_COLUMNS, values)} for values in raw]
    by_source = {row["id"]: row for row in source_rows}
    expected_raw = len(source_rows) if contract.is_fixture else contract.expected("phishload_raw_count", 11215)
    check(len(source_rows) == len(by_source) == expected_raw and all(by_source), "Raw source ID coverage")
    tables = {name: read(output_dir / name, schema) for name, schema in SCHEMAS.items()}
    check(set(report["outputs"]) == set(SCHEMAS), "Six CSV output inventory")
    for name in SCHEMAS:
        check(report["outputs"][name] == {"path": "outputs/datasets/url/canonical/" + name, "sha256": sha(output_dir / name)}, f"Output hash: {name}")
    canonical_identity = {
        "version": "step21b-canonical-url-v1",
        "source": sources,
        "audit_semantic_sha256": step21_report["semantic_sha256"],
        "normalization": "step21-normalize-url-unchanged-v1",
        "label_mapping": "isPhish=0 legitimate; isPhish=1,state=3 phishing; otherwise excluded",
        "conflict_handling": "exclude-entire-normalized-url-identity",
        "leakage_group": "source-urlBasedomain-when-host-consistent-else-literal-ip-else-exclude",
        "outer_split": {"algorithm": "StratifiedGroupKFold", "n_splits": 5,
                        "shuffle": True, "random_state": 42, "holdout_fold": 0},
        "development_cv": {"algorithm": "StratifiedGroupKFold", "n_splits": 5,
                           "shuffle": True, "random_state": 42},
        "counts": report["counts"],
    }
    run_id, semantic_sha = content_run_id("phishload-canonical", canonical_identity)
    check(report["url_run_id"] == run_id and report["semantic_sha256"] == semantic_sha and all(r["url_run_id"] == run_id for rows in tables.values() for r in rows), "Run ID mismatch")
    mapping = tables["source_record_mapping.csv"]
    by_mapping = {row["website_id"]: row for row in mapping}
    check(len(mapping) == len(by_mapping) == len(source_rows) and set(by_mapping) == set(by_source), "Exactly one mapping row per source ID")
    grouped = defaultdict(list)
    for id_, raw_row in by_source.items():
        stored = by_mapping[id_]
        normalized = normalize(raw_row["url"])
        label = "legitimate" if raw_row["isPhish"] == "0" and not raw_row["state"] else "phishing" if raw_row["isPhish"] == "1" and raw_row["state"] == "3" else ""
        for field, value in {"source_record_id": "phishload:website:" + id_, "raw_url": raw_row["url"], "normalized_url": normalized, "candidate_label": label, "isPhish": raw_row["isPhish"], "state": raw_row["state"], "scanned": raw_row["scanned"], "created": raw_row["created"], "parent": raw_row["parent"], "urlBasedomain": raw_row["urlBasedomain"]}.items():
            check(stored[field] == value, f"Mapping source provenance: {id_}:{field}")
        check(stored["mapping_status"] in STATUS, "Unknown final status")
        if normalized and label:
            grouped[normalized].append((raw_row, label))
    conflicts = {url: members for url, members in grouped.items() if len({label for _, label in members}) > 1}
    check(len(conflicts) == 2 and sum(map(len, conflicts.values())) == 4, "Source conflicts changed")
    step21_conflicts = read(root / "outputs/datasets/url/audit/label_conflicts.csv", list(next(csv.DictReader((root / "outputs/datasets/url/audit/label_conflicts.csv").open(encoding="utf-8", newline=""))).keys()))
    check({(r["normalized_url"], r["website_id"]) for r in step21_conflicts} == {(url, row["id"]) for url, members in conflicts.items() for row, _ in members}, "Step-21 conflict identity coverage")
    expected_status = {}
    expected_group = {}
    for id_, raw_row in by_source.items():
        normalized = normalize(raw_row["url"])
        label = "legitimate" if raw_row["isPhish"] == "0" and not raw_row["state"] else "phishing" if raw_row["isPhish"] == "1" and raw_row["state"] == "3" else ""
        if not normalized:
            expected_status[id_] = "invalid_url_excluded"
        elif not label:
            expected_status[id_] = "ambiguous_label_excluded"
        elif normalized in conflicts:
            expected_status[id_] = "conflicting_ground_truth_excluded"
    for url, members in grouped.items():
        if url in conflicts:
            continue
        rows = [row for row, _ in members]
        group = source_group(url, rows)
        if not group:
            for row in rows:
                expected_status[row["id"]] = "missing_leakage_group_excluded"
            continue
        chosen = min(row["id"] for row in rows)
        expected_group[url] = group
        for row in rows:
            expected_status[row["id"]] = "canonical_member" if row["id"] == chosen else "same_label_duplicate_member"
    check(len(expected_status) == len(source_rows) and all(by_mapping[id_]["mapping_status"] == status for id_, status in expected_status.items()), "Final status mismatch")
    status_counts = dict(sorted(Counter(expected_status.values()).items()))
    check(sum(status_counts.values()) == len(source_rows) and report["counts"]["mapping_status_counts"] == status_counts, "Status count reconciliation")
    canonical = tables["url_dataset.csv"]
    by_url = {r["normalized_url"]: r for r in canonical}
    ids = [r["canonical_url_id"] for r in canonical]
    check(len(canonical) == len(by_url) == len(set(ids)) == len(expected_group), "Canonical identity uniqueness/coverage")
    for url, group in expected_group.items():
        r = by_url[url]
        members = grouped[url]
        expected_id = "phishload-canonical:" + hashlib.sha256("\x1f".join(("step21b-canonical-url-v1", "phishload", url)).encode()).hexdigest()
        check(r["canonical_url_id"] == expected_id and r["leakage_group"] == group and r["label"] == members[0][1], "Canonical ID/group/label")
        check(r["representative_raw_url"] == min(m["url"] for m, _ in members) and int(r["source_record_count"]) == len(members), "Raw URL representative or multiplicity")
        check(r["source"] == "phishload" and r["source_role"] == "url_only", "Canonical source role")
    for id_, row in by_mapping.items():
        url = row["normalized_url"]
        expected_id = by_url[url]["canonical_url_id"] if row["mapping_status"] in {"canonical_member", "same_label_duplicate_member"} else ""
        check(row["canonical_url_id"] == expected_id, "Canonical membership of source record")
    exclusions = tables["exclusions.csv"]
    check(len(exclusions) == len({r["website_id"] for r in exclusions}) == sum(s in REASONS for s in expected_status.values()), "Exclusion coverage")
    for row in exclusions:
        source_id = row["website_id"]
        check(expected_status[source_id] in REASONS and row["reason"] == REASONS[expected_status[source_id]] and row["source_record_id"] == by_mapping[source_id]["source_record_id"] and row["raw_url"] == by_mapping[source_id]["raw_url"] and row["normalized_url"] == by_mapping[source_id]["normalized_url"] and row["candidate_label"] == by_mapping[source_id]["candidate_label"], "Exclusion provenance/reason")
    check(report["counts"]["exclusion_reason_counts"] == dict(sorted(Counter(r["reason"] for r in exclusions).items())), "Exclusion report counts")
    resolution = tables["conflict_resolution.csv"]
    check(len(resolution) == len(conflicts) and {r["normalized_url"] for r in resolution} == set(conflicts), "Conflict resolution coverage")
    for r in resolution:
        url = r["normalized_url"]
        check(r["conflict_id"] == "url-conflict:" + hashlib.sha256(url.encode()).hexdigest()[:20] and r["involved_source_record_count"] == str(len(conflicts[url])) and r["observed_labels"] == "legitimate|phishing" and r["resolution"] == "exclude_entire_identity_due_conflicting_ground_truth" and r["excluded_from_supervised"] == "true" and url not in by_url, "Conflict resolution evidence")
    counts = report["counts"]
    check(counts["source_records"] == len(source_rows) and counts["valid_candidate_source_rows"] == sum(len(m) for m in grouped.values()) and counts["unique_candidate_normalized_urls"] == len(grouped) and counts["conflicting_identities"] == len(conflicts) and counts["conflict_source_records"] == sum(map(len, conflicts.values())), "Candidate/conflict report counts")
    check(counts["same_label_duplicate_identities"] == sum(len(m) > 1 for url, m in grouped.items() if url not in conflicts) and counts["duplicate_source_records_collapsed"] == sum(len(grouped[url]) - 1 for url in expected_group if len(grouped[url]) > 1), "Duplicate report counts")
    check(counts["canonical_count"] == len(canonical) and counts["canonical_label_counts"] == dict(sorted(Counter(r["label"] for r in canonical).items())) and counts["leakage_group_count"] == len(set(expected_group.values())), "Canonical report counts")
    sizes = Counter(expected_group.values())
    check(counts["leakage_group_size_distribution"] == {"singletons": sum(n == 1 for n in sizes.values()), "size_2_to_9": sum(2 <= n <= 9 for n in sizes.values()), "size_10_plus": sum(n >= 10 for n in sizes.values()), "max": max(sizes.values(), default=0)}, "Group-size report counts")
    ordered = sorted(canonical, key=lambda r: r["canonical_url_id"])
    idx = list(range(len(ordered)))
    y = [r["label"] for r in ordered]
    g = [r["leakage_group"] for r in ordered]
    outer = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(idx, y, g))
    holdout = set(outer[0][1])
    expected_split = {row["canonical_url_id"]: "holdout" if i in holdout else "development" for i, row in enumerate(ordered)}
    split = tables["split_assignments.csv"]
    check(len(split) == len({r["canonical_url_id"] for r in split}) == len(canonical), "Split ID coverage")
    for row in split:
        base = next((item for item in ordered if item["canonical_url_id"] == row["canonical_url_id"]), None)
        check(base is not None and row["split"] == expected_split[row["canonical_url_id"]] and row["label"] == base["label"] and row["leakage_group"] == base["leakage_group"], "Outer fold assignment")
    split_groups = {name: {r["leakage_group"] for r in split if r["split"] == name} for name in ("development", "holdout")}
    check(not split_groups["development"] & split_groups["holdout"] and all({r["label"] for r in split if r["split"] == name} == {"legitimate", "phishing"} for name in split_groups), "Outer group/class leakage")
    dev = [i for i in idx if i not in holdout]
    inner = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(dev, [y[i] for i in dev], [g[i] for i in dev]))
    expected_cv = {ordered[dev[j]]["canonical_url_id"]: str(k) for k, (_, val) in enumerate(inner) for j in val}
    cv = tables["cv_assignments.csv"]
    check(len(cv) == len({r["canonical_url_id"] for r in cv}) == len(dev) and {r["canonical_url_id"] for r in cv} == set(expected_cv), "CV Development ID coverage")
    fold_groups = defaultdict(set)
    for row in cv:
        base = by_url[next(r["normalized_url"] for r in canonical if r["canonical_url_id"] == row["canonical_url_id"])]
        check(row["cv_fold"] == expected_cv[row["canonical_url_id"]] and row["label"] == base["label"] and row["leakage_group"] == base["leakage_group"], "CV fold assignment")
        fold_groups[row["cv_fold"]].add(row["leakage_group"])
    check(set(fold_groups) == set("01234") and all({r["label"] for r in cv if r["cv_fold"] == fold} == {"legitimate", "phishing"} for fold in fold_groups), "CV fold class coverage")
    check(all(not fold_groups[a] & fold_groups[b] for a in fold_groups for b in fold_groups if a < b), "CV group leakage")
    check(counts["split_counts"] == dict(sorted(Counter(r["split"] for r in split).items())) and counts["split_label_counts"] == {name: dict(sorted(Counter(r["label"] for r in split if r["split"] == name).items())) for name in ("development", "holdout")}, "Split report counts")
    check(counts["cv_fold_counts"] == {str(k): sum(r["cv_fold"] == str(k) for r in cv) for k in range(5)} and counts["cv_fold_label_counts"] == {str(k): dict(sorted(Counter(r["label"] for r in cv if r["cv_fold"] == str(k)).items())) for k in range(5)}, "CV report counts")
    return report


if __name__ == "__main__":
    result = validate()
    print(f"validated Step 21B: {result['counts']['canonical_count']} canonical URL identities")
