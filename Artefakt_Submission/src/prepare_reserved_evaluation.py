
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from src.load_data import (
    NORMALIZED_COLUMNS, _parse_email_bytes, build_record_id,
)
from src.preprocess import normalize_text_field
from src.utils import DATA_RAW_DIR, REPOSITORY_ROOT
from src.execution_contract import current_contract

csv.field_size_limit(sys.maxsize)

EPHISH_PATH = DATA_RAW_DIR / "E-PhishLLM" / "ephishLLM.json"
POT_ROOT = DATA_RAW_DIR / "PhishingPot" / "phishing_pot"
DIDECO_PATH = DATA_RAW_DIR / "DIDECO" / "2026.lrec-1.542.OptionalSupplementaryMaterial.zip"
POT_COMMIT = "49f63777126b0bdb9eb1f6e770a5c3f9df2b0306"
ANNOTATION_COLUMNS = ["annotation_id", "message_record_id", "source_file", "source_record", "round", "annotator", "raw_message_id", "user_id", "displayed_text_empty", "explicit_intents_json", "implicit_intents_json", "raw_intent_json", "raw_annotation_json"]


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_record(dataset: str, subset: str, source_file: str, source_record: str,
                role: str, label: str, provenance: str, subject: str, body: str,
                period: str = "", raw_label: str = "") -> dict[str, str]:
    record = {column: "" for column in NORMALIZED_COLUMNS}
    record.update({
        "record_id": build_record_id(dataset=dataset, source_subset=subset,
                                     source_file=source_file, source_record=source_record),
        "dataset": dataset, "source_subset": subset, "source_file": source_file,
        "source_record": source_record, "source_role": role, "subject": normalize_text_field(subject),
        "body": normalize_text_field(body), "label": label, "raw_label": raw_label or label,
        "label_provenance": provenance, "collection_period": period,
        "inclusion_reason": "reserved evaluation only; excluded from development and selection",
    })
    return record


def load_ephish(path: Path = EPHISH_PATH) -> tuple[list[dict[str, str]], dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("E-PhishLLM root must be a JSON list")
    rows = []
    languages: Counter[str] = Counter()
    source_file = relative(path)
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"Subject", "Body", "Language", "type"}:
            raise ValueError(f"Unexpected E-PhishLLM schema at index {index}")
        language, raw_type = item["Language"], item["type"]
        if not isinstance(language, str) or type(raw_type) is not int or raw_type not in {0, 1}:
            raise ValueError(f"Unexpected E-PhishLLM language/label at index {index}")
        if not isinstance(item["Subject"], str) or not isinstance(item["Body"], str):
            raise ValueError(f"Unexpected E-PhishLLM text at index {index}")
        languages[language] += 1
        if language != "en":
            continue
        rows.append(make_record("ephishllm", "english", source_file, f"json_index={index}",
                                "external_evaluation", "legitimate" if raw_type == 0 else "phishing",
                                "source_field:type;0=legitimate;1=phishing", item["Subject"], item["Body"],
                                raw_label=str(raw_type)))
    return rows, {"raw": len(raw), "retained": len(rows), "languages": dict(sorted(languages.items())),
                  "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
                  "empty_normalized_pair": sum(not row["subject"] and not row["body"] for row in rows),
                  "source_sha256": sha256_file(path)}


def load_phishingpot(root: Path = POT_ROOT) -> tuple[list[dict[str, str]], dict]:
    commit = current_contract(REPOSITORY_ROOT).expected("phishingpot_commit", POT_COMMIT)
    files = sorted(path for path in (root / "email").rglob("*.eml") if path.is_file())
    tracked = [path.relative_to(root).as_posix() for path in files]
    if not tracked:
        raise ValueError("PhishingPot snapshot contains no .eml files")
    rows, failures = [], []
    content_digest = hashlib.sha256()
    empty_subject = empty_body = 0
    source_file = f"{relative(root)}/@{commit}"
    seen = []
    for path in files:
        local = path.relative_to(root).as_posix()
        seen.append(local)
        try:
            payload = path.read_bytes()
            content_digest.update(local.encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
            subject, body = _parse_email_bytes(payload)
        except Exception as exc:
            failures.append({"path": local, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row = make_record("phishingpot", "git_snapshot", source_file, local,
                          "external_evaluation", "phishing", "source_corpus_membership:phishing_pot",
                          subject, body)
        empty_subject += not bool(row["subject"])
        empty_body += not bool(row["body"])
        rows.append(row)
    if seen != tracked:
        raise ValueError("PhishingPot file inventory changed while reading")
    return rows, {"raw": len(tracked), "retained": len(rows), "parse_failures": failures,
                  "empty_subject": empty_subject, "empty_body": empty_body,
                  "empty_normalized_pair": sum(not row["subject"] and not row["body"] for row in rows),
                  "commit": commit, "tracked_path_list_sha256": hashlib.sha256("\n".join(tracked).encode()).hexdigest(),
                  "parsed_snapshot_content_sha256": content_digest.hexdigest()}


def load_dideco(path: Path = DIDECO_PATH) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    messages: dict[str, dict[str, str]] = {}
    annotations: list[dict[str, str]] = []
    rounds, annotators = set(), set()
    source_file = relative(path)
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".jsonl"))
        if len(members) != 30:
            raise ValueError(f"Expected 30 DIDECO JSONL files, got {len(members)}")
        for member in members:
            round_match = re.search(r"round_(\d+)", member)
            annotator_match = re.search(r"annotator(\d+)", member)
            if not round_match or not annotator_match:
                raise ValueError(f"DIDECO member lacks round/annotator: {member}")
            round_id, annotator = round_match.group(1), annotator_match.group(1)
            rounds.add(round_id)
            annotators.add(annotator)
            for index, line in enumerate(archive.read(member).decode("utf-8").splitlines(), 1):
                raw = json.loads(line)
                displayed = raw.get("displayed_text", "")
                if not isinstance(displayed, str):
                    raise ValueError(f"Non-string DIDECO displayed_text: {member}:{index}")
                body = normalize_text_field(displayed)
                message_id = ""
                if displayed.strip():
                    key = hashlib.sha256(displayed.encode("utf-8")).hexdigest()
                    if key not in messages:
                        messages[key] = make_record("dideco", "message", source_file,
                                                    f"displayed_text_sha256={key}", "rq2_evaluation",
                                                    "phishing", "source_corpus_membership:dideco;not_message_gold_label",
                                                    "", displayed)
                    message_id = messages[key]["record_id"]
                intents = raw.get("label_annotations", {}).get("intent_selection", {})
                if not isinstance(intents, dict):
                    raise ValueError(f"Unexpected DIDECO intent schema: {member}:{index}")
                explicit = {k: v for k, v in intents.items() if str(v).isdigit()}
                implicit = {k: v for k, v in intents.items() if str(v).startswith("intent_selection:::")}
                if len(explicit) + len(implicit) != len(intents):
                    raise ValueError(f"Unknown DIDECO intent encoding: {member}:{index}")
                source_record = f"member={member};line={index}"
                annotations.append({
                    "annotation_id": build_record_id(dataset="dideco_annotation", source_subset="raw",
                                                      source_file=source_file, source_record=source_record),
                    "message_record_id": message_id, "source_file": f"{source_file}!{member}",
                    "source_record": source_record, "round": round_id, "annotator": annotator,
                    "raw_message_id": str(raw.get("id", raw.get("instance_id", ""))),
                    "user_id": str(raw.get("user_id", "")),
                    "displayed_text_empty": str(not bool(displayed.strip())).lower(),
                    "explicit_intents_json": json.dumps(explicit, sort_keys=True, ensure_ascii=False),
                    "implicit_intents_json": json.dumps(implicit, sort_keys=True, ensure_ascii=False),
                    "raw_intent_json": json.dumps(intents, sort_keys=True, ensure_ascii=False),
                    "raw_annotation_json": json.dumps(raw, sort_keys=True, ensure_ascii=False),
                })
    rows = sorted(messages.values(), key=lambda row: row["record_id"])
    return rows, annotations, {"raw_annotation_rows": len(annotations), "retained_messages": len(rows),
                               "empty_displayed_text_annotation_rows": sum(a["displayed_text_empty"] == "true" for a in annotations),
                               "rounds": sorted(rounds), "annotators": sorted(annotators),
                               "jsonl_files": len(members), "source_sha256": sha256_file(path)}
