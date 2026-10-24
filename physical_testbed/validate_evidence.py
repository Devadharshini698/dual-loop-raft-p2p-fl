"""Validate physical-run evidence and fail closed on Raft safety violations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    records = _load_records(run_dir)
    faults = _load_jsonl(run_dir / "fault-events.jsonl")
    leaders: dict[int, set[str]] = defaultdict(set)
    terms: dict[str, list[int]] = defaultdict(list)
    commits: dict[int, set[str]] = defaultdict(set)
    for record in records:
        node_id, term = str(record["node_id"]), int(record.get("term", 0))
        terms[node_id].append(term)
        if record.get("role") == "LEADER":
            leaders[term].add(node_id)
        commit_index = int(record.get("commit_index", -1))
        if commit_index >= 0 and record.get("committed_checksum"):
            commits[commit_index].add(str(record["committed_checksum"]))
    checks = {
        "evidence_present": bool(records),
        "minimum_cluster_observed": len(terms) >= 3,
        "run_ids_consistent": all(record.get("run_id") == run_dir.name
                                  for record in [*records, *faults]),
        "one_leader_per_term": all(len(value) <= 1 for value in leaders.values()),
        "terms_monotonic": all(values == sorted(values) for values in terms.values()),
        "committed_logs_consistent": all(len(value) <= 1 for value in commits.values()),
        "fault_ledger_present": _valid_fault_ledger(faults),
    }
    return {
        "run_dir": str(run_dir), "status_records": len(records),
        "nodes": sorted(terms), "leader_terms": {
            str(term): sorted(value) for term, value in sorted(leaders.items())
        },
        "checks": checks, "passed": all(checks.values()),
    }


def write_report(run_dir: str | Path, result: dict[str, Any]) -> Path:
    path = Path(run_dir) / "validation-summary.md"
    rows = ["# Physical Testbed Validation", "", f"Overall: **{'PASS' if result['passed'] else 'FAIL'}**", "",
            "| Check | Result |", "|---|---|"]
    rows.extend(f"| {name.replace('_', ' ').title()} | {'PASS' if passed else 'FAIL'} |"
                for name, passed in result["checks"].items())
    rows += ["", f"Status records: {result['status_records']}",
             f"Nodes: {', '.join(result['nodes'])}", ""]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("node*.jsonl")):
        for record in _load_jsonl(path):
            if record.get("event") == "status":
                records.append(record)
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record must be an object: {path}")
        records.append(record)
    return records


def _valid_fault_ledger(records: list[dict[str, Any]]) -> bool:
    starts = {(item.get("target_node"), item.get("fault_type")) for item in records
              if item.get("event") == "fault_start" and item.get("applied") is True}
    ends = {(item.get("target_node"), item.get("fault_type")) for item in records
            if item.get("event") == "fault_end" and item.get("applied") is True}
    return bool(starts) and starts <= ends


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a physical testbed run")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    result = validate_run(args.run_dir)
    report = write_report(args.run_dir, result)
    print(json.dumps(result, indent=2))
    print(f"Report: {report}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
