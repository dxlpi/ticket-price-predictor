#!/usr/bin/env python3
"""Agent-readiness eval harness — runs acceptance tasks and reports pass-rate.

Each task in ``tasks.json`` is a verifiable check that a coding agent's change to
this repo must keep green (docs resolve, the layer invariant holds, the module
compiles, the decision log stays indexed, ...). The headline metric is the
**pass-rate**; results are written to ``agent-results.json`` as a baseline and for
tooling/telemetry to consume.

This is a v0 structural + guard eval. A richer extension would replay recorded
agent tasks (prompt -> diff -> does it pass?) and log per-task outcomes; the JSON
schema here already carries per-task results to support that.

Usage:
    python evals/run_evals.py            # run all tasks, write agent-results.json
    python evals/run_evals.py --quiet    # summary line + JSON only

Exit status: 0 if pass-rate >= threshold (default 1.0), else 1.
Pure stdlib — no external dependencies.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
TASKS_FILE = EVAL_DIR / "tasks.json"
RESULTS_FILE = EVAL_DIR / "agent-results.json"


def run_task(task: dict) -> dict:
    command = task["command"]
    try:
        proc = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, timeout=120
        )
        passed = proc.returncode == 0
        detail = (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        passed, detail = False, "timeout"
    except Exception as exc:  # noqa: BLE001 - report any launch failure as a fail
        passed, detail = False, f"error: {exc}"
    return {
        "id": task["id"],
        "category": task.get("category", "general"),
        "description": task["description"],
        "passed": passed,
        "detail": detail[:500],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args()

    spec = json.loads(TASKS_FILE.read_text())
    threshold = float(spec.get("threshold", 1.0))
    results = [run_task(t) for t in spec["tasks"]]

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total if total else 0.0

    report = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "threshold": threshold,
        "tasks": results,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2) + "\n")

    if not args.quiet:
        for r in results:
            mark = "✓" if r["passed"] else "✗"
            print(f"  {mark} [{r['category']}] {r['id']}")
            if not r["passed"] and r["detail"]:
                print(f"      {r['detail'].splitlines()[0] if r['detail'] else ''}")
    print(f"\npass-rate {passed}/{total} = {pass_rate:.0%}  (threshold {threshold:.0%})")
    print(f"→ {RESULTS_FILE.relative_to(REPO)}")

    return 0 if pass_rate >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
