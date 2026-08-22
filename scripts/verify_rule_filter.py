"""Generate out/rule_filter_test.log — original stats + 5 corrupted cases."""

from __future__ import annotations

# Ensure utf-8 stdout on Windows
import contextlib
import copy
import json
import pathlib
import sys

with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.rule_filter import evaluate_entries, run_all_rules


def main() -> None:
    in_path = pathlib.Path("data/synthetic_qa.jsonl")
    log_path = pathlib.Path("out/rule_filter_test.log")
    report_path = pathlib.Path("out/rule_report.json")

    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}")

    entries = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Original evaluation
    report = evaluate_entries(entries)

    # Corrupted copy — 5 intentional corruptions
    corrupted = copy.deepcopy(entries)
    # 5: source_chunks empty
    corrupted[5]["source_chunks"] = []
    # 10: empty answer
    corrupted[10]["reference_answer"] = ""
    # 15: fake numeric
    corrupted[15]["reference_answer"] = "매출 999조 원, 영업이익 888조 원 달성"
    # 20: keyword missing
    corrupted[20]["reference_answer"] = "이 답변은 질문과 전혀 관계없는 내용입니다. 사과는 빨갛고 바나나는 노랗습니다."
    # 25: citation doc_id deletion
    if corrupted[25].get("source_chunks") and len(corrupted[25]["source_chunks"]) > 0:
        corrupted[25]["source_chunks"][0].pop("doc_id", None)

    # Run CLI report for original (also write out/rule_report.json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Corrupted per-index check
    targets = {
        5: "source_chunks empty array",
        10: 'reference_answer = ""',
        15: 'reference_answer = fake numeric "매출 999조 원"',
        20: "keyword missing answer",
        25: "citation/doc_id deletion",
    }

    corrupted_results: dict[int, list] = {}
    all_caught = True
    for idx, _desc in targets.items():
        res = run_all_rules(corrupted[idx])
        fails = [r for r in res if not r.passed]
        corrupted_results[idx] = fails
        if not fails:
            all_caught = False

    # Verify negative test conditions
    if report["failed"] == 0 and report["passed"] == report["total"]:
        # All passed would be FAIL per spec
        print("FAIL: all cases passed — negative test required", file=sys.stderr)
        all_caught = False

    if not all_caught:
        print("FAIL: not all 5 corrupted cases were caught", file=sys.stderr)
        # still write log but exit 1
        exit_code = 1
    else:
        exit_code = 0

    # Write log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("=== Rule Filter Test Log ===\n")
        log.write(f"Input: {in_path} ({len(entries)} entries)\n")
        log.write(f"Original total: {report['total']} passed: {report['passed']} failed: {report['failed']}\n")
        if report["taxonomy"]:
            log.write("Original taxonomy (failures by rule):\n")
            for rule, items in report["taxonomy"].items():
                log.write(f"  {rule}: {len(items)} cases\n")
                for item in items[:3]:
                    log.write(f"    - index {item['index']}: {item['reason'][:120]}\n")
                if len(items) > 3:
                    log.write(f"    ... and {len(items)-3} more\n")
        else:
            log.write("Original taxonomy: all passed (WARNING — negative test requires at least one fail)\n")
        log.write("\n")
        log.write("--- Corrupted 5 cases verification ---\n")
        for idx, desc in targets.items():
            fails = corrupted_results[idx]
            q = corrupted[idx].get("question", "")[:80]
            log.write(f"[{idx}] {desc}\n")
            log.write(f"  question: {q}\n")
            if fails:
                log.write(f"  CAUGHT: {len(fails)} rule(s) failed -> {[r.rule for r in fails]}\n")
                for r in fails:
                    log.write(f"    - {r.rule}: {r.reason}\n")
            else:
                log.write("  NOT CAUGHT: 0 failures — ERROR\n")
            log.write("\n")
        log.write(f"Result: {'PASS' if all_caught else 'FAIL'} — 5/5 corrupted caught: {all_caught}\n")
        if report["failed"] == 0:
            log.write("WARNING: original had 0 failures — negative test would be invalid\n")
        log.write(f"Report generated: {report_path}\n")

    # Also print to stdout
    print(log_path.read_text(encoding="utf-8"))
    print(f"\nReport written to {report_path}")
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
