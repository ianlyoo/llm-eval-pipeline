"""Both judge entry points work in a source checkout without an installed package."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_judge(invocation, *args):
    # -S disables installed packages; an allowlisted environment removes inherited
    # PYTHONPATH and provider credentials without breaking Windows startup.
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR") if key in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-S", *invocation, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )


@pytest.mark.parametrize("invocation", [("eval/llm_judge.py",), ("-m", "eval.llm_judge")])
def test_help_from_uninstalled_source_checkout(invocation):
    result = _run_judge(invocation, "--help")

    assert result.returncode == 0, result.stderr
    assert "--input" in result.stdout
    assert "--llm-provider" in result.stdout
    assert "--reliability" in result.stdout


def test_standalone_and_module_scoring_match_without_provider_calls(tmp_path):
    entry = {
        "question": "How long is support available?",
        "reference_answer": "Support is available for 7 days.",
        "source_chunks": [{"doc_id": "fixture", "chunk_id": "fixture-0",
                           "text": "Support is available for 7 days."}],
    }
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    results = []
    for name, invocation in (
        ("script", ("eval/llm_judge.py",)),
        ("module", ("-m", "eval.llm_judge")),
    ):
        output = tmp_path / f"{name}.jsonl"
        result = _run_judge(
            invocation, "--input", str(source), "--output", str(output),
            "--sample", "1", "--llm-provider", "none",
        )
        assert result.returncode == 0, result.stderr
        scored = json.loads(output.read_text(encoding="utf-8"))
        assert scored.pop("latency_ms") >= 1
        assert scored["cost_est"] == 0
        assert scored["candidate_answer"] == entry["reference_answer"]
        results.append(scored)

    assert results[0] == results[1]
