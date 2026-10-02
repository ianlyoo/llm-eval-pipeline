"""Judge entry handling must agree with the deterministic scorer's fallbacks."""

import json
import sys
from types import SimpleNamespace

import pytest

from eval.llm_judge import deterministic_judge, judge_entry, try_llm_judge


@pytest.mark.parametrize("candidate", [None, 17, [], "", "   "])
def test_invalid_candidate_uses_reference_through_complete_judging(candidate):
    entry = {
        "question": "When is support available?",
        "reference_answer": "Support is available for 7 days.",
        "candidate_answer": candidate,
        "source_chunks": [{"text": "Support is available for 7 days."}],
    }

    result = judge_entry(entry)

    assert result["candidate_answer"] == entry["reference_answer"]
    assert result["scores"] == deterministic_judge(entry)
    assert result["tokens_est"] >= 10
    assert result["cost_est"] == 0


@pytest.mark.parametrize("sources", [None, 17, {}, "not a chunk list"])
def test_invalid_source_container_uses_empty_evidence(sources):
    entry = {
        "question": "When is support available?",
        "reference_answer": "Support is available for 7 days.",
        "source_chunks": sources,
    }

    result = judge_entry(entry)

    assert result["scores"] == deterministic_judge({**entry, "source_chunks": []})
    assert result["scores"]["groundedness"]["score"] == 1
    assert result["tokens_est"] >= 10


def test_provider_adapter_normalizes_inputs_before_the_mocked_call(monkeypatch):
    entry = {"question": "When?", "reference_answer": "For 7 days.",
             "candidate_answer": None, "source_chunks": None}
    expected = deterministic_judge(entry)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(expected)))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_kwargs: client))

    result = try_llm_judge(entry, "openai", "fixture-only")

    assert result == {**expected, "_tokens_est": 30}
    assert len(calls) == 1
    assert "CANDIDATE_ANSWER: For 7 days." in calls[0]["messages"][0]["content"]
