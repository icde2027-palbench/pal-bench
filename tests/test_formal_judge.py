import json

from src.benchmark.eval.judge import llm_owner_fact_judge, llm_person_value_judge
from src.llm.base import LLMResponse


class QueueLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": dict(kwargs),
            }
        )
        return LLMResponse(content=self.responses.pop(0), model="fake")


def test_owner_fact_judge_retries_invalid_json() -> None:
    llm = QueueLLM(
        [
            "I think the answer is match, but this is not JSON.",
            json.dumps(
                {
                    "verdict": "match",
                    "score": 1.0,
                    "rationale": "The prediction states the same hobby.",
                    "matched_index": 0,
                }
            ),
        ]
    )

    result = llm_owner_fact_judge(
        "The owner likes hiking.",
        ["The album owner enjoys hiking on weekends."],
        llm,
        allow_fallback=False,
    )

    assert result.verdict == "match"
    assert result.score == 1.0
    assert result.matched_index == 0
    assert len(llm.calls) == 2
    assert llm.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}


def test_owner_fact_judge_repairs_plain_text_verdict() -> None:
    llm = QueueLLM(
        [
            "plain text, no verdict",
            "Verdict: partial. Score: 0.5. matched_index: 1. Rationale: broader than GT.",
        ]
    )

    result = llm_owner_fact_judge(
        "The owner lives in Miami.",
        ["The owner lives in Florida.", "The owner is based somewhere in South Florida."],
        llm,
        allow_fallback=False,
    )

    assert result.verdict == "partial"
    assert result.score == 0.5
    assert result.matched_index == 1


def test_person_value_judge_retries_invalid_json() -> None:
    llm = QueueLLM(
        [
            "not valid json",
            json.dumps(
                {
                    "verdict": "partial",
                    "score": 0.5,
                    "rationale": "The prediction gives a broader relation.",
                }
            ),
        ]
    )

    result = llm_person_value_judge(
        "mother",
        "family member",
        "person_relation",
        None,
        llm,
    )

    assert result.verdict == "partial"
    assert result.score == 0.5
    assert len(llm.calls) == 2
