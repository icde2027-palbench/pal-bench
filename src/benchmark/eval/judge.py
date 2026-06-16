"""Judge functions for album benchmark evaluation.

Supports two modes:
1. Lexical judge (fast, deterministic, for dev iteration)
2. LLM judge (accurate, uses Qwen3.6 local, for final evaluation)

Includes:
- llm_owner_fact_judge / llm_person_value_judge — value-level correctness
- llm_reasoning_fidelity_judge — Reasoning-path Fidelity Score (RFS)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

Verdict = Literal["match", "partial", "mismatch"]
RFSVerdict = Literal["equivalent", "partial", "weak", "mismatch"]
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]{2,}")

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    verdict: Verdict
    score: float
    rationale: str
    matched_index: int | None = None
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "rationale": self.rationale,
            "matched_index": self.matched_index,
            "fallback_used": self.fallback_used,
        }


@dataclass
class RFSResult:
    """Reasoning Fidelity Score for one (target, agent reasoning_path) pair."""

    verdict: RFSVerdict
    score: float                # one of {1.0, 0.66, 0.33, 0.0}
    rationale: str
    grounded: bool = False      # True iff the agent's reasoning cites concrete evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "rationale": self.rationale,
            "grounded": self.grounded,
        }


_RFS_VERDICT_TO_SCORE: dict[str, float] = {
    "equivalent": 1.0,
    "partial":    0.66,
    "weak":       0.33,
    "mismatch":   0.0,
}


# ─── LLM Judge ────────────────────────────────────────────────────────────────

_OWNER_FACT_JUDGE_SYSTEM = """You are an evaluation judge. Given a ground truth fact about an album owner and a list of predicted facts, determine whether the ground truth is expressed or entailed by any prediction.

Rules:
- "match" (score 1.0): The prediction clearly states or directly entails the GT fact.
- "partial" (score 0.5): The prediction is in the right direction but vague, incomplete, or uses a broader term.
- "mismatch" (score 0.0): No prediction covers the GT fact, or predictions contradict it.

For names: first name OR full name match → match. Wrong name → mismatch.
For age: ±2 years tolerance → match.
For city: same city → match. Same country but different city → partial.
For occupation: same profession → match. Related field → partial.
For behaviors: same activity → match. Related activity → partial.

Output ONLY a JSON object:
{"verdict": "match|partial|mismatch", "score": 1.0|0.5|0.0, "rationale": "brief reason", "matched_index": index_or_null}"""

_PERSON_VALUE_JUDGE_SYSTEM = """You are an evaluation judge for person attributes. Given a ground truth value and a predicted value for a specific person (identified by face_id), determine if they match.

For person_name:
- Full name match or unique first name → "match" (1.0)
- Ambiguous partial name or nickname → "partial" (0.5)
- Wrong name, empty, or null → "mismatch" (0.0)

For person_relation:
- Same specific relation (partner=boyfriend/girlfriend/significant other) → "match" (1.0)
- Broader category correct (mother→family member) → "partial" (0.5)
- Wrong relation (friend vs coworker, mother vs father) → "mismatch" (0.0)

For relation_category:
- Same category → "match" (1.0)
- Defensible alternative → "partial" (0.5)
- Wrong category → "mismatch" (0.0)

Output ONLY a JSON object:
{"verdict": "match|partial|mismatch", "score": 1.0|0.5|0.0, "rationale": "brief reason"}"""


def llm_owner_fact_judge(
    gt_text: str,
    candidate_texts: list[str],
    llm: Any,
    *,
    allow_fallback: bool = True,
) -> JudgeResult:
    """Use LLM to judge whether GT owner fact is covered by predictions."""
    if not candidate_texts:
        return JudgeResult("mismatch", 0.0, "No predictions available.")

    candidates_formatted = "\n".join(f"  [{i}] {t}" for i, t in enumerate(candidate_texts))
    user_prompt = f"""Ground truth fact: "{gt_text}"

Predicted facts:
{candidates_formatted}

Is the ground truth fact expressed or entailed by any of the predictions?"""

    result = _call_judge_llm(llm, _OWNER_FACT_JUDGE_SYSTEM, user_prompt)
    if result is None:
        if not allow_fallback:
            raise RuntimeError("LLM owner fact judge failed or returned invalid JSON.")
        judge = lexical_judge(gt_text, candidate_texts)
        judge.fallback_used = True
        return judge

    verdict = str(result.get("verdict", "mismatch"))
    if verdict not in ("match", "partial", "mismatch"):
        verdict = "mismatch"
    score = {"match": 1.0, "partial": 0.5, "mismatch": 0.0}[verdict]
    rationale = str(result.get("rationale", ""))
    matched_idx = result.get("matched_index")
    if matched_idx is not None:
        try:
            matched_idx = int(matched_idx)
            if matched_idx < 0 or matched_idx >= len(candidate_texts):
                matched_idx = None
        except (TypeError, ValueError):
            matched_idx = None

    return JudgeResult(verdict, score, rationale, matched_idx)


def llm_person_value_judge(gt_value: str, pred_value: str | None, target_type: str, aliases: list[str] | None, llm: Any) -> JudgeResult:
    """Use LLM to judge person attribute match."""
    pred_text = str(pred_value or "").strip()
    if not pred_text:
        return JudgeResult("mismatch", 0.0, "No prediction for this person.")

    alias_info = f"\nAccepted aliases: {', '.join(aliases)}" if aliases else ""
    user_prompt = f"""Target type: {target_type}
Ground truth value: "{gt_value}"{alias_info}
Predicted value: "{pred_text}"

Does the prediction match the ground truth?"""

    result = _call_judge_llm(llm, _PERSON_VALUE_JUDGE_SYSTEM, user_prompt)
    if result is None:
        return value_judge(gt_value, pred_value, aliases)

    verdict = str(result.get("verdict", "mismatch"))
    if verdict not in ("match", "partial", "mismatch"):
        verdict = "mismatch"
    score = {"match": 1.0, "partial": 0.5, "mismatch": 0.0}[verdict]
    rationale = str(result.get("rationale", ""))

    return JudgeResult(verdict, score, rationale)


def _call_judge_llm(llm: Any, system: str, user_prompt: str) -> dict[str, Any] | None:
    """Call LLM judge and parse JSON response with one strict-format retry."""
    if llm is None:
        return None

    last_raw = ""
    for attempt in range(2):
        actual_prompt = user_prompt
        if attempt:
            actual_prompt = (
                user_prompt
                + "\n\nIMPORTANT: Return ONLY one valid JSON object matching the requested schema. "
                "No markdown, no prose outside JSON."
            )
        try:
            raw = _llm_json_text(llm, system=system, user_prompt=actual_prompt, max_tokens=256)
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            continue

        last_raw = str(raw or "")
        parsed = _extract_json(last_raw)
        if isinstance(parsed, dict):
            return parsed

    repaired = _repair_value_judge_json_from_text(last_raw)
    if repaired:
        return repaired
    return None


def _llm_json_text(llm: Any, *, system: str, user_prompt: str, max_tokens: int) -> str:
    """Prefer chat/json mode when available, then fall back to a plain chat call."""
    from src.llm.base import LLMMessage

    messages = [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user_prompt),
    ]
    try:
        response = llm.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    return str(getattr(response, "content", "") or "")


def _repair_value_judge_json_from_text(raw: Any) -> dict[str, Any] | None:
    """Recover simple verdict/score answers when a judge ignores JSON formatting."""
    text = " ".join(str(raw or "").split())
    if not text:
        return None
    lower = text.lower()
    verdict = ""
    if re.search(r"\bmismatch\b", lower) or "not match" in lower or "does not match" in lower:
        verdict = "mismatch"
    elif re.search(r"\bpartial\b", lower):
        verdict = "partial"
    elif re.search(r"\bmatch\b", lower):
        verdict = "match"
    if not verdict:
        return None

    matched_index = None
    index_match = re.search(r"\b(?:matched_)?index\b[^0-9-]*(-?\d+|null)", lower)
    if index_match and index_match.group(1) != "null":
        try:
            matched_index = int(index_match.group(1))
        except ValueError:
            matched_index = None

    return {
        "verdict": verdict,
        "score": {"match": 1.0, "partial": 0.5, "mismatch": 0.0}[verdict],
        "rationale": text[:500],
        "matched_index": matched_index,
        "json_repaired": True,
    }


# ─── Lexical Judge (fallback) ─────────────────────────────────────────────────

def lexical_judge(gt: str, candidates: list[str]) -> JudgeResult:
    """Token-overlap based judge (fast fallback)."""
    gt_tokens = _tokens(gt)
    if not gt_tokens or not candidates:
        return JudgeResult("mismatch", 0.0, "No comparable prediction.")
    best_i = None
    best = 0.0
    for i, cand in enumerate(candidates):
        cand_tokens = _tokens(cand)
        if not cand_tokens:
            continue
        overlap = len(gt_tokens & cand_tokens) / max(len(gt_tokens), 1)
        if overlap > best:
            best = overlap
            best_i = i
    if best >= 0.55:
        return JudgeResult("match", 1.0, "Prediction substantially overlaps the target fact.", best_i)
    if best >= 0.25:
        return JudgeResult("partial", 0.5, "Prediction partially overlaps the target fact.", best_i)
    return JudgeResult("mismatch", 0.0, "Prediction does not cover the target fact.", best_i)


def value_judge(gt: str, pred: str | None, aliases: list[str] | None = None) -> JudgeResult:
    """Value-based judge for person attributes (fast fallback)."""
    pred_text = str(pred or "").strip()
    gt_text = str(gt or "").strip()
    if not pred_text or not gt_text:
        return JudgeResult("mismatch", 0.0, "Missing prediction or ground truth.")
    values = [gt_text, *(aliases or [])]
    if any(_norm(pred_text) == _norm(v) for v in values if v):
        return JudgeResult("match", 1.0, "Prediction matches an accepted value.")
    if any(_norm(pred_text) in _norm(v) or _norm(v) in _norm(pred_text) for v in values if v):
        return JudgeResult("match", 1.0, "Prediction is an identifying alias or partial name.")
    gt_tokens = _tokens(gt_text)
    pred_tokens = _tokens(pred_text)
    if gt_tokens & pred_tokens:
        return JudgeResult("partial", 0.5, "Prediction shares some identifying content.")
    return JudgeResult("mismatch", 0.0, "Prediction does not match the target value.")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD_RE.findall(text or "") if len(t) > 2}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    # Try direct parse
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    # Find first JSON object
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


# ─── RFS — Reasoning Fidelity Score ───────────────────────────────────────────

_RFS_JUDGE_SYSTEM = """You are an evaluation judge scoring an AGENT's reasoning_path against a GROUND-TRUTH reasoning_path for the SAME inference target (an attribute of an album owner or a person in the album).

You will be given:
- target_type: one of owner_fact_atom, person_name, person_relation, person_category
- target_key: a short identifier of what is being inferred (e.g. the GT value)
- gt_value: the ground-truth value (e.g. "partner", "Darius Campbell", "Jasmine is an attorney based in Miami")
- gt_reasoning_path: a natural-language explanation of HOW the ground truth is derivable from the album
- agent_prediction: the agent's predicted value for this target (may be empty / null)
- agent_reasoning_path: the agent's reasoning chain in its own words. May contain section headers like "## CandidateProposer ...", "## AlignmentVerifier ...", "## RelationInferer ...", "## OwnerFactAuditor ..." followed by bullet steps citing photo_ids.

Your task: judge the FIDELITY of the agent's reasoning to the ground-truth reasoning. You are NOT primarily judging whether the prediction is correct — agent_prediction can match GT yet have a hollow reasoning_path, and a reasoning_path can be coherent yet land on a wrong answer. Score the chain itself, conditioned on the prediction.

Scoring rubric (output exactly one verdict):
- "equivalent" (1.0):  Agent's reasoning is grounded (cites specific photo_ids or named evidence) AND covers the SAME core evidence pillars as the GT path AND its prediction is consistent with that chain. Phrasing differences are fine.
- "partial"   (0.66): Agent's reasoning covers a SUBSET of the GT pillars, or covers them but with shallow grounding (e.g. only one cited photo for a multi-pillar GT chain). Prediction may be correct or partially correct.
- "weak"      (0.33): Agent's reasoning is mostly generic / hand-wavy / a single weak observation. Even if the prediction happens to match GT, the path does NOT explain how it was derived.
- "mismatch"  (0.0):  Agent's reasoning is empty, irrelevant, contradictory, or invokes evidence that does not exist; OR the prediction strongly disagrees with GT and the path supplies no compensating reasoning.

Additional rules:
- If agent_prediction is empty/null AND agent_reasoning_path is empty → "mismatch".
- If agent_prediction is empty/null but the reasoning_path correctly identifies a substantial fraction of GT pillars → at most "partial".
- "Grounded" means the path cites at least one specific photo_id (photo_NNNN) OR a specific named entity from the album (a name, a venue, a screenshot type). Pure restatements of the prediction are NOT grounded.
- Photo-id mismatch with GT key_photo_ids is OK as long as the agent cites SOME photo and its evidence type aligns with the GT chain. We are not checking exact photo overlap.

Output ONLY a JSON object with this exact shape:
{
  "verdict": "equivalent" | "partial" | "weak" | "mismatch",
  "grounded": true | false,
  "rationale": "1-3 sentences explaining which pillars the agent did/did not cover and whether the chain is grounded."
}"""


def llm_reasoning_fidelity_judge(
    *,
    target_type: str,
    target_key: str,
    gt_value: str,
    gt_reasoning_path: str,
    agent_prediction: str | None,
    agent_reasoning_path: str,
    llm: Any,
) -> RFSResult:
    """Judge the fidelity of agent's reasoning_path against GT reasoning_path.

    Returns an RFSResult with a 4-tier verdict and score in {0.0, 0.33, 0.66, 1.0}.
    Falls back to a deterministic heuristic RFS when llm is unavailable.
    """
    agent_pred_text = str(agent_prediction or "").strip()
    agent_path_text = str(agent_reasoning_path or "").strip()
    gt_path_text = str(gt_reasoning_path or "").strip()

    if not agent_path_text and not agent_pred_text:
        return RFSResult("mismatch", 0.0, "Agent produced neither prediction nor reasoning_path.", grounded=False)

    if llm is None:
        return _heuristic_rfs(
            agent_pred_text=agent_pred_text,
            agent_path_text=agent_path_text,
            gt_value=gt_value,
            gt_path_text=gt_path_text,
        )

    user_prompt = f"""target_type: {target_type}
target_key: {target_key}

## Ground-truth value
{gt_value or "(unspecified)"}

## Ground-truth reasoning_path
{gt_path_text or "(empty)"}

## Agent prediction
{agent_pred_text or "(empty)"}

## Agent reasoning_path
{agent_path_text or "(empty)"}

Score the fidelity of the agent's reasoning_path to the ground-truth reasoning_path."""

    try:
        raw = llm.simple(prompt=user_prompt, system=_RFS_JUDGE_SYSTEM, temperature=0.0, max_tokens=384)
    except Exception as exc:
        logger.warning("RFS judge call failed: %s — falling back to heuristic", exc)
        return _heuristic_rfs(
            agent_pred_text=agent_pred_text,
            agent_path_text=agent_path_text,
            gt_value=gt_value,
            gt_path_text=gt_path_text,
        )

    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return _heuristic_rfs(
            agent_pred_text=agent_pred_text,
            agent_path_text=agent_path_text,
            gt_value=gt_value,
            gt_path_text=gt_path_text,
        )

    verdict = str(parsed.get("verdict", "mismatch")).strip().lower()
    if verdict not in _RFS_VERDICT_TO_SCORE:
        # Try lenient mapping
        if verdict in ("strong", "good", "complete"):
            verdict = "equivalent"
        elif verdict in ("acceptable", "ok", "moderate"):
            verdict = "partial"
        elif verdict in ("shallow", "thin"):
            verdict = "weak"
        else:
            verdict = "mismatch"

    score = _RFS_VERDICT_TO_SCORE[verdict]
    grounded = bool(parsed.get("grounded", False))
    rationale = str(parsed.get("rationale", "")).strip()

    return RFSResult(
        verdict=verdict,  # type: ignore[arg-type]
        score=score,
        rationale=rationale,
        grounded=grounded,
    )


_PHOTO_ID_RE = re.compile(r"\bphoto_\d{3,5}\b")


def _heuristic_rfs(
    *,
    agent_pred_text: str,
    agent_path_text: str,
    gt_value: str,
    gt_path_text: str,
) -> RFSResult:
    """Fallback RFS when no LLM judge is available.

    Heuristic: cap at "partial" — without a real judge we can't confirm
    semantic equivalence. Drops to "weak"/"mismatch" when the path lacks
    grounding or the agent has no prediction.
    """
    if not agent_path_text:
        return RFSResult("mismatch", 0.0, "Heuristic: empty agent reasoning_path.", grounded=False)

    grounded = bool(_PHOTO_ID_RE.search(agent_path_text)) or any(
        h in agent_path_text for h in ("CandidateProposer", "AlignmentVerifier", "RelationInferer", "OwnerFactAuditor")
    )

    gt_tokens = _tokens(gt_path_text) if gt_path_text else set()
    agent_tokens = _tokens(agent_path_text)
    overlap = (
        len(gt_tokens & agent_tokens) / max(len(gt_tokens), 1)
        if gt_tokens else 0.0
    )

    pred_consistent = bool(agent_pred_text) and any(
        _norm(t) in _norm(agent_pred_text) or _norm(agent_pred_text) in _norm(t)
        for t in (gt_value or "").split()
        if len(t) > 2
    )

    if grounded and overlap >= 0.4 and pred_consistent:
        return RFSResult("partial", 0.66, "Heuristic: grounded path with substantial token overlap with GT path.", grounded=True)
    if grounded and (overlap >= 0.2 or pred_consistent):
        return RFSResult("weak", 0.33, "Heuristic: grounded path but limited overlap with GT path.", grounded=True)
    if grounded:
        return RFSResult("weak", 0.33, "Heuristic: path is grounded but unrelated to GT pillars.", grounded=True)
    return RFSResult("mismatch", 0.0, "Heuristic: ungrounded reasoning_path.", grounded=False)
