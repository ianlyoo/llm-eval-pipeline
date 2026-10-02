"""Pure deterministic judge scoring, independent of API clients and CLI I/O."""

from __future__ import annotations

import random
import re

from eval.judge_inputs import resolve_candidate, resolve_sources

# ---------------------------------------------------------------------------
# Tokenization / helpers (shared pattern with rule_filter)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)

_KO_PARTICLES = [
    "에서", "에게", "한테", "부터", "까지", "으로", "로서", "로써",
    "으로서", "으로써", "와", "과", "은", "는", "이", "가", "을", "를",
    "의", "에", "로", "도", "만", "조차", "마저", "이다", "입니다",
    "인가", "인가요", "이며", "이고", "라는", "이라는",
]

_STOPWORDS = {
    "what", "is", "the", "and", "are", "a", "an", "of", "in", "on", "for", "to", "with",
    "how", "when", "where", "why", "who", "does", "do", "did", "be", "been", "being",
    "have", "has", "had", "was", "were", "will", "would", "can", "could", "should",
    "mean", "means", "or", "but", "if", "then", "than", "this", "that", "these", "those",
    "it", "its", "by", "as", "at", "from", "up", "out", "about", "into", "over", "after",
    "before", "under", "again", "further",
    "무엇인가", "무엇", "얼마인가", "얼마", "어떻게", "언제", "어디", "누가", "왜",
    "인가", "인가요", "은", "는", "이", "가", "을", "를", "에", "의", "로", "과", "와",
    "하다", "한다", "있나", "있나요", "되는가", "있는가", "관한", "대한", "대해",
    "대하여", "관련", "질문", "답변", "정책", "무슨", "어느", "어떤",
}


def _strip_ko_particle(token: str) -> str:
    for p in sorted(_KO_PARTICLES, key=len, reverse=True):
        if token.endswith(p) and len(token) > len(p):
            return token[: -len(p)]
    return token


def _normalize_kw(kw: str) -> str:
    kw = kw.lower()
    return _strip_ko_particle(kw)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _keywords_from_question(q: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(q)]
    kws: list[str] = []
    for tok in toks:
        if tok in _STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        norm = _normalize_kw(tok)
        if len(norm) < 2:
            continue
        if norm in _STOPWORDS:
            continue
        kws.append(norm)
    # dedup preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def _numeric_tokens(text: str) -> list[str]:
    # capture integers and decimals, strip % sign
    return re.findall(r"\d+(?:\.\d+)?", text)


def _ratio_to_score(ratio: float) -> int:
    if ratio >= 1.0 - 1e-9:
        return 5
    if ratio >= 0.8:
        return 4
    if ratio >= 0.5:
        return 3
    if ratio >= 0.25:
        return 2
    return 1


def _clamp_score(s: int) -> int:
    return max(1, min(5, s))


# ---------------------------------------------------------------------------
# 4 metric scorers (deterministic)
# ---------------------------------------------------------------------------

def score_correctness(reference: str, candidate: str) -> tuple[int, str]:
    ref_nums = _numeric_tokens(reference)
    ref_toks = [t for t in _tokens(reference) if t not in _STOPWORDS and len(t) >= 2]
    cand_set = set(_tokens(candidate))

    if not reference.strip():
        return 1, "correctness: reference empty → 1 (no ground truth)"

    # Numeric coverage matters most when reference has numbers
    if ref_nums:
        matched_nums = sum(1 for n in ref_nums if n in candidate)
        # also check numeric overlap ratio; if candidate has numbers, check they align
        ratio_num = matched_nums / len(ref_nums) if ref_nums else 1.0
        # fallback to token overlap
        matched_toks = sum(1 for t in ref_toks if t in cand_set) if ref_toks else 0
        ratio_tok = matched_toks / len(ref_toks) if ref_toks else 0.0
        # weighted: 60% numeric, 40% token
        ratio = 0.6 * ratio_num + 0.4 * ratio_tok
        score = _ratio_to_score(ratio)
        reason = (
            f"correctness: numeric {matched_nums}/{len(ref_nums)} matched ({ratio_num:.2f}), "
            f"token {matched_toks}/{len(ref_toks)} ({ratio_tok:.2f}) → blended {ratio:.2f} → {score}/5"
        )
        return score, reason
    # No numeric in reference: purely token overlap
    if not ref_toks:
        return 3, "correctness: no extractable reference tokens → 3 (neutral)"
    matched = sum(1 for t in ref_toks if t in cand_set)
    ratio = matched / len(ref_toks)
    score = _ratio_to_score(ratio)
    reason = f"correctness: token {matched}/{len(ref_toks)} matched ({ratio:.2f}) → {score}/5"
    return score, reason


def score_groundedness(candidate: str, source_chunks: list[dict]) -> tuple[int, str]:
    if not candidate.strip():
        return 1, "groundedness: candidate empty → 1 (ungrounded)"
    cand_nums = _numeric_tokens(candidate)
    source_text = " ".join([c.get("text", "") for c in source_chunks if isinstance(c, dict)])
    if not cand_nums:
        # No numeric to verify - check token grounding: candidate tokens in source?
        cand_toks = [t for t in _tokens(candidate) if len(t) >= 2 and t not in _STOPWORDS]
        if not cand_toks:
            return 5, "groundedness: no numeric & no content tokens → 5 (trivially grounded)"
        source_set = set(_tokens(source_text))
        matched = sum(1 for t in cand_toks if t in source_set)
        ratio = matched / len(cand_toks) if cand_toks else 1.0
        # Map more leniently: high token overlap = grounded
        score = _ratio_to_score(ratio)
        reason = (
            f"groundedness: numeric none, token {matched}/{len(cand_toks)} in source ({ratio:.2f}) → {score}/5"
        )
        return score, reason
    missing = [n for n in cand_nums if n not in source_text]
    matched_n = len(cand_nums) - len(missing)
    ratio = matched_n / len(cand_nums) if cand_nums else 1.0
    score = _ratio_to_score(ratio)
    if missing:
        reason = (
            f"groundedness: numeric {matched_n}/{len(cand_nums)} in source ({ratio:.2f}), "
            f"missing {missing} → {score}/5"
        )
    else:
        reason = f"groundedness: all {len(cand_nums)} numeric tokens found in source ({ratio:.2f}) → {score}/5"
    return score, reason


def score_relevance(question: str, candidate: str) -> tuple[int, str]:
    if not question.strip() or not candidate.strip():
        return 1, "relevance: question or candidate empty → 1"
    kws = _keywords_from_question(question)
    if not kws:
        return 3, "relevance: no question keywords → 3 (neutral)"
    cand_low = candidate.lower()
    matched = [kw for kw in kws if kw in cand_low]
    ratio = len(matched) / len(kws) if kws else 0.0
    score = _ratio_to_score(ratio)
    reason = f"relevance: keywords {len(matched)}/{len(kws)} matched {matched} vs {kws} ({ratio:.2f}) → {score}/5"
    return score, reason


def score_completeness(reference: str, candidate: str) -> tuple[int, str]:
    if not reference.strip():
        return 1, "completeness: reference empty → 1"
    if not candidate.strip():
        return 1, "completeness: candidate empty → 1 (nothing covered)"
    ref_toks = [t for t in _tokens(reference) if t not in _STOPWORDS and len(t) >= 2]
    if not ref_toks:
        return 3, "completeness: no extractable reference tokens → 3 (neutral)"
    cand_set = set(_tokens(candidate))
    matched = sum(1 for t in ref_toks if t in cand_set)
    ratio = matched / len(ref_toks)
    score = _ratio_to_score(ratio)
    reason = f"completeness: {matched}/{len(ref_toks)} ref tokens covered ({ratio:.2f}) → {score}/5"
    return score, reason


def _apply_profile(score: int, profile: str) -> int:
    if profile == "deterministic-strict":
        return _clamp_score(score - 1)
    if profile == "deterministic-lenient":
        return _clamp_score(score + 1)
    return score


def _apply_noise(score: int, rng: random.Random) -> int:
    # small jitter: -1 with 10%, +1 with 10%, else 0 → variance <1
    r = rng.random()
    if r < 0.10:
        return _clamp_score(score - 1)
    if r < 0.20:
        return _clamp_score(score + 1)
    return score


# ---------------------------------------------------------------------------
# Deterministic judge engine
# ---------------------------------------------------------------------------

def deterministic_judge(
    entry: dict,
    candidate_override: str | None = None,
    profile: str = "default",
    noise_rng: random.Random | None = None,
) -> dict:
    """Run deterministic 4-metric judge on one entry. Returns scores dict."""
    question = entry.get("question", "")
    reference = entry.get("reference_answer", "")
    candidate = resolve_candidate(entry, candidate_override)
    source_chunks = resolve_sources(entry)

    s_corr, r_corr = score_correctness(reference, candidate)
    s_ground, r_ground = score_groundedness(candidate, source_chunks)
    s_rel, r_rel = score_relevance(question, candidate)
    s_comp, r_comp = score_completeness(reference, candidate)

    # Apply profile shift
    s_corr_p = _apply_profile(s_corr, profile)
    s_ground_p = _apply_profile(s_ground, profile)
    s_rel_p = _apply_profile(s_rel, profile)
    s_comp_p = _apply_profile(s_comp, profile)

    # Apply noise if rng given (reliability simulation)
    if noise_rng is not None:
        s_corr_p = _apply_noise(s_corr_p, noise_rng)
        s_ground_p = _apply_noise(s_ground_p, noise_rng)
        s_rel_p = _apply_noise(s_rel_p, noise_rng)
        s_comp_p = _apply_noise(s_comp_p, noise_rng)
        # annotate reasons with noise hint
        r_corr += " [noise-applied]"
        r_ground += " [noise-applied]"
        r_rel += " [noise-applied]"
        r_comp += " [noise-applied]"

    # Ensure reasons are non-empty - FAIL guard
    for r in [r_corr, r_ground, r_rel, r_comp]:
        if not r or not r.strip():
            raise ValueError("Judge reason must not be empty - FAIL")

    return {
        "correctness": {"score": s_corr_p, "reason": r_corr},
        "groundedness": {"score": s_ground_p, "reason": r_ground},
        "relevance": {"score": s_rel_p, "reason": r_rel},
        "completeness": {"score": s_comp_p, "reason": r_comp},
    }
