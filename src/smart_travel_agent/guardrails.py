"""Guardrails: lightweight, deterministic checks that catch failure patterns a prompt
instruction alone can't guarantee against. Regex/set-based, not another LLM call, so
they add no extra hallucination risk or API cost of their own — a model checking
another model's output is still just guessing, whereas these are auditable and testable
in isolation.
"""

import re

# Absolute-certainty phrases that read as overconfident regardless of subject matter.
# These get hedged, not removed or flagged as false — the underlying claim may well be
# accurate; the guardrail targets the STYLE of unwarranted certainty (which a reader
# could mistake for verified fact), not the content's truth, which this check has no
# way to verify.
_OVERCONFIDENCE_MARKERS = {
    "guaranteed": "generally expected",
    "always": "typically",
    "never": "rarely",
    "completely safe": "generally considered safe",
    "totally safe": "generally considered safe",
    "perfectly safe": "generally considered safe",
    "absolutely safe": "generally considered safe",
    "no risk": "low risk",
    "zero risk": "low risk",
    "risk-free": "low-risk",
    "100%": "highly likely",
    "certainly": "likely",
    "undoubtedly": "evidence suggests",
    "definitely": "current guidance suggests",
    "without exception": "in most cases",
}


def check_confidence_calibration(text: str) -> dict:
    """During-generation guardrail: flags absolute-certainty language (e.g. "guaranteed",
    "100% safe") in agent output and produces a hedged rewrite.

    Prompts can ask a model to hedge, but that's a probabilistic nudge, not a guarantee
    (see activity_planner's anti-filler instruction, which measurably helped but didn't
    eliminate the pattern across every run). This check catches overconfident phrasing
    deterministically regardless of whether the model complied.

    Returns a dict with:
        flagged_phrases      : list of (marker, context, hedged_alternative) tuples
        overconfidence_score : fraction of sentences containing at least one marker
        verdict               : 'PASS' (<0.3) | 'WARN' (<0.5) | 'FAIL' (>=0.5)
        hedged_text           : text with every marker replaced by its hedged alternative
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 10]

    flagged_phrases: list[tuple[str, str, str]] = []
    flagged_sentence_indices: set[int] = set()
    hedged_text = text

    for marker, alternative in _OVERCONFIDENCE_MARKERS.items():
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        for i, sent in enumerate(sentences):
            match = pattern.search(sent)
            if match:
                start = max(0, match.start() - 20)
                end = min(len(sent), match.end() + 20)
                flagged_phrases.append((marker, f"...{sent[start:end]}...", alternative))
                flagged_sentence_indices.add(i)
        hedged_text = pattern.sub(alternative, hedged_text)

    total = len(sentences) or 1
    score = len(flagged_sentence_indices) / total

    if score < 0.3:
        verdict = "PASS"
    elif score < 0.5:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "flagged_phrases": flagged_phrases,
        "overconfidence_score": round(score, 3),
        "verdict": verdict,
        "hedged_text": hedged_text,
    }
