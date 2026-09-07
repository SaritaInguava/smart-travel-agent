"""LLM-as-judge eval for itinerary coherence/redundancy.

Unlike eval_agent_outputs.py's deterministic keyword checks, this asks a second real
model call to judge a subjective quality: does the itinerary repeat the same specific
restaurant/attraction across days, or lean on generic filler ("enjoy the local culture")
instead of concrete, named activities?

This is a genuinely weaker signal than the other evals, and worth being honest about —
confirmed by an actual run: the judge once claimed "Uobei Sushi" was repeated on Day 2
and Day 4, but Day 4 never mentioned it at all. What really happened is Day 2's evening
line offered two alternative dining options in one sentence ("Uobei Sushi or Shibuya
Yakiniku"), and the judge misattributed one option to a different day — a factual claim
about the very text it was reading, not just a differing opinion. So repeated_items
claims are now verified programmatically (see _verify) rather than trusted outright:
the judge must cite which day numbers an item appears on, and a claim only counts if
the item's text is actually found on at least two of those cited days. This catches
exact misattributions like the Uobei case; it doesn't make the judge infallible (it
could still misquote an item's text so the substring check itself misses, or hallucinate
something no verification can catch), but it turns "trust the model" into "trust but
verify the specific, checkable part of its claim."

Deliberately not attempted here: groundedness/factuality (does a named attraction
actually exist in that city). destination_researcher has no tool access, so verifying
its claims would need an actual fact-check (e.g. a search tool) — another LLM guessing
whether a fact is true is checking a guess with a guess, which isn't worth much.

Run with:

    uv run python -m evals.eval_itinerary_quality
"""

import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from smart_travel_agent.agents import activity_planner
from smart_travel_agent.config import get_llm
from smart_travel_agent.state import TravelPlanState

COHERENCE_PASS_THRESHOLD = 4  # out of 5 — see JUDGE_SYSTEM_PROMPT's rubric

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous travel-itinerary editor. Evaluate the given multi-day "
    "itinerary against this rubric, being strict — don't give benefit of the doubt on "
    "vague wording:\n\n"
    "1. Repeated items: is the same specific restaurant, attraction, or activity named "
    "on two or more DIFFERENT days (excluding an intentional repeat like a home-base "
    "hotel)? Two alternative options mentioned within the SAME day (e.g. 'try X or Y "
    "for dinner') are NOT a repeat — only count it if the identical item appears under "
    "separate day headings. For each repeated item, quote it exactly as written in the "
    "itinerary and list every day number (1-indexed) it actually appears on.\n"
    "2. Generic filler: are there vague statements that could apply to any city on any "
    "trip (e.g. 'explore the local culture', 'enjoy the vibrant atmosphere') without a "
    "specific, named place or activity attached? List each example verbatim.\n"
    "3. Overall coherence, 1-5: 5 = no redundancy or filler, every activity specific and "
    "distinct; 1 = mostly repeated or generic. Judge the whole itinerary, not just the "
    "presence of any single issue."
)


class RepeatedItem(BaseModel):
    item: str = Field(description="The repeated item, quoted exactly as written in the itinerary")
    days: list[int] = Field(description="Every day number (1-indexed) this exact item appears on")


class ItineraryJudgment(BaseModel):
    repeated_items: list[RepeatedItem] = Field(description="Items repeated across two or more different days")
    generic_filler_examples: list[str] = Field(description="Vague, non-specific statements found")
    coherence_score: int = Field(description="1-5 per the rubric")
    reasoning: str = Field(description="Brief justification for the score")


def _judge(itinerary: str) -> ItineraryJudgment:
    return (
        get_llm(temperature=0)
        .with_structured_output(ItineraryJudgment)
        .invoke([SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=itinerary)])
    )


# Matches a line starting a day's section, allowing for markdown noise (#, *, -) before
# "Day N" — e.g. "**Day 2: Shibuya**" or "### Day 4: Art and Shopping Day". Not a general
# markdown parser, just enough structure to bucket lines under the day they belong to.
_DAY_HEADER_RE = re.compile(r"^[#*\s\-]*day\s+(\d+)\b", re.IGNORECASE)


def _split_by_day(itinerary: str) -> dict[int, str]:
    segments: dict[int, list[str]] = {0: []}
    current_day = 0
    for line in itinerary.splitlines():
        match = _DAY_HEADER_RE.match(line)
        if match:
            current_day = int(match.group(1))
            segments.setdefault(current_day, [])
        segments[current_day].append(line)
    return {day: "\n".join(lines) for day, lines in segments.items()}


def _verify(claim: RepeatedItem, day_segments: dict[int, str]) -> bool:
    """A claim only counts if its item text actually appears on at least 2 DISTINCT
    days the judge cited — catches misattribution like claiming a same-day alternative
    is a cross-day repeat (see the Uobei case in the module docstring). Distinct
    matters: a claim citing the same day twice (e.g. days=[2, 2]) isn't a repeat at
    all, and must not pass just because that one day happens to contain the item."""
    unique_days = set(claim.days)
    if len(unique_days) < 2:
        return False
    hits = sum(1 for d in unique_days if claim.item.lower() in day_segments.get(d, "").lower())
    return hits >= 2


def run() -> bool:
    # A longer trip gives the model more days to potentially run out of distinct ideas
    # and repeat itself — a 3-day itinerary rarely has room to show redundancy at all.
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=6,
        budget=6000,
        passengers=2,
        interests="food, culture, history",
        destination_brief="Tokyo is a vibrant city with great food, culture, and history.",
        date_range="March 10-16, 2027",
    )
    itinerary = activity_planner(state)["itinerary"]
    judgment = _judge(itinerary)

    day_segments = _split_by_day(itinerary)
    verified = [c for c in judgment.repeated_items if _verify(c, day_segments)]
    rejected = [c for c in judgment.repeated_items if c not in verified]

    passed = judgment.coherence_score >= COHERENCE_PASS_THRESHOLD and not verified

    print(f"[{'PASS' if passed else 'FAIL'}] 6-day itinerary is coherent and non-redundant")
    print(f"       coherence_score={judgment.coherence_score}/5 (pass threshold: {COHERENCE_PASS_THRESHOLD})")
    print(f"       verified repeated items: {[(c.item, c.days) for c in verified]}")
    if rejected:
        print(f"       judge also claimed (but couldn't verify, so ignored): {[(c.item, c.days) for c in rejected]}")
    print(f"       generic_filler_examples={judgment.generic_filler_examples!r}")
    print(f"       judge's reasoning: {judgment.reasoning}")
    return passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
