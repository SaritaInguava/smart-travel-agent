"""Agent-output evals: real-model checks for regressions found during manual testing.

Unlike tests/test_agents.py (which mocks the LLM entirely), these make real OpenAI
calls against the actual agent functions — slower, cost real usage, and mildly flaky
since LLM output isn't deterministic — but they catch quality regressions a mocked
unit test can't, by construction. Each case below corresponds to a real bug found and
fixed earlier in this project:

  - destination_researcher used to default to a stale remembered break instead of the
    trip's own stated preferred_break (see agents.py's break_line).
  - destination_researcher and activity_planner used to duplicate budget_analyst's job
    by including budget/cost content of their own.
  - calendar_keeper used to run a school-calendar search on any preferred_break input,
    including random text that isn't describing a break at all.

The remaining cases check general constraint satisfaction, format adherence, and
interest coverage rather than a specific past bug. One deliberate choice: none of them
assert the budget breakdown's total is <= the stated budget. BUDGET_SYSTEM_PROMPT
explicitly allows exceeding budget as long as it's flagged clearly (real example seen
in manual testing: "the cheapest options I found are above budget") — a strict <=
assertion would be wrong by design. Instead, the budget eval checks that a fit/overage
statement is present at all.

Run with:

    uv run python -m evals.eval_agent_outputs
"""

import sys
from dataclasses import dataclass
from typing import Callable

from smart_travel_agent.agents import activity_planner, budget_analyst, calendar_keeper, destination_researcher
from smart_travel_agent.state import TravelPlanState

BUDGET_KEYWORDS = ["$", "budget breakdown", "budget overview", "cost estimate"]


def _contains_any(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [k for k in keywords if k.lower() in lowered]


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def eval_destination_brief_mentions_preferred_break() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Paris, France",
        num_days=5,
        budget=5000,
        passengers=2,
        preferred_break="President's Day",
    )
    brief = destination_researcher(state)["destination_brief"]
    passed = "president" in brief.lower()
    detail = "brief mentions President's Day" if passed else f"brief did not mention it:\n{brief[:300]}"
    return Result("destination_brief mentions the stated preferred_break", passed, detail)


def eval_destination_brief_excludes_budget() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=5,
        budget=5000,
        passengers=2,
        interests="food, culture",
    )
    brief = destination_researcher(state)["destination_brief"]
    hits = _contains_any(brief, BUDGET_KEYWORDS)
    passed = not hits
    detail = "no budget keywords found" if passed else f"found budget keywords {hits}"
    return Result("destination_brief excludes budget/cost content", passed, detail)


def eval_itinerary_excludes_budget() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=3,
        budget=3000,
        passengers=2,
        interests="food",
        destination_brief="Tokyo is a vibrant city with great food and culture.",
        date_range="March 10-13, 2027",
    )
    itinerary = activity_planner(state)["itinerary"]
    hits = _contains_any(itinerary, BUDGET_KEYWORDS)
    passed = not hits
    detail = "no budget keywords found" if passed else f"found budget keywords {hits}"
    return Result("itinerary excludes budget/cost content", passed, detail)


def eval_calendar_keeper_rejects_non_break_input() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=5,
        budget=5000,
        passengers=1,
        preferred_break="what is the capital of france",
    )
    trace = calendar_keeper(state)["trace"]
    tool_calls = [tc for entry in trace for tc in entry.get("tool_calls", [])]
    passed = not tool_calls
    detail = "made no tool calls" if passed else f"made tool calls: {tool_calls}"
    return Result("calendar_keeper doesn't search when given a non-break input", passed, detail)


def eval_itinerary_covers_requested_days() -> Result:
    num_days = 4
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=num_days,
        budget=4000,
        passengers=2,
        interests="food, culture, history",
        destination_brief="Tokyo is a vibrant city with great food, culture, and history.",
        date_range="March 10-14, 2027",
    )
    itinerary = activity_planner(state)["itinerary"]
    missing_days = [d for d in range(1, num_days + 1) if f"Day {d}" not in itinerary]
    passed = not missing_days
    detail = f"all {num_days} days present" if passed else f"missing day markers: {missing_days}"
    return Result(f"itinerary covers all {num_days} requested days", passed, detail)


def eval_itinerary_follows_daypart_format() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=3,
        budget=3000,
        passengers=2,
        interests="food",
        destination_brief="Tokyo is a vibrant city with great food and culture.",
        date_range="March 10-13, 2027",
    )
    itinerary = activity_planner(state)["itinerary"].lower()
    required = ["morning", "afternoon", "evening"]
    missing = [p for p in required if p not in itinerary]
    passed = not missing
    detail = "morning/afternoon/evening all present" if passed else f"missing sections: {missing}"
    return Result("itinerary follows the morning/afternoon/evening format", passed, detail)


# A model covering "food" as an interest will reliably say "cuisine"/"dining"/
# "restaurant" rather than the literal word "food" — checking only the literal word
# produced a false failure on a genuinely on-topic itinerary. Matching on synonyms
# instead avoids penalizing valid paraphrasing.
INTEREST_SYNONYMS = {
    "food": ["food", "cuisine", "culinary", "restaurant", "dining"],
    "culture": ["culture", "cultural"],
    "history": ["history", "historical", "heritage"],
}


def _interest_covered(interest: str, text: str) -> bool:
    return any(s in text for s in INTEREST_SYNONYMS.get(interest, [interest]))


def eval_itinerary_covers_stated_interests() -> Result:
    interests = ["food", "culture", "history"]
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=4,
        budget=4000,
        passengers=2,
        interests=", ".join(interests),
        destination_brief="Tokyo is a vibrant city with great food, culture, and history.",
        date_range="March 10-14, 2027",
    )
    itinerary = activity_planner(state)["itinerary"].lower()
    missing = [i for i in interests if not _interest_covered(i, itinerary)]
    passed = not missing
    detail = "all stated interests appear in the itinerary" if passed else f"missing interests: {missing}"
    return Result("itinerary reflects the stated interests", passed, detail)


def eval_budget_breakdown_states_fit_clearly() -> Result:
    state = TravelPlanState(
        origin="San Francisco, CA",
        destination="Tokyo, Japan",
        num_days=3,
        budget=1000,  # deliberately tight, likely to be exceeded
        passengers=2,
        tickets={"summary": "ANA | SFO-NRT | 11h | departs 10:00 arrives 14:00 | $1,200"},
        itinerary="Day 1: Explore Shibuya, dinner at a mid-range restaurant ($40/person).",
    )
    breakdown = budget_analyst(state)["budget_breakdown"]["summary"].lower()
    fit_phrases = [
        "within budget",
        "under budget",
        "over budget",
        "exceeds budget",
        "does not fit",
        "fits the budget",
        "fits within",
    ]
    passed = any(p in breakdown for p in fit_phrases)
    detail = "explicitly states budget fit/overage" if passed else "no clear fit-vs-budget statement found"
    return Result("budget breakdown clearly states whether the trip fits the budget", passed, detail)


EVALS: list[Callable[[], Result]] = [
    eval_destination_brief_mentions_preferred_break,
    eval_destination_brief_excludes_budget,
    eval_itinerary_excludes_budget,
    eval_calendar_keeper_rejects_non_break_input,
    eval_itinerary_covers_requested_days,
    eval_itinerary_follows_daypart_format,
    eval_itinerary_covers_stated_interests,
    eval_budget_breakdown_states_fit_clearly,
]


def run() -> bool:
    all_passed = True
    for eval_fn in EVALS:
        result = eval_fn()
        all_passed = all_passed and result.passed
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}")
        print(f"       {result.detail}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
