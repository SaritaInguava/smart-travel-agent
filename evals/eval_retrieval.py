"""Retrieval eval for search_school_calendar.

calendar_keeper's prompt (see agents.py's CALENDAR_SYSTEM_PROMPT) instructs the agent to
query the school-calendar index once per school, by name — e.g. "Stratford <break>", then
separately "Harker Upper School <break>" — because Stratford's calendar is one dense chunk
while Harker's is split into many small weekly chunks, so a generic, school-less query
tends to surface only Harker results and miss Stratford entirely. This eval checks that
the strategy the prompt relies on actually works: that naming a school in the query
reliably surfaces the CORRECT chunk — right school AND right dates, not just the right
school label — within the results the agent actually sees.

TOP_K below matches search_school_calendar's own k=5 exactly (see retriever.py) rather
than checking rank #1 alone: rank #1 proved too strict (see git history) — Harker's
weekly chunks compete with each other closely enough that the correct week can rank
below #1 even when the query names the school correctly, and the agent isn't limited to
rank #1 either, since it reads all 5 results. Checking dates matters regardless of rank:
a top result can have the right `school` in its metadata while still being the wrong
week's chunk (Harker's calendar is split into ~50 small per-week chunks that compete
with each other on relevance, unlike Stratford's one all-encompassing chunk). Each
case's `expected_substrings` are exact phrases pulled from the real source data for that
break, so a wrong-week match still fails even though the school matches.

Unlike tests/test_agents.py (which mocks the LLM entirely), this makes real OpenAI
embedding API calls and requires the FAISS index to already be built:

    uv run python -m smart_travel_agent.retrieval.build_index

Run with:

    uv run python -m evals.eval_retrieval
"""

import sys
from dataclasses import dataclass

from smart_travel_agent.retrieval.retriever import search

# Matches search_school_calendar's own k=5 — check what the agent actually sees, not an
# arbitrarily stricter window.
TOP_K = 5


@dataclass
class Case:
    query: str
    expected_school: str
    expected_substrings: list[str]  # phrases the correct chunk's text must contain


CASES = [
    Case("Stratford Thanksgiving break", "Stratford", ["Thanksgiving Break", "November 25"]),
    Case("Harker Upper School Thanksgiving break", "Harker Upper School", ["Thanksgiving Vacation"]),
    Case("Stratford winter break", "Stratford", ["Winter Break", "December 21"]),
    Case("Harker Upper School spring break", "Harker Upper School", ["Spring Break"]),
]


def _matches(case: Case, text: str, meta: dict) -> bool:
    return meta.get("school") == case.expected_school and all(s in text for s in case.expected_substrings)


def run() -> bool:
    all_passed = True
    for case in CASES:
        top_k = search(case.query, k=TOP_K)
        passed = any(_matches(case, text, meta) for text, _score, meta in top_k)
        all_passed = all_passed and passed

        print(f"[{'PASS' if passed else 'FAIL'}] {case.query!r}")
        if passed:
            print(f"       a top-{TOP_K} result matched the expected school and date/break phrases")
        else:
            for rank, (text, _score, meta) in enumerate(top_k, start=1):
                missing = [s for s in case.expected_substrings if s not in text]
                print(f"       #{rank}: school={meta.get('school')!r} missing={missing!r}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
