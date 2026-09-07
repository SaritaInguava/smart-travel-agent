"""Retrieval eval for search_school_calendar.

calendar_keeper's prompt (see agents.py's CALENDAR_SYSTEM_PROMPT) instructs the agent to
query the school-calendar index once per school, by name — e.g. "Stratford <break>", then
separately "Harker Upper School <break>" — because Stratford's calendar is one dense chunk
while Harker's is split into many small weekly chunks, so a generic, school-less query
tends to surface only Harker results and miss Stratford entirely. This eval checks that
the strategy the prompt relies on actually works: that naming a school in the query
reliably puts that school's chunk at rank #1.

Unlike tests/test_agents.py (which mocks the LLM entirely), this makes real OpenAI
embedding API calls and requires the FAISS index to already be built:

    uv run python -m smart_travel_agent.retrieval.build_index

Run with:

    uv run python -m evals.eval_retrieval
"""

import sys
from dataclasses import dataclass

from smart_travel_agent.retrieval.retriever import search


@dataclass
class Case:
    query: str
    expected_school: str


CASES = [
    Case("Stratford Thanksgiving break", "Stratford"),
    Case("Harker Upper School Thanksgiving break", "Harker Upper School"),
    Case("Stratford winter break", "Stratford"),
    Case("Harker Upper School spring break", "Harker Upper School"),
]


def run() -> bool:
    all_passed = True
    for case in CASES:
        results = search(case.query, k=5)
        top_school = results[0][2].get("school") if results else None
        passed = top_school == case.expected_school
        all_passed = all_passed and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {case.query!r}")
        print(f"       expected top result from {case.expected_school!r}, got {top_school!r}")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
