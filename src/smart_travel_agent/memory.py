import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from smart_travel_agent.config import get_llm

# Long-term memory: durable facts per traveler, written to this file so they survive
# across process restarts (not just across turns within one run, like graph state does).
MEMORY_PATH = Path(__file__).parent / "memories.json"


def _load() -> dict[str, list[str]]:
    if not MEMORY_PATH.exists():
        return {}
    return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))


def _save(memories: dict[str, list[str]]) -> None:
    MEMORY_PATH.write_text(json.dumps(memories, indent=2), encoding="utf-8")


class _ExtractedFacts(BaseModel):
    facts: list[str]


_EXTRACT_PROMPT = (
    "You extract durable travel preferences/facts about a family from a completed trip "
    "request — the kind of thing worth remembering for FUTURE trips, not specific to this "
    "one trip's destination or dates.\n\n"
    'Examples to extract: "Usually travels from San Francisco, CA.", "Prefers nonstop '
    'flights over cheaper layover options.", "Typically budgets $5,000-6,000 for a family '
    'of 3.", "Interested in food, culture, and history when traveling."\n\n'
    "Do NOT extract: this trip's specific destination or dates, anything that's just normal "
    "one-off variation, or anything speculative. Return an empty list if nothing durable "
    "stands out — that's the common, correct answer for most trips."
)


def _extract_facts(state: dict) -> list[str]:
    summary = (
        f"Origin: {state.get('origin')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Days: {state.get('num_days')}\n"
        f"Budget: {state.get('budget')}\n"
        f"Passengers: {state.get('passengers')}\n"
        f"Interests: {state.get('interests')}\n"
        f"Preferred break: {state.get('preferred_break')}\n"
        f"Chosen dates: {state.get('date_range')}\n"
        f"Ticket outcome: {(state.get('tickets') or {}).get('summary', '')[:500]}\n"
    )
    parsed = (
        get_llm()
        .with_structured_output(_ExtractedFacts)
        .invoke([SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content=summary)])
    )
    return [f.strip() for f in parsed.facts if f.strip()]


class _MergedFacts(BaseModel):
    facts: list[str]


_RECONCILE_PROMPT = (
    "You maintain a short list of durable travel preferences/facts about a family. Given "
    "the EXISTING facts and the NEW facts just learned, return the merged, deduplicated "
    "list: if a new fact updates or replaces an existing one, keep only the updated "
    "version; if it's genuinely new information, add it; drop anything the new facts make "
    "obsolete or contradict. Keep the list short and durable."
)


def _reconcile(existing: list[str], new: list[str]) -> list[str]:
    if not new:
        return existing
    if not existing:
        return new
    prompt = (
        "EXISTING:\n" + "\n".join(f"- {f}" for f in existing) + "\n\nNEW:\n" + "\n".join(f"- {f}" for f in new)
    )
    parsed = (
        get_llm()
        .with_structured_output(_MergedFacts)
        .invoke([SystemMessage(content=_RECONCILE_PROMPT), HumanMessage(content=prompt)])
    )
    return [f.strip() for f in parsed.facts if f.strip()]


def get_memories(user_name: str | None) -> list[str]:
    if not user_name:
        return []
    return _load().get(user_name, [])


def remember(user_name: str | None, state: dict) -> str:
    """Extracts durable facts from a completed trip, reconciles them against what's already
    known about this user, and saves the result to memories.json. Returns a short
    description of what happened, for the Memory tab."""
    if not user_name:
        return "No name given — long-term memory needs a name to save under."

    new_facts = _extract_facts(state)
    if not new_facts:
        return "Nothing new and durable to remember from this trip."

    memories = _load()
    existing = memories.get(user_name, [])
    merged = _reconcile(existing, new_facts)
    memories[user_name] = merged
    _save(memories)

    added = [f for f in merged if f not in existing]
    if added:
        return "Learned: " + "; ".join(added)
    return "Reconciled existing memory (no new distinct facts)."
