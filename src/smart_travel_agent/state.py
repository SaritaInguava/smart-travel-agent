import operator
from typing import Annotated

from pydantic import BaseModel, Field


class TravelPlanState(BaseModel):
    origin: str
    destination: str
    num_days: int
    budget: float
    passengers: int = 1
    interests: str = ""
    preferred_break: str | None = None
    # Freeform flight-specific asks for tickets_scouter — round-trip vs one-way, nonstop
    # only, a preferred airline, refundable fare, etc. Kept freeform rather than a narrow
    # field like is_roundtrip: bool so it covers whatever specific preference comes up
    # next, not just this one.
    ticket_preferences: str = ""
    user_name: str | None = None

    destination_brief: str | None = None
    date_range: str | None = None
    tickets: dict = Field(default_factory=dict)
    itinerary: str | None = None
    budget_breakdown: dict = Field(default_factory=dict)

    # Empty means "no skip info available" — every agent runs. Set by follow-up calls to name
    # exactly which agents need to recompute; agents not listed skip and keep their prior value.
    dirty_nodes: list[str] = Field(default_factory=list)

    # Raw LLM/tool message log for this run, for the UI's Trace/Tools panels. Uses operator.add
    # as its reducer so agents running in the same superstep (e.g. destination_researcher and
    # calendar_keeper, both off START) each append their own messages without clobbering the
    # other's — see agents.py's _run_agent/_arun_agent for how entries get built.
    trace: Annotated[list[dict], operator.add] = Field(default_factory=list)
