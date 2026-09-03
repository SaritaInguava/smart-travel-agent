from pydantic import BaseModel, Field


class TravelPlanState(BaseModel):
    destination: str
    num_days: int
    budget: float
    passengers: int = 1
    interests: str = ""

    destination_brief: str | None = None
    date_range: str | None = None
    tickets: dict = Field(default_factory=dict)
    itinerary: str | None = None
    budget_breakdown: dict = Field(default_factory=dict)
