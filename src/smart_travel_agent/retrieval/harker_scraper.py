from datetime import date

import requests
from bs4 import BeautifulSoup

EVENTS_URL = "https://www.harker.org/about/events"
SCHOOL_YEAR_MONTHS = [
    date(2026, 8, 1),
    date(2026, 9, 1),
    date(2026, 10, 1),
    date(2026, 11, 1),
    date(2026, 12, 1),
    date(2027, 1, 1),
    date(2027, 2, 1),
    date(2027, 3, 1),
    date(2027, 4, 1),
    date(2027, 5, 1),
    date(2027, 6, 1),
]


def _parse_month(html: str) -> list[dict]:
    """Parses the calendar grid (fsCalendarDaybox), not the fixed "upcoming events" list
    widget elsewhere on the page — that list is the same regardless of cal_date and misses
    breaks/holidays that aren't among the next few upcoming events."""
    soup = BeautifulSoup(html, "html.parser")
    events = []

    for daybox in soup.select("div.fsCalendarDaybox"):
        date_tag = daybox.select_one("div.fsCalendarDate")
        if not date_tag:
            continue

        day = f"{int(date_tag['data-year']):04d}-{int(date_tag['data-month']) + 1:02d}-{int(date_tag['data-day']):02d}"

        for title_tag in daybox.select("a.fsCalendarEventTitle"):
            events.append(
                {
                    "start": f"{day}T00:00:00",
                    "end": f"{day}T23:59:59",
                    "summary": title_tag.get_text(strip=True),
                }
            )

    return events


def fetch_school_year_events(months: list[date] = SCHOOL_YEAR_MONTHS) -> list[dict]:
    """Fetches Harker's events list-view for each month in the school year and merges them.

    Deduplicates by (start, summary) since a month's page can include a few days that
    spill over from the adjacent month.
    """
    all_events: dict[tuple[str, str], dict] = {}

    for month in months:
        response = requests.get(
            EVENTS_URL,
            params={"cal_date": month.isoformat()},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        for event in _parse_month(response.text):
            all_events[(event["start"], event["summary"])] = event

    return sorted(all_events.values(), key=lambda e: e["start"])


if __name__ == "__main__":
    events = fetch_school_year_events()
    print(f"Fetched {len(events)} events across {len(SCHOOL_YEAR_MONTHS)} months.")
    for event in events[:10]:
        print(event)
