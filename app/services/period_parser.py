"""Rule-based date-range parser for spending questions. No LLM involved here
by design: every number in a spending answer must come from the database, so
the period boundaries that gate that query must be deterministic and
testable, not generated. An LLM may extract a candidate period as JSON
upstream (see app/services/spending.py), but it is re-validated through this
same parser before use, and the LLM never sees or computes a total itself."""
import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_NAMES.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


@dataclass
class Period:
    start: date
    end: date
    label: str


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year, month, calendar.monthrange(year, month)[1])
    return start, end


def default_period(today: date | None = None) -> Period:
    today = today or datetime.now().date()
    return Period(today - timedelta(days=29), today, "the last 30 days")


def previous_period(period: Period) -> Period:
    length = (period.end - period.start).days + 1
    prev_end = period.start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    return Period(prev_start, prev_end, "the period before")


def parse_period(text: str, today: date | None = None) -> Period | None:
    """Parses a free-text period reference such as "last month", "August",
    "March 2026", "last 3 months", or "120 days ago". Returns None if nothing
    recognisable is found — callers should ask the user to rephrase rather
    than guessing."""
    today = today or datetime.now().date()
    t = text.lower().strip()

    if re.search(r"\bthis month\b", t):
        start, end = _month_range(today.year, today.month)
        return Period(start, min(end, today), today.strftime("%B %Y"))

    if re.search(r"\blast month\b", t):
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - timedelta(days=1)
        start, end = _month_range(last_month_end.year, last_month_end.month)
        return Period(start, end, start.strftime("%B %Y"))

    m = re.search(r"last (\d+) months?\b", t)
    if m:
        n = int(m.group(1))
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = first_of_this_month
        for _ in range(n):
            start = (start - timedelta(days=1)).replace(day=1)
        return Period(start, end, f"the last {n} months")

    m = re.search(r"\blast (\d+) days\b", t)
    if m:
        n = int(m.group(1))
        return Period(today - timedelta(days=n - 1), today, f"the last {n} days")

    m = re.search(r"\b(\d{1,4})\s*days?\s*(?:ago|back)\b", t)
    if m:
        n = int(m.group(1))
        return Period(today - timedelta(days=n - 1), today, f"the last {n} days")

    m = re.search(r"\b([a-z]{3,9})\s+(\d{4})\b", t)
    if m and m.group(1) in MONTH_NAMES:
        month = MONTH_NAMES[m.group(1)]
        year = int(m.group(2))
        start, end = _month_range(year, month)
        return Period(start, end, start.strftime("%B %Y"))

    for word in re.findall(r"[a-z]{3,9}", t):
        if word in MONTH_NAMES:
            month = MONTH_NAMES[word]
            year = today.year if month <= today.month else today.year - 1
            start, end = _month_range(year, month)
            return Period(start, end, start.strftime("%B %Y"))

    if re.search(r"\blast 30 days\b", t):
        return default_period(today)

    return None
