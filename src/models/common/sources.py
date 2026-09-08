"""Source lines built from the data actually loaded, rather than typed by hand.

Every loader in `src.data` returns a `DataSeries` carrying `source` ("ABS",
"RBA", "NY Fed") and, for ABS series, `cat`. The chart footers were hardcoded
strings maintained separately from the loading code, and they had drifted: the
joint y*/u* model named 6202.0 for an unemployment rate that comes from
1364.0.15.003, and no footer anywhere named the GSCPI.

`SourceSet` closes the gap by recording the metadata at the point of loading:

    sources = SourceSet()
    columns = {"log_gdp": sources.take(get_log_gdp(), "log GDP")}
    ...
    print(sources.footer())   # Built using: ABS 5206.0

`take` returns the series' data, so the loading code reads much as it did. The
records are plain tuples, so they pickle into the `constants` dict each model
already saves, and the footer is rebuilt from the run rather than from whatever
the charting module last believed.

Series with `source == "Model"` (the expectations model's saved output) are
recorded but kept out of the footer. They are not a data source, and their own
ABS and survey inputs would otherwise be counted twice.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

from src.data.dataseries import DataSeries

# Sources that name model output rather than a data provider.
DERIVED_SOURCES = frozenset({"Model"})

# The provider whose catalogue numbers are worth listing individually. Every
# other source is named by organisation alone.
_ABS = "ABS"

# "Built using" rather than "Source": these lines name the inputs a model was
# estimated on, which is a wider claim than the provenance of a plotted series.
_PREFIX = "Built using:"

# " (gaps filled)" and similar. The qualifier belongs in the chart's left
# footer or its notes, not in the source line.
_QUALIFIER = re.compile(r"\s*\([^)]*\)")

SourceRecord = tuple[str, str | None]

# A saved record is (source, cat). Anything else in the pickle is not one.
_RECORD_LEN = 2


def _clean(source: str) -> list[str]:
    """Split a free-text source into the organisations it names.

    `source` is not a controlled vocabulary: `get_corporate_spread` reports
    "Bloomberg; RBA" and `get_indexed_yield_filled` "RBA F2 (gaps filled)".
    """
    parts = (_QUALIFIER.sub("", part).strip() for part in source.split(";"))
    return [part for part in parts if part]


def _collapse_to_general(names: list[str]) -> list[str]:
    """Drop a name that extends another, keeping the more general one.

    `rstar` reads the indexed yield from "RBA F2" and the cash rate from "RBA".
    Keeping both is noise, and keeping only "RBA F2" would say the whole line
    came from that one table, which is worse than saying "RBA". A footer names
    providers; the tables belong in the model's notes.
    """
    return [
        name for name in names
        if not any(other != name and name.startswith(other + " ") for other in names)
    ]


def render_footer(records: list[SourceRecord] | tuple[SourceRecord, ...]) -> str:
    """Render "Built using: ..." from `(source, cat)` records."""
    abs_cats = sorted({cat for source, cat in records if source == _ABS and cat})
    others = sorted({
        name
        for source, _ in records
        if source not in DERIVED_SOURCES and source != _ABS
        for name in _clean(source)
    })
    if any(source == _ABS for source, _ in records) and not abs_cats:
        others.insert(0, _ABS)

    groups = []
    if abs_cats:
        groups.append(f"{_ABS} " + ", ".join(abs_cats))
    groups.extend(_collapse_to_general(others))
    return f"{_PREFIX} " + "; ".join(groups) if groups else ""


@dataclass
class SourceSet:
    """The providers behind one set of observations, collected as they load."""

    records: list[SourceRecord] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def take(self, series: DataSeries, label: str | None = None, key: str | None = None) -> pd.Series:
        """Record where `series` came from and return its data.

        `label` is the human name for the verbose input listing; it is stored
        against `key`, defaulting to the label itself.
        """
        self.add(series.source, series.cat)
        if label is not None:
            self.labels[key or label] = self.describe(series, label)
        return series.data

    def add(self, source: str, cat: str | None = None) -> None:
        """Record a source directly, for data that arrives without a `DataSeries`."""
        record = (source, cat)
        if record not in self.records:
            self.records.append(record)

    def merge(self, other: SourceSet) -> SourceSet:
        """Return a new set holding both sides' records and labels."""
        merged = SourceSet(records=list(self.records), labels=dict(self.labels))
        for source, cat in other.records:
            merged.add(source, cat)
        merged.labels.update(other.labels)
        return merged

    @staticmethod
    def describe(series: DataSeries, label: str) -> str:
        """Return "label (5206.0)", or the source's name where there is no catalogue."""
        if series.cat:
            return f"{label} ({series.cat})"
        if series.source in DERIVED_SOURCES:
            return f"{label} (model output)"
        return f"{label} ({series.source})"

    def note(self, key: str, label: str) -> None:
        """Record a label for a series derived from ones already taken.

        The Solow residual has no provider of its own: it is built from series
        whose sources are already recorded, so it needs a name and nothing else.
        """
        self.labels[key] = label

    def label(self, key: str) -> str:
        """Return the recorded label for `key`, or `key` where none was given."""
        return self.labels.get(key, key)

    def footer(self) -> str:
        """Return the "Built using: ..." line for everything recorded here."""
        return render_footer(self.records)

    def to_records(self) -> list[SourceRecord]:
        """Return the records as plain tuples, for saving alongside a run."""
        return list(self.records)

    @classmethod
    def from_records(cls, records: object) -> SourceSet | None:
        """Rebuild a set from saved records, or None where a run saved none."""
        if not isinstance(records, list | tuple):
            return None
        rebuilt = cls()
        for record in records:
            if isinstance(record, list | tuple) and len(record) == _RECORD_LEN:
                source, cat = record
                if isinstance(source, str) and (cat is None or isinstance(cat, str)):
                    rebuilt.add(source, cat)
        return rebuilt if rebuilt.records else None


def footer_from_constants(constants: dict[str, object], key: str = "sources") -> str | None:
    """Return the "Built using" line saved with a run, or None if it saved none."""
    collected = SourceSet.from_records(constants.get(key))
    return collected.footer() if collected is not None else None
