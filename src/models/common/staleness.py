"""Whether a saved run is from today, which decides if a comparison re-runs it."""

from datetime import datetime
from pathlib import Path


def is_current(path: Path) -> bool:
    """Return True if the file at `path` exists and was written today, local time."""
    if not path.exists():
        return False
    written = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    return written.date() == datetime.now().astimezone().date()
