"""Where the project's directories are.

Resolved from this file rather than the working directory, so a script run from
anywhere finds the same tree. Defined once because the alternative is every
module counting its own depth back to the root, and a module moved one level
then reads a directory that does not exist, or worse, one that does.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CACHE = ROOT / ".readabs_cache"
CHARTS = ROOT / "charts"
INPUT_DATA = ROOT / "input_data"
MODEL_OUTPUTS = ROOT / "model_outputs"
OUTPUT = ROOT / "output"
