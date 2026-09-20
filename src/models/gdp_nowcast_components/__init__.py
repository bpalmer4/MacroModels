"""Components (expenditure-identity) GDP nowcast."""

# Here rather than in `model`, because `diagnostics` writes to the same
# directory and `model` calls `diagnostics`: holding it in either one makes the
# two import each other.
CHART_DIR = "./charts/GDP-Nowcast-Components/"
