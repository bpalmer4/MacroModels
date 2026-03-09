"""Model configuration for NAIRU + Output Gap estimation.

ModelConfig is the single source of truth for all model options. It controls:
- Which equations are included
- State equation variants (Gaussian vs Student-t, Normal vs SkewNormal)
- Regime switching
- Per-equation fixed constants
- Human-readable label (shown on all charts and saved with results)

ModelConfig is serializable — it is saved alongside the trace and observations
so that analyse.py and forecast.py know exactly what produced the results.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# Regime boundary periods (Phillips curve slope regimes)
REGIME_GFC_START = pd.Period("2008Q4")
REGIME_COVID_START = pd.Period("2021Q1")


@dataclass
class ModelConfig:
    """Complete specification of a NAIRU + Output Gap model variant.

    The label field appears on all charts and is persisted with saved results,
    so every output is self-describing regardless of file location.

    Attributes:
        label: Human-readable variant name (shown on charts)

        # State equation options
        student_t_nairu: Use Student-t(nu=4) innovations instead of Gaussian
        skewnormal_potential: Use SkewNormal innovations for potential output

        # Observation equation inclusion
        include_price_inflation: Include price Phillips curve (always True)
        include_wage_growth: Include ULC wage Phillips curve
        include_hourly_coe: Include hourly COE wage Phillips curve
        include_okun: Include Okun's Law
        include_is_curve: Include IS curve
        include_participation: Include participation rate equation
        include_employment: Include employment equation
        include_exchange_rate: Include exchange rate equation
        include_import_price: Include import price pass-through equation
        include_net_exports: Include net exports equation

        # Equation variants
        regime_switching: Use regime-switching Phillips curve slopes
        okun_gap_form: Use gap-to-gap Okun's Law instead of simple change form
        wage_expectations: Include inflation expectations in wage equations
        wage_price_passthrough: Include demand deflator pass-through in wages

        # Supply-side controls in Phillips curve
        include_import_price_control: Include lagged import price growth
        include_gscpi_control: Include GSCPI (COVID supply chain pressure)

        # Per-equation fixed constants
        nairu_const: Fixed values for NAIRU equation
        potential_const: Fixed values for potential output equation
        price_inflation_const: Fixed values for price Phillips curve
        wage_growth_const: Fixed values for wage Phillips curve
        hourly_coe_const: Fixed values for hourly COE Phillips curve
        okun_const: Fixed values for Okun's Law
        is_curve_const: Fixed values for IS curve
        participation_const: Fixed values for participation equation
        employment_const: Fixed values for employment equation
        exchange_rate_const: Fixed values for exchange rate equation
        import_price_const: Fixed values for import price equation
        net_exports_const: Fixed values for net exports equation

    """

    label: str = "default"

    # State equation options
    student_t_nairu: bool = False
    skewnormal_potential: bool = False

    # Observation equation inclusion
    include_price_inflation: bool = True
    include_wage_growth: bool = True
    include_hourly_coe: bool = True
    include_okun: bool = True
    include_is_curve: bool = True
    include_participation: bool = False
    include_employment: bool = False
    include_exchange_rate: bool = False
    include_import_price: bool = False
    include_net_exports: bool = False

    # Equation variants
    regime_switching: bool = False
    okun_gap_form: bool = False
    wage_expectations: bool = True
    wage_price_passthrough: bool = False

    # Supply-side controls in Phillips curve
    include_import_price_control: bool = True
    include_gscpi_control: bool = True

    # Per-equation fixed constants (None = use equation defaults)
    # nairu_const default depends on student_t_nairu (set in __post_init__)
    nairu_const: dict[str, Any] | None = None
    potential_const: dict[str, Any] = field(default_factory=lambda: {"potential_innovation": 0.3})
    price_inflation_const: dict[str, Any] | None = None
    wage_growth_const: dict[str, Any] | None = None
    hourly_coe_const: dict[str, Any] | None = None
    okun_const: dict[str, Any] | None = None
    is_curve_const: dict[str, Any] | None = None
    participation_const: dict[str, Any] | None = None
    employment_const: dict[str, Any] | None = None
    exchange_rate_const: dict[str, Any] | None = None
    import_price_const: dict[str, Any] | None = None
    net_exports_const: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Set default NAIRU innovation based on distribution type."""
        if self.nairu_const is None:
            innovation = 0.10 if self.student_t_nairu else 0.15
            self.nairu_const = {"nairu_innovation": innovation}

    @property
    def rfooter(self) -> str:
        """Right footer string for charts."""
        return f"NAIRU-{self.label}"

    @property
    def chart_dir_name(self) -> str:
        """Default chart subdirectory name derived from label."""
        return f"nairu_{self.label.lower().replace(' ', '_')}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for saving with results."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Reconstruct from saved dict."""
        return cls(**d)

    def _observation_equations(self) -> list[str]:  # noqa: C901 — flat feature-flag list, not genuinely complex
        """List active observation equations."""
        regime = " [regime-switching]" if self.regime_switching else ""
        eqs = []
        if self.include_okun:
            form = "gap-to-gap" if self.okun_gap_form else "simple change"
            eqs.append(f"  Okun's Law ({form})")
        if self.include_price_inflation:
            eqs.append(f"  Price Phillips curve{regime}")
        if self.include_wage_growth:
            eqs.append(f"  Wage Phillips curve (ULC){regime}")
        if self.include_hourly_coe:
            eqs.append(f"  Hourly COE Phillips curve{regime}")
        if self.include_is_curve:
            eqs.append("  IS curve")
        if self.include_participation:
            eqs.append("  Participation rate")
        if self.include_employment:
            eqs.append("  Employment")
        if self.include_exchange_rate:
            eqs.append("  Exchange rate (UIP)")
        if self.include_import_price:
            eqs.append("  Import price pass-through")
        if self.include_net_exports:
            eqs.append("  Net exports")
        return eqs

    def summary(self) -> str:
        """Human-readable summary of active equations."""
        lines = [f"Model variant: {self.label}"]
        lines.append("")

        # State equations
        nairu_type = "Student-t(nu=4)" if self.student_t_nairu else "Gaussian"
        potential_type = "SkewNormal" if self.skewnormal_potential else "Gaussian"
        lines.append(f"  NAIRU:      {nairu_type} random walk")
        lines.append(f"  Potential:  Cobb-Douglas + {potential_type} innovations")

        # Observation equations
        lines.append("")
        eqs = self._observation_equations()
        lines.extend(eqs)
        lines.append(f"  Total: 2 state + {len(eqs)} observation equations")

        # Controls
        controls = []
        if self.include_import_price_control:
            controls.append("import prices")
        if self.include_gscpi_control:
            controls.append("GSCPI")
        if controls:
            lines.append(f"  Phillips controls: {', '.join(controls)}")

        return "\n".join(lines)


# --- Preset Configurations ---

SIMPLE = ModelConfig(label="simple")

COMPLEX = ModelConfig(
    label="complex",
    student_t_nairu=True,
    regime_switching=True,
    okun_gap_form=True,
    include_import_price_control=True,
    include_exchange_rate=True,
    include_import_price=True,
    include_participation=True,
    include_employment=True,
    include_net_exports=True,
    wage_expectations=True,
)

# Registry of named presets
PRESETS: dict[str, ModelConfig] = {
    "default": ModelConfig(),
    "simple": SIMPLE,
    "complex": COMPLEX,
}
