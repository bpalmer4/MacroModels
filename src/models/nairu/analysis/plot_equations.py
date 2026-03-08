"""Render model equations as a publication-quality image for blog posts.

Dynamically extracts equations from the model's equation functions
(stored on model._equations by build_model), so the plot always matches
the actual model specification.
"""

import inspect
import re
from pathlib import Path
from typing import Any, Callable

import mgplot as mg
import matplotlib.pyplot as plt

# Regex to extract the 'Model:' line from a function's docstring
_MODEL_LINE_RE = re.compile(r"^\s*Model:\s*(.+)$", re.MULTILINE)

# Unicode → LaTeX substitution rules (applied in order)
_LATEX_SUBS = [
    # Multi-character patterns first (longest match wins)
    # Compound terms containing Greek letters
    ("Δ4ρm", r"\Delta_4\rho^m"),
    ("Δ4twi", r"\Delta_4\mathrm{twi}"),
    ("Δ4oil", r"\Delta_4\mathrm{oil}"),
    ("ΔMFP", r"\Delta\mathrm{MFP}"),
    ("Δulc", r"\Delta\mathrm{ulc}"),
    ("Δhcoe", r"\Delta\mathrm{hcoe}"),
    ("Δemp", r"\Delta\mathrm{emp}"),
    ("Δtwi", r"\Delta\mathrm{twi}"),
    ("Δ(NX/Y)", r"\Delta(\mathrm{NX}/Y)"),
    ("Δpr", r"\Delta\mathrm{pr}"),
    ("Δe", r"\Delta e"),
    ("ΔK", r"\Delta K"),
    ("ΔL", r"\Delta L"),
    ("ΔU", r"\Delta U"),
    # Named terms (before Greek letters they might contain)
    ("NAIRU", r"\mathrm{NAIRU}"),
    ("MFP", r"\mathrm{MFP}"),
    ("GSCPI", r"\mathrm{GSCPI}"),
    ("StudentT", r"\mathrm{StudentT}"),
    ("SkewNormal", r"\mathrm{SkewNormal}"),
    ("output_gap", r"Y^\mathrm{gap}"),
    ("real_wage_gap", r"\mathrm{rwg}"),
    ("u_gap", r"u^\mathrm{gap}"),
    ("U_gap", r"U^\mathrm{gap}"),
    ("Y_gap", r"Y^\mathrm{gap}"),
    ("y_gap", r"y^\mathrm{gap}"),
    ("r_gap", r"(r - r^*)"),
    ("fiscal", r"\mathrm{fiscal}"),
    ("controls", r"\mathrm{controls}"),
    ("quarterly", r"\mathrm{quarterly}"),
    # Subscripted Greek (before bare Greek)
    ("π_exp", r"\pi^e"),
    ("α_t", r"\alpha_t"),
    ("β_is", r"\beta"),
    ("β_pr", r"\beta"),
    ("β_pt", r"\beta_\mathrm{pt}"),
    ("β_oil", r"\beta_\mathrm{oil}"),
    ("β_r", r"\beta"),
    ("β_ygap", r"\beta_y"),
    ("β_wage", r"\beta_w"),
    ("β₁", r"\beta_1"),
    ("β₂", r"\beta_2"),
    ("γ_regime", r"\gamma_\mathrm{regime}"),
    ("γ_fi", r"\gamma"),
    ("γ_pi", r"\gamma"),
    ("ρ_is", r"\rho"),
    ("τ₁", r"\tau_1"),
    ("τ₂", r"\tau_2"),
    # Bare Greek letters (after all compound patterns)
    ("α", r"\alpha"),
    ("β", r"\beta"),
    ("γ", r"\gamma"),
    ("δ", r"\delta"),
    ("ε", r"\varepsilon"),
    ("η", r"\eta"),
    ("λ", r"\lambda"),
    ("π", r"\pi"),
    ("ρ", r"\rho"),
    ("σ", r"\sigma"),
    ("ψ", r"\psi"),
    ("ξ", r"\xi"),
    ("ν", r"\nu"),
    # Special symbols
    ("r*", r"r^*"),
    ("U*", r"U^*"),
    ("Y*", r"Y^*"),
    # Operators
    ("×", r"\,"),
    ("~", r"\sim"),
]


def _unicode_to_latex(model_line: str) -> str:
    """Convert a Unicode model equation to LaTeX."""
    result = model_line
    for old, new in _LATEX_SUBS:
        result = result.replace(old, new)
    return result


def _get_model_line(func: Callable) -> str | None:
    """Extract the 'Model:' line from a function's docstring."""
    doc = inspect.getdoc(func)
    if doc is None:
        return None
    m = _MODEL_LINE_RE.search(doc)
    return m.group(1).strip() if m else None


def _get_equation_label(func: Callable) -> str:
    """Extract the first line of a function's docstring as the equation name."""
    doc = inspect.getdoc(func) or ""
    return doc.split("\n")[0].rstrip(".")


# Section classification based on function module/name
_SECTION_KEYWORDS = {
    "State equations": ["nairu_equation", "nairu_student_t", "potential_output"],
    "Core observation equations": ["okun", "price_inflation", "wage_growth", "hourly_coe"],
    "Demand side": ["is_equation", "is_curve"],
    "Labour supply": ["participation"],
    "Employment": ["employment"],
    "Open economy": ["exchange_rate", "import_price"],
    "Trade": ["net_exports"],
}

_SECTION_ORDER = [
    "State equations",
    "Core observation equations",
    "Demand side",
    "Labour supply",
    "Employment",
    "Open economy",
    "Trade",
]


def _classify_equation(func: Callable) -> str:
    """Classify an equation function into a section."""
    name = func.__name__
    for section, keywords in _SECTION_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return section
    return "Other"


def _build_sections(equations: list[Callable]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Group equations into sections with (label, latex) pairs."""
    section_eqs: dict[str, list[tuple[str, str]]] = {}
    for func in equations:
        model_line = _get_model_line(func)
        if model_line is None:
            continue
        label = _get_equation_label(func)
        latex = _unicode_to_latex(model_line)
        section = _classify_equation(func)
        if section not in section_eqs:
            section_eqs[section] = []
        section_eqs[section].append((label, latex))

    return [
        (section, section_eqs[section])
        for section in _SECTION_ORDER
        if section in section_eqs
    ]


def plot_equations(
    equations: list[Callable] | None = None,
    constants: dict[str, Any] | None = None,
    dpi: int = 200,
    show: bool = False,
) -> None:
    """Render model equations as a publication-quality image."""
    if equations is None:
        return

    sections = _build_sections(equations)
    n_equations = sum(len(eqs) for _, eqs in sections)
    has_constants = bool(constants)

    fontsize_eq = 13
    fontsize_label = 11
    fontsize_section = 12
    fontsize_title = 16
    fontsize_const = 11
    line_height = 0.055
    section_gap = 0.025
    eq_indent = 0.04

    n_lines = n_equations + len(sections)
    if has_constants:
        n_lines += 1 + len(constants)
    total_height = n_lines * line_height + (len(sections) + has_constants) * section_gap + 0.15
    fig_width = 14
    fig_height = max(total_height * 20, 7.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.96
    ax.text(
        0.5, y, "NAIRU + Output Gap Model \u2014 Equations",
        fontsize=fontsize_title, fontweight="bold", ha="center", va="top",
        fontfamily="serif",
    )
    y -= 0.06

    eq_num = 1
    for section_title, eqs in sections:
        ax.text(
            0.02, y, section_title,
            fontsize=fontsize_section, fontweight="bold", va="top",
            fontstyle="italic", fontfamily="serif", color="#444444",
        )
        y -= line_height * 0.7

        for label, latex in eqs:
            ax.text(
                eq_indent, y, f"{eq_num}. {label}",
                fontsize=fontsize_label, va="top",
                fontfamily="serif", color="#333333",
            )
            y -= line_height * 0.85

            ax.text(
                eq_indent + 0.04, y, f"${latex}$",
                fontsize=fontsize_eq, va="top",
                fontfamily="serif",
            )
            y -= line_height * 1.1

            eq_num += 1

        y -= section_gap

    if has_constants:
        ax.text(
            0.02, y, "Fixed constants",
            fontsize=fontsize_section, fontweight="bold", va="top",
            fontstyle="italic", fontfamily="serif", color="#444444",
        )
        y -= line_height * 0.7

        for name, value in constants.items():
            latex_name = _unicode_to_latex(name)
            ax.text(
                eq_indent + 0.04, y,
                f"${latex_name} = {value}$",
                fontsize=fontsize_const, va="top",
                fontfamily="serif", color="#333333",
            )
            y -= line_height * 0.9

    chart_dir = Path(mg.get_setting("chart_dir"))
    chart_dir.mkdir(parents=True, exist_ok=True)
    output_path = chart_dir / "plot-model-equations.png"

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)
