"""Maximum likelihood estimation for the NK DSGE model with wages.

Uses the Kalman filter to compute the log-likelihood, then optimizes
over the parameter space to find the MLE.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from src.models.dsge.nk_model import NKModel, NKParameters, NKSolution
from src.models.dsge.kalman import kalman_filter


@dataclass
class EstimationResult:
    """Results from DSGE estimation.

    Attributes:
        params: Estimated parameters
        log_likelihood: Log-likelihood at optimum
        solution: Model solution at estimated parameters
        std_errors: Standard errors (from Hessian inverse)
        convergence: Optimizer convergence info
        n_obs: Number of observations used

    """

    params: NKParameters
    log_likelihood: float
    solution: NKSolution
    std_errors: dict[str, float] | None
    convergence: dict
    n_obs: int


# Parameter bounds (prevents numerical issues)
PARAM_BOUNDS = {
    "sigma": (0.1, 5.0),  # IES
    "beta": (0.9, 0.9999),  # Discount factor
    "kappa_p": (0.01, 0.5),  # Price Phillips curve slope
    "kappa_w": (0.01, 0.5),  # Wage Phillips curve slope
    "omega": (0.1, 1.0),  # Okun's law coefficient
    "phi_pi": (1.01, 3.0),  # Taylor rule inflation response (>1 for determinacy)
    "phi_y": (0.0, 2.0),  # Taylor rule output gap response
    "rho_i": (0.5, 0.95),  # Interest rate smoothing
    "rho_demand": (0.01, 0.99),  # Shock persistence
    "rho_supply": (0.01, 0.99),
    "rho_wage": (0.01, 0.99),
    "rho_monetary": (0.01, 0.99),
    "sigma_demand": (0.01, 2.0),  # Shock volatility
    "sigma_supply": (0.01, 2.0),
    "sigma_wage": (0.01, 2.0),
    "sigma_monetary": (0.01, 2.0),
}

# Parameters to estimate (can fix some at prior values)
# Note: rho_monetary removed since monetary shock is iid (persistence via rho_i)
DEFAULT_ESTIMATE_PARAMS = [
    "kappa_p",
    "kappa_w",
    "phi_pi",
    "phi_y",
    "rho_i",
    "rho_demand",
    "rho_supply",
    "rho_wage",
    "sigma_demand",
    "sigma_supply",
    "sigma_wage",
    "sigma_monetary",
]


def params_to_vector(
    params: NKParameters,
    estimate_params: list[str],
) -> np.ndarray:
    """Convert NKParameters to optimization vector."""
    param_dict = params.to_dict()
    return np.array([param_dict[name] for name in estimate_params])


def vector_to_params(
    x: np.ndarray,
    estimate_params: list[str],
    fixed_params: NKParameters,
) -> NKParameters:
    """Convert optimization vector to NKParameters."""
    param_dict = fixed_params.to_dict()
    for i, name in enumerate(estimate_params):
        param_dict[name] = x[i]
    return NKParameters(**param_dict)


def get_bounds(estimate_params: list[str]) -> list[tuple[float, float]]:
    """Get optimization bounds for specified parameters."""
    return [PARAM_BOUNDS[name] for name in estimate_params]


def compute_log_likelihood(
    y: np.ndarray,
    params: NKParameters,
    n_observables: int = 2,
) -> float:
    """Compute log-likelihood for given parameters.

    Args:
        y: Observations (T × n_obs)
        params: Model parameters
        n_observables: Number of observables (2, 3, 4, or 5)

    Returns:
        Log-likelihood (or -1e10 if model fails)

    """
    try:
        model = NKModel(params=params)

        # Check determinacy
        is_det, _ = model.check_determinacy()
        if not is_det:
            return -1e10

        # Solve model
        solution = model.solve()

        # Get state-space matrices (now returns 5 values including H)
        T, R, Z, Q, H = model.state_space_matrices(solution, n_observables=n_observables)

        # Run Kalman filter
        kf = kalman_filter(y, T, R, Z, Q, H=H)

        return kf.log_likelihood

    except Exception:
        return -1e10


def negative_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    estimate_params: list[str],
    fixed_params: NKParameters,
    n_observables: int = 2,
) -> float:
    """Negative log-likelihood for optimization (minimization)."""
    params = vector_to_params(x, estimate_params, fixed_params)
    ll = compute_log_likelihood(y, params, n_observables=n_observables)
    return -ll


def estimate_dsge(
    y: np.ndarray,
    initial_params: NKParameters | None = None,
    estimate_params: list[str] | None = None,
    method: str = "L-BFGS-B",
    options: dict | None = None,
    n_observables: int = 2,
) -> EstimationResult:
    """Estimate DSGE model by maximum likelihood.

    Args:
        y: Observations (T × n_obs)
        initial_params: Starting values for optimization
        estimate_params: Which parameters to estimate (others fixed)
        method: Optimization method (default L-BFGS-B)
        options: Options for scipy.optimize.minimize
        n_observables: Number of observables (2, 3, or 4)

    Returns:
        EstimationResult with estimated parameters and diagnostics

    """
    if initial_params is None:
        initial_params = NKParameters()

    if estimate_params is None:
        estimate_params = DEFAULT_ESTIMATE_PARAMS

    if options is None:
        options = {"maxiter": 1000}

    # Initial vector and bounds
    x0 = params_to_vector(initial_params, estimate_params)
    bounds = get_bounds(estimate_params)

    # Optimize
    result = optimize.minimize(
        negative_log_likelihood,
        x0,
        args=(y, estimate_params, initial_params, n_observables),
        method=method,
        bounds=bounds,
        options=options,
    )

    # Extract estimated parameters
    estimated_params = vector_to_params(result.x, estimate_params, initial_params)

    # Solve model at optimum
    model = NKModel(params=estimated_params)
    solution = model.solve()

    # Compute standard errors from Hessian (if available)
    std_errors = None
    if hasattr(result, "hess_inv"):
        try:
            if isinstance(result.hess_inv, np.ndarray):
                hess_inv = result.hess_inv
            else:
                hess_inv = result.hess_inv.todense()
            variances = np.diag(hess_inv)
            if np.all(variances > 0):
                std_errors = {
                    name: np.sqrt(variances[i])
                    for i, name in enumerate(estimate_params)
                }
        except Exception:
            pass

    return EstimationResult(
        params=estimated_params,
        log_likelihood=-result.fun,
        solution=solution,
        std_errors=std_errors,
        convergence={
            "success": result.success,
            "message": result.message,
            "nit": result.nit,
            "nfev": result.nfev,
        },
        n_obs=len(y),
    )


def print_estimation_results(result: EstimationResult) -> None:
    """Print formatted estimation results."""
    print("=" * 60)
    print("DSGE MODEL ESTIMATION RESULTS")
    print("=" * 60)

    print(f"\nObservations: {result.n_obs}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Convergence: {'Yes' if result.convergence['success'] else 'No'}")
    print(f"  Iterations: {result.convergence['nit']}")
    print(f"  Function evals: {result.convergence['nfev']}")

    print("\n" + "-" * 60)
    print("ESTIMATED PARAMETERS")
    print("-" * 60)

    p = result.params
    se = result.std_errors or {}

    def fmt_param(name: str, value: float, se_val: float | None) -> str:
        if se_val is not None:
            return f"  {name:20s} = {value:8.4f}  (SE: {se_val:.4f})"
        return f"  {name:20s} = {value:8.4f}"

    print("\nStructural parameters:")
    print(fmt_param("sigma (IES)", p.sigma, se.get("sigma")))
    print(fmt_param("beta", p.beta, se.get("beta")))
    print(fmt_param("kappa_p (price PC)", p.kappa_p, se.get("kappa_p")))
    print(fmt_param("kappa_w (wage PC)", p.kappa_w, se.get("kappa_w")))

    print("\nTaylor rule:")
    print(fmt_param("phi_pi", p.phi_pi, se.get("phi_pi")))
    print(fmt_param("phi_y", p.phi_y, se.get("phi_y")))
    print(fmt_param("rho_i (smoothing)", p.rho_i, se.get("rho_i")))

    print("\nShock persistence:")
    print(fmt_param("rho_demand", p.rho_demand, se.get("rho_demand")))
    print(fmt_param("rho_supply", p.rho_supply, se.get("rho_supply")))
    print(fmt_param("rho_wage", p.rho_wage, se.get("rho_wage")))

    print("\nShock volatilities:")
    print(fmt_param("sigma_demand", p.sigma_demand, se.get("sigma_demand")))
    print(fmt_param("sigma_supply", p.sigma_supply, se.get("sigma_supply")))
    print(fmt_param("sigma_wage", p.sigma_wage, se.get("sigma_wage")))
    print(fmt_param("sigma_monetary", p.sigma_monetary, se.get("sigma_monetary")))

    print("\n" + "-" * 60)
    print("POLICY FUNCTION")
    print("-" * 60)
    R = result.solution.R
    print("\n[ŷ, π, π_w]' = R @ [ε_d, ε_s, ε_w, i]'")
    print(f"\n             ε_demand   ε_supply   ε_wage        i")
    print(f"  ŷ:       {R[0, 0]:9.4f}  {R[0, 1]:9.4f}  {R[0, 2]:9.4f}  {R[0, 3]:9.4f}")
    print(f"  π:       {R[1, 0]:9.4f}  {R[1, 1]:9.4f}  {R[1, 2]:9.4f}  {R[1, 3]:9.4f}")
    print(f"  π_w:     {R[2, 0]:9.4f}  {R[2, 1]:9.4f}  {R[2, 2]:9.4f}  {R[2, 3]:9.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    from src.models.dsge.data_loader import get_estimation_arrays

    # Test with 4 observables
    print("Loading data with 4 observables...")
    y, dates = get_estimation_arrays(start="1993Q1", n_observables=4)
    print(f"Data: {len(y)} observations from {dates[0]} to {dates[-1]}")
    print(f"Observables: output_gap, inflation, interest_rate, wage_inflation")

    print("\nEstimating DSGE model with 4 observables...")
    result = estimate_dsge(y, n_observables=4)

    print_estimation_results(result)
