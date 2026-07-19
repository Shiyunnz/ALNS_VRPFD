"""Supply-class deprivation cost functions.

Implements the Holguin-Wang exponential projection: each supply class has a
class-specific exponential growth rate beta_c projected from Wang et al.'s
logistic DLF front segment, combined with a global cost-strength parameter
lambda and class importance weights omega_c.

The unified delay cost is:

    f_{c_i}(tau) = lambda * omega_c * (exp(1.5031 + beta_c * rho * tau) - exp(1.5031))

or in normalized form:

    f_{c_i}(tau) = lambda * omega_c *
        (exp(1.5031 + beta_c * rho * tau) - exp(1.5031))
        / (exp(1.5031 + beta_c * x_tau_max) - exp(1.5031))

Reference: code/docs/holguin_wang_exponential_projection.md
"""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = [
    "SupplyClassSpec",
    "WANG_SUPPLY_CLASSES",
    "DEFAULT_SUPPLY_CLASS_SEQUENCE",
    "deprivation_cost",
]


@dataclass(frozen=True)
class SupplyClassSpec:
    """Parameters for supply-class deadlines and deprivation costs."""

    key: str
    label: str
    deadline_optimal_delta_hours: tuple[float, float]
    deadline_latest_delta_hours: tuple[float, float]
    beta: float
    omega: float


WANG_SUPPLY_CLASSES: dict[str, SupplyClassSpec] = {
    "medicine": SupplyClassSpec(
        key="medicine",
        label="Medicine",
        deadline_optimal_delta_hours=(0.50, 1.00),
        deadline_latest_delta_hours=(0.80, 1.50),
        beta=0.4558,
        omega=1.35,
    ),
    "water": SupplyClassSpec(
        key="water",
        label="Drinking water",
        deadline_optimal_delta_hours=(1.00, 1.80),
        deadline_latest_delta_hours=(1.20, 2.20),
        beta=0.4525,
        omega=1.35,
    ),
    "food": SupplyClassSpec(
        key="food",
        label="Food",
        deadline_optimal_delta_hours=(1.80, 3.00),
        deadline_latest_delta_hours=(1.80, 3.00),
        beta=0.4464,
        omega=1.0,
    ),
    "tent": SupplyClassSpec(
        key="tent",
        label="Tent",
        deadline_optimal_delta_hours=(2.50, 4.00),
        deadline_latest_delta_hours=(2.50, 4.50),
        beta=0.4469,
        omega=0.75,
    ),
}

DEFAULT_SUPPLY_CLASS_SEQUENCE: tuple[str, ...] = (
    "medicine",
    "water",
    "food",
    "tent",
)

HOLGUIN_INTERCEPT: float = 1.5031

MAX_TARDINESS_HOURS: float = 4.4947


def _normalise_supply_class(supply_class: str | None) -> str:
    if supply_class is None:
        return "water"
    key = str(supply_class).strip().lower()
    aliases = {
        "health": "medicine",
        "medical": "medicine",
        "med": "medicine",
        "wash": "water",
        "drinking_water": "water",
        "shelter": "tent",
    }
    key = aliases.get(key, key)
    if key not in WANG_SUPPLY_CLASSES:
        return "water"
    return key


def deprivation_cost(
    tau_hours: float,
    supply_class: str | None = "water",
    *,
    cost_lambda: float = 12.0,
    rho: float = 1.0,
    normalized: bool = True,
) -> float:
    """Return class-specific exponential deprivation cost.

    Parameters
    ----------
    tau_hours : float
        Operational tardiness beyond the soft deadline, in hours.
    supply_class : str or None
        Supply class key (medicine, water, food, tent).
    cost_lambda : float
        Global cost-strength parameter controlling the delay-cost share.
    rho : float
        Time-scale mapping from operational hours to literature deprivation days.
        rho=1/24 treats 1 operational hour as 1 actual hour in Wang's day-scale.
        Larger rho compresses more operational delay into the exponential curve.
    normalized : bool
        If True, divide by the cost at the maximum tardiness horizon to separate
        curve shape from class importance. Recommended for calibration.
    """
    tau = max(0.0, float(tau_hours))
    if tau == 0.0:
        return 0.0
    key = _normalise_supply_class(supply_class)
    spec = WANG_SUPPLY_CLASSES[key]
    beta_c = spec.beta
    omega_c = spec.omega

    exp_intercept = math.exp(HOLGUIN_INTERCEPT)
    raw_cost = math.exp(HOLGUIN_INTERCEPT + beta_c * rho * tau) - exp_intercept

    if normalized:
        x_tau_max = rho * MAX_TARDINESS_HOURS
        normalizer = math.exp(HOLGUIN_INTERCEPT + beta_c * x_tau_max) - exp_intercept
        normalized_cost = raw_cost / normalizer
    else:
        normalized_cost = raw_cost

    return cost_lambda * omega_c * normalized_cost