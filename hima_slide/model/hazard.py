"""Hazard: failure probability under a stated triggering scenario.

Susceptibility asks whether terrain is predisposed to fail. Hazard asks how
likely it is to fail given a trigger of stated severity. In a physically based
model that is not a separate calculation but the same one with the trigger
written into the force balance, so this module supplies the two quantities the
stability model needs and nothing else:

* **Rainfall** raises recharge. R appears in the wetness term only as the ratio
  R/T, so a storm enters as a multiplier on R/T. Wetter conditions push more of
  the soil column towards saturation, which cuts the effective normal stress
  and lowers the factor of safety.

* **Earthquakes** add an inertial force. That enters as the horizontal seismic
  coefficient k_h in the pseudo-static factor of safety.

Both reduce to one scalar the stability model already accepts, so hazard and
susceptibility come from the same code path with different arguments.

The recharge multiplier
-----------------------

Annual maximum daily rainfall is well described by a Gumbel distribution, under
which the depth with return period T is

    I(T) = mu + sigma * (sqrt(6)/pi) * (y_T - gamma),    y_T = -ln(-ln(1 - 1/T))

The stability parameters are fitted against whatever recharge prevailed during
the inventory's triggering events, so what matters is the *ratio* of a scenario
depth to that reference. Taking the ratio cancels mu and leaves a dependence on
the coefficient of variation of the annual maxima alone:

    m(T) = [1 + cv*k(T)] / [1 + cv*k(T_ref)],    k(T) = (sqrt(6)/pi)(y_T - gamma)

``cv`` is the one parameter here that is not derived from the data in this
package. For monsoon Asia, station analyses put the coefficient of variation of
annual maximum 24 h rainfall at roughly 0.25 to 0.35; 0.30 is the default. It
is a single interpretable number, and the sensitivity of the result to it can
be checked by rerunning with the ends of that range.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..utility.grid import map_raster

HAZARD_NODATA = -9999.0

_EULER = 0.5772156649015329
_SQRT6_PI = math.sqrt(6.0) / math.pi

#: Coefficient of variation of annual maximum 24 h rainfall. See module notes.
DEFAULT_RAINFALL_CV = 0.30

#: Return period the fitted parameters are taken to represent. Two years is the
#: near-median annual maximum: a wet season that happens most years, which is
#: the condition a multi-year inventory mostly samples.
REFERENCE_RETURN_PERIOD_YR = 2.0

#: Fraction of peak ground acceleration used as the pseudo-static coefficient.
#: A pseudo-static analysis applies a sustained force where the real loading is
#: a brief oscillation, so using the full peak is markedly conservative;
#: one half is the long-standing convention (Hynes-Griffin & Franklin 1984).
DEFAULT_PGA_FRACTION = 0.5


# ---------------------------------------------------------------------------
# rainfall
# ---------------------------------------------------------------------------

def _gumbel_factor(return_period_yr: float) -> float:
    """The reduced-variate term k(T) in the Gumbel quantile."""
    if return_period_yr <= 1.0:
        raise ValueError("return period must exceed 1 year")
    y = -math.log(-math.log(1.0 - 1.0 / return_period_yr))
    return _SQRT6_PI * (y - _EULER)


def recharge_multiplier(return_period_yr: float,
                        cv: float = DEFAULT_RAINFALL_CV,
                        reference_yr: float = REFERENCE_RETURN_PERIOD_YR
                        ) -> float:
    """Ratio of scenario recharge to the recharge the parameters were fitted at.

    Multiplies R/T in the wetness term. Returns 1.0 at the reference return
    period by construction.
    """
    num = 1.0 + cv * _gumbel_factor(return_period_yr)
    den = 1.0 + cv * _gumbel_factor(reference_yr)
    return float(num / den)


def return_period_from_normalised(z: np.ndarray) -> np.ndarray:
    """Return period (years) of a normalised 24 h rainfall z = (I - mu)/sigma.

    The inverse of the Gumbel quantile, for users who supply a standardised
    rainfall grid rather than a single scenario value.
    """
    y = z / _SQRT6_PI + _EULER
    with np.errstate(over="ignore"):
        p = 1.0 - np.exp(-np.exp(-y))
    return 1.0 / np.clip(p, 1e-9, 1.0)


# ---------------------------------------------------------------------------
# earthquakes
# ---------------------------------------------------------------------------

def seismic_coefficient(pga_g: float,
                        fraction: float = DEFAULT_PGA_FRACTION) -> float:
    """Pseudo-static horizontal seismic coefficient from peak ground acceleration."""
    return float(fraction * pga_g)


def seismic_coefficient_raster(pga_path: str, out_path: str,
                               fraction: float = DEFAULT_PGA_FRACTION,
                               block: int = 1024) -> str:
    """Convert a PGA raster (in g) to a seismic-coefficient raster."""
    def fn(pga: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(pga), HAZARD_NODATA, fraction * pga)

    return map_raster(pga_path, out_path, fn, "float32", HAZARD_NODATA,
                      block=block)


# ---------------------------------------------------------------------------
# putting a scenario together
# ---------------------------------------------------------------------------

def scenario_terms(trigger: str, return_period_yr: float = 100.0,
                   pga_g: float = 0.30, cv: float = DEFAULT_RAINFALL_CV,
                   pga_fraction: float = DEFAULT_PGA_FRACTION,
                   reference_yr: float = REFERENCE_RETURN_PERIOD_YR) -> dict:
    """The two scalars a trigger contributes to the factor of safety.

    Returns ``{"recharge_multiplier": m, "k_h": k}``. A rainfall scenario
    raises recharge and leaves k_h at zero; an earthquake scenario adds k_h and
    leaves recharge at the fitted level, since the shaking says nothing about
    how wet the ground already is. Antecedent wetness for a seismic scenario is
    set by passing a return period as well.
    """
    if trigger == "rainfall":
        return {"recharge_multiplier":
                recharge_multiplier(return_period_yr, cv, reference_yr),
                "k_h": 0.0}
    if trigger == "earthquake":
        return {"recharge_multiplier": 1.0,
                "k_h": seismic_coefficient(pga_g, pga_fraction)}
    raise ValueError(f"unknown trigger {trigger!r}")


def describe_scenario(trigger: str, terms: dict,
                      return_period_yr: Optional[float] = None,
                      pga_g: Optional[float] = None) -> str:
    """One-line human-readable summary of a scenario, for run reports."""
    if trigger == "rainfall":
        return (f"{return_period_yr:g}-year rainfall, recharge "
                f"{terms['recharge_multiplier']:.2f}x the fitted level")
    return (f"PGA {pga_g:g} g, seismic coefficient k_h = "
            f"{terms['k_h']:.3f}")
