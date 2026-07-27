"""Risk stage - not implemented.

Risk is the expected consequence of landsliding, and requires two inputs this
package does not yet have:

    risk = hazard x exposure x vulnerability

``hazard`` is produced by :mod:`giri_landslide.model.hazard` as a probability
per scenario event. ``exposure`` and ``vulnerability`` are absent.

Scope required to complete this stage
-------------------------------------
Exposure
    Assets and their locations. The manuscript uses road and railway
    centrelines, segmented into 1 km units with a 300 m buffer each side, and
    assigns each unit the maximum susceptibility within its buffer. OpenStreetMap
    supplies road and rail geometry for the region; WorldPop supplies population.

Vulnerability
    The damage expected to an asset given an impact, by asset class. The
    manuscript treats a landslide intersecting a 1 km unit as a full impact on
    that unit, which is a binary simplification rather than a damage function.

Annualisation
    Scenario hazard is conditional on a stated trigger. Converting it to an
    expected annual loss requires integrating over the return periods of the
    triggering events, weighted by their annual exceedance probabilities.

Blocking dependency
-------------------
The rainfall hazard matrix in :mod:`giri_landslide.config` is a placeholder.
The manuscript publishes it only as a figure, so the values in
``RAINFALL_MATRIX`` are illustrative and the absolute rainfall-triggered
probabilities are not calibrated. Risk figures derived from them would carry
that error through to monetary or casualty estimates. The earthquake matrix is
transcribed exactly and does not have this problem.

Until the matrix is calibrated against observed landslide frequency, this stage
should not be used to produce absolute loss estimates.
"""

from __future__ import annotations


def compute_risk(*args, **kwargs):
    """Placeholder. See the module docstring for the required inputs."""
    raise NotImplementedError(
        "The risk stage is not implemented: it needs an exposure layer (roads, "
        "rail, population), a vulnerability model, and a calibrated rainfall "
        "hazard matrix. See giri_landslide/model/risk.py for the full scope."
    )
