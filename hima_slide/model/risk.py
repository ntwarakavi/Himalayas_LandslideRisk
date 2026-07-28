"""Risk stage - not implemented.

Risk is the expected consequence of landsliding, and requires two inputs this
package does not yet have:

    risk = hazard x exposure x vulnerability

``hazard`` is produced by :mod:`hima_slide.model.hazard` as a probability
per scenario event. ``exposure`` and ``vulnerability`` are absent.

Scope required to complete this stage
-------------------------------------
Exposure
    Assets and their locations. The usual treatment for linear infrastructure
    is to segment road and railway centrelines into 1 km units with a buffer of
    a few hundred metres either side, and assign each unit the maximum failure
    probability within its buffer. OpenStreetMap supplies road and rail
    geometry for the region; WorldPop supplies population.

Vulnerability
    The damage expected to an asset given an impact, by asset class. Treating a
    landslide intersecting a unit as a full impact on that unit is the usual
    first approximation, but it is a binary simplification rather than a damage
    function, and runout is not modelled here at all: the stability model says
    where material detaches, not where it arrives.

Annualisation
    Scenario hazard is conditional on a stated trigger. Converting it to an
    expected annual loss requires integrating over the return periods of the
    triggering events, weighted by their annual exceedance probabilities.

Blocking dependency
-------------------
The failure probability this model produces is a *relative* field. Its level
depends on how background points were drawn during fitting, and on two trigger
conventions that are not fitted here - the rainfall coefficient of variation
and the pseudo-static fraction of PGA. Differences between pixels are
meaningful; the value at a pixel is not a frequency of failure per year.

Risk figures carry that level straight through to monetary or casualty
estimates, so absolute losses cannot be computed until the probability is
anchored against observed landslide frequency over a known period.
"""

from __future__ import annotations


def compute_risk(*args, **kwargs):
    """Placeholder. See the module docstring for the required inputs."""
    raise NotImplementedError(
        "The risk stage is not implemented: it needs an exposure layer (roads, "
        "rail, population), a vulnerability model, a runout model, and a "
        "failure probability anchored to observed landslide frequency. See "
        "hima_slide/model/risk.py for the full scope."
    )
