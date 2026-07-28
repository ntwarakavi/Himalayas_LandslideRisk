"""Climate scenarios: present-day baseline and CMIP6 futures.

Climate enters the stability model at exactly one place. Recharge ``R`` appears
in the wetness term only through the ratio ``R/T``, so a change in rainfall is a
multiplier on that ratio:

    w = min( R a / (T sin theta), 1 )    ->    R/T  ->  (R/T) * (P / P_ref)

``P`` is wettest-month precipitation, the season when the soil column is closest
to saturation and when the inventories were mostly filled. ``P_ref`` is a fixed
reference in millimetres, recorded when the soil parameters are fitted.

The reference is the part that matters and the part that is easy to get wrong.
A future field must be normalised by the **present-day** reference, not by its
own median. Normalising each scenario by its own statistics would divide out
exactly the signal being looked for: a uniformly wetter future would come back
looking identical to today.

What a scenario does and does not change
----------------------------------------

Changed
    The recharge field, and therefore wetness, the factor of safety, and the
    failure probability.

Unchanged
    The soil parameters, which are properties of soil rather than of climate;
    the terrain; and the definition of a triggering return period. A "100-year
    storm" keeps its present-day meaning, because the terrain takes far longer
    than a century to adjust to a new regime and because redefining the trigger
    at the same time would confound two effects in one map.

Scenarios come from WorldClim's downscaled CMIP6 archive, which offers four
shared socioeconomic pathways over four twenty-year windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Shared socioeconomic pathways, low to high forcing.
SSPS: Tuple[str, ...] = ("ssp126", "ssp245", "ssp370", "ssp585")

#: Twenty-year windows the downscaled archive provides.
PERIODS: Tuple[str, ...] = ("2021-2040", "2041-2060", "2061-2080", "2081-2100")

#: Default general circulation model. Mid-range for climate sensitivity among
#: the CMIP6 ensemble, so it is neither the optimistic nor the pessimistic edge.
DEFAULT_GCM = "IPSL-CM6A-LR"

#: CMIP6 grid. The 30 s files are about 22 GB each, which is rarely worth it
#: given that the recharge field is a smooth multiplier.
DEFAULT_GCM_RESOLUTION = "2.5m"

_SSP_LABEL = {
    "ssp126": "SSP1-2.6, low forcing",
    "ssp245": "SSP2-4.5, intermediate forcing",
    "ssp370": "SSP3-7.0, high forcing",
    "ssp585": "SSP5-8.5, very high forcing",
}


@dataclass(frozen=True)
class ClimateScenario:
    """One climate state the model can be evaluated under."""

    key: str
    ssp: Optional[str]          # None for the present-day baseline
    period: Optional[str]       # None for the present-day baseline
    gcm: str = DEFAULT_GCM
    resolution: str = DEFAULT_GCM_RESOLUTION

    @property
    def is_baseline(self) -> bool:
        return self.ssp is None

    @property
    def label(self) -> str:
        if self.is_baseline:
            return "present day (WorldClim v2.1, 1970-2000 baseline)"
        return f"{_SSP_LABEL.get(self.ssp, self.ssp)}, {self.period}, {self.gcm}"

    def as_dict(self) -> dict:
        return {"key": self.key, "ssp": self.ssp, "period": self.period,
                "gcm": self.gcm, "resolution": self.resolution,
                "label": self.label}


#: The present-day baseline. Every future scenario is measured against it, and
#: the recharge reference in millimetres is fitted under it.
BASELINE = ClimateScenario(key="current", ssp=None, period=None)


def scenario(spec: str, gcm: str = DEFAULT_GCM,
             resolution: str = DEFAULT_GCM_RESOLUTION) -> ClimateScenario:
    """Parse a scenario specification.

    Accepts ``"current"``, an SSP with a period (``"ssp585:2061-2080"``), or a
    bare SSP, which takes the last period in :data:`PERIODS`.
    """
    spec = spec.strip().lower()
    if spec in ("current", "baseline", "present"):
        return BASELINE

    ssp, _, period = spec.partition(":")
    if ssp not in SSPS:
        raise ValueError(f"unknown pathway {ssp!r}; expected one of "
                         f"{', '.join(SSPS)} or 'current'")
    period = period or PERIODS[-1]
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}; expected one of "
                         f"{', '.join(PERIODS)}")
    return ClimateScenario(key=f"{ssp}_{period}", ssp=ssp, period=period,
                           gcm=gcm, resolution=resolution)


def from_dict(d: dict) -> ClimateScenario:
    """Rebuild a scenario from :meth:`ClimateScenario.as_dict`.

    Round-tripping through :func:`scenario` would not work: ``key`` joins the
    pathway and period with an underscore because it names files, while the
    specification syntax uses a colon.
    """
    if not d.get("ssp"):
        return BASELINE
    return ClimateScenario(key=d.get("key") or f"{d['ssp']}_{d['period']}",
                           ssp=d["ssp"], period=d.get("period"),
                           gcm=d.get("gcm") or DEFAULT_GCM,
                           resolution=d.get("resolution")
                           or DEFAULT_GCM_RESOLUTION)


def parse_all(specs: Sequence[str], gcm: str = DEFAULT_GCM,
              resolution: str = DEFAULT_GCM_RESOLUTION
              ) -> List[ClimateScenario]:
    """Parse several specifications, keeping order and dropping duplicates."""
    out: List[ClimateScenario] = []
    for s in specs:
        sc = scenario(s, gcm, resolution)
        if sc.key not in {x.key for x in out}:
            out.append(sc)
    return out


def suite(ssps: Sequence[str] = SSPS, period: str = "2061-2080",
          gcm: str = DEFAULT_GCM,
          resolution: str = DEFAULT_GCM_RESOLUTION) -> List[ClimateScenario]:
    """Baseline plus one period across several pathways.

    The usual comparison: hold the window fixed and vary the forcing, so the
    spread between maps is the pathway rather than a mixture of pathway and
    date.
    """
    return [BASELINE] + [scenario(f"{s}:{period}", gcm, resolution)
                         for s in ssps]


def trajectory(ssp: str = "ssp585", periods: Sequence[str] = PERIODS,
               gcm: str = DEFAULT_GCM,
               resolution: str = DEFAULT_GCM_RESOLUTION
               ) -> List[ClimateScenario]:
    """Baseline plus one pathway across several windows, to see the time course."""
    return [BASELINE] + [scenario(f"{ssp}:{p}", gcm, resolution)
                         for p in periods]
