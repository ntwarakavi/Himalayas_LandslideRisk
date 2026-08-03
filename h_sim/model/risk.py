"""What the susceptibility means for a town or a road: reaching susceptibility.

The obvious way to score a settlement is to sample the susceptibility map at
its coordinates. That is wrong, and wrong in the direction that matters: it
reports safety.

Towns sit on flat ground - valley floors, terraces, the insides of meanders -
where the factor of safety is high and failure probability is near zero. The
stability model is right about that: the ground under the town is not going to
slide. What destroys mountain towns is material arriving *from above*. Sampling
at the point asks the wrong question, and reliably answers "safe" for exactly
the settlements most at risk.

Angle of reach
--------------

What is needed is the susceptibility of ground that could deliver material to
the asset. The standard empirical screening tool is the angle of reach, also
called the Fahrboschung or Heim ratio: debris from a source can reach a target
if the line between them is steeper than a limiting travel angle.

    (z_source - z_target) / horizontal_distance  >  tan(alpha)

Reported values of ``alpha`` cluster around 11 to 25 degrees for channelised
debris flows and shallow slides in mountain terrain, falling with volume
(Corominas 1996; Rickenmann 1999; Hunter & Fell 2003). A smaller ``alpha``
means a longer reach and a more conservative screen.

Why the score is a weighted mean and not a maximum
--------------------------------------------------

The first version of this module scored an asset by the *highest* failure
probability among the cells that could reach it. On real terrain that number
saturates and stops discriminating. In Himalayan relief a 2 km search radius
puts a few thousand cells above a typical valley settlement; if seven per cent
of the landscape exceeds a failure probability of 0.6 - which is what the
Gorkha 30 m fit gives - then the chance that *none* of several thousand upslope
cells does is negligible. Scored that way, 56% of settlements in the Gorkha
box came out in the top band, which is not a finding about the Himalaya, it is
an artefact of taking a maximum over a large sample.

The score is therefore the **proximity-weighted mean failure probability over
the ground positioned to reach the asset**: of the terrain that could deliver
material here, what fraction of it does the stability model call unstable.
That is stable under window size, monotone in the underlying probabilities,
and interpretable.

The weights come from one geometric argument. A debris path spreads laterally
roughly in proportion to how far it has travelled, so a target of fixed width
occupies a share of the possible path fan that falls about as 1/d. Sources
close above an asset are therefore weighted far more heavily than sources at
the edge of the search radius, which is also what the runout literature
implies. No other tuning enters.

``reaching_max`` is still recorded, because "the worst single cell above this
town" is a useful diagnostic, but it is not banded and not the headline: it is
the statistic whose saturation caused the problem.

What this is and is not
-----------------------

A **screening indicator**, not a runout model. It uses no volume, no channel
geometry, no entrainment and no rheology. It answers "how much of the ground
positioned to reach this place is unstable", which is what a regional map can
honestly address. It does not answer "will it arrive, how much, or how fast".

Nor is it risk in the full sense. Risk needs

    risk = hazard x exposure x vulnerability

and only the first two are here. There is no damage function, so nothing in
this module converts to expected loss, casualties or cost. Population is
carried through for ranking, never multiplied into anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Default travel angle in degrees. 18 sits inside the reported band for
#: channelised debris flows, deliberately towards the conservative (longer
#: reach) end: this is a screening product, and a false alarm costs an
#: inspection while a missed settlement costs more.
DEFAULT_TRAVEL_ANGLE_DEG = 18.0

#: How far upslope to look, in metres. Beyond a couple of kilometres the angle
#: criterion is doing all the work, the 1/d weight has decayed, and window cost
#: grows with the square.
DEFAULT_SEARCH_RADIUS_M = 2000.0

#: Bands for a legend, and nothing more. The underlying score is continuous
#: and should be used continuously. The edges are quantitative statements
#: about the weighted unstable fraction, not calibrated damage thresholds:
#: "moderate" means a fifth of the ground that can reach you is called
#: unstable.
RISK_BANDS = ((0.02, "very low"), (0.08, "low"), (0.20, "moderate"),
              (0.40, "high"), (1.01, "very high"))


def band(score: float) -> str:
    for edge, label in RISK_BANDS:
        if score < edge:
            return label
    return RISK_BANDS[-1][1]


BAND_ORDER = [b for _, b in RISK_BANDS]

#: Compass octants, clockwise from north, for the worst-sector report.
SECTORS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

#: Score at or above which an asset is reported as exposed in summaries.
EXPOSED_THRESHOLD = RISK_BANDS[1][0]


@dataclass
class AssetScore:
    """What reaches one asset, and from where."""

    on_site: float                  # failure probability of the asset's cell
    reaching: float                 # proximity-weighted mean over sources
    reaching_max: float             # worst single source cell (diagnostic)
    n_sources: int                  # cells satisfying the angle criterion
    source_relief_m: float          # height of the worst source above target
    source_distance_m: float        # its horizontal distance
    #: Expected unstable area positioned to reach the asset, in m2:
    #: sum of p_i x cell area over the sources. An expectation, so it is
    #: valid under the strong positive dependence the shared parameter draw
    #: induces between cells - which the union probability 1 - prod(1 - p_i)
    #: is not. It is what separates thirty unstable source cells from three
    #: thousand when their mean is the same. Relative like everything else:
    #: compare it between assets, never read it as absolute supply.
    delivering_m2: float = 0.0
    #: Compass octant whose sources carry the highest weighted mean failure
    #: probability, and that mean - "the threat is from the NE". Empty when
    #: nothing reaches.
    sector: str = ""
    sector_reaching: float = 0.0

    @property
    def score(self) -> float:
        """The headline number: the greater of standing on it or under it.

        Both terms are on 0-1 but they are not the same quantity - one is the
        failure probability of a single cell, the other a weighted fraction of
        upslope ground. Taking the greater keeps an asset built on genuinely
        unstable ground from being scored only by what is above it, which for
        a hillside settlement is the mechanism that matters.
        """
        return float(max(self.on_site, self.reaching))

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "band": band(self.score),
                "on_site": round(self.on_site, 4),
                "reaching": round(self.reaching, 4),
                "reaching_max": round(self.reaching_max, 4),
                "n_sources": self.n_sources,
                "source_relief_m": round(self.source_relief_m, 1),
                "source_distance_m": round(self.source_distance_m, 1),
                "delivering_m2": round(self.delivering_m2, 0),
                "sector": self.sector,
                "sector_reaching": round(self.sector_reaching, 4)}


EMPTY_SCORE = AssetScore(0.0, 0.0, 0.0, 0, 0.0, 0.0)


@dataclass
class Reach:
    """The cells that can deliver to one asset, and how much each counts.

    Pure geometry: it depends on the DEM and the travel angle, never on a
    susceptibility map. That separation is what makes scoring the same
    settlement under six climate scenarios cost one window search, not six.
    """

    row: int
    col: int
    rows: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, int))
    cols: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, int))
    weight: np.ndarray = field(repr=False,
                               default_factory=lambda: np.empty(0, float))
    relief: np.ndarray = field(repr=False,
                               default_factory=lambda: np.empty(0, float))
    dist: np.ndarray = field(repr=False,
                             default_factory=lambda: np.empty(0, float))
    #: Compass azimuth from the target to each source, degrees clockwise
    #: from north.
    az: np.ndarray = field(repr=False,
                           default_factory=lambda: np.empty(0, float))
    #: Cell area in m2, for the expected delivering area.
    cell_m2: float = 0.0

    def score(self, prob: np.ndarray) -> AssetScore:
        """Score this reach against one susceptibility map."""
        on = prob[self.row, self.col]
        on = float(on) if np.isfinite(on) else 0.0
        if self.rows.size == 0:
            return AssetScore(on, 0.0, 0.0, 0, 0.0, 0.0)

        p = prob[self.rows, self.cols]
        ok = np.isfinite(p)
        if not ok.any():
            return AssetScore(on, 0.0, 0.0, 0, 0.0, 0.0)

        p, w = p[ok], self.weight[ok]
        total = w.sum()
        weighted = float((p * w).sum() / total) if total > 0 else 0.0
        k = int(np.argmax(p))

        # The octant the threat comes from: the compass sector whose sources
        # carry the highest weighted mean. Reported so a field visit knows
        # which slope to walk, not just that one exists.
        sector, sector_val = "", 0.0
        if self.az.size:
            oct_idx = (np.round(self.az[ok] / 45.0).astype(int)) % 8
            for i, label in enumerate(SECTORS):
                m = oct_idx == i
                if not m.any():
                    continue
                tw = w[m].sum()
                v = float((p[m] * w[m]).sum() / tw) if tw > 0 else 0.0
                if v > sector_val:
                    sector, sector_val = label, v

        return AssetScore(on_site=on,
                          reaching=weighted,
                          reaching_max=float(p[k]),
                          n_sources=int(ok.sum()),
                          source_relief_m=float(self.relief[ok][k]),
                          source_distance_m=float(self.dist[ok][k]),
                          delivering_m2=float(p.sum() * self.cell_m2),
                          sector=sector,
                          sector_reaching=sector_val)


class ReachIndex:
    """Finds, for any coordinate, the cells positioned to reach it.

    Holds the DEM and the grid together because every asset needs the same
    three things - elevation, transform, and cell size in metres - and passing
    them separately invites mixing grids.
    """

    def __init__(self, dem: np.ndarray, transform,
                 dx_m: float, dy_m: float,
                 travel_angle_deg: float = DEFAULT_TRAVEL_ANGLE_DEG,
                 search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
                 flow: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 connectivity_floor: float = 0.2):
        """``flow``, when given, is ``(filled_dem, dinf_angle)`` and switches
        on connectivity weighting: each source's weight is scaled by the
        fraction of its D-infinity flow that passes through the target patch
        (Tarboton's dependence, cf. the spreading algorithms of Holmgren 1994
        as used regionally by Horton et al. 2013, Flow-R). Debris paths are
        mostly channelised, so ground draining *toward* an asset should count
        for more than ground merely above it - but not exclusively, which is
        what ``connectivity_floor`` preserves: the minimum share of weight
        every cone source keeps, so near-field planar delivery that follows
        no channel is never zeroed out.

        Off by default (``flow=None``), pending the held-out measurement in
        analysis/08_connectivity.py: the cone alone is the published SINMAP-
        adjacent screen, and this repository adopts refinements when a number
        says they help, not because they sound right.
        """
        self.dem = np.asarray(dem, "float64")
        self.transform = transform
        self.dx, self.dy = float(dx_m), float(dy_m)
        self.tan_alpha = math.tan(math.radians(travel_angle_deg))
        self.travel_angle_deg = travel_angle_deg
        self.radius_m = float(search_radius_m)
        self.rx = max(int(round(self.radius_m / self.dx)), 1)
        self.ry = max(int(round(self.radius_m / self.dy)), 1)
        self._offsets = self._build_offsets()
        self.flow = flow
        self.connectivity_floor = float(connectivity_floor)
        if flow is not None and flow[0].shape != self.dem.shape:
            raise ValueError("flow grids and DEM have different shapes")

    def _build_offsets(self):
        """Row/column offsets inside the search radius, with their distances."""
        rr, cc = np.mgrid[-self.ry:self.ry + 1, -self.rx:self.rx + 1]
        dist = np.hypot(rr * self.dy, cc * self.dx)
        keep = (dist <= self.radius_m) & (dist > 0)
        return rr[keep], cc[keep], dist[keep]

    def rowcol(self, lon: float, lat: float) -> Tuple[int, int]:
        inv = ~self.transform
        col, row = inv * (lon, lat)
        return int(row), int(col)

    def reach(self, lon: float, lat: float) -> Optional[Reach]:
        """The reach of one location. None if it falls outside the grid."""
        h, w = self.dem.shape
        r0, c0 = self.rowcol(lon, lat)
        if not (0 <= r0 < h and 0 <= c0 < w):
            return None

        z0 = self.dem[r0, c0]
        if not np.isfinite(z0):
            return Reach(r0, c0)

        dr, dc, dist = self._offsets
        rr, cc = r0 + dr, c0 + dc
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc, dist = rr[inside], cc[inside], dist[inside]
        if rr.size == 0:
            return Reach(r0, c0)

        relief = self.dem[rr, cc] - z0
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (np.isfinite(relief) & (relief > 0)
                  & (relief / dist > self.tan_alpha))
        if not ok.any():
            return Reach(r0, c0)

        # A debris path widens roughly in proportion to distance travelled, so
        # the share of the fan a fixed-width target occupies falls as 1/d.
        rr, cc, d = rr[ok], cc[ok], dist[ok]
        wt = 1.0 / d

        # Connectivity: scale by the fraction of each source's flow routed
        # through the target patch, floored so unchannelised delivery keeps
        # a share. Computed on the search window alone - beyond it the angle
        # criterion has already said no.
        if self.flow is not None:
            from . import hydrology as H
            filled, ang = self.flow
            wr0, wr1 = max(r0 - self.ry, 0), min(r0 + self.ry + 1, h)
            wc0, wc1 = max(c0 - self.rx, 0), min(c0 + self.rx + 1, w)
            dep = H.dinf_dependence(filled[wr0:wr1, wc0:wc1],
                                    ang[wr0:wr1, wc0:wc1],
                                    r0 - wr0, c0 - wc0)
            f = self.connectivity_floor
            wt = wt * (f + (1.0 - f) * dep[rr - wr0, cc - wc0])

        # Compass azimuth target -> source, clockwise from north.
        az = np.degrees(np.arctan2((cc - c0) * self.dx,
                                   -(rr - r0) * self.dy)) % 360.0
        return Reach(r0, c0, rr, cc, wt, relief[ok], d, az,
                     cell_m2=self.dx * self.dy)

    # convenience wrappers -------------------------------------------------

    def score_point(self, prob: np.ndarray, lon: float, lat: float
                    ) -> Optional[AssetScore]:
        rch = self.reach(lon, lat)
        return None if rch is None else rch.score(prob)

    def line_reaches(self, coords: Sequence[Tuple[float, float]]
                     ) -> List[Reach]:
        """Reaches of a short line's vertices, deduplicated by cell.

        Segments are short by construction (see :func:`segment_line`), so their
        vertices often land in the same cell; scoring each once is both faster
        and gives the same answer.
        """
        seen = set()
        out = []
        for lon, lat in coords:
            rch = self.reach(lon, lat)
            if rch is None:
                continue
            key = (rch.row, rch.col)
            if key in seen:
                continue
            seen.add(key)
            out.append(rch)
        return out


def score_reaches(prob: np.ndarray, reaches: Sequence[Reach]
                  ) -> Optional[AssetScore]:
    """Summarise several reaches - a line's vertices - as one score.

    The worst vertex sets the headline, which is right for a short segment: a
    500 m stretch of road is closed by its most exposed point, not its average.
    """
    scores = [r.score(prob) for r in reaches]
    if not scores:
        return None
    best = max(scores, key=lambda s: s.score)
    return AssetScore(on_site=max(s.on_site for s in scores),
                      reaching=best.reaching,
                      reaching_max=max(s.reaching_max for s in scores),
                      n_sources=max(s.n_sources for s in scores),
                      source_relief_m=best.source_relief_m,
                      source_distance_m=best.source_distance_m,
                      # The largest expected supply anywhere on the segment;
                      # sector travels with the vertex that set the headline.
                      delivering_m2=max(s.delivering_m2 for s in scores),
                      sector=best.sector,
                      sector_reaching=best.sector_reaching)


# ---------------------------------------------------------------------------
# road failure mechanisms beyond burial from above
# ---------------------------------------------------------------------------

#: Adjacent upslope gradient at or above which a segment is flagged as
#: cut-slope susceptible. Reported hillslope thresholds for cut-slope failure
#: along Himalayan roads sit around 30-40 degrees; 35 is the middle of that
#: band. This is the *hillslope* gradient at grid resolution, standing proxy
#: for the engineered cut face the grid cannot see.
DEFAULT_CUT_SLOPE_DEG = 35.0

#: Specific catchment area, in metres, at or above which a cell is treated as
#: a channel for the washout flag. Hillslopes in this region carry SCA of
#: hundreds of metres; convergent hollows low thousands; anything above 5000 m
#: is drainage that can deliver a debris flow to a crossing.
DEFAULT_WASHOUT_SCA_M = 5000.0


class MechanismIndex:
    """Geometric flags for the two road failure modes the reach score misses.

    The reaching score covers **burial from above** - unstable ground
    positioned to deliver material onto the road. Himalayan roads are lost at
    least as often to two mechanisms that SINMAP has no term for, and that
    honesty requires labelling as geometry rather than model output:

    * **cut_slope** - the road traverses ground whose immediately adjacent
      upslope gradient exceeds a threshold. Where a road crosses such ground
      it does so on a cut, and cut faces oversteepen the very slope the
      infinite-slope model assumes undisturbed. Measured on the Shimla
      inventory, that assumption fails to the point of inversion (README,
      limits 2-3): this flag marks where that blind spot is, it does not
      score it.
    * **washout** - the segment touches a cell whose specific catchment area
      marks it as a channel. Culverts and causeways at such crossings are
      taken out by debris flows and flood scour arriving *along the channel*,
      from far beyond any local search radius.

    Both are properties of the terrain alone, so they are computed once per
    segment and do not vary with climate scenario. Combine them with the
    exposure score visually: a red segment carrying a washout flag is a
    crossing fed by ground the model calls unstable.
    """

    def __init__(self, dem: np.ndarray, sca: np.ndarray, transform,
                 dx_m: float, dy_m: float,
                 cut_slope_deg: float = DEFAULT_CUT_SLOPE_DEG,
                 washout_sca_m: float = DEFAULT_WASHOUT_SCA_M):
        self.dem = np.asarray(dem, "float64")
        self.sca = np.asarray(sca, "float64")
        if self.dem.shape != self.sca.shape:
            raise ValueError(f"DEM {self.dem.shape} and SCA {self.sca.shape} "
                             "are on different grids")
        self.transform = transform
        self.cut_tan = math.tan(math.radians(cut_slope_deg))
        self.cut_slope_deg = cut_slope_deg
        self.washout_sca_m = float(washout_sca_m)
        d = math.hypot(dx_m, dy_m)
        self._noff = ((-1, -1, d), (-1, 0, dy_m), (-1, 1, d),
                      (0, -1, dx_m), (0, 1, dx_m),
                      (1, -1, d), (1, 0, dy_m), (1, 1, d))

    def _rowcol(self, lon: float, lat: float) -> Tuple[int, int]:
        inv = ~self.transform
        col, row = inv * (lon, lat)
        return int(row), int(col)

    def assess(self, coords: Sequence[Tuple[float, float]]) -> dict:
        """Flags for one segment, from its vertices' unique cells."""
        h, w = self.dem.shape
        cells = []
        seen = set()
        for lon, lat in coords:
            rc = self._rowcol(lon, lat)
            if rc in seen or not (0 <= rc[0] < h and 0 <= rc[1] < w):
                continue
            seen.add(rc)
            cells.append(rc)

        max_tan, max_sca = 0.0, 0.0
        for r0, c0 in cells:
            z0 = self.dem[r0, c0]
            s = self.sca[r0, c0]
            if np.isfinite(s):
                max_sca = max(max_sca, float(s))
            if not np.isfinite(z0):
                continue
            for dr, dc, dist in self._noff:
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                rise = self.dem[r, c] - z0
                if np.isfinite(rise) and rise > 0:
                    max_tan = max(max_tan, float(rise) / dist)

        return {
            "cut_slope": bool(max_tan >= self.cut_tan),
            "cut_slope_deg": round(math.degrees(math.atan(max_tan)), 1),
            "washout": bool(max_sca >= self.washout_sca_m),
            "washout_sca_m": round(max_sca, 0),
        }


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def segment_line(coords: Sequence[Tuple[float, float]],
                 length_m: float = 500.0
                 ) -> List[List[Tuple[float, float]]]:
    """Cut a way into pieces of roughly ``length_m``.

    A road is scored per segment rather than per way because a way can be fifty
    kilometres long: one number for all of it would be the maximum over its
    worst kilometre, which tells a maintainer nothing about where to go.
    Vertices are never split, so a segment is at least one vertex pair even
    when that pair is longer than ``length_m``.
    """
    if len(coords) < 2:
        return []
    out, cur, run = [], [coords[0]], 0.0
    for a, b in zip(coords, coords[1:]):
        run += _haversine_m(a, b)
        cur.append(b)
        if run >= length_m:
            out.append(cur)
            cur, run = [b], 0.0
    if len(cur) > 1:
        out.append(cur)
    elif out:
        out[-1].extend(cur[1:])
    return out


def line_length_m(coords: Sequence[Tuple[float, float]]) -> float:
    return sum(_haversine_m(a, b) for a, b in zip(coords, coords[1:]))


def midpoint(coords: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ---------------------------------------------------------------------------
# batch scoring, over one or many climate scenarios
# ---------------------------------------------------------------------------

def _apply(rec: dict, scores: Dict[str, AssetScore], baseline: str) -> dict:
    """Attach per-scenario scores, promoting the baseline to the top level."""
    rec["scenarios"] = {k: v.as_dict() for k, v in scores.items()}
    base = scores.get(baseline) or next(iter(scores.values()), EMPTY_SCORE)
    rec.update(base.as_dict())
    if len(scores) > 1:
        worst = max(scores.values(), key=lambda s: s.score)
        rec["score_max"] = round(worst.score, 4)
        rec["delta_max"] = round(worst.score - base.score, 4)
    return rec


#: Footprint radius by OSM place type, metres. A settlement is an area, not
#: its centroid: OSM place nodes sit at the centre while the hillside wards a
#: reach score exists for sit at the edge. These radii are a stated heuristic
#: standing in for mapped built-up extent (GHS-BUILT or building footprints
#: would replace them behind the same interface); they scale with how far a
#: place of that type plausibly extends.
FOOTPRINT_RADIUS_M = {"city": 1000.0, "town": 500.0,
                      "village": 250.0, "hamlet": 100.0}
DEFAULT_FOOTPRINT_M = 250.0

#: The footprint headline is this quantile of the per-cell scores, not the
#: maximum: the max over many cells is exactly the saturation artefact the
#: module docstring warns about, while the 90th percentile still reports the
#: exposed edge of town rather than its safe centre.
FOOTPRINT_QUANTILE = 0.9


def footprint_reaches(index: ReachIndex, lon: float, lat: float,
                      radius_m: float) -> List[Reach]:
    """Reaches of every grid cell within ``radius_m`` of a point."""
    rc = index.rowcol(lon, lat)
    h, w = index.dem.shape
    nx = max(int(math.ceil(radius_m / index.dx)), 0)
    ny = max(int(math.ceil(radius_m / index.dy)), 0)
    out = []
    for r in range(rc[0] - ny, rc[0] + ny + 1):
        for c in range(rc[1] - nx, rc[1] + nx + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if math.hypot((r - rc[0]) * index.dy,
                          (c - rc[1]) * index.dx) > radius_m:
                continue
            cx, cy = index.transform * (c + 0.5, r + 0.5)
            rch = index.reach(cx, cy)
            if rch is not None:
                out.append(rch)
    return out


def score_footprint(reaches: Sequence[Reach], prob: np.ndarray,
                    q: float = FOOTPRINT_QUANTILE) -> AssetScore:
    """One score for a footprint: the cell at the ``q`` rank of the scores.

    A whole cell's record is reported rather than mixing quantiles of
    different fields, so the sector, source and supply all describe the same
    place - the exposed edge that set the headline.
    """
    scores = sorted((r.score(prob) for r in reaches), key=lambda s: s.score)
    return scores[int(round(q * (len(scores) - 1)))]


def score_settlements(index: ReachIndex, settlements,
                      probs: Dict[str, np.ndarray],
                      baseline: str = "current",
                      footprints: bool = True) -> List[dict]:
    """Score every settlement under every scenario; drop those off-grid.

    With ``footprints`` on, a settlement is scored over the cells of a disc
    scaled by its place type and headlined at the 90th-percentile cell, so a
    town whose centre is safe but whose edge sits under a slope is no longer
    reported by its safest point.
    """
    out = []
    for s in settlements:
        rec = {"name": s.name, "lon": s.lon, "lat": s.lat, "place": s.place,
               "population": s.population, "source": s.source}
        if footprints:
            radius = FOOTPRINT_RADIUS_M.get(s.place, DEFAULT_FOOTPRINT_M)
            reaches = footprint_reaches(index, s.lon, s.lat, radius)
            if not reaches:
                continue
            rec["footprint_m"] = radius
            rec["n_cells"] = len(reaches)
            scores = {k: score_footprint(reaches, p)
                      for k, p in probs.items()}
        else:
            rch = index.reach(s.lon, s.lat)
            if rch is None:
                continue
            scores = {k: rch.score(p) for k, p in probs.items()}
        out.append(_apply(rec, scores, baseline))
    out.sort(key=lambda r: -r["score"])
    return out


def score_roads(index: ReachIndex, roads, probs: Dict[str, np.ndarray],
                segment_m: float = 500.0,
                baseline: str = "current") -> List[dict]:
    """Cut every road into segments and score each one, every scenario."""
    out = []
    for road in roads:
        for i, seg in enumerate(segment_line(road.coords, segment_m)):
            reaches = index.line_reaches(seg)
            if not reaches:
                continue
            scores = {k: score_reaches(p, reaches) for k, p in probs.items()}
            scores = {k: v for k, v in scores.items() if v is not None}
            if not scores:
                continue
            rec = {"name": road.name, "highway": road.highway,
                   "segment": i, "coords": seg,
                   "length_m": round(line_length_m(seg), 1),
                   "source": road.source}
            out.append(_apply(rec, scores, baseline))
    out.sort(key=lambda r: -r["score"])
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def _band_counts(rows: Sequence[dict], key: Optional[str],
                 weight: Optional[str] = None,
                 scale: float = 1.0) -> Dict[str, float]:
    acc = {b: 0.0 for b in BAND_ORDER}
    for r in rows:
        rec = r["scenarios"][key] if key else r
        acc[rec["band"]] += (r.get(weight) or 0) if weight else 1
    return {k: round(v * scale, 1) for k, v in acc.items()}


def scenario_stats(settlements: Sequence[dict], roads: Sequence[dict],
                   key: Optional[str] = None) -> dict:
    """Headline counts under one scenario."""
    def sc(r):
        return (r["scenarios"][key]["score"] if key else r["score"])

    exposed = [s for s in settlements if sc(s) >= EXPOSED_THRESHOLD]
    road_km = sum(r["length_m"] for r in roads) / 1000.0
    exposed_km = sum(r["length_m"] for r in roads
                     if sc(r) >= EXPOSED_THRESHOLD) / 1000.0
    return {
        "n_settlements": len(settlements),
        "settlements_by_band": _band_counts(settlements, key),
        "n_settlements_exposed": len(exposed),
        "population_exposed": sum(s["population"] or 0 for s in exposed),
        "mean_settlement_score": round(
            float(np.mean([sc(s) for s in settlements])), 4)
        if settlements else 0.0,
        "n_road_segments": len(roads),
        "road_km_total": round(road_km, 1),
        "road_km_exposed": round(exposed_km, 1),
        "road_pct_exposed": round(100.0 * exposed_km / road_km, 1)
        if road_km else 0.0,
        # length_m is metres; the field is kilometres, so scale on the way out.
        "road_km_by_band": _band_counts(roads, key, "length_m", 1e-3),
    }


def summarise(settlements: Sequence[dict], roads: Sequence[dict],
              scenarios: Optional[Sequence[str]] = None,
              baseline: str = "current") -> dict:
    """Per-scenario counts, plus what changes between them."""
    keys = list(scenarios) if scenarios else []
    out = {
        "baseline": baseline,
        "scenarios": {k: scenario_stats(settlements, roads, k) for k in keys},
        "exposed_threshold": EXPOSED_THRESHOLD,
        "note": "Screening by angle of reach, not a runout model, and not "
                "risk: there is no vulnerability or damage function here. "
                "The score is the proximity-weighted fraction of upslope "
                "ground positioned to reach the asset that the stability "
                "model calls unstable. Road cut_slope and washout are "
                "geometric flags from terrain alone - where the model's "
                "blind spots are, not scores from it.",
    }
    # Mechanism flags are terrain geometry, so unlike everything above they do
    # not vary by scenario and are rolled up once.
    if roads and "cut_slope" in roads[0]:
        km = lambda pred: round(sum(r["length_m"] for r in roads  # noqa: E731
                                    if pred(r)) / 1000.0, 1)
        out["mechanisms"] = {
            "road_km_cut_slope": km(lambda r: r["cut_slope"]),
            "road_km_washout": km(lambda r: r["washout"]),
            "road_km_both": km(lambda r: r["cut_slope"] and r["washout"]),
        }
    if not keys:
        out["scenarios"] = {baseline: scenario_stats(settlements, roads, None)}
        keys = [baseline]

    base = out["scenarios"].get(baseline)
    if base:
        out["change"] = {
            k: {
                "settlements_exposed": (out["scenarios"][k]
                                        ["n_settlements_exposed"]
                                        - base["n_settlements_exposed"]),
                "road_km_exposed": round(out["scenarios"][k]["road_km_exposed"]
                                         - base["road_km_exposed"], 1),
                "mean_settlement_score": round(
                    out["scenarios"][k]["mean_settlement_score"]
                    - base["mean_settlement_score"], 4),
            }
            for k in keys if k != baseline
        }
    # Keep the flat headline figures the CLI prints.
    if base:
        out.update({k: v for k, v in base.items()})
    return out


def compute_risk(*args, **kwargs):
    """Old name, kept so existing scripts fail loudly rather than silently."""
    raise NotImplementedError(
        "compute_risk has been replaced. Use ReachIndex with "
        "score_settlements / score_roads, via pipeline.run_risk. Note the "
        "result is exposure screening, not risk: there is still no "
        "vulnerability model or damage function.")
