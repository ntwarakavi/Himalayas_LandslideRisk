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
                "source_distance_m": round(self.source_distance_m, 1)}


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
        return AssetScore(on_site=on,
                          reaching=weighted,
                          reaching_max=float(p[k]),
                          n_sources=int(ok.sum()),
                          source_relief_m=float(self.relief[ok][k]),
                          source_distance_m=float(self.dist[ok][k]))


class ReachIndex:
    """Finds, for any coordinate, the cells positioned to reach it.

    Holds the DEM and the grid together because every asset needs the same
    three things - elevation, transform, and cell size in metres - and passing
    them separately invites mixing grids.
    """

    def __init__(self, dem: np.ndarray, transform,
                 dx_m: float, dy_m: float,
                 travel_angle_deg: float = DEFAULT_TRAVEL_ANGLE_DEG,
                 search_radius_m: float = DEFAULT_SEARCH_RADIUS_M):
        self.dem = np.asarray(dem, "float64")
        self.transform = transform
        self.dx, self.dy = float(dx_m), float(dy_m)
        self.tan_alpha = math.tan(math.radians(travel_angle_deg))
        self.travel_angle_deg = travel_angle_deg
        self.radius_m = float(search_radius_m)
        self.rx = max(int(round(self.radius_m / self.dx)), 1)
        self.ry = max(int(round(self.radius_m / self.dy)), 1)
        self._offsets = self._build_offsets()

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
        d = dist[ok]
        return Reach(r0, c0, rr[ok], cc[ok], 1.0 / d, relief[ok], d)

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
                      source_distance_m=best.source_distance_m)


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


def score_settlements(index: ReachIndex, settlements,
                      probs: Dict[str, np.ndarray],
                      baseline: str = "current") -> List[dict]:
    """Score every settlement under every scenario; drop those off-grid."""
    out = []
    for s in settlements:
        rch = index.reach(s.lon, s.lat)
        if rch is None:
            continue
        rec = {"name": s.name, "lon": s.lon, "lat": s.lat, "place": s.place,
               "population": s.population, "source": s.source}
        out.append(_apply(rec, {k: rch.score(p) for k, p in probs.items()},
                          baseline))
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
                "model calls unstable.",
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
