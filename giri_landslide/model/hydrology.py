"""Terrain hydrology: depression filling, D-infinity flow, catchment area.

The physically based stability model needs to know how much water arrives at a
cell from upslope, which means routing flow across the DEM. The methods here
follow TauDEM (Tarboton, Utah State University):

* **Depression filling** by priority flood (Barnes, Lehman & Mulla 2014). Raw
  DEMs contain sinks - real ones, and artefacts of the sensor and of
  interpolation - which trap flow and truncate catchments. Filling raises each
  sink to the level of its lowest outlet so every cell drains to the edge.

* **D-infinity flow directions** (Tarboton 1997). Flow leaves a cell along a
  single angle rather than being forced into one of eight neighbours, and is
  divided between the two neighbours that bracket that angle. On the planar
  hillslopes that matter for shallow failure, D8's 45-degree quantisation
  produces artificial parallel flow lines; D-infinity does not.

* **Specific catchment area** ``a``, the upslope area draining through unit
  contour width. This is the quantity the wetness term in the stability model
  needs, not total upslope area.

Everything works on a projected metric grid. Geographic degrees are converted
to metres per row using the local latitude, as elsewhere in the package.
"""

from __future__ import annotations

import heapq
import math
from typing import Tuple

import numpy as np

# Neighbour offsets, counter-clockwise from east. Index matches the facet
# tables below.
_D_ROW = np.array([0, -1, -1, -1, 0, 1, 1, 1])
_D_COL = np.array([1, 1, 0, -1, -1, -1, 0, 1])

# Tarboton (1997) table 1: for each of the eight facets, the cardinal and
# diagonal neighbour indices, and the coefficients that map the facet-local
# angle onto a global one.
_FACET = [
    # (cardinal neighbour, diagonal neighbour, ac, af)
    (0, 1, 0, 1), (2, 1, 1, -1), (2, 3, 1, 1), (4, 3, 2, -1),
    (4, 5, 2, 1), (6, 5, 3, -1), (6, 7, 3, 1), (0, 7, 4, -1),
]


def fill_depressions(dem: np.ndarray, nodata_mask: np.ndarray | None = None
                     ) -> np.ndarray:
    """Priority-flood depression filling.

    Cells are raised to the lowest elevation from which they can still drain to
    the grid edge. A small increment is added along the way so filled areas keep
    a defined drainage direction instead of forming flat pools, which would
    leave flow routing undefined.
    """
    out = np.array(dem, dtype="float64", copy=True)
    if nodata_mask is None:
        nodata_mask = ~np.isfinite(out)
    out[nodata_mask] = np.inf

    rows, cols = out.shape
    closed = np.zeros(out.shape, dtype=bool)
    closed[nodata_mask] = True

    heap: list = []
    # Seed with the grid edge: those cells already drain out of the domain.
    for r in range(rows):
        for c in (0, cols - 1):
            if not closed[r, c]:
                heapq.heappush(heap, (out[r, c], r, c))
                closed[r, c] = True
    for c in range(cols):
        for r in (0, rows - 1):
            if not closed[r, c]:
                heapq.heappush(heap, (out[r, c], r, c))
                closed[r, c] = True

    eps = 1e-6
    while heap:
        elev, r, c = heapq.heappop(heap)
        for k in range(8):
            rr, cc = r + _D_ROW[k], c + _D_COL[k]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols or closed[rr, cc]:
                continue
            # Raise the neighbour to just above the outlet if it sits lower.
            out[rr, cc] = max(out[rr, cc], elev + eps)
            closed[rr, cc] = True
            heapq.heappush(heap, (out[rr, cc], rr, cc))

    out[nodata_mask] = np.nan
    return out


def dinf_flow_direction(filled: np.ndarray, dx: float, dy: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """D-infinity flow angle and slope magnitude (Tarboton 1997).

    Returns ``(angle, slope)``. The angle is measured counter-clockwise from
    east in radians; the slope is the steepest downslope gradient found across
    the eight triangular facets. Cells with no downslope facet - pits and flats
    that survived filling - get angle NaN and slope 0.
    """
    rows, cols = filled.shape
    z = np.pad(filled, 1, mode="edge")

    # Neighbour elevation stack, in the offset order above.
    nb = np.empty((8, rows, cols), dtype="float64")
    for k in range(8):
        r0, c0 = 1 + _D_ROW[k], 1 + _D_COL[k]
        nb[k] = z[r0:r0 + rows, c0:c0 + cols]

    best_slope = np.zeros((rows, cols), dtype="float64")
    best_angle = np.full((rows, cols), np.nan, dtype="float64")

    for (i_card, i_diag, ac, af) in _FACET:
        # Cardinal spacing depends on whether the facet's cardinal neighbour is
        # east/west or north/south.
        d1 = dx if _D_ROW[i_card] == 0 else dy
        d2 = dy if _D_ROW[i_card] == 0 else dx

        e0 = filled
        e1 = nb[i_card]
        e2 = nb[i_diag]

        with np.errstate(invalid="ignore", divide="ignore"):
            s1 = (e0 - e1) / d1                 # along the cardinal edge
            s2 = (e1 - e2) / d2                 # across to the diagonal
            r = np.arctan2(s2, s1)
            s = np.hypot(s1, s2)

            # Constrain the direction to lie inside the facet.
            r_max = math.atan2(d2, d1)
            outside_low = r < 0
            r = np.where(outside_low, 0.0, r)
            s = np.where(outside_low, s1, s)
            outside_high = r > r_max
            r = np.where(outside_high, r_max, r)
            s = np.where(outside_high, (e0 - e2) / math.hypot(d1, d2), s)

        take = np.isfinite(s) & (s > best_slope)
        best_slope = np.where(take, s, best_slope)
        # Map the facet-local angle to a global one.
        global_angle = af * r + ac * (math.pi / 2.0)
        best_angle = np.where(take, global_angle, best_angle)

    best_angle = np.where(best_slope > 0, np.mod(best_angle, 2 * math.pi),
                          np.nan)
    best_slope = np.where(np.isfinite(filled), best_slope, np.nan)
    return best_angle, best_slope


def dinf_accumulation(filled: np.ndarray, angle: np.ndarray,
                      cell_area: np.ndarray | float) -> np.ndarray:
    """D-infinity contributing area.

    Each cell's flow is split between the two neighbours bracketing its flow
    angle, in proportion to how close the angle lies to each. Cells are
    processed from high to low so a cell's own inflow is complete before it
    passes anything on.
    """
    rows, cols = filled.shape
    area = np.full(filled.shape, np.nan, dtype="float64")
    valid = np.isfinite(filled)
    area[valid] = (cell_area[valid] if isinstance(cell_area, np.ndarray)
                   else float(cell_area))

    order = np.argsort(np.where(valid, filled, -np.inf), axis=None)[::-1]
    n_valid = int(valid.sum())
    order = order[:n_valid]

    ang = angle
    for idx in order:
        r, c = divmod(int(idx), cols)
        a = ang[r, c]
        if not np.isfinite(a):
            continue                      # pit or flat: flow stops here
        # Which two neighbour directions bracket this angle.
        k = a / (math.pi / 4.0)
        k0 = int(math.floor(k)) % 8
        k1 = (k0 + 1) % 8
        p1 = k - math.floor(k)            # share to the second neighbour
        contrib = area[r, c]
        if not np.isfinite(contrib):
            continue
        for kk, share in ((k0, 1.0 - p1), (k1, p1)):
            if share <= 0:
                continue
            rr, cc = r + _D_ROW[kk], c + _D_COL[kk]
            if 0 <= rr < rows and 0 <= cc < cols and np.isfinite(area[rr, cc]):
                area[rr, cc] += contrib * share
    return area


def specific_catchment_area(dem: np.ndarray, dx: float, dy: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Specific catchment area ``a`` (m) and slope (m/m) for a DEM block.

    ``a`` is contributing area divided by the contour width the flow crosses,
    which is what the wetness term in the stability model requires. Total
    contributing area would scale with cell size and make the result
    resolution-dependent.
    """
    filled = fill_depressions(dem)
    angle, slope = dinf_flow_direction(filled, dx, dy)
    cell_area = dx * dy
    total = dinf_accumulation(filled, angle, cell_area)
    contour_width = 0.5 * (dx + dy)       # mean cell width
    return total / contour_width, slope
