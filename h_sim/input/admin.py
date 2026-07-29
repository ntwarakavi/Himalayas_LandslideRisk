"""Administrative units: the tiles a regional run is actually decomposed into.

The Hindu Kush Himalaya is roughly 4,400 by 2,500 km. At 30 m that is thirteen
billion cells, and flow routing is not tiled, so a single pass over the region
is out of the question. Something has to cut it up.

States and provinces are the natural cut. They are not hydrological units - a
provincial border crosses catchments without regard for drainage - but they are
the units that hazard maps get requested in, budgeted for, and acted on. A
province-by-province sweep produces maps somebody can use, in an order somebody
can prioritise.

The hydrology has to be protected from that choice, which is what
:func:`buffered_bbox` is for; see :func:`h_sim.pipeline.run_admin_unit`.

Source
------

Natural Earth 1:10m admin-1 states and provinces. Public domain, no
registration, no Git LFS, one 15 MB download for the world. Its boundaries are
generalised - good to a few hundred metres, not survey grade - which is
irrelevant here because the polygon only decides which cells belong to which
map, and a cell is 30 to 90 m anyway.

geoBoundaries would be more precise and is CC-BY, but ships its geometry
through Git LFS, which is frequently unreachable from restricted networks. If
you have it locally, pass it as ``admin_path``: any polygon layer with a name
field works.
"""

from __future__ import annotations

import os
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

NE_ADMIN1_URL = ("https://naciscdn.org/naturalearth/10m/cultural/"
                 "ne_10m_admin_1_states_provinces.zip")
NE_ADMIN1_SHP = "ne_10m_admin_1_states_provinces.shp"

ADMIN_SOURCE_INFO = (
    "Administrative units: Natural Earth 1:10m admin-1 states and provinces, "
    f"downloaded automatically from {NE_ADMIN1_URL} (public domain). Supply "
    "your own polygon layer via config.admin_path to use a national dataset "
    "instead; any layer with a name attribute works."
)

#: Candidate attribute names for the unit's own name and its country, in the
#: order they are tried. Natural Earth uses the first of each; other datasets
#: (GADM, geoBoundaries) use the later ones.
_NAME_FIELDS = ("name", "shapeName", "NAME_1", "ADM1_EN", "adm1_name")
_COUNTRY_FIELDS = ("admin", "shapeGroup", "NAME_0", "ADM0_EN", "country")


@dataclass
class AdminUnit:
    """One state or province, and the box a run over it needs."""

    name: str
    country: str
    bbox: Tuple[float, float, float, float]
    geometry: dict = field(repr=False, default_factory=dict)
    #: Filled by :func:`load_units` when an elevation raster is supplied:
    #: mountain fraction, max and median elevation. Empty otherwise.
    relief: Dict[str, float] = field(repr=False, default_factory=dict)

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, unique across countries."""
        def clean(s: str) -> str:
            out = "".join(c.lower() if c.isalnum() else "_" for c in s)
            return "_".join(p for p in out.split("_") if p)
        return f"{clean(self.country)}_{clean(self.name)}"

    def span_deg(self) -> Tuple[float, float]:
        w, s, e, n = self.bbox
        return (e - w, n - s)

    def cell_count(self, resolution_deg: float,
                   buffer_deg: float = 0.0) -> int:
        w, s, e, n = buffered_bbox(self.bbox, buffer_deg)
        return (int(round((e - w) / resolution_deg))
                * int(round((n - s) / resolution_deg)))

    def as_dict(self) -> dict:
        d = {"name": self.name, "country": self.country,
             "slug": self.slug, "bbox": list(self.bbox)}
        if self.relief:
            d["relief"] = self.relief
        return d


def buffered_bbox(bbox: Sequence[float], buffer_deg: float
                  ) -> Tuple[float, float, float, float]:
    """Grow a bounding box by ``buffer_deg`` on every side.

    A province boundary cuts drainage networks. Routing flow over the clipped
    box alone starts every catchment at the border, so cells just inside it are
    handed too little upslope area. Running over the buffered box and clipping
    the *output* back to the province lets flow enter from outside.

    The error this avoids is real but local: measured on 30 m Himalayan terrain
    it touched 1% of a province's cells and moved failure probability by more
    than 0.05 in 0.4% of them, all in the outermost ring. A 3 km buffer removed
    it completely. See analysis/07_boundary_buffer.py.
    """
    w, s, e, n = bbox
    return (w - buffer_deg, s - buffer_deg, e + buffer_deg, n + buffer_deg)


def download_admin1(data_dir: str) -> Optional[str]:
    """Fetch and extract the Natural Earth admin-1 layer. Cached after once."""
    from .sources import download_file

    out_dir = os.path.join(data_dir, "admin")
    shp = os.path.join(out_dir, NE_ADMIN1_SHP)
    if os.path.exists(shp):
        return shp

    dest = os.path.join(out_dir, "ne_10m_admin_1_states_provinces.zip")
    got = download_file(NE_ADMIN1_URL, dest, timeout=300)
    if not got:
        print("  " + ADMIN_SOURCE_INFO)
        return None
    with zipfile.ZipFile(got) as zf:
        zf.extractall(out_dir)
    return shp if os.path.exists(shp) else None


def _first(props: dict, keys: Sequence[str]) -> str:
    for k in keys:
        v = props.get(k)
        if v:
            return str(v)
    return ""


def _intersect(a: Sequence[float], b: Sequence[float]
               ) -> Optional[Tuple[float, float, float, float]]:
    w = max(a[0], b[0]); s = max(a[1], b[1])
    e = min(a[2], b[2]); n = min(a[3], b[3])
    return (w, s, e, n) if (w < e and s < n) else None


#: A cell at or above this elevation counts as mountain. 1,000 m is low for
#: the Hindu Kush Himalaya and deliberately so: it takes in the Siwaliks and
#: the Middle Hills, where most of the region's landslides and almost all of
#: its people are, while excluding the Gangetic and Indus plains.
DEFAULT_MOUNTAIN_ELEVATION_M = 1000.0

#: ...and it must also be rugged. Elevation alone cannot tell a mountain range
#: from a high plain, and the difference matters here because a high plain has
#: no hillslopes to fail on. Inner Mongolia's Alashan plateau sits above
#: 1,000 m across the part of it inside the region box and has a median local
#: relief of 62 m; Nepal's Bagmati has 1,512 m over the same window.
#:
#: Requiring both elevation and local elevation range is the standard
#: definition of mountain terrain (Kapos et al. 2000, used by UNEP-WCMC).
#: Kapos uses 300 m within a 7 km radius; the grid here is about 9 km, so a
#: 3x3 window spans roughly 27 km and the threshold is raised to match.
DEFAULT_LOCAL_RELIEF_M = 500.0

#: A province qualifies when at least this share of it is mountain.
DEFAULT_MOUNTAIN_FRACTION = 0.10

#: ...or when it has this much mountain outright, however small a share of the
#: whole that is. A fraction test alone penalises a large state with a genuine
#: mountainous corner: West Bengal is 1.9 % mountain and that 1.9 % is
#: Darjeeling and Kalimpong, among the most landslide-prone ground in India.
DEFAULT_MOUNTAIN_AREA_KM2 = 1000.0

#: The area test needs a second condition or it readmits the Eastern Ghats:
#: Odisha also clears 1,000 km2 above 1,000 m. Requiring real altitude
#: separates them - Odisha tops out at 1,110 m on this grid, West Bengal at
#: 2,938 m. The threshold is 1,400 rather than 2,000 m because a 9 km grid
#: flattens peaks: Mizoram's Phawngpui is 2,157 m in the field and 1,435 m
#: here, and Mizoram has landslides.
DEFAULT_MOUNTAIN_PEAK_M = 1400.0


def _no_relief() -> Dict[str, float]:
    return {"mountain_fraction": 0.0, "mountain_area_km2": 0.0,
            "max_elevation_m": 0.0, "median_elevation_m": 0.0,
            "median_local_relief_m": 0.0, "cells": 0}


def is_mountainous(relief: Dict[str, float],
                   min_fraction: float = DEFAULT_MOUNTAIN_FRACTION,
                   min_area_km2: float = DEFAULT_MOUNTAIN_AREA_KM2,
                   min_peak_m: float = DEFAULT_MOUNTAIN_PEAK_M) -> bool:
    """Is this unit in the mountains, on either of two tests?

    Mostly mountain, **or** carrying a substantial and genuinely high massif.
    The second clause exists for provinces the first would wrongly drop, and
    its altitude condition exists so the second clause does not readmit
    provinces the first rightly dropped. See the constants above.
    """
    return (relief.get("mountain_fraction", 0.0) >= min_fraction
            or (relief.get("mountain_area_km2", 0.0) >= min_area_km2
                and relief.get("max_elevation_m", 0.0) >= min_peak_m))


#: Units that pass every relief test and are still not in the Hindu Kush
#: Himalaya. Relief can tell a mountain from a plain; it cannot tell *which*
#: mountains, and these are the two that clear the bar on ranges outside the
#: arc - the Helan Shan in Inner Mongolia and the Yunnan-Guizhou plateau.
#: Naming them is more honest than tuning a threshold until they disappear and
#: taking half the Himalaya with them. Override with config.admin_exclude.
NOT_HKH = ("Inner Mongol", "Guizhou")


def relief_stats(unit: "AdminUnit", elev_path: str,
                 threshold_m: float = DEFAULT_MOUNTAIN_ELEVATION_M,
                 local_relief_m: float = DEFAULT_LOCAL_RELIEF_M
                 ) -> Dict[str, float]:
    """How mountainous a unit is, from a coarse global elevation raster.

    The polygon is rasterised onto the elevation grid and the cells inside it
    are summarised. A cell counts as mountain only if it is both high and
    rugged - see :data:`DEFAULT_LOCAL_RELIEF_M` for why both are needed. The
    grid is about 9 km, which is useless for stability and entirely adequate
    for "is there a mountain range here".
    """
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds

    from numpy.lib.stride_tricks import sliding_window_view

    with rasterio.open(elev_path) as src:
        w, s, e, n = unit.bbox
        try:
            win = from_bounds(w, s, e, n, src.transform)
            arr = src.read(1, window=win, boundless=True,
                           fill_value=src.nodata if src.nodata is not None
                           else -32768)
            tr = src.window_transform(win)
        except Exception:                                # noqa: BLE001
            return {"mountain_fraction": 0.0, "max_elevation_m": 0.0,
                    "median_elevation_m": 0.0, "cells": 0}
        nod = src.nodata

    if arr.size == 0:
        return _no_relief()

    inside = ~geometry_mask([unit.geometry], out_shape=arr.shape,
                            transform=tr, invert=False)
    z = arr.astype("float64")
    ok = inside & np.isfinite(z)
    if nod is not None:
        ok &= (arr != nod)
    ok &= z > -1000.0                                    # sea fill values
    if not ok.any():
        return _no_relief()

    # Local elevation range over a 3x3 window, edge-padded so the array keeps
    # its shape and border cells are judged on what is actually beside them.
    zp = np.pad(np.where(ok, z, np.nan), 1, mode="edge")
    win = sliding_window_view(zp, (3, 3))
    with warnings.catch_warnings():
        # An all-nodata window is legitimate at a coastline; it means no
        # relief, which is what nan_to_num turns it into.
        warnings.simplefilter("ignore", RuntimeWarning)
        rel = (np.nanmax(win, axis=(2, 3)) - np.nanmin(win, axis=(2, 3)))
    rel = np.nan_to_num(rel, nan=0.0)

    mountain = ok & (z >= threshold_m) & (rel >= local_relief_m)
    vals = z[ok]
    high = mountain[ok]

    # Cell area varies with latitude; the grid is regular in degrees.
    rows = np.nonzero(ok)[0]
    lat = tr.f + (rows + 0.5) * tr.e
    dx_km = abs(tr.a) * 111.32 * np.cos(np.radians(lat))
    dy_km = abs(tr.e) * 110.54
    area_km2 = float((dx_km * dy_km)[high].sum())

    return {"mountain_fraction": round(float(high.mean()), 4),
            "mountain_area_km2": round(area_km2, 1),
            "max_elevation_m": round(float(vals.max()), 1),
            "median_elevation_m": round(float(np.median(vals)), 1),
            "median_local_relief_m": round(float(np.median(rel[ok])), 1),
            "cells": int(ok.sum())}


def load_units(shp_path: str, region_bbox: Sequence[float],
               countries: Optional[Sequence[str]] = None,
               names: Optional[Sequence[str]] = None,
               min_span_deg: float = 0.02,
               elevation_path: Optional[str] = None,
               min_mountain_fraction: float = DEFAULT_MOUNTAIN_FRACTION,
               mountain_elevation_m: float = DEFAULT_MOUNTAIN_ELEVATION_M,
               local_relief_m: float = DEFAULT_LOCAL_RELIEF_M,
               min_mountain_area_km2: float = DEFAULT_MOUNTAIN_AREA_KM2,
               min_mountain_peak_m: float = DEFAULT_MOUNTAIN_PEAK_M,
               exclude: Optional[Sequence[str]] = None) -> List[AdminUnit]:
    """Read admin-1 polygons and keep the ones actually in the mountains.

    A unit's bounding box is intersected with ``region_bbox``, so a province
    straddling the edge is run only over the part inside the study region and
    units entirely outside are dropped. ``min_span_deg`` discards slivers -
    city-states and offshore fragments - that are too small to route.

    **A bounding box is not a mountain range.** The Hindu Kush Himalaya box
    spans 60-105 E and 16-39 N, which contains the whole Gangetic plain and
    most of peninsular India: selecting on it alone returns Odisha, Madhya
    Pradesh, Telangana and Andhra Pradesh, none of which have a Himalayan
    hillslope in them. When ``elevation_path`` is given, each unit is tested
    against a coarse global elevation raster and kept only if at least
    ``min_mountain_fraction`` of it sits above ``mountain_elevation_m``.

    Units that pass carry their relief figures in ``.relief``, so a run can
    report why each province was included.
    """
    import fiona
    from rasterio.warp import transform_geom

    want_c = {c.lower() for c in countries} if countries else None
    want_n = {n.lower() for n in names} if names else None
    skip = {x.lower() for x in exclude} if exclude else set()

    units: List[AdminUnit] = []
    with fiona.open(shp_path) as src:
        crs = src.crs or "EPSG:4326"
        try:
            import rasterio.crs
            same = rasterio.crs.CRS.from_user_input(crs).to_epsg() == 4326
        except Exception:                                # noqa: BLE001
            same = True
        for feat in src:
            props = feat["properties"]
            name = _first(props, _NAME_FIELDS)
            country = _first(props, _COUNTRY_FIELDS)
            if not name:
                continue
            # A layer that carries no country attribute at all - somebody's
            # own province shapefile, passed as admin_path - must not be
            # silently emptied by a country filter it cannot answer.
            if want_c and country and country.lower() not in want_c:
                continue
            if want_n and name.lower() not in want_n:
                continue
            if name.lower() in skip:
                continue
            geom = feat["geometry"]
            if geom is None:
                continue
            if not same:
                geom = transform_geom(crs, "EPSG:4326", geom)

            bounds = _geom_bounds(geom)
            if bounds is None:
                continue
            clipped = _intersect(bounds, region_bbox)
            if clipped is None:
                continue
            if (clipped[2] - clipped[0] < min_span_deg
                    or clipped[3] - clipped[1] < min_span_deg):
                continue
            units.append(AdminUnit(name=name, country=country, bbox=clipped,
                                   geometry=geom))

    if elevation_path and os.path.exists(elevation_path):
        kept = []
        for u in units:
            u.relief = relief_stats(u, elevation_path, mountain_elevation_m,
                                    local_relief_m)
            if is_mountainous(u.relief, min_mountain_fraction,
                              min_mountain_area_km2, min_mountain_peak_m):
                kept.append(u)
        units = kept

    units.sort(key=lambda u: (u.country, u.name))
    return units


def _geom_bounds(geom: dict) -> Optional[Tuple[float, float, float, float]]:
    """Bounding box of a GeoJSON geometry, without pulling in shapely."""
    xs: List[float] = []
    ys: List[float] = []

    def walk(coords) -> None:
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            xs.append(float(coords[0])); ys.append(float(coords[1]))
            return
        for c in coords:
            walk(c)

    walk(geom.get("coordinates"))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def unit_mask(unit: AdminUnit, grid) -> "object":
    """Boolean array, True inside the unit, on ``grid``.

    Used to clip a buffered run back to the province it belongs to, so
    neighbouring provinces do not each claim the same border cells.
    """
    import numpy as np
    from rasterio.features import rasterize

    arr = rasterize([(unit.geometry, 1)], out_shape=grid.shape,
                    transform=grid.transform, fill=0, dtype="uint8")
    return arr.astype(bool) if arr is not None else np.ones(grid.shape, bool)


def summarise(units: Sequence[AdminUnit], resolution_deg: float,
              buffer_deg: float = 0.0) -> List[Dict[str, object]]:
    """Per-unit cost table, for deciding what to run and at what resolution."""
    rows = []
    for u in units:
        dw, dh = u.span_deg()
        rows.append({"slug": u.slug, "country": u.country, "name": u.name,
                     "bbox": list(u.bbox),
                     "span_deg": [round(dw, 3), round(dh, 3)],
                     "cells": u.cell_count(resolution_deg, buffer_deg),
                     "mountain_fraction": u.relief.get("mountain_fraction"),
                     "mountain_area_km2": u.relief.get("mountain_area_km2"),
                     "max_elevation_m": u.relief.get("max_elevation_m")})
    rows.sort(key=lambda r: -r["cells"])
    return rows
