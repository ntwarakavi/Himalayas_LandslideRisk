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
        return {"name": self.name, "country": self.country,
                "slug": self.slug, "bbox": list(self.bbox)}


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


def load_units(shp_path: str, region_bbox: Sequence[float],
               countries: Optional[Sequence[str]] = None,
               names: Optional[Sequence[str]] = None,
               min_span_deg: float = 0.02) -> List[AdminUnit]:
    """Read admin-1 polygons, keep those inside the region, clip their boxes.

    A unit's bounding box is intersected with ``region_bbox``, so a province
    straddling the edge is run only over the part inside the study region and
    units entirely outside are dropped. ``min_span_deg`` discards slivers -
    city-states and offshore fragments - that are too small to route.
    """
    import fiona
    from rasterio.warp import transform_geom

    want_c = {c.lower() for c in countries} if countries else None
    want_n = {n.lower() for n in names} if names else None

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
            if want_c and country.lower() not in want_c:
                continue
            if want_n and name.lower() not in want_n:
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
                     "cells": u.cell_count(resolution_deg, buffer_deg)})
    rows.sort(key=lambda r: -r["cells"])
    return rows
